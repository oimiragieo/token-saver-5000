"""Phase 0 observability validation for handler-side metrics and warnings."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import networkx as nx
import pytest

from src.handlers import compression_handlers
from src.observability import ObservabilityManager


def _make_ingest_context():
    skeleton = SimpleNamespace(
        file_id="phase0_doc",
        total_nodes=3,
        total_tokens=120,
        skeleton_tokens=40,
        compression_ratio=3.0,
        skeleton_text="Observed skeleton",
    )
    return skeleton, {
        "compressor": Mock(
            ingest_file_async=AsyncMock(return_value=skeleton),
            graphs={"phase0_doc": nx.Graph()},
            file_metadata={},
            chunks={},
        ),
        "resource_manager": Mock(
            check_document_size_async=AsyncMock(return_value=(True, "")),
            register_document_async=AsyncMock(),
        ),
        "sync_manager": Mock(export_metadata=Mock(return_value=[]), register_file=Mock()),
        "version_manager": Mock(add_version_async=AsyncMock()),
        "persistence": Mock(
            save_document=Mock(return_value=True), save_file_sync_metadata=Mock(return_value=True)
        ),
        "retrieval_history": {},
        "path_validator": Mock(),
    }


@pytest.mark.asyncio
async def test_handle_ingest_logs_metrics_warning_on_metrics_failure(caplog):
    skeleton, context = _make_ingest_context()

    with patch("src.handlers.compression_handlers_ingest.CompressionAdvisor") as advisor_cls:
        advisor = Mock()
        advisor.estimate_compression.return_value = SimpleNamespace(
            compression_ratio=2.9, original_tokens=skeleton.total_tokens, estimated_compressed=41
        )
        advisor_cls.return_value = advisor

        with patch("src.handlers.compression_handlers_ingest.get_metrics") as get_metrics:
            get_metrics.return_value = Mock(
                record_compression_ratio=Mock(side_effect=RuntimeError("metrics down")),
                increment_documents_processed=Mock(),
                set_active_documents=Mock(),
            )
            with caplog.at_level(logging.WARNING, logger="semantic-modulator"):
                payload = json.loads(
                    await compression_handlers.handle_ingest(
                        context,
                        {
                            "text": "This document is long enough for observability characterization.",
                            "file_id": "phase0_doc",
                        },
                    )
                )

    assert payload["status"] == "success"
    assert "Metrics recording failed for 'phase0_doc': metrics down" in caplog.text


@pytest.mark.asyncio
async def test_handle_ingest_logs_persistence_error_but_returns_success(caplog):
    _, context = _make_ingest_context()
    context["persistence"].save_document.side_effect = RuntimeError("disk unavailable")

    with patch("src.handlers.compression_handlers_ingest.CompressionAdvisor") as advisor_cls:
        advisor = Mock()
        advisor.estimate_compression.return_value = SimpleNamespace(
            compression_ratio=2.9, original_tokens=120, estimated_compressed=41
        )
        advisor_cls.return_value = advisor

        with caplog.at_level(logging.ERROR, logger="semantic-modulator"):
            payload = json.loads(
                await compression_handlers.handle_ingest(
                    context,
                    {
                        "text": "This document is long enough for persistence error characterization.",
                        "file_id": "phase0_doc",
                    },
                )
            )

    assert payload["status"] == "success"
    assert "Failed to persist phase0_doc: disk unavailable" in caplog.text


@pytest.mark.asyncio
async def test_handle_search_semantic_logs_access_tracking_warning(caplog):
    compressor = Mock(
        search_semantic_with_scores=Mock(return_value=[("phase0_doc_0", 0.88)]),
        chunks={
            "phase0_doc_0": SimpleNamespace(
                text="Temporal context retrieval keeps scoped memory relevant.",
                importance=0.63,
                metadata={"tokens": 11},
            )
        },
        _generate_summary=Mock(return_value="Scoped memory summary"),
        _access_tracker=SimpleNamespace(
            record_access=Mock(side_effect=RuntimeError("tracker offline"))
        ),
    )

    with caplog.at_level(logging.WARNING, logger="semantic-modulator"):
        payload = json.loads(
            await compression_handlers.handle_search_semantic(
                {"compressor": compressor},
                {"query": "scoped memory", "file_id": "phase0_doc", "top_k": 1},
            )
        )

    assert payload["total_results"] == 1
    assert "Access tracking failed during semantic search: tracker offline" in caplog.text


def test_observability_defaults_to_local_only_export_in_development():
    original_instance = ObservabilityManager._instance
    ObservabilityManager._instance = None
    manager = None
    try:
        manager = ObservabilityManager(environment="development")
        assert manager.enable_otlp_export is False
    finally:
        if manager is not None and manager.is_enabled():
            manager.shutdown()
        ObservabilityManager._instance = original_instance
