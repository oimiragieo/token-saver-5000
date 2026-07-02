"""World-Class Compression Sprint — Batch 1 regression locks (2026-07-01).

Task 1 (council-blessed ship-first, codex gpt-5.5 + droid glm-5.1): the MCP
``ingest_context`` response must surface the honest estimate the advisor already
computes — ``estimated_compressed`` / ``reasoning`` / ``confidence`` — AND a
small-doc "too small to compress" ``note`` (the REST path already emits one at
``api/app/routers/v1/compress.py``).

Ground-truth (2026-07-01, direct read): pre-fix the response carried only
``estimate.{estimated_ratio, accuracy}`` (compression_handlers.py:700) and no
``note``; ``estimate.reasoning``/``confidence`` were computed at :598 then logged
and dropped. This is the activation gap behind a connected user who compressed
nothing — the pre-flight signal that says "is this worth it?" never reached them.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.handlers import compression_handlers as ch


def _make_context():
    compressor = Mock()
    compressor.graphs = {}
    compressor.chunks = {}
    compressor.file_metadata = {}
    resource_manager = Mock()
    resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
    resource_manager.register_document_async = AsyncMock()
    version_manager = Mock()
    version_manager.add_version_async = AsyncMock()
    persistence = Mock()
    persistence.save_document.return_value = True
    persistence.save_file_sync_metadata.return_value = True
    sync_manager = Mock()
    sync_manager.export_metadata.return_value = []
    context = {
        "compressor": compressor,
        "persistence": persistence,
        "resource_manager": resource_manager,
        "sync_manager": sync_manager,
        "version_manager": version_manager,
        "retrieval_history": {},
    }
    return context, compressor


def _make_skeleton(total_tokens: int, skeleton_tokens: int) -> Mock:
    sk = Mock()
    sk.total_nodes = 5
    sk.total_tokens = total_tokens
    sk.skeleton_tokens = skeleton_tokens
    sk.compression_ratio = (total_tokens / skeleton_tokens) if skeleton_tokens else 1.0
    sk.skeleton_text = "Mock skeleton"
    return sk


def _make_estimate(
    *, ratio=1.1, compressed=190, confidence="low", reasoning="Overhead dominates."
) -> Mock:
    est = Mock()
    est.compression_ratio = ratio
    est.original_tokens = 200
    est.estimated_compressed = compressed
    est.confidence = confidence
    est.reasoning = reasoning
    return est


@pytest.mark.asyncio
async def test_ingest_surfaces_reasoning_confidence_and_estimated_compressed():
    """A large doc must return the enriched estimate + note=None."""
    context, compressor = _make_context()
    compressor.ingest_file_async = AsyncMock(return_value=_make_skeleton(1000, 100))
    args = {"text": "x" * 4000, "file_id": "wc_estimate"}
    with (
        patch.object(ch, "CompressionAdvisor") as MockAdvisor,
        patch.object(ch, "validate_file_id"),
    ):
        MockAdvisor.return_value.estimate_compression.return_value = _make_estimate(
            ratio=9.5, compressed=105, confidence="high", reasoning="Large structured doc."
        )
        result = await ch.handle_ingest(context, args)
    data = json.loads(result)
    assert data["estimate"]["reasoning"] == "Large structured doc."
    assert data["estimate"]["confidence"] == "high"
    assert data["estimate"]["estimated_compressed"] == 105
    assert data["note"] is None  # 90% savings on a 1000-token doc → no small-doc note


@pytest.mark.asyncio
async def test_ingest_small_doc_returns_honesty_note():
    """A tiny input that expands must return a truthful small-doc note (not silent growth)."""
    context, compressor = _make_context()
    # 60-token input yielding a 90-token skeleton = negative savings (expansion).
    compressor.ingest_file_async = AsyncMock(return_value=_make_skeleton(60, 90))
    args = {"text": "Small snippet that is over twenty characters long.", "file_id": "wc_small"}
    with (
        patch.object(ch, "CompressionAdvisor") as MockAdvisor,
        patch.object(ch, "validate_file_id"),
    ):
        MockAdvisor.return_value.estimate_compression.return_value = _make_estimate()
        result = await ch.handle_ingest(context, args)
    data = json.loads(result)
    assert data["note"] is not None
    assert "too small" in data["note"].lower()
    assert data["token_savings_percent"] <= 0
