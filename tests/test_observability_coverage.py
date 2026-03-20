"""Coverage tests for handler-level observability spans."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.connector_registry import ConnectorRegistry
from src.memory_api import MemoryAPI
from src.structured_bundles import HandoffBundleRegistry


class _FakeTracker:
    def get_access_info(self, doc_id):
        return {"first_accessed": 1.0, "last_accessed": 2.0, "access_count": 3}


class _FakeReplay:
    def get_history(self, doc_id):
        return [{"ratio": 5.0}]


class _FakeTemporal:
    def get_active_facts(self, doc_id, as_of=None, include_invalidated=False):
        return [{"fact_id": f"{doc_id}_n0", "summary": "login flow", "active": True}]

    def search_timeline(self, **kwargs):
        return [{"event_type": "document_ingested", "timestamp": "2026-03-16T07:00:00Z"}]


class _FakeCompressor:
    def __init__(self):
        self.graphs = {"scope__w=acme__f=auth_doc": object()}
        self.chunks = {
            "scope__w=acme__f=auth_doc_n0": SimpleNamespace(
                importance=0.71,
                text="Login flow handles credentials and session issuance.",
                metadata={"tokens": 12},
            )
        }
        self._access_tracker = _FakeTracker()
        self._compression_replay = _FakeReplay()
        self._temporal_graph = _FakeTemporal()

    def _generate_skeleton(self, file_id, query=None, exclude_node_ids=None):
        return SimpleNamespace(
            file_id=file_id,
            total_nodes=4,
            total_tokens=120,
            skeleton_tokens=24,
            compression_ratio=5.0,
            skeleton_text="AUTH SKELETON",
            node_map={f"{file_id}_n0": "ANCHOR: Login flow"},
        )

    def search_semantic_with_scores(self, query, file_id=None, top_k=5):
        return [(f"{file_id}_n0", 0.981)]

    def _generate_summary(self, text, max_length=100):
        return text[:max_length]

    def _count_tokens(self, text):
        return max(1, len(text.split()))


@pytest.fixture
def memory_context():
    MemoryAPI.reset_singleton()
    return {"memory_api": MemoryAPI()}


@pytest.fixture
def connector_context():
    ConnectorRegistry.reset_singleton()
    compressor = Mock()
    compressor.graphs = {"scope__w=acme__f=docs-web%3Aweb-example.com-docs": object()}
    compressor.file_metadata = {}
    compressor.chunks = {}
    compressor.ingest_file_async = AsyncMock(
        return_value=SimpleNamespace(total_tokens=100, skeleton_tokens=20, compression_ratio=5.0)
    )
    return {
        "connector_registry": ConnectorRegistry.get_registry(),
        "compressor": compressor,
        "resource_manager": Mock(
            check_connector_batch_async=AsyncMock(return_value=(True, None)),
            check_document_size_async=AsyncMock(return_value=(True, None)),
            register_document_async=AsyncMock(),
        ),
        "sync_manager": Mock(register_file=Mock(), export_metadata=Mock(return_value={})),
        "version_manager": Mock(add_version_async=AsyncMock()),
        "persistence": Mock(
            save_document=Mock(return_value=True), save_file_sync_metadata=Mock(return_value=True)
        ),
        "retrieval_history": {},
    }


@pytest.fixture(autouse=True)
def reset_bundle_registry():
    HandoffBundleRegistry.reset_singleton()


def _observe_mock():
    observe = Mock()
    observe.trace.return_value.__enter__ = Mock(return_value=Mock())
    observe.trace.return_value.__exit__ = Mock(return_value=None)
    return observe


@pytest.mark.asyncio
async def test_memory_handlers_emit_trace(memory_context):
    from src.handlers.memory_handlers import handle_add_memory

    observe = _observe_mock()
    with patch("src.handlers.memory_handlers.get_observability", return_value=observe):
        payload = json.loads(
            await handle_add_memory(
                memory_context,
                {"text": "Prefer pytest fixtures", "workspace_id": "acme", "user_id": "alice"},
            )
        )

    assert payload["status"] == "success"
    observe.trace.assert_called_once()
    assert observe.trace.call_args.args[0] == "memory_api.add"


@pytest.mark.asyncio
async def test_connector_handlers_emit_trace(connector_context):
    from src.handlers.connector_handlers import handle_create_connector_feed

    observe = _observe_mock()
    with patch("src.handlers.connector_handlers.get_observability", return_value=observe):
        payload = json.loads(
            await handle_create_connector_feed(
                connector_context,
                {
                    "name": "docs-web",
                    "connector_type": "web",
                    "config": {
                        "pages": [{"url": "https://example.com/docs", "content": "Docs body"}]
                    },
                },
            )
        )

    assert payload["status"] == "success"
    assert observe.trace.call_args.args[0] == "connector_registry.create_feed"


@pytest.mark.asyncio
async def test_bundle_handlers_emit_trace():
    from src.handlers.bundle_handlers import handle_create_handoff_bundle

    observe = _observe_mock()
    context = {"compressor": _FakeCompressor()}
    with patch("src.handlers.bundle_handlers.get_observability", return_value=observe):
        payload = json.loads(
            await handle_create_handoff_bundle(
                context,
                {
                    "file_id": "auth_doc",
                    "workspace_id": "acme",
                    "query": "authentication flow",
                    "bundle_id": "bundle-1",
                    "created_at": "2026-03-16T07:00:00Z",
                },
            )
        )

    assert payload["status"] == "success"
    assert observe.trace.call_args.args[0] == "handoff_bundle.create"


@pytest.mark.asyncio
async def test_model_handlers_emit_trace():
    from src.handlers.model_handlers import handle_estimate_model_cost

    observe = _observe_mock()
    with patch("src.handlers.model_handlers.get_observability", return_value=observe):
        payload = json.loads(
            await handle_estimate_model_cost(
                {},
                {
                    "model": "gpt-5.4",
                    "original_tokens": 100_000,
                    "compressed_tokens": 20_000,
                },
            )
        )

    assert payload["status"] == "success"
    assert observe.trace.call_args.args[0] == "model_optimization.estimate_cost"
