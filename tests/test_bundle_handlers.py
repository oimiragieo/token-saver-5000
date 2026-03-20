import json
from types import SimpleNamespace

import pytest

from src.handlers.bundle_handlers import (
    handle_create_handoff_bundle,
    handle_get_handoff_bundle,
    handle_list_handoff_bundles,
    handle_replay_handoff_bundle,
)
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


@pytest.fixture(autouse=True)
def reset_registry():
    HandoffBundleRegistry.reset_singleton()


@pytest.fixture
def handler_context():
    return {"compressor": _FakeCompressor()}


@pytest.mark.asyncio
async def test_create_and_fetch_handoff_bundle(handler_context):
    created = json.loads(
        await handle_create_handoff_bundle(
            handler_context,
            {
                "file_id": "auth_doc",
                "workspace_id": "acme",
                "query": "authentication flow",
                "bundle_id": "bundle-1",
                "created_at": "2026-03-16T07:00:00Z",
            },
        )
    )

    assert created["status"] == "success"
    assert created["bundle"]["bundle_id"] == "bundle-1"
    assert created["bundle"]["doc_id"] == "auth_doc"

    fetched = json.loads(
        await handle_get_handoff_bundle(
            handler_context,
            {"bundle_id": "bundle-1", "workspace_id": "acme"},
        )
    )
    assert fetched["bundle"]["artifact"]["summary"].startswith("Distilled handoff")


@pytest.mark.asyncio
async def test_list_and_replay_handoff_bundle(handler_context):
    await handle_create_handoff_bundle(
        handler_context,
        {
            "file_id": "auth_doc",
            "workspace_id": "acme",
            "query": "authentication flow",
            "bundle_id": "bundle-1",
            "created_at": "2026-03-16T07:00:00Z",
        },
    )

    listing = json.loads(
        await handle_list_handoff_bundles(handler_context, {"workspace_id": "acme"})
    )
    assert listing["total_bundles"] == 1

    replay = json.loads(
        await handle_replay_handoff_bundle(
            handler_context,
            {"bundle_id": "bundle-1", "workspace_id": "acme"},
        )
    )
    assert replay["replay"]["bundle_id"] == "bundle-1"
    assert "AUTH SKELETON" in replay["replay"]["replay_text"]
