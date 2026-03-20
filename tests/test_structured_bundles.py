import pytest

from src.evidence_bundle import EvidenceBundle, reset_evidence_store
from src.structured_bundles import HandoffBundleRegistry


def _sample_evidence() -> EvidenceBundle:
    return EvidenceBundle.create(
        operation="handoff_bundle",
        input_data="input context",
        output_data="distilled replay",
        input_token_count=100,
        output_token_count=25,
        parameters={"selection_mode": "query_guided"},
    )


def _sample_artifact() -> dict:
    return {
        "bundle_version": "1.0",
        "doc_id": "auth_doc",
        "query": "authentication flow",
        "summary": "Distilled authentication handoff",
        "skeleton": {
            "text": "AUTH SKELETON",
            "total_nodes": 4,
            "skeleton_tokens": 20,
            "compression_ratio": 5.0,
            "node_map": {"auth_doc_n0": "ANCHOR: Login flow"},
        },
        "search_results": [
            {
                "node_id": "auth_doc_n0",
                "similarity": 0.98,
                "importance": 0.74,
                "summary": "Login flow anchor",
            }
        ],
        "context_block": {
            "summary": "2 active facts; 1 recent event",
            "active_facts": [{"fact_id": "auth_doc_n0"}],
            "recent_events": [],
            "cache_stable_prefix": "AUTH SKELETON",
        },
        "replay_text": "Replay this bundle to understand authentication flow.",
    }


def setup_function():
    HandoffBundleRegistry.reset_singleton()
    reset_evidence_store()


def test_registry_stores_and_lists_scoped_bundles():
    registry = HandoffBundleRegistry.get_registry()
    evidence = _sample_evidence()

    created = registry.create_bundle(
        bundle_id="bundle-1",
        scoped_doc_id="scope__w=acme__f=auth_doc",
        artifact=_sample_artifact(),
        replay_text="Replay this bundle",
        evidence_bundle=evidence,
        metadata={"owner": "team-auth"},
        created_at="2026-03-16T07:00:00Z",
    )

    assert created["bundle_id"] == "bundle-1"
    assert created["doc_id"] == "auth_doc"
    assert created["audit"]["bundle_hash"] == evidence.bundle_hash

    listing = registry.list_bundles(workspace_id="acme")
    assert len(listing) == 1
    assert listing[0]["bundle_id"] == "bundle-1"
    assert listing[0]["doc_id"] == "auth_doc"


def test_registry_rejects_scope_mismatch_on_get():
    registry = HandoffBundleRegistry.get_registry()
    registry.create_bundle(
        bundle_id="bundle-1",
        scoped_doc_id="scope__w=acme__f=auth_doc",
        artifact=_sample_artifact(),
        replay_text="Replay this bundle",
        evidence_bundle=_sample_evidence(),
        created_at="2026-03-16T07:00:00Z",
    )

    with pytest.raises(ValueError, match="not found"):
        registry.get_bundle("bundle-1", workspace_id="other")


def test_registry_replays_bundle_in_requested_format():
    registry = HandoffBundleRegistry.get_registry()
    registry.create_bundle(
        bundle_id="bundle-1",
        scoped_doc_id="auth_doc",
        artifact=_sample_artifact(),
        replay_text="Replay this bundle",
        evidence_bundle=_sample_evidence(),
        created_at="2026-03-16T07:00:00Z",
    )

    replay = registry.replay_bundle("bundle-1")

    assert replay["bundle_id"] == "bundle-1"
    assert replay["replay_text"] == "Replay this bundle"
    assert "artifact_toon" in replay
    assert replay["artifact"]["doc_id"] == "auth_doc"
