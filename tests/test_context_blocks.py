"""Tests for lifecycle-aware context block assembly."""

from src.context_blocks import build_context_block


def test_build_context_block_summarizes_access_and_compression():
    block = build_context_block(
        doc_id="doc1",
        active_facts=[{"fact_id": "doc1_n0", "content": "alpha", "active": True}],
        recent_events=[{"event_type": "read_skeleton", "timestamp": "2026-01-01T00:00:00Z"}],
        access_info={"first_accessed": 100.0, "last_accessed": 200.0, "access_count": 3},
        compression_history=[{"ratio": 4.5}],
        skeleton_text="stable prefix",
        max_facts=5,
    )

    assert block["doc_id"] == "doc1"
    assert block["active_fact_count"] == 1
    assert block["compression"]["latest_ratio"] == 4.5
    assert "active facts" in block["summary"]
