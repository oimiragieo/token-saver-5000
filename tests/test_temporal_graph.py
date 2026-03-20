"""Tests for temporal fact lifecycle tracking."""

from src.temporal_graph import TemporalGraph


def test_record_document_state_and_invalidation():
    graph = TemporalGraph()
    graph.record_document_state(
        "doc1",
        [
            {"fact_id": "doc1_n0", "content": "alpha"},
            {"fact_id": "doc1_n1", "content": "beta"},
        ],
        timestamp=100.0,
    )
    graph.invalidate_fact("doc1_n0", reason="stale", timestamp=200.0)

    active_facts = graph.get_active_facts("doc1", as_of=300.0)
    history = graph.list_fact_history(doc_id="doc1", as_of=300.0)

    assert {fact["fact_id"] for fact in active_facts} == {"doc1_n1"}
    assert any(
        item["fact_id"] == "doc1_n0" and item["invalidation_reason"] == "stale" for item in history
    )


def test_reingest_invalidates_removed_facts():
    graph = TemporalGraph()
    graph.record_document_state(
        "doc2",
        [
            {"fact_id": "doc2_n0", "content": "old alpha"},
            {"fact_id": "doc2_n1", "content": "old beta"},
        ],
        timestamp=100.0,
    )
    graph.record_document_state(
        "doc2",
        [{"fact_id": "doc2_n1", "content": "old beta"}],
        timestamp=150.0,
    )

    assert graph.is_fact_active("doc2_n1", as_of=200.0) is True
    assert graph.is_fact_active("doc2_n0", as_of=200.0) is False


def test_search_timeline_filters_by_query():
    graph = TemporalGraph()
    graph.record_event(
        "document_ingested", doc_id="doc3", summary="Captured auth policy", timestamp=100.0
    )
    graph.record_event(
        "search_semantic", doc_id="doc3", summary="billing question", timestamp=120.0
    )

    events = graph.search_timeline(query="auth", doc_id="doc3")

    assert len(events) == 1
    assert events[0]["event_type"] == "document_ingested"
