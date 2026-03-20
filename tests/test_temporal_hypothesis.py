"""Property tests for temporal fact lifecycle semantics."""

from hypothesis import given, strategies as st

from src.temporal_graph import TemporalGraph


@given(
    observed_at=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    delta=st.floats(min_value=0.1, max_value=1_000.0, allow_nan=False, allow_infinity=False),
)
def test_invalidated_facts_are_visible_before_but_not_after(observed_at, delta):
    graph = TemporalGraph()
    invalidated_at = observed_at + delta

    graph.record_document_state(
        "doc",
        [{"fact_id": "doc_n0", "content": "alpha"}],
        timestamp=observed_at,
    )
    graph.invalidate_fact("doc_n0", reason="stale", timestamp=invalidated_at)

    assert graph.is_fact_active("doc_n0", as_of=observed_at + (delta / 2)) is True
    assert graph.is_fact_active("doc_n0", as_of=invalidated_at + 0.001) is False
