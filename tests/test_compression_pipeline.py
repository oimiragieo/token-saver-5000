"""Tests for the composable read-skeleton compression pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.compression_pipeline import run_read_skeleton_pipeline


def _skeleton(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        file_id="doc1",
        total_nodes=5,
        total_tokens=100,
        skeleton_tokens=20,
        compression_ratio=5.0,
        skeleton_text=f"{name} skeleton",
        node_map={"doc1_n0": name},
    )


def test_read_skeleton_pipeline_runs_baseline_then_evidence_stages():
    compressor = Mock()
    compressor._generate_skeleton.side_effect = [
        _skeleton("baseline"),
        _skeleton("query_guided"),
        _skeleton("evidence_aware"),
    ]
    compressor.retrieve_evidence.return_value = SimpleNamespace(
        sufficient=True,
        best_score=0.9,
        threshold=0.35,
        used_expanded_search=False,
        message="ok",
        node_ids=["doc1_n1"],
    )

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="evidence_aware",
        query="retry behavior",
        top_k=2,
        min_similarity=0.4,
        excluded_node_ids=set(),
    )

    assert [stage["name"] for stage in result["stages"]] == [
        "baseline",
        "query_guided",
        "evidence_aware",
    ]
    assert result["final_stage"] == "evidence_aware"
    assert result["evidence"]["sufficient"] is True


# ---------------------------------------------------------------------------
# F3: auto selection_mode tests
# ---------------------------------------------------------------------------

_STRUCTURED_DOC = """\
# Security Audit Report: Example Service

## Authentication Findings

1. Missing rate-limiting on /login endpoint
2. JWT tokens have no expiry enforcement
3. CSRF token not validated on state-changing routes

## Verdict

The service has CRITICAL vulnerabilities that must be fixed before launch.

## Authorization Findings

4. Role check bypassed when user_id is null
5. Admin panel accessible without elevated session
6. Tenant isolation missing on shared DB queries

## Conclusion

Multiple HIGH severity issues confirmed.
"""

_PLAIN_DOC = """\
This document describes the general approach to building web services.
We should think carefully about architecture and keep things simple.
Maintainability matters more than premature optimisation in most cases.
"""


def _make_compressor_for_auto(skeleton_results):
    """Return a Mock compressor whose _generate_skeleton cycles through the given skeletons."""
    compressor = Mock()
    compressor._generate_skeleton.side_effect = list(skeleton_results)
    compressor.retrieve_evidence.return_value = SimpleNamespace(
        sufficient=True,
        best_score=0.85,
        threshold=0.35,
        used_expanded_search=False,
        message="ok",
        node_ids=["doc1_n2"],
    )
    return compressor


def test_auto_mode_resolves_to_evidence_aware_for_structured_doc():
    """auto mode detects 3+ H2 + 3+ findings + verdict → evidence_aware."""
    compressor = _make_compressor_for_auto(
        [_skeleton("baseline"), _skeleton("query_guided"), _skeleton("evidence_aware")]
    )

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query=None,
        top_k=2,
        min_similarity=0.35,
        raw_text=_STRUCTURED_DOC,
    )

    assert result["final_stage"] == "evidence_aware"
    assert result["selection_mode_resolved"] == "auto-detected: evidence_aware"
    # All three stages must have run
    assert [s["name"] for s in result["stages"]] == [
        "baseline",
        "query_guided",
        "evidence_aware",
    ]


def test_auto_mode_resolves_to_baseline_for_plain_prose():
    """auto mode falls back to baseline for plain prose docs."""
    compressor = _make_compressor_for_auto([_skeleton("baseline")])

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query=None,
        top_k=2,
        min_similarity=0.35,
        raw_text=_PLAIN_DOC,
    )

    assert result["final_stage"] == "baseline"
    assert result["selection_mode_resolved"] == "auto-detected: baseline"
    assert len(result["stages"]) == 1


def test_auto_mode_resolves_to_baseline_when_raw_text_is_none():
    """auto mode with no raw_text falls back to baseline without error."""
    compressor = _make_compressor_for_auto([_skeleton("baseline")])

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query=None,
        top_k=2,
        min_similarity=0.35,
        raw_text=None,
    )

    assert result["final_stage"] == "baseline"
    assert result["selection_mode_resolved"] == "auto-detected: baseline"


def test_auto_mode_synthesises_h1_as_query_when_no_query_supplied():
    """auto mode in evidence_aware path uses the H1 heading as synthetic query."""
    compressor = _make_compressor_for_auto(
        [_skeleton("baseline"), _skeleton("query_guided"), _skeleton("evidence_aware")]
    )

    run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query=None,
        top_k=2,
        min_similarity=0.35,
        raw_text=_STRUCTURED_DOC,
    )

    # The second and third _generate_skeleton calls (query_guided + evidence_aware)
    # must carry the synthetic H1 query.
    calls = compressor._generate_skeleton.call_args_list
    # calls[1] is query_guided; calls[2] is evidence_aware
    for call in calls[1:]:
        _, kwargs = call
        assert "query" in kwargs
        assert kwargs["query"] == "Security Audit Report: Example Service"


def test_auto_mode_uses_explicit_query_when_supplied():
    """If caller passes an explicit query, auto mode must honour it over H1 synthesis."""
    compressor = _make_compressor_for_auto(
        [_skeleton("baseline"), _skeleton("query_guided"), _skeleton("evidence_aware")]
    )

    run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query="custom caller query",
        top_k=2,
        min_similarity=0.35,
        raw_text=_STRUCTURED_DOC,
    )

    calls = compressor._generate_skeleton.call_args_list
    for call in calls[1:]:
        _, kwargs = call
        assert kwargs["query"] == "custom caller query"


def test_selection_mode_resolved_present_in_baseline_path():
    """selection_mode_resolved is always included, even for baseline mode."""
    compressor = _make_compressor_for_auto([_skeleton("baseline")])

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="baseline",
        query=None,
        top_k=2,
        min_similarity=0.35,
    )

    assert "selection_mode_resolved" in result
    assert result["selection_mode_resolved"] == "baseline"


def test_selection_mode_resolved_present_in_query_guided_path():
    """selection_mode_resolved is always included for query_guided mode."""
    compressor = _make_compressor_for_auto([_skeleton("baseline"), _skeleton("query_guided")])

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="query_guided",
        query="some query",
        top_k=2,
        min_similarity=0.35,
    )

    assert "selection_mode_resolved" in result
    assert result["selection_mode_resolved"] == "query_guided"


# ---------------------------------------------------------------------------
# _resolve_auto_mode and _extract_h1_query unit tests
# ---------------------------------------------------------------------------


def test_resolve_auto_mode_detects_structured_doc():
    from src.compression_pipeline import _resolve_auto_mode

    mode, label = _resolve_auto_mode(_STRUCTURED_DOC)
    assert mode == "evidence_aware"
    assert "evidence_aware" in label


def test_resolve_auto_mode_falls_back_for_plain_prose():
    from src.compression_pipeline import _resolve_auto_mode

    mode, label = _resolve_auto_mode(_PLAIN_DOC)
    assert mode == "baseline"
    assert "baseline" in label


def test_resolve_auto_mode_returns_baseline_for_empty_text():
    from src.compression_pipeline import _resolve_auto_mode

    mode, label = _resolve_auto_mode("")
    assert mode == "baseline"

    mode2, label2 = _resolve_auto_mode(None)
    assert mode2 == "baseline"


def test_extract_h1_query_returns_heading_text():
    from src.compression_pipeline import _extract_h1_query

    text = "# My Report Title\n\nSome body text."
    assert _extract_h1_query(text) == "My Report Title"


def test_extract_h1_query_returns_none_when_no_h1():
    from src.compression_pipeline import _extract_h1_query

    assert _extract_h1_query("## Only H2 here\n\ntext.") is None
    assert _extract_h1_query("") is None
    assert _extract_h1_query(None) is None


def test_auto_mode_with_caller_query_on_prose_resolves_to_query_guided():
    """#92 (2026-06-12): auto + a CALLER-SUPPLIED query on plain prose must
    honor the query (query_guided). Pre-fix, auto resolved to baseline and
    silently ignored the query, returning all nodes (dogfood + codex
    production find)."""
    compressor = _make_compressor_for_auto([_skeleton("baseline"), _skeleton("query_guided")])

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query="where is the budget loop?",
        top_k=2,
        min_similarity=0.35,
        raw_text=_PLAIN_DOC,
    )

    assert result["final_stage"] == "query_guided"
    assert result["selection_mode_resolved"] == "auto-resolved: query_guided (caller query honored)"


def test_auto_mode_with_caller_query_on_structured_doc_stays_evidence_aware():
    """Guard: the prose fix must not change structured-doc behavior — a
    caller query there already flows into the evidence_aware stage."""
    compressor = _make_compressor_for_auto(
        [_skeleton("baseline"), _skeleton("query_guided"), _skeleton("evidence_aware")]
    )

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="auto",
        query="rate limiting",
        top_k=2,
        min_similarity=0.35,
        raw_text=_STRUCTURED_DOC,
    )

    assert result["final_stage"] == "evidence_aware"
    assert result["selection_mode_resolved"] == "auto-detected: evidence_aware"
