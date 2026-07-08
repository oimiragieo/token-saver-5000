"""Regression locks + measurement smoke for the #250 F11 multi-node fixture
corpus (``tests/fixtures/f11_multi_node_fixtures.py``) and its comparison
harness (``tests/f11_fixture_harness.py``).

Closes the coverage gap documented in
``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 3: the 5
existing sealed fixtures in ``tests/fixtures/quality_gate_fixtures.py`` each
collapse to ONE real chunk, so F11_RANKER_PATH (which only matters with >=2
ranking candidates) had ZERO test coverage. This file proves three things:

1. The corpus construction claims are TRUE, model-free (fast, no embedding
   model needed): every ``pure_paraphrase`` query has zero non-stopword
   token overlap with its gold section; every ``lexical_trap`` query's trap
   term is genuinely more frequent in the decoy section than the gold one.
2. The corpus actually ENGAGES F11 with the real engine: each fixture doc
   chunks into strictly more than 3 real nodes (the COMI coarse-filter
   threshold) -- the receipt the orchestrator asked for.
3. The comparison harness runs end-to-end on both ranker paths and returns
   well-formed per-class data -- a measurement smoke test, NOT a strict
   ship-bar pass/fail gate (that gate is future work once the corpus grows
   toward the full 12-16 doc / ~240 query spec).
"""

from __future__ import annotations

import pytest

from tests.f11_fixture_harness import (
    compare_paths,
    per_class_summary,
    probe_model_load,
    run_fixture,
)
from tests.fixtures.f11_multi_node_fixtures import (
    ALL_F11_FIXTURES,
    ALL_QUERY_CLASSES,
    QUERY_CLASS_LEXICAL_TRAP,
    QUERY_CLASS_MULTI_HOP,
    QUERY_CLASS_PURE_PARAPHRASE,
    content_words,
    term_count,
)

_MODEL_AVAILABLE, _MODEL_SKIP_REASON = probe_model_load()


# ===========================================================================
# Model-free construction-validity locks
# ===========================================================================


class TestParaphraseQueriesHaveZeroContentOverlap:
    """LOAD-BEARING: the design memo requires paraphrase queries to be
    'construct[ed] by synonym-rewriting a gold section, then
    programmatically verif[ied for] zero non-stopword token overlap with
    the gold section' -- this is that verification, not an eyeball check."""

    def test_every_pure_paraphrase_query_has_zero_overlap(self) -> None:
        checked = 0
        for fixture in ALL_F11_FIXTURES:
            for query in fixture.queries:
                if query.query_class != QUERY_CLASS_PURE_PARAPHRASE:
                    continue
                gold_text = fixture.section_text(query.gold_markers[0])
                overlap = content_words(query.query_text) & content_words(gold_text)
                assert not overlap, (
                    f"{fixture.fixture_id}/{query.query_id}: paraphrase query "
                    f"shares content words with its gold section: {overlap}"
                )
                checked += 1
        assert checked >= 3, "expected at least one pure_paraphrase query per fixture"


class TestLexicalTrapConstructionIsValid:
    """LOAD-BEARING: a lexical_trap query's decoy section must contain the
    trap term MORE often than the true gold section -- this is the
    construction-level claim the #1 gated-fusion idea (design memo) exists
    to defend against. Checked on raw section text, independent of chunking
    or any ranker's behavior."""

    def test_every_lexical_trap_decoy_outranks_gold_on_raw_term_frequency(self) -> None:
        checked = 0
        for fixture in ALL_F11_FIXTURES:
            for query in fixture.queries:
                if query.query_class != QUERY_CLASS_LEXICAL_TRAP:
                    continue
                gold_text = fixture.section_text(query.gold_markers[0])
                decoy_text = fixture.section_text(query.decoy_marker)
                gold_count = term_count(gold_text, query.trap_term)
                decoy_count = term_count(decoy_text, query.trap_term)
                assert decoy_count > gold_count, (
                    f"{fixture.fixture_id}/{query.query_id}: trap term "
                    f"{query.trap_term!r} must be MORE frequent in decoy "
                    f"section {query.decoy_marker!r} ({decoy_count}) than in "
                    f"gold section {query.gold_markers[0]!r} ({gold_count})"
                )
                checked += 1
        assert checked >= 3, "expected at least one lexical_trap query per fixture"


class TestQueryClassCoverage:
    """Regression lock: every one of the 7 query classes from the design
    memo must have at least one representative query SOMEWHERE in the
    corpus. Guards against silently losing class coverage in a future
    edit (the whole point of #250 is per-class, never-blended reporting)."""

    def test_all_seven_query_classes_represented(self) -> None:
        seen = {query.query_class for fixture in ALL_F11_FIXTURES for query in fixture.queries}
        missing = set(ALL_QUERY_CLASSES) - seen
        assert not missing, f"query classes with zero coverage: {missing}"

    def test_multi_hop_queries_name_exactly_two_gold_markers(self) -> None:
        for fixture in ALL_F11_FIXTURES:
            for query in fixture.queries:
                if query.query_class == QUERY_CLASS_MULTI_HOP:
                    assert len(query.gold_markers) == 2
                    assert len(query.answer_spans) == 2


# ===========================================================================
# Real-engine chunk-count proof -- the receipt proving the corpus engages F11.
# ===========================================================================


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason=_MODEL_SKIP_REASON)
class TestChunkCountEngagesF11:
    """LOAD-BEARING: F11_RANKER_PATH / COMI's coarse-filter only matter with
    >3 candidate nodes. Every #250 fixture must produce strictly more than 3
    real chunks with the real chunker -- the exact gap the 5 existing
    single-node sealed fixtures could not close."""

    @pytest.mark.parametrize("fixture", ALL_F11_FIXTURES, ids=lambda f: f.fixture_id)
    def test_fixture_chunks_into_more_than_three_nodes(self, fixture) -> None:
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor(skeleton_ratio=0.34)
        file_id = f"corpus_test_{fixture.fixture_id}"
        compressor.ingest_file(fixture.source_text, file_id)
        node_ids = list(compressor.graphs[file_id].nodes())
        assert len(node_ids) > 3, (
            f"{fixture.fixture_id} chunked into only {len(node_ids)} nodes "
            "-- F11_RANKER_PATH has no effect below 4 candidate nodes"
        )
        # Every section should map to exactly one node (1:1 header-aware
        # chunking) -- proves the corpus's marker-based gold labels are
        # meaningful, not a coincidence of some other chunk boundary.
        assert len(node_ids) == len(fixture.section_order), (
            f"{fixture.fixture_id}: expected 1 node per section "
            f"({len(fixture.section_order)}), got {len(node_ids)}"
        )

    @pytest.mark.parametrize("fixture", ALL_F11_FIXTURES, ids=lambda f: f.fixture_id)
    def test_every_gold_marker_resolves_to_a_real_node(self, fixture) -> None:
        """Every query's gold_markers (and decoy_marker, where present) must
        actually appear in the real chunker's output -- a typo'd marker
        name would otherwise silently make a query ungradeable."""
        from src.semantic_compressor import SemanticCompressor

        from tests.f11_fixture_harness import node_section_marker

        compressor = SemanticCompressor(skeleton_ratio=0.34)
        file_id = f"corpus_marker_test_{fixture.fixture_id}"
        compressor.ingest_file(fixture.source_text, file_id)
        node_ids = list(compressor.graphs[file_id].nodes())
        real_markers = {node_section_marker(compressor, nid) for nid in node_ids}

        for query in fixture.queries:
            for marker in query.gold_markers:
                assert marker in real_markers, (
                    f"{fixture.fixture_id}/{query.query_id}: gold marker "
                    f"{marker!r} not found among real chunked nodes {real_markers}"
                )
            if query.decoy_marker:
                assert query.decoy_marker in real_markers, (
                    f"{fixture.fixture_id}/{query.query_id}: decoy marker "
                    f"{query.decoy_marker!r} not found among real chunked "
                    f"nodes {real_markers}"
                )


# ===========================================================================
# Harness smoke test -- proves the A-vs-C comparison runs end-to-end and
# returns well-formed per-class data. NOT a pass/fail ship gate (see module
# docstring + design memo section 3 for the eventual ship bar).
# ===========================================================================


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason=_MODEL_SKIP_REASON)
class TestHarnessRunsEndToEndOnBothPaths:
    def test_run_fixture_returns_one_result_per_query_per_path(self) -> None:
        for fixture in ALL_F11_FIXTURES:
            for path in ("a", "c"):
                results = run_fixture(fixture, path)
                assert len(results) == len(fixture.queries)
                for result in results:
                    assert result.path == path
                    assert result.query_class in ALL_QUERY_CLASSES
                    assert 0.0 <= result.answerability_score <= 1.0
                    if result.rank_of_best_gold is not None:
                        assert 1 <= result.rank_of_best_gold <= 5

    def test_compare_paths_covers_identical_query_ids_on_both_paths(self) -> None:
        by_path = compare_paths()
        a_ids = {r.query_id for r in by_path["a"]}
        c_ids = {r.query_id for r in by_path["c"]}
        assert a_ids == c_ids
        expected_total = sum(len(f.queries) for f in ALL_F11_FIXTURES)
        assert len(a_ids) == expected_total

    def test_per_class_summary_covers_every_represented_class(self) -> None:
        by_path = compare_paths()
        summaries = per_class_summary(by_path)
        represented_classes = {
            query.query_class for fixture in ALL_F11_FIXTURES for query in fixture.queries
        }
        assert set(summaries.keys()) == represented_classes
        for summary in summaries.values():
            assert summary.n >= 1
            assert 0.0 <= summary.recall_at_5_rate_a <= 1.0
            assert 0.0 <= summary.recall_at_5_rate_c <= 1.0
            assert summary.wins_c + summary.losses_c + summary.ties == summary.n

    def test_per_class_report_prints_without_error(self, capsys) -> None:
        """Informational: prints the per-class A-vs-C table to test output
        (visible with `pytest -s`) so the comparison is visible in CI logs
        without needing to inspect a separate receipt file."""
        from tests.f11_fixture_harness import format_report

        by_path = compare_paths()
        summaries = per_class_summary(by_path)
        report = format_report(summaries)
        print("\n" + report)
        assert "class" in report and "TOTAL" in report
