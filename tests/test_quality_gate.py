"""Deterministic, INDEPENDENT-oracle compression quality gate (Phase 0).

Load-bearing constraint (codex MF1, compression-quality-eval skill): the
EXISTING ``benchmark_harness.py::_quality_overlap_metrics`` defines relevance
USING the live ``search_semantic_with_scores`` ranker. A gate built on that
grades a ranker change AGAINST ITSELF (circular). Every grader in this module
instead reads FIXED LABELS hand-authored in
``tests/fixtures/quality_gate_fixtures.py`` -- never anything computed by a
ranker/embedder under test.

Four graders, each proven BIDIRECTIONALLY (a correct skeleton PASSES, a
deliberately-lobotomized skeleton FAILS -- a grader that can't fail a wrong
answer is broken, per the skill's "broken-oracle trap"):

1. ``grade_answerability``       -- answer spans survive verbatim somewhere
   in the skeleton.
2. ``grade_byte_identity``       -- load-bearing numbers/identifiers/code
   tokens survive byte-identical.
3. ``grade_source_order``        -- surviving anchored sections appear in
   the same relative order as the source.
4. ``grade_modulate_region_roundtrip`` -- a ``[HIDDEN]``/node_id marker
   resolves back to real source content via ``modulate_region``.

Graders 1-3 are pure functions over a ``skeleton_text`` string -- 100%
model-free (the bidirectional RED/GREEN pairs manipulate hand-crafted
skeleton strings; no compressor, no embeddings). Grader 4 needs a
``SemanticCompressor`` instance but not a model -- ``modulate_region`` reads
only ``self.chunks``, so the bidirectional pair uses the
``object.__new__(SemanticCompressor)`` model-free pattern from
``test_audit_compression_correctness.py``.

One integration test (``TestRealCompressorIntegration``) runs the ACTUAL
compressor + a real embedding model against ``INTEGRATION_FIXTURE`` and
re-runs the answerability/byte-identity/round-trip graders against the real
output. It is deliberately single-paragraph (not one of the 3 richer sealed
fixtures) because ``_generate_skeleton``'s default chunker merges short
paragraphs into ONE chunk (max_chunk_size=512 tokens), and an ANCHOR node's
``Summary`` line is only the chunk's FIRST substantive sentence -- so a
multi-paragraph small doc would silently drop paragraphs 2+ from the
rendered skeleton, which is a real (if orthogonal) engine property, not a
grader bug. See the "blockers for the orchestrator" note at the bottom of
this docstring.

FIXED (world-class compression audit #2, 2026-07-07): ``_generate_skeleton``
used to sort ``file_nodes`` by PageRank ``importance`` descending
(semantic_compressor.py:1434/1492) AND render in that same order
(semantic_compressor.py:1639), so ``grade_source_order`` was NOT guaranteed
end-to-end for a multi-node document -- it only passed incidentally when
nodes were otherwise-equal-importance ties (Python's stable sort then
preserves insertion order). The fix is a pure RENDER reorder: node
SELECTION (skeleton_nodes / anchors / COMI / MMR) is untouched; immediately
before the render loop, nodes-to-render are re-sorted by
``metadata["position"]`` (original document order) with a ``node_id``
tiebreak for determinism. ``TestRealCompressorIntegration`` still uses a
single-node fixture (order is vacuous there -- see that class's docstring
for why), but ``TestRealCompressorSourceOrderEndToEnd`` below exercises the
real chunker + real PageRank + the fix end-to-end on a 5-node document where
the ACTUAL importance ranking (not just a contrived one) diverges from
document order.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from src.quality_gate import (
    CorpusFixtureReport,
    CorpusReport,
    empty_compressor,
    evaluate_compressor,
    first_paragraph_compressor,
    grade_answerability,
    grade_byte_identity,
    grade_modulate_region_roundtrip,
    grade_source_order,
    identity_compressor,
)
from src.semantic_compressor import FidelityLevel, SemanticCompressor, SemanticNode
from tests.fixtures.quality_gate_fixtures import (
    ALL_FIXTURES,
    CODE_FIXTURE,
    INTEGRATION_FIXTURE,
    MIXED_FIXTURE,
    PROSE_FIXTURE,
    QualityGateFixture,
)

# NOTE (2026-07-07): ``GradeResult`` + the 4 graders (``grade_answerability``,
# ``grade_byte_identity``, ``grade_source_order``,
# ``grade_modulate_region_roundtrip``) used to be defined INLINE in this test
# file. They are now promoted, byte-for-byte unchanged, to
# ``src/quality_gate.py`` so a real ratio-flip / reranker-flip / TOON-routing
# change (design doc §8 Wave 3) can call the SAME oracle a CI gate would use
# instead of re-deriving the grading logic inline in a test. This file keeps
# every existing test unchanged (just imports the logic instead of defining
# it) and adds ``TestBidirectionalCompressorEvaluation`` below, which proves
# the corpus-level entry point (``evaluate_compressor``) discriminates a
# known-good, a known-bad, and a partial reference compressor.


# ===========================================================================
# Hand-authored skeleton builder (mirrors the real Skeleton-Version: 2
# render format -- see semantic_compressor.py:1567-1624 -- WITHOUT calling
# the real compressor). Used only by the bidirectional grader tests below.
# ===========================================================================


def _sections(fixture: QualityGateFixture) -> List[str]:
    return [s.strip() for s in fixture.source_text.split("\n\n") if s.strip()]


def _build_correct_skeleton(fixture: QualityGateFixture) -> str:
    """A skeleton where every section is a full-fidelity ANCHOR node, in
    original source order -- must pass all 3 pure graders."""
    sections = _sections(fixture)
    lines = [
        f"=== SEMANTIC SKELETON: {fixture.fixture_id} ===",
        "Skeleton-Version: 2",
        f"Total nodes: {len(sections)} | Skeleton nodes: {len(sections)}",
        "Compression: 100% of content shown",
        "Hidden regions expand via modulate_region(node_id).\n",
    ]
    for idx, section in enumerate(sections):
        node_id = f"{fixture.fixture_id}_n{idx}"
        lines.append(
            f"[{node_id}] [rag:{node_id}] [ANCHOR] (importance: 0.500)\n" f"  Summary: {section}\n"
        )
    return "\n".join(lines)


# ===========================================================================
# Bidirectional tests -- grade_answerability
# ===========================================================================


class TestAnswerabilityGrader:
    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
    def test_green_correct_skeleton_passes(self, fixture: QualityGateFixture) -> None:
        skeleton = _build_correct_skeleton(fixture)
        result = grade_answerability(skeleton, fixture.answer_spans)
        assert result.passed, f"answerability should PASS on a correct skeleton: {result.missing}"
        assert result.score == 1.0

    def test_red_redacted_answer_span_fails(self) -> None:
        """LOAD-BEARING: a skeleton with the answer span stripped MUST fail."""
        skeleton = _build_correct_skeleton(PROSE_FIXTURE)
        lobotomized = skeleton.replace("$49 per month", "[REDACTED]")
        result = grade_answerability(lobotomized, PROSE_FIXTURE.answer_spans)
        assert not result.passed, "answerability must FAIL when the answer span is redacted"
        assert "Pro plan costs $49 per month" in result.missing
        assert result.score < 1.0


# ===========================================================================
# Bidirectional tests -- grade_byte_identity
# ===========================================================================


class TestByteIdentityGrader:
    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
    def test_green_correct_skeleton_passes(self, fixture: QualityGateFixture) -> None:
        skeleton = _build_correct_skeleton(fixture)
        result = grade_byte_identity(skeleton, fixture.load_bearing_tokens)
        assert result.passed, f"byte-identity should PASS on a correct skeleton: {result.missing}"
        assert result.score == 1.0

    def test_red_rounded_number_fails(self) -> None:
        """LOAD-BEARING: a rounded/altered number MUST fail byte-identity."""
        skeleton = _build_correct_skeleton(PROSE_FIXTURE)
        # "$49" -> "$50" is exactly the extractive-compression failure mode
        # P2 in compression-quality-eval warns about: a refiner rounding a
        # currency figure inside an anchor node.
        lobotomized = skeleton.replace("$49", "$50")
        result = grade_byte_identity(lobotomized, PROSE_FIXTURE.load_bearing_tokens)
        assert not result.passed, "byte-identity must FAIL when a number is altered"
        assert "$49" in result.missing

    def test_red_misspelled_identifier_fails(self) -> None:
        """LOAD-BEARING: a mangled code identifier MUST fail byte-identity."""
        skeleton = _build_correct_skeleton(CODE_FIXTURE)
        lobotomized = skeleton.replace("verify_api_key", "verify_apikey")
        result = grade_byte_identity(lobotomized, CODE_FIXTURE.load_bearing_tokens)
        assert not result.passed, "byte-identity must FAIL when an identifier is mangled"
        assert "verify_api_key" in result.missing


# ===========================================================================
# Bidirectional tests -- grade_source_order
# ===========================================================================


class TestSourceOrderGrader:
    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.fixture_id)
    def test_green_correct_skeleton_passes(self, fixture: QualityGateFixture) -> None:
        skeleton = _build_correct_skeleton(fixture)
        result = grade_source_order(skeleton, fixture.order_markers)
        assert result.passed, "source-order should PASS when sections are in original order"
        assert result.score == 1.0

    def test_red_shuffled_sections_fails(self) -> None:
        """LOAD-BEARING: reordering the anchored sections MUST fail order check."""
        sections = list(reversed(_sections(MIXED_FIXTURE)))
        lines = [
            f"=== SEMANTIC SKELETON: {MIXED_FIXTURE.fixture_id} ===",
            "Skeleton-Version: 2",
            "Hidden regions expand via modulate_region(node_id).\n",
        ]
        for idx, section in enumerate(sections):
            node_id = f"{MIXED_FIXTURE.fixture_id}_n{idx}"
            lines.append(f"[{node_id}] [ANCHOR]\n  Summary: {section}\n")
        shuffled_skeleton = "\n".join(lines)

        result = grade_source_order(shuffled_skeleton, MIXED_FIXTURE.order_markers)
        assert not result.passed, "source-order must FAIL when sections are reordered"
        assert result.score == 0.0

    def test_red_missing_marker_fails(self) -> None:
        """A marker dropped entirely (e.g. its whole section got pruned) must fail,
        not vacuously pass because 'the remaining markers are in order'."""
        skeleton = _build_correct_skeleton(PROSE_FIXTURE)
        lobotomized = skeleton.replace("BETA_GATEWAY_SECTION: ", "")
        result = grade_source_order(lobotomized, PROSE_FIXTURE.order_markers)
        assert not result.passed
        assert "BETA_GATEWAY_SECTION" in result.missing


# ===========================================================================
# Bidirectional tests -- grade_modulate_region_roundtrip (model-free: only
# self.chunks is touched, per test_audit_compression_correctness.py's
# object.__new__(SemanticCompressor) pattern)
# ===========================================================================


def _make_compressor_with_raw_chunks(chunks: Dict[str, str]) -> SemanticCompressor:
    """Minimal SemanticCompressor backed by hand-built nodes -- no model load.

    ``modulate_region`` only reads ``self.chunks[node_id].text`` /
    ``.metadata['tokens']`` / ``.importance`` -- it never touches
    ``self.model`` or ``.embedding``, so this helper is simpler than the
    ``_make_compressor_with_chunks`` fake-embedder variant in
    ``test_audit_compression_correctness.py``.
    """
    compressor = object.__new__(SemanticCompressor)
    compressor.chunks = {}
    for node_id, text in chunks.items():
        node = object.__new__(SemanticNode)
        node.text = text
        node.importance = 0.5
        node.metadata = {"tokens": len(text.split()), "entities": []}
        compressor.chunks[node_id] = node
    return compressor


class TestModulateRegionRoundtripGrader:
    def test_green_real_node_roundtrips_to_raw_content(self) -> None:
        compressor = _make_compressor_with_raw_chunks(
            {
                "qg_rt_n0": "The refund policy allows 30 days for a full refund of $49.",
            }
        )
        result = grade_modulate_region_roundtrip(compressor, "qg_rt_n0", "30 days")
        assert result.passed, f"round-trip should PASS for a real node: {result.missing}"
        # Byte-identity check on the recovered content too.
        raw = compressor.modulate_region(["qg_rt_n0"], fidelity_level=FidelityLevel.RAW)
        assert "$49" in raw

    def test_red_dangling_node_id_fails(self) -> None:
        """LOAD-BEARING: a node_id referenced in the skeleton but absent from
        ``self.chunks`` (a broken node-map / dangling reference) MUST fail
        the round-trip, not silently pass."""
        compressor = _make_compressor_with_raw_chunks(
            {
                "qg_rt_n0": "The refund policy allows 30 days for a full refund of $49.",
            }
        )
        result = grade_modulate_region_roundtrip(compressor, "qg_rt_missing", "30 days")
        assert not result.passed, "round-trip must FAIL for a dangling/missing node_id"
        assert "30 days" in result.missing


# ===========================================================================
# Bidirectional self-check -- a known-empty/garbage skeleton must fail every
# grader (P6 in compression-quality-eval: bidirectionally validate the gate
# before trusting a batch).
# ===========================================================================


class TestGateItselfIsNotBroken:
    def test_empty_skeleton_fails_every_grader(self) -> None:
        empty = "=== SEMANTIC SKELETON: qg_empty ===\nTotal nodes: 0 | Skeleton nodes: 0\n"
        for fixture in ALL_FIXTURES:
            assert not grade_answerability(empty, fixture.answer_spans).passed
            assert not grade_byte_identity(empty, fixture.load_bearing_tokens).passed
            assert not grade_source_order(empty, fixture.order_markers).passed


# ===========================================================================
# Integration test -- runs the ACTUAL compressor + a real embedding model.
# ===========================================================================


def _probe_model_load() -> tuple[bool, str]:
    """Best-effort model load + one HF-cache repair attempt (per
    compression-engine-sota skill: a JSONDecodeError / exit-5 on load means
    the local HF_HOME cache has 0-byte JSON configs, NOT "offline")."""
    try:
        SemanticCompressor()
        return True, ""
    except Exception as first_exc:  # noqa: BLE001 -- broad probe, not app logic
        try:
            from huggingface_hub import snapshot_download

            snapshot_download("BAAI/bge-small-en-v1.5")
            snapshot_download("sentence-transformers/all-MiniLM-L6-v2")
            SemanticCompressor()
            return True, ""
        except Exception as second_exc:  # noqa: BLE001
            return False, (
                f"model load failed ({first_exc!r}); repair attempt also failed "
                f"({second_exc!r})"
            )


_MODEL_AVAILABLE, _MODEL_SKIP_REASON = _probe_model_load()


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason=_MODEL_SKIP_REASON)
class TestRealCompressorIntegration:
    """Runs the REAL compressor on INTEGRATION_FIXTURE and re-grades its
    actual output. See the module docstring for why this fixture is a single
    short paragraph and why grade_source_order is not exercised here."""

    def test_real_skeleton_passes_answerability_and_byte_identity(self) -> None:
        compressor = SemanticCompressor(skeleton_ratio=1.0)
        file_id = "qg_integration_real"
        compressor.ingest_file(INTEGRATION_FIXTURE.source_text, file_id)
        response = compressor._generate_skeleton(file_id, query=None)

        answerability = grade_answerability(
            response.skeleton_text, INTEGRATION_FIXTURE.answer_spans
        )
        assert answerability.passed, (
            f"real compressor output failed answerability: missing={answerability.missing}\n"
            f"--- skeleton ---\n{response.skeleton_text}"
        )

        byte_identity = grade_byte_identity(
            response.skeleton_text, INTEGRATION_FIXTURE.load_bearing_tokens
        )
        assert byte_identity.passed, (
            f"real compressor output failed byte-identity: missing={byte_identity.missing}\n"
            f"--- skeleton ---\n{response.skeleton_text}"
        )

    def test_real_modulate_region_roundtrip(self) -> None:
        compressor = SemanticCompressor(skeleton_ratio=1.0)
        file_id = "qg_integration_real_rt"
        compressor.ingest_file(INTEGRATION_FIXTURE.source_text, file_id)
        response = compressor._generate_skeleton(file_id, query=None)

        # Concatenate the RAW round-trip of every node the real chunker
        # produced -- avoids depending on how many nodes it split the doc
        # into or which node_id maps to which content.
        all_raw = "".join(
            compressor.modulate_region([node_id], fidelity_level=FidelityLevel.RAW)
            for node_id in response.node_map
        )
        result = grade_byte_identity(all_raw, INTEGRATION_FIXTURE.load_bearing_tokens)
        assert result.passed, (
            f"real modulate_region round-trip failed byte-identity: missing={result.missing}\n"
            f"--- recovered raw ---\n{all_raw}"
        )


# ===========================================================================
# End-to-end source-order proof (world-class compression audit #2,
# 2026-07-07) -- runs the REAL compressor (real chunker, real embedding
# model, real PageRank) on a 5-section document deliberately shaped so a
# LATER section (a "recap" paragraph that echoes vocabulary from three
# earlier sections) becomes a similarity-graph hub and out-ranks earlier,
# topically-isolated sections on PageRank importance. Pre-fix this is
# RENDERED first (importance-descending render order); post-fix it renders
# in its true document position. skeleton_ratio=1.0 keeps every node an
# ANCHOR so every order_marker survives verbatim (no [HIDDEN] truncation
# noise in the order check).
# ===========================================================================

_SOURCE_ORDER_DOC = (
    "## ALPHA_AUTH_SECTION\n\n"
    "Authentication in gotcontext.ai resolves an inbound request to a "
    "(user_id, plan) tuple using one of three mechanisms: a Clerk-issued "
    "session JWT verified against the JWKS endpoint, a gc_ prefixed API key "
    "verified via HMAC signature, or a self-hosted Ed25519 license token. "
    "Each mechanism populates request.state with the resolved plan so "
    "downstream middleware can apply plan-gating without a second lookup.\n\n"
    "## BETA_BILLING_SECTION\n\n"
    "Billing is handled entirely by Polar as the merchant of record. Webhook "
    "events for subscription.created, subscription.updated, and "
    "subscription.canceled are verified with Svix signatures before the "
    "handler mutates the local subscriptions table. A daily reconciliation "
    "cron treats the Polar API as the source of truth and never downgrades "
    "a user's plan on an empty or unparseable provider response.\n\n"
    "## GAMMA_CACHE_SECTION\n\n"
    "The semantic cache stores compressed skeleton output keyed by a hash "
    "of the source document plus the requested fidelity level. Cache "
    "entries live in Upstash Redis with a five minute TTL for the plan "
    "cache and a longer TTL for the semantic cache proper, using pgvector "
    "HNSW indexing for approximate nearest-neighbor lookups.\n\n"
    "## DELTA_WEBHOOK_SECTION\n\n"
    "Outbound webhook delivery is best effort with a durable retry queue "
    "backed by a webhook_deliveries table. A drain cron runs hourly and "
    "attempts redelivery for any row still marked pending, applying "
    "exponential backoff between attempts and giving up after a bounded "
    "number of retries so a permanently dead endpoint cannot loop forever.\n\n"
    "## EPSILON_RECAP_SECTION\n\n"
    "To recap: authentication resolves a JWT or API key to a plan, billing "
    "runs through Polar webhooks reconciled daily against the provider, the "
    "semantic cache uses Redis and pgvector HNSW for fast lookups, and "
    "outbound webhook delivery retries through a durable queue with "
    "exponential backoff until a bounded retry ceiling is reached."
)
_SOURCE_ORDER_MARKERS = [
    "ALPHA_AUTH_SECTION",
    "BETA_BILLING_SECTION",
    "GAMMA_CACHE_SECTION",
    "DELTA_WEBHOOK_SECTION",
    "EPSILON_RECAP_SECTION",
]


@pytest.mark.skipif(not _MODEL_AVAILABLE, reason=_MODEL_SKIP_REASON)
class TestRealCompressorSourceOrderEndToEnd:
    """LOAD-BEARING: proves the render-reorder fix on the REAL engine, not
    just on hand-authored skeleton strings. The recap section's real
    PageRank importance genuinely out-ranks the earlier, topically-isolated
    sections (verified empirically, not assumed) -- this is not a contrived
    ``importance=`` override on a fake node."""

    def test_real_multi_node_skeleton_preserves_document_order(self) -> None:
        compressor = SemanticCompressor(skeleton_ratio=1.0)
        file_id = "qg_source_order_real"
        compressor.ingest_file(_SOURCE_ORDER_DOC, file_id)
        response = compressor._generate_skeleton(file_id, query=None)

        result = grade_source_order(response.skeleton_text, _SOURCE_ORDER_MARKERS)
        assert result.passed, (
            "real compressor output should render sections in original "
            f"document order: missing={result.missing}\n"
            f"--- skeleton ---\n{response.skeleton_text}"
        )
        assert result.score == 1.0

    def test_real_multi_node_importance_actually_diverges_from_position(self) -> None:
        """Guards against the test above passing VACUOUSLY (i.e. the engine
        happening to keep PageRank importance monotonic with position, which
        would make this fixture no better than the single-node integration
        fixture). Confirms the recap section (last in the doc) has
        importance strictly greater than at least one earlier section --
        the precondition that makes the order-preservation test above
        actually exercise the render-reorder fix."""
        compressor = SemanticCompressor(skeleton_ratio=1.0)
        file_id = "qg_source_order_divergence_check"
        compressor.ingest_file(_SOURCE_ORDER_DOC, file_id)
        nodes = sorted(
            (n for nid, n in compressor.chunks.items() if nid.startswith(file_id)),
            key=lambda n: n.metadata["position"],
        )
        assert len(nodes) >= 4, "fixture must yield multiple real nodes, not merge to one"
        recap_importance = nodes[-1].importance
        earlier_importances = [n.importance for n in nodes[:-1]]
        assert any(recap_importance > imp for imp in earlier_importances), (
            "fixture precondition failed: recap section must out-rank at least "
            f"one earlier section on PageRank importance; got recap={recap_importance}, "
            f"earlier={earlier_importances}"
        )


# ===========================================================================
# Bidirectional CORPUS-level evaluation (MF1 hard prerequisite) -- proves
# `evaluate_compressor`, the entry point any future ratio/reranker/routing
# flip (design doc §8 Wave 3) must call, actually discriminates: a known-good
# compressor passes the whole sealed corpus, a known-bad compressor fails the
# whole corpus, and a deterministic PARTIAL compressor lands strictly between
# the two extremes -- proving the oracle grades degrees of quality loss, not
# just a binary pass/fail (compression-quality-eval skill, P6: "bidirectionally
# validate the gate before trusting a batch").
# ===========================================================================


class TestBidirectionalCompressorEvaluation:
    def test_identity_compressor_passes_entire_corpus(self) -> None:
        """KNOWN-GOOD direction: a passthrough compressor must pass every
        sealed fixture (prose/code/mixed/json/qa) -- it returns the fixture's
        own source text, which by construction contains every one of its
        own hand-labelled answer_spans/load_bearing_tokens."""
        report = evaluate_compressor(identity_compressor, ALL_FIXTURES)
        assert isinstance(report, CorpusReport)
        assert report.all_passed, (
            f"identity compressor should pass every fixture; " f"failed={report.failed_fixture_ids}"
        )
        for fixture_report in report.fixture_reports:
            assert isinstance(fixture_report, CorpusFixtureReport)
            assert fixture_report.answerability.score == 1.0
            assert fixture_report.byte_identity.score == 1.0

    def test_empty_compressor_fails_entire_corpus(self) -> None:
        """KNOWN-BAD direction: a compressor that emits nothing must fail
        every sealed fixture -- if it didn't, the oracle would be the exact
        broken-oracle trap the skill warns about (a stub passing as if it
        were a capability)."""
        report = evaluate_compressor(empty_compressor, ALL_FIXTURES)
        assert not report.all_passed
        assert set(report.failed_fixture_ids) == {f.fixture_id for f in ALL_FIXTURES}
        for fixture_report in report.fixture_reports:
            assert fixture_report.answerability.score == 0.0
            assert fixture_report.byte_identity.score == 0.0

    def test_partial_compressor_scores_strictly_between_pass_and_fail(self) -> None:
        """A deterministic PARTIAL compressor (keeps only the fixture's first
        of three sections) must land STRICTLY BETWEEN the identity-compressor
        ceiling (1.0) and the empty-compressor floor (0.0) -- proving the
        oracle can discriminate a mid-fidelity regression, not just detect
        the two extremes above."""
        report = evaluate_compressor(first_paragraph_compressor, [PROSE_FIXTURE])
        (fixture_report,) = report.fixture_reports
        assert 0.0 < fixture_report.answerability.score < 1.0, fixture_report.answerability
        assert 0.0 < fixture_report.byte_identity.score < 1.0, fixture_report.byte_identity
        assert not fixture_report.passed
        assert not report.all_passed
