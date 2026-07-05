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

BLOCKER FOR THE ORCHESTRATOR (found while building this gate, not fixed --
this task only adds the gate): ``_generate_skeleton`` sorts ``file_nodes`` by
PageRank ``importance`` descending (semantic_compressor.py:1397/1453), NOT by
original document position. So ``grade_source_order`` is NOT currently
guaranteed end-to-end by the real engine for a multi-node document -- it is
only satisfied incidentally when nodes are otherwise-equal-importance ties
(Python's stable sort then preserves insertion order). The
``TestRealCompressorIntegration`` test therefore does not exercise
``grade_source_order`` (it uses a single-node fixture where order is
vacuous); the grader itself IS fully bidirectionally proven against
hand-authored skeleton text. If per-document source-order becomes a hard
product requirement, ``_generate_skeleton`` would need an explicit
"preserve original order among skeleton_nodes" render step -- that is an
engine change, out of scope for this gate-only task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import pytest

from src.semantic_compressor import FidelityLevel, SemanticCompressor, SemanticNode
from tests.fixtures.quality_gate_fixtures import (
    ALL_FIXTURES,
    CODE_FIXTURE,
    INTEGRATION_FIXTURE,
    MIXED_FIXTURE,
    PROSE_FIXTURE,
    QualityGateFixture,
)

# ===========================================================================
# Grader result type
# ===========================================================================


@dataclass(frozen=True)
class GradeResult:
    """Pass/fail + score for one grader run against one skeleton."""

    passed: bool
    score: float
    missing: List[str] = field(default_factory=list)


# ===========================================================================
# Graders (pure functions -- no ranker/embedder involvement)
# ===========================================================================


def grade_answerability(skeleton_text: str, answer_spans: Sequence[str]) -> GradeResult:
    """Each fixed answer span must survive verbatim somewhere in the skeleton."""
    if not answer_spans:
        return GradeResult(passed=True, score=1.0, missing=[])
    missing = [span for span in answer_spans if span not in skeleton_text]
    hits = len(answer_spans) - len(missing)
    return GradeResult(
        passed=not missing,
        score=hits / len(answer_spans),
        missing=missing,
    )


def grade_byte_identity(skeleton_text: str, load_bearing_tokens: Sequence[str]) -> GradeResult:
    """Numbers/identifiers/code tokens must appear BYTE-IDENTICAL in the skeleton."""
    if not load_bearing_tokens:
        return GradeResult(passed=True, score=1.0, missing=[])
    missing = [tok for tok in load_bearing_tokens if tok not in skeleton_text]
    hits = len(load_bearing_tokens) - len(missing)
    return GradeResult(
        passed=not missing,
        score=hits / len(load_bearing_tokens),
        missing=missing,
    )


def grade_source_order(skeleton_text: str, order_markers: Sequence[str]) -> GradeResult:
    """Surviving order markers must appear in the SAME relative order as given.

    A marker that is entirely missing means order cannot be verified for it,
    so a missing marker fails the grader (it does not vacuously pass).
    """
    if not order_markers:
        return GradeResult(passed=True, score=1.0, missing=[])
    positions: List[int] = []
    missing: List[str] = []
    for marker in order_markers:
        idx = skeleton_text.find(marker)
        if idx == -1:
            missing.append(marker)
        else:
            positions.append(idx)
    if missing:
        return GradeResult(passed=False, score=0.0, missing=missing)
    in_order = positions == sorted(positions)
    return GradeResult(passed=in_order, score=1.0 if in_order else 0.0, missing=[])


def grade_modulate_region_roundtrip(
    compressor: SemanticCompressor,
    node_id: str,
    expected_substring: str,
) -> GradeResult:
    """A ``[HIDDEN]``/node_id marker must resolve back to real source content."""
    output = compressor.modulate_region([node_id], fidelity_level=FidelityLevel.RAW)
    passed = "[WARN] Node not found" not in output and expected_substring in output
    return GradeResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        missing=[] if passed else [expected_substring],
    )


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
