"""Independent, sealed-fixture utility oracle for compression-quality gating (MF1).

This module is the callable/product form of the graders proven in
``tests/test_quality_gate.py`` -- promoted out of the test file so a real
ratio-flip / reranker-flip / TOON-routing change (see the design doc
``docs/superpowers/plans/2026-07-06-ultimate-compression-architecture.md``,
section 8, Wave 3) can call the SAME oracle a CI gate would use, instead of
re-deriving the grading logic inline in a test.

THE DEFECT THIS ROUTES AROUND: ``benchmark_harness.py::_quality_overlap_metrics``
(``token-saver-5000/src/benchmark_harness.py:132-153``) defines "relevant"
node ids by calling ``compressor.search_semantic_with_scores(query, ...)`` --
the very ranker/embedder a compression change would modify. Precision/
recall/F1 computed that way grades a ranker change against ITS OWN output: a
reranker that gets systematically WORSE would still score perfectly as long
as it reranks *consistently* with itself, because the "ground truth" moves
whenever the ranker does. This oracle never asks the engine what is
relevant. Every grader below reads FIXED labels hand-authored in
``tests/fixtures/quality_gate_fixtures.py`` -- author-fixed ground truth,
never engine-derived.

Sealed + deterministic: no model load, no network, no randomness in this
module. Every grader is a pure substring / ordering check over plain text.
``compressor`` callables accepted by ``evaluate_compressor`` are plain
``Callable[[str], str]`` -- the oracle only ever does string containment
against them; it never inspects what produced the string (a real engine
pipeline, a passthrough, a stub, or a partial/broken reference compressor
all work identically as inputs).

Bidirectional validation (compression-quality-eval skill, P6 -- "the
broken-oracle trap"): see
``tests/test_quality_gate.py::TestBidirectionalCompressorEvaluation`` for the
proof that ``identity_compressor`` (passthrough) PASSES the whole sealed
corpus, ``empty_compressor`` FAILS the whole corpus, and
``first_paragraph_compressor`` (a deterministic partial compressor) scores
STRICTLY BETWEEN 0 and 1 -- proving the oracle discriminates degrees of
quality loss rather than only detecting the two extremes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Protocol, Sequence, Tuple

from .semantic_compressor import FidelityLevel, SemanticCompressor

# ===========================================================================
# Grader result type
# ===========================================================================


@dataclass(frozen=True)
class GradeResult:
    """Pass/fail + score for one grader run against one compressed text."""

    passed: bool
    score: float
    missing: List[str] = field(default_factory=list)


# ===========================================================================
# Graders (pure functions -- no ranker/embedder involvement)
# ===========================================================================


def grade_answerability(compressed_text: str, answer_spans: Sequence[str]) -> GradeResult:
    """Each fixed answer span must survive verbatim somewhere in the output."""
    if not answer_spans:
        return GradeResult(passed=True, score=1.0, missing=[])
    missing = [span for span in answer_spans if span not in compressed_text]
    hits = len(answer_spans) - len(missing)
    return GradeResult(
        passed=not missing,
        score=hits / len(answer_spans),
        missing=missing,
    )


def grade_byte_identity(compressed_text: str, load_bearing_tokens: Sequence[str]) -> GradeResult:
    """Numbers/identifiers/code tokens must appear BYTE-IDENTICAL in the output."""
    if not load_bearing_tokens:
        return GradeResult(passed=True, score=1.0, missing=[])
    missing = [tok for tok in load_bearing_tokens if tok not in compressed_text]
    hits = len(load_bearing_tokens) - len(missing)
    return GradeResult(
        passed=not missing,
        score=hits / len(load_bearing_tokens),
        missing=missing,
    )


def grade_source_order(compressed_text: str, order_markers: Sequence[str]) -> GradeResult:
    """Surviving order markers must appear in the SAME relative order as given.

    A marker that is entirely missing means order cannot be verified for it,
    so a missing marker fails the grader (it does not vacuously pass).
    """
    if not order_markers:
        return GradeResult(passed=True, score=1.0, missing=[])
    positions: List[int] = []
    missing: List[str] = []
    for marker in order_markers:
        idx = compressed_text.find(marker)
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
# Corpus-level evaluation -- the entry point a future ratio/reranker/routing
# flip must call before shipping (design doc §8 Wave 3 gate; MF1).
# ===========================================================================


class UtilityFixtureLike(Protocol):
    """Structural type for a sealed fixture.

    Deliberately a ``Protocol`` (duck-typed), not an import of the concrete
    ``QualityGateFixture`` dataclass -- this module lives in ``src/`` and must
    not depend on ``tests/fixtures/...``. Any object with these four
    attributes (the real ``QualityGateFixture`` included) works.
    """

    fixture_id: str
    doc_type: str
    source_text: str
    answer_spans: List[str]
    load_bearing_tokens: List[str]


@dataclass(frozen=True)
class CorpusFixtureReport:
    """Answerability + byte-identity verdict for one fixture."""

    fixture_id: str
    doc_type: str
    answerability: GradeResult
    byte_identity: GradeResult

    @property
    def passed(self) -> bool:
        return self.answerability.passed and self.byte_identity.passed


@dataclass(frozen=True)
class CorpusReport:
    """Aggregate verdict across the whole sealed fixture set for one compressor."""

    fixture_reports: Tuple[CorpusFixtureReport, ...]

    @property
    def all_passed(self) -> bool:
        return all(report.passed for report in self.fixture_reports)

    @property
    def failed_fixture_ids(self) -> Tuple[str, ...]:
        return tuple(report.fixture_id for report in self.fixture_reports if not report.passed)


def evaluate_compressor(
    compressor: Callable[[str], str],
    fixtures: Sequence[UtilityFixtureLike],
) -> CorpusReport:
    """Run ``compressor`` over every sealed fixture's raw text and grade the result.

    ``compressor`` is any ``Callable[[str], str]`` -- a real engine's
    ``ingest_file`` + ``_generate_skeleton().skeleton_text`` pipeline, a
    passthrough, an empty stub, or a deterministic partial reference
    compressor. The oracle never inspects ``compressor`` internals; it only
    checks whether each fixture's hand-labelled ground truth survives in
    whatever text comes back.
    """
    reports: List[CorpusFixtureReport] = []
    for fixture in fixtures:
        compressed = compressor(fixture.source_text)
        reports.append(
            CorpusFixtureReport(
                fixture_id=fixture.fixture_id,
                doc_type=fixture.doc_type,
                answerability=grade_answerability(compressed, fixture.answer_spans),
                byte_identity=grade_byte_identity(compressed, fixture.load_bearing_tokens),
            )
        )
    return CorpusReport(fixture_reports=tuple(reports))


# ===========================================================================
# Reference compressors for bidirectional validation (P6) -- deterministic,
# no model, no network. See TestBidirectionalCompressorEvaluation.
# ===========================================================================


def identity_compressor(text: str) -> str:
    """KNOWN-GOOD reference: passthrough. Must PASS every fixture."""
    return text


def empty_compressor(text: str) -> str:
    """KNOWN-BAD reference: returns nothing. Must FAIL every fixture."""
    return ""


def first_paragraph_compressor(text: str) -> str:
    r"""Deterministic MID-FIDELITY reference compressor.

    Keeps only the first ``\n\n``-delimited paragraph/section, drops the
    rest. Every sealed fixture in ``quality_gate_fixtures.py`` is authored as
    three ``\n\n``-separated sections with roughly a third of the ground
    truth (answer spans / load-bearing tokens) in each section, so this
    reference compressor deterministically lands at PARTIAL recall -- proving
    the oracle reports a score strictly between 0 and 1 for a compressor that
    is neither perfect nor empty, not just a binary pass/fail.
    """
    return text.split("\n\n", 1)[0]
