"""Reusable A-vs-C comparison harness for the #250 multi-node fixture corpus.

Runs every query in ``tests/fixtures/f11_multi_node_fixtures.py`` through
both ``F11_RANKER_PATH="a"`` (dense cosine, default) and ``"c"`` (BM25 +
Reciprocal Rank Fusion), grades the resulting evidence-aware skeleton with
the SHIPPED, sealed oracle in ``src/quality_gate.py``, and reports PER QUERY
CLASS -- never blended. The design memo's "1W/1L/6T wash" headline was
Simpson's-paradox aggregation of two opposite-signed per-class effects
(``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 3); this
harness exists so that never happens silently again.

THIS IS MEASUREMENT INFRASTRUCTURE, NOT A PASS/FAIL CI GATE (yet). The
eventual ship bar (design memo section 3) is: zero statistically-significant
regression on paraphrase + lexical-trap classes, a significant win on >=1
lexical class, deterministic gates green, p95 latency in budget. At n~=3
queries/class this corpus cannot resolve significance -- it proves the
corpus ENGAGES the ranker (>3 real nodes per fixture) and gives the #187
gated-fusion build real per-class signal to iterate against, instead of
zero coverage.

Deliberate scope note: the design memo (section 3) suggests extending
``src/quality_gate.py`` with this harness. Build #250 is TEST-ONLY
infrastructure and does not touch ``src/`` -- this module lives in
``tests/`` instead and imports (never modifies) the sealed graders from
``src/quality_gate.py``. Promoting parts of it into ``src/`` is a decision
for whoever builds the actual gated-fusion ranker (#187).

Usage::

    python -m tests.f11_fixture_harness   # prints the per-class table

Or import ``compare_paths`` / ``per_class_summary`` from a test.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Allow `python tests/f11_fixture_harness.py` as well as `-m tests....`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import src.semantic_compressor as semantic_compressor_module  # noqa: E402
from src.semantic_compressor import SemanticCompressor  # noqa: E402
from src.quality_gate import grade_answerability, grade_byte_identity  # noqa: E402

from tests.fixtures.f11_multi_node_fixtures import (  # noqa: E402
    ALL_F11_FIXTURES,
    ALL_QUERY_CLASSES,
    F11MultiNodeFixture,
    F11Query,
    QUERY_CLASS_MULTI_HOP,
)

RANKER_PATHS: Tuple[str, str] = ("a", "c")


# ===========================================================================
# Model availability probe (mirrors tests/test_quality_gate.py's pattern --
# duplicated rather than imported so this module has no test-file coupling
# and stays importable standalone, e.g. by a future #187 CI gate).
# ===========================================================================


def probe_model_load() -> Tuple[bool, str]:
    """Best-effort real-embedding-model load + one HF-cache repair attempt
    (compression-engine-sota skill: a JSONDecodeError / exit-5 on load means
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


def set_ranker_path(path: str) -> None:
    """Flip F11_RANKER_PATH at runtime -- mutates the module-global copy
    ``semantic_compressor.py`` reads at CALL time (imported once at module
    load, so the env var itself is not the lever post-import; this is the
    same monkeypatch ``tests/test_f11_path_c.py`` and the prior subagent's
    scratchpad scripts already use)."""
    semantic_compressor_module.F11_RANKER_PATH = path


def node_section_marker(compressor: SemanticCompressor, node_id: str) -> Optional[str]:
    """Best-effort: which '## MARKER_SECTION' heading does this node's raw
    chunk text start with. Returns None if no heading line is found (should
    not happen for this corpus -- every section is header-aware chunked
    1:1, see TestChunkCountEngagesF11)."""
    text = compressor.chunks[node_id].text
    for line in text.split("\n"):
        stripped = line.strip().lstrip("#").strip()
        if stripped.endswith("_SECTION"):
            return stripped
    return None


# ===========================================================================
# Per-query result + per-class aggregation
# ===========================================================================


@dataclass(frozen=True)
class PerQueryResult:
    fixture_id: str
    query_id: str
    query_class: str
    path: str
    gold_markers: Tuple[str, ...]
    ranked_markers: Tuple[str, ...]  # top-K markers in rank order, path's ranker
    rank_of_best_gold: Optional[int]  # 1-indexed; None if no gold marker in top-K
    top1_hit: Optional[bool]  # None for multi_hop (not a well-defined metric there)
    recall_at_3: bool
    recall_at_5: bool
    mrr_at_5: float
    answerability_score: float
    answerability_passed: bool
    byte_identity_passed: bool
    evidence_sufficient: bool


def _rank_metrics(
    gold_markers: List[str], ranked_markers: List[str]
) -> Tuple[Optional[int], bool, bool, float]:
    """Returns (rank_of_best_gold, recall_at_3, recall_at_5, mrr_at_5).

    recall_at_N requires ALL gold_markers present within the first N ranked
    markers (single-hop: trivially "the one gold marker is in top-N";
    multi-hop: both gold markers must appear).
    """
    positions = {}
    for idx, marker in enumerate(ranked_markers, start=1):
        if marker in gold_markers and marker not in positions:
            positions[marker] = idx

    if len(positions) < len(gold_markers):
        # Not every gold marker was retrieved at all within the candidate list.
        best_rank = min(positions.values()) if positions else None
        return best_rank, False, False, 0.0

    best_rank = min(positions.values())
    worst_rank = max(positions.values())
    recall_at_3 = worst_rank <= 3
    recall_at_5 = worst_rank <= 5
    mrr_at_5 = 1.0 / best_rank if best_rank <= 5 else 0.0
    return best_rank, recall_at_3, recall_at_5, mrr_at_5


def run_query(
    compressor: SemanticCompressor,
    file_id: str,
    query: F11Query,
    *,
    rank_top_k: int = 5,
    evidence_top_k: int = 3,
    min_similarity: float = 0.35,
) -> Tuple[Optional[int], bool, bool, float, List[str], str]:
    """Run one query against an already-ingested compressor. Returns
    (rank_of_best_gold, recall_at_3, recall_at_5, mrr_at_5, ranked_markers,
    evidence_aware_skeleton_text)."""
    ranked = compressor.search_semantic_with_scores(
        query.query_text, file_id=file_id, top_k=rank_top_k
    )
    ranked_markers = [
        m for m in (node_section_marker(compressor, nid) for nid, _score in ranked) if m
    ]
    rank_of_best_gold, recall_at_3, recall_at_5, mrr_at_5 = _rank_metrics(
        query.gold_markers, ranked_markers
    )

    skeleton_text = compressor.read_skeleton_evidence_aware(
        file_id=file_id,
        query=query.query_text,
        top_k=evidence_top_k,
        min_similarity=min_similarity,
    )
    return rank_of_best_gold, recall_at_3, recall_at_5, mrr_at_5, ranked_markers, skeleton_text


def run_fixture(
    fixture: F11MultiNodeFixture,
    path: str,
    *,
    skeleton_ratio: float = 0.34,
    rank_top_k: int = 5,
    evidence_top_k: int = 3,
    min_similarity: float = 0.35,
) -> List[PerQueryResult]:
    """Ingest ``fixture`` fresh under a FIXED ranker path, run every query,
    and grade the evidence-aware skeleton with the shipped oracle."""
    set_ranker_path(path)
    compressor = SemanticCompressor(skeleton_ratio=skeleton_ratio)
    file_id = f"{fixture.fixture_id}_{path}"
    compressor.ingest_file(fixture.source_text, file_id)

    results: List[PerQueryResult] = []
    for query in fixture.queries:
        (
            rank_of_best_gold,
            recall_at_3,
            recall_at_5,
            mrr_at_5,
            ranked_markers,
            skeleton_text,
        ) = run_query(
            compressor,
            file_id,
            query,
            rank_top_k=rank_top_k,
            evidence_top_k=evidence_top_k,
            min_similarity=min_similarity,
        )

        top1_hit: Optional[bool]
        if query.query_class == QUERY_CLASS_MULTI_HOP:
            top1_hit = None
        else:
            top1_hit = bool(ranked_markers) and ranked_markers[0] == query.gold_markers[0]

        answerability = grade_answerability(skeleton_text, query.answer_spans)
        byte_identity = grade_byte_identity(skeleton_text, query.load_bearing_tokens)
        evidence_sufficient = "SUFFICIENT" in skeleton_text.split("\n", 1)[0]

        results.append(
            PerQueryResult(
                fixture_id=fixture.fixture_id,
                query_id=query.query_id,
                query_class=query.query_class,
                path=path,
                gold_markers=tuple(query.gold_markers),
                ranked_markers=tuple(ranked_markers),
                rank_of_best_gold=rank_of_best_gold,
                top1_hit=top1_hit,
                recall_at_3=recall_at_3,
                recall_at_5=recall_at_5,
                mrr_at_5=mrr_at_5,
                answerability_score=answerability.score,
                answerability_passed=answerability.passed,
                byte_identity_passed=byte_identity.passed,
                evidence_sufficient=evidence_sufficient,
            )
        )
    return results


def compare_paths(
    fixtures: List[F11MultiNodeFixture] = ALL_F11_FIXTURES,
    **run_kwargs,
) -> Dict[str, List[PerQueryResult]]:
    """Run every fixture's queries through both ranker paths.

    Returns ``{"a": [...], "c": [...]}``, each a flat list of
    ``PerQueryResult`` across all fixtures (query order preserved so
    ``results["a"][i]`` and ``results["c"][i]`` are the same query).
    """
    by_path: Dict[str, List[PerQueryResult]] = {p: [] for p in RANKER_PATHS}
    for fixture in fixtures:
        for path in RANKER_PATHS:
            by_path[path].extend(run_fixture(fixture, path, **run_kwargs))
    return by_path


# ===========================================================================
# Per-class summary (McNemar-style paired win/loss/tie + descriptive means)
# ===========================================================================


@dataclass
class ClassSummary:
    query_class: str
    n: int
    top1_hit_rate_a: Optional[float]
    top1_hit_rate_c: Optional[float]
    recall_at_5_rate_a: float
    recall_at_5_rate_c: float
    mean_answerability_a: float
    mean_answerability_c: float
    byte_identity_rate_a: float
    byte_identity_rate_c: float
    wins_c: int  # A wrong (recall@5), C right
    losses_c: int  # A right, C wrong
    ties: int


def per_class_summary(by_path: Dict[str, List[PerQueryResult]]) -> Dict[str, ClassSummary]:
    a_results = {r.query_id: r for r in by_path["a"]}
    c_results = {r.query_id: r for r in by_path["c"]}
    assert a_results.keys() == c_results.keys(), "path A and C must cover the same queries"

    summaries: Dict[str, ClassSummary] = {}
    for qclass in ALL_QUERY_CLASSES:
        ids = [qid for qid, r in a_results.items() if r.query_class == qclass]
        if not ids:
            continue
        a_rows = [a_results[qid] for qid in ids]
        c_rows = [c_results[qid] for qid in ids]
        n = len(ids)

        top1_a = [r.top1_hit for r in a_rows if r.top1_hit is not None]
        top1_c = [r.top1_hit for r in c_rows if r.top1_hit is not None]

        wins_c = losses_c = ties = 0
        for qid in ids:
            a_ok = a_results[qid].recall_at_5
            c_ok = c_results[qid].recall_at_5
            if a_ok == c_ok:
                ties += 1
            elif c_ok and not a_ok:
                wins_c += 1
            else:
                losses_c += 1

        summaries[qclass] = ClassSummary(
            query_class=qclass,
            n=n,
            top1_hit_rate_a=(sum(top1_a) / len(top1_a)) if top1_a else None,
            top1_hit_rate_c=(sum(top1_c) / len(top1_c)) if top1_c else None,
            recall_at_5_rate_a=sum(r.recall_at_5 for r in a_rows) / n,
            recall_at_5_rate_c=sum(r.recall_at_5 for r in c_rows) / n,
            mean_answerability_a=sum(r.answerability_score for r in a_rows) / n,
            mean_answerability_c=sum(r.answerability_score for r in c_rows) / n,
            byte_identity_rate_a=sum(r.byte_identity_passed for r in a_rows) / n,
            byte_identity_rate_c=sum(r.byte_identity_passed for r in c_rows) / n,
            wins_c=wins_c,
            losses_c=losses_c,
            ties=ties,
        )
    return summaries


def format_report(summaries: Dict[str, ClassSummary]) -> str:
    def fmt_pct(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.0%}"

    lines = [
        f"{'class':<16} {'n':>3} {'top1 A':>7} {'top1 C':>7} {'rec5 A':>7} {'rec5 C':>7} "
        f"{'ans A':>6} {'ans C':>6} {'byte A':>7} {'byte C':>7} {'W/L/T':>8}",
        "-" * 100,
    ]
    for qclass in ALL_QUERY_CLASSES:
        s = summaries.get(qclass)
        if s is None:
            continue
        lines.append(
            f"{qclass:<16} {s.n:>3} {fmt_pct(s.top1_hit_rate_a):>7} {fmt_pct(s.top1_hit_rate_c):>7} "
            f"{s.recall_at_5_rate_a:>6.0%} {s.recall_at_5_rate_c:>6.0%} "
            f"{s.mean_answerability_a:>5.2f} {s.mean_answerability_c:>5.2f} "
            f"{s.byte_identity_rate_a:>6.0%} {s.byte_identity_rate_c:>6.0%} "
            f"{s.wins_c}/{s.losses_c}/{s.ties:>2}"
        )
    total_wins = sum(s.wins_c for s in summaries.values())
    total_losses = sum(s.losses_c for s in summaries.values())
    total_ties = sum(s.ties for s in summaries.values())
    lines.append("-" * 100)
    lines.append(
        f"TOTAL (all classes, recall@5 win/loss/tie for Path C): "
        f"{total_wins}W / {total_losses}L / {total_ties}T"
    )
    return "\n".join(lines)


def main() -> None:
    available, reason = probe_model_load()
    if not available:
        print(f"SKIPPED: real embedding model unavailable ({reason})")
        return

    by_path = compare_paths()
    summaries = per_class_summary(by_path)
    print(format_report(summaries))

    receipt_path = os.path.join(os.path.dirname(__file__), "f11_fixture_harness_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": {path: [r.__dict__ for r in rows] for path, rows in by_path.items()},
                "summary": {qc: s.__dict__ for qc, s in summaries.items()},
            },
            f,
            indent=2,
            default=list,
        )
    print(f"\nReceipt written to {receipt_path}")


if __name__ == "__main__":
    main()
