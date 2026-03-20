"""
Benchmark harness for stable token-savings regression tracking.

Week 2 modernization objective:
- fixed benchmark corpus
- golden token-savings thresholds
- machine-readable benchmark artifact
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional

from .extractive_baseline import ExtractiveCompressor
from .semantic_compressor import SemanticCompressor


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark input with minimum expected compression targets."""

    case_id: str
    name: str
    text: str
    min_compression_ratio: float
    min_token_savings_pct: float
    query: Optional[str] = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured output for a single benchmark case."""

    case_id: str
    name: str
    mode: str
    original_tokens: int
    skeleton_tokens: int
    compression_ratio: float
    token_savings_pct: float
    meets_ratio_target: bool
    meets_savings_target: bool
    quality_metrics_available: bool = False
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    f1_at_k: float = 0.0

    @property
    def passed(self) -> bool:
        return self.meets_ratio_target and self.meets_savings_target


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark output used for CI checks and trend tracking."""

    generated_at_utc: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    avg_compression_ratio: float
    avg_token_savings_pct: float
    avg_precision_at_k: float
    avg_recall_at_k: float
    avg_f1_at_k: float
    quality_cases_count: int
    results: List[BenchmarkResult]

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0


@dataclass(frozen=True)
class BenchmarkExecutionDetail:
    """Detailed benchmark execution used by experiment tracking."""

    case: BenchmarkCase
    result: BenchmarkResult
    skeleton_text: str
    node_map: dict[str, str]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_corpus_path() -> Path:
    return _project_root() / "tests" / "fixtures" / "benchmark_corpus.json"


def load_benchmark_cases(corpus_path: Path | str) -> List[BenchmarkCase]:
    """Load benchmark cases from JSON corpus fixture."""
    path = Path(corpus_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", [])
    if not raw_cases:
        raise ValueError(f"No benchmark cases found in {path}")

    cases: List[BenchmarkCase] = []
    for raw in raw_cases:
        cases.append(
            BenchmarkCase(
                case_id=raw["case_id"],
                name=raw["name"],
                text=raw["text"],
                min_compression_ratio=float(raw["min_compression_ratio"]),
                min_token_savings_pct=float(raw["min_token_savings_pct"]),
                query=raw.get("query"),
            )
        )
    return cases


def _savings_pct(compression_ratio: float) -> float:
    if compression_ratio <= 0:
        return 0.0
    return (1.0 - (1.0 / compression_ratio)) * 100.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _quality_overlap_metrics(
    *,
    compressor: SemanticCompressor,
    file_id: str,
    query: str,
    selected_node_ids: set[str],
    top_k: int = 5,
) -> tuple[float, float, float]:
    if not query.strip():
        return 0.0, 0.0, 0.0
    relevant = {
        node_id
        for node_id, _score in compressor.search_semantic_with_scores(
            query, file_id=file_id, top_k=top_k
        )
    }
    if not relevant:
        return 0.0, 0.0, 0.0
    overlap = selected_node_ids & relevant
    precision = len(overlap) / max(len(selected_node_ids), 1)
    recall = len(overlap) / max(len(relevant), 1)
    return precision, recall, _f1(precision, recall)


def run_benchmark_cases_detailed(
    cases: Iterable[BenchmarkCase],
    mode: str = "baseline",
    similarity_threshold: float = 0.75,
    skeleton_ratio: float = 0.2,
) -> List[BenchmarkExecutionDetail]:
    """Run benchmark cases and return detailed execution artifacts."""
    compressor = SemanticCompressor(
        similarity_threshold=similarity_threshold,
        skeleton_ratio=skeleton_ratio,
    )

    details: List[BenchmarkExecutionDetail] = []
    for case in cases:
        file_id = f"bench_{case.case_id}"
        response = compressor.ingest_file(case.text, file_id)
        if mode == "query_guided" and case.query:
            response = compressor._generate_skeleton(file_id, query=case.query)
        elif mode == "evidence_aware" and case.query:
            evidence = compressor.retrieve_evidence(case.query, file_id=file_id, top_k=5)
            response = compressor._generate_skeleton(
                file_id=file_id,
                query=case.query,
                anchor_node_ids=set(evidence.node_ids),
            )

        ratio = float(response.compression_ratio)
        savings = _savings_pct(ratio)
        selected_node_ids = {
            node_id
            for node_id, summary in response.node_map.items()
            if isinstance(summary, str) and summary.startswith("ANCHOR:")
        }
        quality_available = bool(case.query and mode in {"query_guided", "evidence_aware"})
        precision_at_k = 0.0
        recall_at_k = 0.0
        f1_at_k = 0.0
        if quality_available:
            precision_at_k, recall_at_k, f1_at_k = _quality_overlap_metrics(
                compressor=compressor,
                file_id=file_id,
                query=case.query or "",
                selected_node_ids=selected_node_ids,
                top_k=5,
            )

        result = BenchmarkResult(
            case_id=case.case_id,
            name=case.name,
            mode=mode,
            original_tokens=response.total_tokens,
            skeleton_tokens=response.skeleton_tokens,
            compression_ratio=ratio,
            token_savings_pct=savings,
            meets_ratio_target=ratio >= case.min_compression_ratio,
            meets_savings_target=savings >= case.min_token_savings_pct,
            quality_metrics_available=quality_available,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
        )
        details.append(
            BenchmarkExecutionDetail(
                case=case,
                result=result,
                skeleton_text=response.skeleton_text,
                node_map=dict(response.node_map),
            )
        )

    return details


def run_benchmark_cases(
    cases: Iterable[BenchmarkCase],
    mode: str = "baseline",
    similarity_threshold: float = 0.75,
    skeleton_ratio: float = 0.2,
) -> BenchmarkSummary:
    """Run all benchmark cases and return aggregate summary."""
    details = run_benchmark_cases_detailed(
        cases,
        mode=mode,
        similarity_threshold=similarity_threshold,
        skeleton_ratio=skeleton_ratio,
    )
    results = [detail.result for detail in details]
    passed = sum(1 for item in results if item.passed)
    failed = len(results) - passed
    return BenchmarkSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=failed,
        avg_compression_ratio=mean(item.compression_ratio for item in results),
        avg_token_savings_pct=mean(item.token_savings_pct for item in results),
        avg_precision_at_k=mean(item.precision_at_k for item in results) if results else 0.0,
        avg_recall_at_k=mean(item.recall_at_k for item in results) if results else 0.0,
        avg_f1_at_k=mean(item.f1_at_k for item in results) if results else 0.0,
        quality_cases_count=sum(1 for item in results if item.quality_metrics_available),
        results=results,
    )


def filter_cases(
    cases: Iterable[BenchmarkCase], case_ids: Optional[Iterable[str]]
) -> List[BenchmarkCase]:
    """Select a subset of cases by case_id. Returns all if case_ids is empty."""
    case_list = list(cases)
    if not case_ids:
        return case_list
    wanted = set(case_ids)
    return [case for case in case_list if case.case_id in wanted]


def summary_to_dict(summary: BenchmarkSummary) -> dict:
    """Convert summary to a JSON-serializable dictionary."""
    payload = asdict(summary)
    for item in payload["results"]:
        item["passed"] = item["meets_ratio_target"] and item["meets_savings_target"]
    payload["all_passed"] = summary.all_passed
    return payload


def write_summary(summary: BenchmarkSummary, output_path: Path | str) -> Path:
    """Write benchmark summary to JSON and return resolved output path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_to_dict(summary), indent=2), encoding="utf-8")
    return path


def run_method_comparison(cases: Iterable[BenchmarkCase]) -> dict[str, object]:
    """Compare semantic baseline against a local extractive baseline."""
    case_list = list(cases)
    semantic_summary = run_benchmark_cases(case_list, mode="baseline")
    extractive = ExtractiveCompressor()
    extractive_ratios: list[float] = []
    extractive_savings: list[float] = []
    for case in case_list:
        result = extractive.compress_text(case.text, query=case.query, target_tokens=None)
        ratio = (
            result["original_tokens"] / result["compressed_tokens"]
            if result["compressed_tokens"] > 0
            else 0.0
        )
        extractive_ratios.append(ratio)
        extractive_savings.append(_savings_pct(ratio))

    return {
        "total_cases": semantic_summary.total_cases,
        "methods": {
            "semantic_baseline": {
                "avg_compression_ratio": semantic_summary.avg_compression_ratio,
                "avg_token_savings_pct": semantic_summary.avg_token_savings_pct,
            },
            "extractive_baseline": {
                "avg_compression_ratio": mean(extractive_ratios) if extractive_ratios else 0.0,
                "avg_token_savings_pct": mean(extractive_savings) if extractive_savings else 0.0,
            },
        },
    }
