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

from .semantic_compressor import SemanticCompressor


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark input with minimum expected compression targets."""

    case_id: str
    name: str
    text: str
    min_compression_ratio: float
    min_token_savings_pct: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured output for a single benchmark case."""

    case_id: str
    name: str
    original_tokens: int
    skeleton_tokens: int
    compression_ratio: float
    token_savings_pct: float
    meets_ratio_target: bool
    meets_savings_target: bool

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
    results: List[BenchmarkResult]

    @property
    def all_passed(self) -> bool:
        return self.failed_cases == 0


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
            )
        )
    return cases


def _savings_pct(compression_ratio: float) -> float:
    if compression_ratio <= 0:
        return 0.0
    return (1.0 - (1.0 / compression_ratio)) * 100.0


def run_benchmark_cases(
    cases: Iterable[BenchmarkCase],
    similarity_threshold: float = 0.75,
    skeleton_ratio: float = 0.2,
) -> BenchmarkSummary:
    """Run all benchmark cases and return aggregate summary."""
    compressor = SemanticCompressor(
        similarity_threshold=similarity_threshold,
        skeleton_ratio=skeleton_ratio,
    )

    results: List[BenchmarkResult] = []
    for case in cases:
        response = compressor.ingest_file(case.text, f"bench_{case.case_id}")
        ratio = float(response.compression_ratio)
        savings = _savings_pct(ratio)

        result = BenchmarkResult(
            case_id=case.case_id,
            name=case.name,
            original_tokens=response.total_tokens,
            skeleton_tokens=response.skeleton_tokens,
            compression_ratio=ratio,
            token_savings_pct=savings,
            meets_ratio_target=ratio >= case.min_compression_ratio,
            meets_savings_target=savings >= case.min_token_savings_pct,
        )
        results.append(result)

    passed = sum(1 for item in results if item.passed)
    failed = len(results) - passed
    return BenchmarkSummary(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        total_cases=len(results),
        passed_cases=passed,
        failed_cases=failed,
        avg_compression_ratio=mean(item.compression_ratio for item in results),
        avg_token_savings_pct=mean(item.token_savings_pct for item in results),
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
