#!/usr/bin/env python3
"""Run fixed-corpus token-savings benchmarks and emit JSON report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_harness import (  # noqa: E402
    default_corpus_path,
    filter_cases,
    load_benchmark_cases,
    run_benchmark_cases,
    write_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run token-savings regression benchmark on fixed corpus.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=default_corpus_path(),
        help="Path to benchmark corpus JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "latest.json",
        help="Where to write benchmark summary JSON.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case ID to run (repeatable). If omitted, runs all cases.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Semantic similarity threshold for graph edges.",
    )
    parser.add_argument(
        "--skeleton-ratio",
        type=float,
        default=0.2,
        help="Skeleton ratio used for compression.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return success even if one or more benchmark cases fail golden thresholds.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    cases = load_benchmark_cases(args.corpus)
    selected_cases = filter_cases(cases, args.case)
    if not selected_cases:
        print("No benchmark cases selected. Check --case values.", file=sys.stderr)
        return 2

    summary = run_benchmark_cases(
        selected_cases,
        similarity_threshold=args.similarity_threshold,
        skeleton_ratio=args.skeleton_ratio,
    )
    output_path = write_summary(summary, args.output)

    print(f"Benchmark report: {output_path}")
    print(
        "Cases: "
        f"{summary.passed_cases}/{summary.total_cases} passed | "
        f"avg ratio={summary.avg_compression_ratio:.2f}x | "
        f"avg savings={summary.avg_token_savings_pct:.1f}%"
    )
    for result in summary.results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"- [{status}] {result.case_id}: ratio={result.compression_ratio:.2f}x "
            f"savings={result.token_savings_pct:.1f}% "
            f"(target {result.meets_ratio_target}/{result.meets_savings_target})"
        )

    if summary.all_passed or args.allow_failures:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
