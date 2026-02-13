#!/usr/bin/env python3
"""Fail CI if benchmark reports regress below configured thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_guard import evaluate_report_against_thresholds, load_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check benchmark reports against guard thresholds."
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks" / "golden_thresholds.json",
        help="Path to benchmark threshold config.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "benchmarks",
        help="Directory containing latest_<mode>.json reports.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,query_guided,evidence_aware",
        help="Comma-separated modes to validate.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    thresholds_payload = load_json(args.thresholds)
    thresholds = thresholds_payload.get("modes", {})
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]

    all_violations = []
    for mode in modes:
        report_path = args.reports_dir / f"latest_{mode}.json"
        if not report_path.exists():
            all_violations.append(f"[{mode}] missing report: {report_path}")
            continue
        report = load_json(report_path)
        violations = evaluate_report_against_thresholds(
            mode=mode,
            report=report,
            thresholds=thresholds,
        )
        if violations:
            for violation in violations:
                all_violations.append(f"[{violation.mode}] {violation.metric}: {violation.message}")
        else:
            print(f"[PASS] {mode}: benchmark thresholds satisfied")

    if all_violations:
        print("[FAIL] Benchmark guard violations detected:", file=sys.stderr)
        for item in all_violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("[PASS] Benchmark guard checks passed for all modes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
