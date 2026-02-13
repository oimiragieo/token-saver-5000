#!/usr/bin/env python3
"""Fail CI if benchmark reports regress below configured thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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
    parser.add_argument(
        "--summary-file",
        type=Path,
        help="Optional path to write markdown summary report.",
    )
    parser.add_argument(
        "--strict-case-set",
        action="store_true",
        help=(
            "Fail when report case IDs differ from threshold per_case IDs "
            "(missing or unexpected cases)."
        ),
    )
    return parser


def _evaluate_case_set(
    *,
    mode: str,
    report: dict,
    mode_thresholds: dict,
) -> List[str]:
    expected_case_ids = {
        str(case_id)
        for case_id in mode_thresholds.get("per_case", {}).keys()
        if str(case_id).strip()
    }
    if not expected_case_ids:
        return []

    actual_case_ids = {
        str(item.get("case_id")).strip()
        for item in report.get("results", [])
        if str(item.get("case_id", "")).strip()
    }
    missing_case_ids = sorted(expected_case_ids - actual_case_ids)
    unexpected_case_ids = sorted(actual_case_ids - expected_case_ids)
    violations: List[str] = []
    if missing_case_ids:
        violations.append(
            f"[{mode}] case_set: missing expected cases: {', '.join(missing_case_ids)}"
        )
    if unexpected_case_ids:
        violations.append(f"[{mode}] case_set: unexpected cases: {', '.join(unexpected_case_ids)}")
    return violations


def _build_markdown_summary(
    *,
    modes: List[str],
    thresholds: dict,
    reports_dir: Path,
    all_violations: List[str],
) -> str:
    lines = ["## Benchmark Guard Summary", ""]
    for mode in modes:
        mode_thresholds = thresholds.get(mode, {})
        report_path = reports_dir / f"latest_{mode}.json"
        lines.append(f"### Mode: `{mode}`")
        if not report_path.exists():
            lines.append(f"- Status: FAIL (missing report `{report_path}`)")
            lines.append("")
            continue

        report = load_json(report_path)
        avg_ratio = float(report.get("avg_compression_ratio", 0.0))
        avg_savings = float(report.get("avg_token_savings_pct", 0.0))
        min_avg_ratio = float(mode_thresholds.get("min_avg_compression_ratio", 0.0))
        min_avg_savings = float(mode_thresholds.get("min_avg_token_savings_pct", 0.0))
        lines.append(
            f"- avg_compression_ratio: `{avg_ratio:.3f}` (threshold `{min_avg_ratio:.3f}`, delta `{avg_ratio - min_avg_ratio:+.3f}`)"
        )
        lines.append(
            f"- avg_token_savings_pct: `{avg_savings:.3f}` (threshold `{min_avg_savings:.3f}`, delta `{avg_savings - min_avg_savings:+.3f}`)"
        )

        per_case = mode_thresholds.get("per_case", {})
        report_cases = {item.get("case_id"): item for item in report.get("results", [])}
        if per_case:
            lines.append("- per_case:")
            for case_id, case_limits in per_case.items():
                case = report_cases.get(case_id, {})
                ratio = float(case.get("compression_ratio", 0.0))
                savings = float(case.get("token_savings_pct", 0.0))
                min_ratio = float(case_limits.get("min_compression_ratio", 0.0))
                min_savings = float(case_limits.get("min_token_savings_pct", 0.0))
                lines.append(
                    f"  - `{case_id}` ratio `{ratio:.3f}` vs `{min_ratio:.3f}` (delta `{ratio - min_ratio:+.3f}`), "
                    f"savings `{savings:.3f}` vs `{min_savings:.3f}` (delta `{savings - min_savings:+.3f}`)"
                )
        lines.append("")

    if all_violations:
        lines.append("### Violations")
        for violation in all_violations:
            lines.append(f"- {violation}")
    else:
        lines.append("### Result")
        lines.append("- PASS: all benchmark guard thresholds satisfied")
    lines.append("")
    return "\n".join(lines)


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
        if args.strict_case_set:
            mode_thresholds = thresholds.get(mode, {})
            all_violations.extend(
                _evaluate_case_set(
                    mode=mode,
                    report=report,
                    mode_thresholds=mode_thresholds,
                )
            )
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
        exit_code = 1
    else:
        print("[PASS] Benchmark guard checks passed for all modes")
        exit_code = 0

    if args.summary_file:
        markdown = _build_markdown_summary(
            modes=modes,
            thresholds=thresholds,
            reports_dir=args.reports_dir,
            all_violations=all_violations,
        )
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(markdown, encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
