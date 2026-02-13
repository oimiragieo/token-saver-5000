"""Benchmark regression guard utilities."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class GuardViolation:
    """One benchmark guard failure."""

    mode: str
    metric: str
    message: str


def load_json(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_report_against_thresholds(
    *,
    mode: str,
    report: Dict[str, Any],
    thresholds: Dict[str, Any],
) -> List[GuardViolation]:
    """Evaluate one benchmark report against threshold config."""
    violations: List[GuardViolation] = []
    mode_thresholds = thresholds.get(mode, {})
    if not mode_thresholds:
        violations.append(
            GuardViolation(
                mode=mode, metric="config", message=f"Missing thresholds for mode '{mode}'"
            )
        )
        return violations

    min_avg_ratio = float(mode_thresholds.get("min_avg_compression_ratio", 0.0))
    min_avg_savings = float(mode_thresholds.get("min_avg_token_savings_pct", 0.0))

    avg_ratio = float(report.get("avg_compression_ratio", 0.0))
    avg_savings = float(report.get("avg_token_savings_pct", 0.0))
    if avg_ratio < min_avg_ratio:
        violations.append(
            GuardViolation(
                mode=mode,
                metric="avg_compression_ratio",
                message=f"{avg_ratio:.3f} < required {min_avg_ratio:.3f}",
            )
        )
    if avg_savings < min_avg_savings:
        violations.append(
            GuardViolation(
                mode=mode,
                metric="avg_token_savings_pct",
                message=f"{avg_savings:.3f} < required {min_avg_savings:.3f}",
            )
        )

    report_cases = {item["case_id"]: item for item in report.get("results", [])}
    per_case = mode_thresholds.get("per_case", {})
    for case_id, case_limits in per_case.items():
        if case_id not in report_cases:
            violations.append(
                GuardViolation(
                    mode=mode,
                    metric=f"case:{case_id}",
                    message="Case missing from report",
                )
            )
            continue
        case_report = report_cases[case_id]
        min_case_ratio = float(case_limits.get("min_compression_ratio", 0.0))
        min_case_savings = float(case_limits.get("min_token_savings_pct", 0.0))

        case_ratio = float(case_report.get("compression_ratio", 0.0))
        case_savings = float(case_report.get("token_savings_pct", 0.0))
        if case_ratio < min_case_ratio:
            violations.append(
                GuardViolation(
                    mode=mode,
                    metric=f"{case_id}.compression_ratio",
                    message=f"{case_ratio:.3f} < required {min_case_ratio:.3f}",
                )
            )
        if case_savings < min_case_savings:
            violations.append(
                GuardViolation(
                    mode=mode,
                    metric=f"{case_id}.token_savings_pct",
                    message=f"{case_savings:.3f} < required {min_case_savings:.3f}",
                )
            )

    return violations
