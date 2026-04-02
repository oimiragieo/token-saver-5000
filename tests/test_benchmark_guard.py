"""Tests for benchmark guard threshold checks."""

import json
import subprocess
import sys
from pathlib import Path

from src.benchmark_guard import evaluate_report_against_thresholds

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_evaluate_report_against_thresholds_passes_with_good_metrics():
    report = {
        "avg_compression_ratio": 8.5,
        "avg_token_savings_pct": 88.0,
        "results": [
            {"case_id": "c1", "compression_ratio": 6.0, "token_savings_pct": 80.0},
        ],
    }
    thresholds = {
        "baseline": {
            "min_avg_compression_ratio": 8.0,
            "min_avg_token_savings_pct": 87.0,
            "per_case": {
                "c1": {"min_compression_ratio": 5.5, "min_token_savings_pct": 79.0},
            },
        }
    }
    assert (
        evaluate_report_against_thresholds(mode="baseline", report=report, thresholds=thresholds)
        == []
    )


def test_evaluate_report_against_thresholds_detects_regression():
    report = {
        "avg_compression_ratio": 7.0,
        "avg_token_savings_pct": 80.0,
        "results": [
            {"case_id": "c1", "compression_ratio": 4.0, "token_savings_pct": 70.0},
        ],
    }
    thresholds = {
        "baseline": {
            "min_avg_compression_ratio": 8.0,
            "min_avg_token_savings_pct": 87.0,
            "per_case": {
                "c1": {"min_compression_ratio": 5.5, "min_token_savings_pct": 79.0},
            },
        }
    }
    violations = evaluate_report_against_thresholds(
        mode="baseline",
        report=report,
        thresholds=thresholds,
    )
    assert violations
    assert any(v.metric == "avg_compression_ratio" for v in violations)


def test_check_benchmark_guard_script_passes_and_fails(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tmp_path / "thresholds.json"

    thresholds_payload = {
        "modes": {
            "baseline": {
                "min_avg_compression_ratio": 8.0,
                "min_avg_token_savings_pct": 87.0,
                "per_case": {},
            }
        }
    }
    _write_json(thresholds, thresholds_payload)

    good_report = {
        "avg_compression_ratio": 8.5,
        "avg_token_savings_pct": 88.0,
        "results": [],
    }
    _write_json(reports_dir / "latest_baseline.json", good_report)

    cmd = [
        sys.executable,
        "scripts/benchmarks/check_benchmark_guard.py",
        "--thresholds",
        str(thresholds),
        "--reports-dir",
        str(reports_dir),
        "--modes",
        "baseline",
    ]
    good = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert good.returncode == 0
    assert "[PASS]" in good.stdout

    bad_report = {
        "avg_compression_ratio": 7.0,
        "avg_token_savings_pct": 80.0,
        "results": [],
    }
    _write_json(reports_dir / "latest_baseline.json", bad_report)
    bad = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert bad.returncode == 1
    assert "[FAIL]" in bad.stderr


def test_check_benchmark_guard_writes_summary_markdown(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tmp_path / "thresholds.json"
    summary_file = tmp_path / "summary.md"

    thresholds_payload = {
        "modes": {
            "baseline": {
                "min_avg_compression_ratio": 8.0,
                "min_avg_token_savings_pct": 87.0,
                "per_case": {"c1": {"min_compression_ratio": 5.0, "min_token_savings_pct": 75.0}},
            }
        }
    }
    _write_json(thresholds, thresholds_payload)
    report = {
        "avg_compression_ratio": 8.5,
        "avg_token_savings_pct": 88.0,
        "results": [
            {"case_id": "c1", "compression_ratio": 6.0, "token_savings_pct": 80.0},
        ],
    }
    _write_json(reports_dir / "latest_baseline.json", report)

    cmd = [
        sys.executable,
        "scripts/benchmarks/check_benchmark_guard.py",
        "--thresholds",
        str(thresholds),
        "--reports-dir",
        str(reports_dir),
        "--modes",
        "baseline",
        "--summary-file",
        str(summary_file),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert summary_file.exists()
    content = summary_file.read_text(encoding="utf-8")
    assert "Benchmark Guard Summary" in content
    assert "baseline" in content
    assert "avg_compression_ratio" in content


def test_check_benchmark_guard_strict_case_set(tmp_path: Path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tmp_path / "thresholds.json"

    thresholds_payload = {
        "modes": {
            "baseline": {
                "min_avg_compression_ratio": 8.0,
                "min_avg_token_savings_pct": 87.0,
                "per_case": {
                    "c1": {"min_compression_ratio": 5.0, "min_token_savings_pct": 75.0},
                    "c2": {"min_compression_ratio": 5.0, "min_token_savings_pct": 75.0},
                },
            }
        }
    }
    _write_json(thresholds, thresholds_payload)
    # Missing c2, and has unexpected c3.
    report = {
        "avg_compression_ratio": 8.5,
        "avg_token_savings_pct": 88.0,
        "results": [
            {"case_id": "c1", "compression_ratio": 6.0, "token_savings_pct": 80.0},
            {"case_id": "c3", "compression_ratio": 6.0, "token_savings_pct": 80.0},
        ],
    }
    _write_json(reports_dir / "latest_baseline.json", report)

    cmd = [
        sys.executable,
        "scripts/benchmarks/check_benchmark_guard.py",
        "--thresholds",
        str(thresholds),
        "--reports-dir",
        str(reports_dir),
        "--modes",
        "baseline",
        "--strict-case-set",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert (
        "missing expected cases" in result.stderr.lower()
        or "unexpected cases" in result.stderr.lower()
    )
