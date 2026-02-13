"""Tests for fixed-corpus benchmark harness."""

from pathlib import Path

from src.benchmark_harness import (
    default_corpus_path,
    filter_cases,
    load_benchmark_cases,
    run_benchmark_cases,
    summary_to_dict,
    write_summary,
)


def test_load_benchmark_cases_has_expected_fixture():
    corpus_path = default_corpus_path()
    assert corpus_path.exists()

    cases = load_benchmark_cases(corpus_path)
    assert len(cases) >= 2
    assert all(case.case_id for case in cases)
    assert all(case.min_compression_ratio > 0 for case in cases)


def test_filter_cases_selects_subset():
    cases = load_benchmark_cases(default_corpus_path())
    filtered = filter_cases(cases, ["medium_architecture"])
    assert len(filtered) == 1
    assert filtered[0].case_id == "medium_architecture"


def test_run_benchmark_cases_meets_golden_thresholds():
    cases = load_benchmark_cases(default_corpus_path())
    selected = filter_cases(cases, ["medium_architecture", "large_repetitive_tech"])

    summary = run_benchmark_cases(selected)
    assert summary.total_cases == 2
    assert summary.failed_cases == 0
    assert summary.avg_compression_ratio >= 2.0
    assert summary.avg_token_savings_pct >= 60.0


def test_summary_to_dict_and_write_output(tmp_path: Path):
    cases = load_benchmark_cases(default_corpus_path())
    selected = filter_cases(cases, ["medium_architecture"])
    summary = run_benchmark_cases(selected)

    payload = summary_to_dict(summary)
    assert payload["total_cases"] == 1
    assert "results" in payload
    assert "all_passed" in payload

    output_file = tmp_path / "benchmark_report.json"
    written = write_summary(summary, output_file)
    assert written.exists()
    assert written == output_file
