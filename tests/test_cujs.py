"""Tests for Critical User Journey (CUJ) baselines.

Verifies that each of the 6 CUJs produces expected results and that
the benchmark script produces a JSON-serializable baseline.

Run with:
    pytest tests/test_cujs.py -v --no-cov
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

# Ensure the project root is on sys.path when running directly
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_cujs import (
    CUJBaseline,
    CUJResult,
    GIT_DIFF_FIXTURE,
    NPM_INSTALL_FIXTURE,
    PYTEST_FIXTURE,
    run_all_cujs,
    run_cuj_1_solo_dev_codebase,
    run_cuj_2_long_document,
    run_cuj_3_cli_output_filtering,
    run_cuj_4_query_focused_search,
    run_cuj_5_session_recovery,
    run_cuj_6_savings_report,
)


# ---------------------------------------------------------------------------
# CUJ 1: Solo Dev Codebase Compression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj1() -> CUJResult:
    return run_cuj_1_solo_dev_codebase(verbose=False)


def test_cuj_1_passes(cuj1: CUJResult) -> None:
    assert cuj1.passed, f"CUJ 1 failed: {cuj1.error}"


def test_cuj_1_finds_relevant_files(cuj1: CUJResult) -> None:
    search_step = next(s for s in cuj1.steps if s.name == "search_then_compress")
    files_matched = search_step.extra.get("files_matched", 0)
    assert files_matched > 0, "Expected at least one file matched by auth/JWT query"


def test_cuj_1_compresses_codebase(cuj1: CUJResult) -> None:
    search_step = next(s for s in cuj1.steps if s.name == "search_then_compress")
    ratio = search_step.extra.get("compression_ratio", 0)
    assert ratio > 1.0, f"Expected compression ratio > 1.0, got {ratio}"


def test_cuj_1_search_beats_naive(cuj1: CUJResult) -> None:
    """Search+compress must produce fewer tokens than naive (all files raw)."""
    assert cuj1.total_output_tokens < cuj1.total_input_tokens, (
        f"Output tokens ({cuj1.total_output_tokens}) should be less than "
        f"input tokens ({cuj1.total_input_tokens})"
    )


def test_cuj_1_savings_above_50pct(cuj1: CUJResult) -> None:
    assert (
        cuj1.total_savings_pct > 50.0
    ), f"Expected > 50% savings for search+compress, got {cuj1.total_savings_pct:.1f}%"


def test_cuj_1_has_configure_step(cuj1: CUJResult) -> None:
    step_names = [s.name for s in cuj1.steps]
    assert "configure_for_client" in step_names


# ---------------------------------------------------------------------------
# CUJ 2: Long Document Compression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj2() -> CUJResult:
    return run_cuj_2_long_document(verbose=False)


def test_cuj_2_passes(cuj2: CUJResult) -> None:
    assert cuj2.passed, f"CUJ 2 failed: {cuj2.error}"


def test_cuj_2_compresses_document(cuj2: CUJResult) -> None:
    compress_step = next(s for s in cuj2.steps if s.name == "compress_text")
    ratio = compress_step.extra.get("compression_ratio", 0)
    assert ratio > 5.0, f"Expected compression ratio > 5.0 on large.txt, got {ratio}"


def test_cuj_2_result_has_steps(cuj2: CUJResult) -> None:
    assert len(cuj2.steps) >= 2, "Expected at least read + compress steps"


def test_cuj_2_savings_positive(cuj2: CUJResult) -> None:
    assert (
        cuj2.total_savings_pct > 0
    ), f"Expected positive savings, got {cuj2.total_savings_pct:.1f}%"


def test_cuj_2_output_tokens_reduced(cuj2: CUJResult) -> None:
    assert cuj2.total_output_tokens < cuj2.total_input_tokens, (
        f"Compressed ({cuj2.total_output_tokens}) should be less than "
        f"original ({cuj2.total_input_tokens})"
    )


# ---------------------------------------------------------------------------
# CUJ 3: CLI Output Filtering
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj3() -> CUJResult:
    return run_cuj_3_cli_output_filtering(verbose=False)


def test_cuj_3_passes(cuj3: CUJResult) -> None:
    assert cuj3.passed, f"CUJ 3 failed: {cuj3.error}"


def test_cuj_3_git_diff_filtered(cuj3: CUJResult) -> None:
    git_step = next((s for s in cuj3.steps if "git_diff" in s.name), None)
    assert git_step is not None, "Expected a git_diff filtering step"
    orig_lines = git_step.extra.get("original_lines", 0)
    filt_lines = git_step.extra.get("filtered_lines", orig_lines)
    assert (
        filt_lines < orig_lines
    ), f"git_diff: filtered lines ({filt_lines}) should be less than original ({orig_lines})"


def test_cuj_3_pytest_focuses_failures(cuj3: CUJResult) -> None:
    pytest_step = next((s for s in cuj3.steps if "pytest" in s.name), None)
    assert pytest_step is not None, "Expected a pytest filtering step"
    # Verify the filter retained failure lines
    from src.cli_output_optimizer import CLIOutputOptimizer

    result = CLIOutputOptimizer().filter(PYTEST_FIXTURE, command_hint="test_output")
    assert (
        "FAILED" in result.filtered_text or "failed" in result.filtered_text
    ), "pytest filter output should contain failure information"


def test_cuj_3_npm_strips_progress(cuj3: CUJResult) -> None:
    npm_step = next((s for s in cuj3.steps if "npm" in s.name), None)
    assert npm_step is not None, "Expected an npm filtering step"
    from src.cli_output_optimizer import CLIOutputOptimizer

    result = CLIOutputOptimizer().filter(NPM_INSTALL_FIXTURE, command_hint="install_output")
    # timing/idealTree lines should be stripped
    assert "npm timing" not in result.filtered_text, "npm timing lines should be stripped"


def test_cuj_3_all_three_steps_present(cuj3: CUJResult) -> None:
    names = [s.name for s in cuj3.steps]
    assert any("git_diff" in n for n in names), "Missing git_diff step"
    assert any("pytest" in n for n in names), "Missing pytest step"
    assert any("npm" in n for n in names), "Missing npm step"


def test_cuj_3_savings_positive(cuj3: CUJResult) -> None:
    assert (
        cuj3.total_savings_pct > 0
    ), f"Expected positive aggregate savings, got {cuj3.total_savings_pct:.1f}%"


# ---------------------------------------------------------------------------
# CUJ 4: Query-Focused Code Search + Compression
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj4() -> CUJResult:
    return run_cuj_4_query_focused_search(verbose=False)


def test_cuj_4_passes(cuj4: CUJResult) -> None:
    assert cuj4.passed, f"CUJ 4 failed: {cuj4.error}"


def test_cuj_4_search_finds_cache(cuj4: CUJResult) -> None:
    search_step = next((s for s in cuj4.steps if s.name == "search_then_compress"), None)
    assert search_step is not None
    matched = search_step.extra.get("matched_files", [])
    assert any(
        "cache" in f.lower() for f in matched
    ), f"Expected cache.py in matched files, got: {matched}"


def test_cuj_4_beats_compress_all(cuj4: CUJResult) -> None:
    compress_all_step = next(s for s in cuj4.steps if s.name == "compress_all_files")
    search_step = next(s for s in cuj4.steps if s.name == "search_then_compress")
    assert search_step.output_tokens < compress_all_step.output_tokens, (
        f"Search+compress ({search_step.output_tokens}) should use fewer tokens "
        f"than compress-all ({compress_all_step.output_tokens})"
    )


def test_cuj_4_three_strategies_compared(cuj4: CUJResult) -> None:
    step_names = [s.name for s in cuj4.steps]
    assert "naive_all_files" in step_names
    assert "compress_all_files" in step_names
    assert "search_then_compress" in step_names


def test_cuj_4_savings_above_90pct(cuj4: CUJResult) -> None:
    assert (
        cuj4.total_savings_pct > 90.0
    ), f"Expected > 90% savings for search+compress vs naive, got {cuj4.total_savings_pct:.1f}%"


# ---------------------------------------------------------------------------
# CUJ 5: Session Recovery After Compaction
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj5() -> CUJResult:
    return run_cuj_5_session_recovery(verbose=False)


def test_cuj_5_passes(cuj5: CUJResult) -> None:
    assert cuj5.passed, f"CUJ 5 failed: {cuj5.error}"


def test_cuj_5_recovery_has_files(cuj5: CUJResult) -> None:
    recover_step = next(s for s in cuj5.steps if s.name == "recover_session")
    count = recover_step.extra.get("ingested_files_count", 0)
    assert count > 0, f"Expected ingested files in recovery, got {count}"


def test_cuj_5_recovery_has_config(cuj5: CUJResult) -> None:
    recover_step = next(s for s in cuj5.steps if s.name == "recover_session")
    config = recover_step.extra.get("client_config")
    assert config is not None, "Expected client_config to be set after recovery"
    assert "model_id" in config


def test_cuj_5_recovery_compact(cuj5: CUJResult) -> None:
    """Recovery summary must be under 500 tokens."""
    recover_step = next(s for s in cuj5.steps if s.name == "recover_session")
    summary_tokens = recover_step.extra.get("summary_tokens", 9999)
    assert summary_tokens < 500, f"Recovery summary should be < 500 tokens, got {summary_tokens}"


def test_cuj_5_savings_vs_reingest(cuj5: CUJResult) -> None:
    assert (
        cuj5.total_output_tokens < cuj5.total_input_tokens
    ), "Recovery summary should use fewer tokens than re-ingesting all original files"


# ---------------------------------------------------------------------------
# CUJ 6: Savings Report (ROI Justification)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuj6() -> CUJResult:
    return run_cuj_6_savings_report(verbose=False)


def test_cuj_6_passes(cuj6: CUJResult) -> None:
    assert cuj6.passed, f"CUJ 6 failed: {cuj6.error}"


def test_cuj_6_dollars_saved_positive(cuj6: CUJResult) -> None:
    report_step = next(s for s in cuj6.steps if s.name == "get_report")
    saved = report_step.extra.get("total_dollars_saved", 0.0)
    assert saved > 0, f"Expected positive dollars saved, got {saved}"


def test_cuj_6_roi_computed(cuj6: CUJResult) -> None:
    report_step = next(s for s in cuj6.steps if s.name == "get_report")
    roi = report_step.extra.get("roi_vs_pro_plan", 0.0)
    assert roi > 0, f"Expected ROI > 0, got {roi}"


def test_cuj_6_breakeven_reasonable(cuj6: CUJResult) -> None:
    report_step = next(s for s in cuj6.steps if s.name == "get_report")
    breakeven = report_step.extra.get("breakeven_operations", 999_999)
    assert breakeven < 1000, f"Expected breakeven < 1000 operations, got {breakeven}"


def test_cuj_6_monthly_savings_exceeds_pro_plan(cuj6: CUJResult) -> None:
    """Projected monthly savings should exceed the $29 Pro plan price."""
    report_step = next(s for s in cuj6.steps if s.name == "get_report")
    monthly = report_step.extra.get("monthly_projected_savings", 0.0)
    assert (
        monthly > 29.0
    ), f"Expected monthly projected savings > $29 (Pro plan), got ${monthly:.2f}"


def test_cuj_6_compression_ratio_above_10x(cuj6: CUJResult) -> None:
    report_step = next(s for s in cuj6.steps if s.name == "get_report")
    ratio = report_step.extra.get("avg_compression_ratio", 0.0)
    assert ratio > 10.0, f"Expected avg compression ratio > 10x, got {ratio}"


# ---------------------------------------------------------------------------
# Cross-cutting: run all CUJs together
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_baseline() -> CUJBaseline:
    return run_all_cujs(verbose=False)


def test_all_cujs_pass(full_baseline: CUJBaseline) -> None:
    failures = [j for j in full_baseline.journeys if not j.passed]
    assert len(failures) == 0, "Some CUJs failed: " + ", ".join(
        f"CUJ{j.journey_id}: {j.error}" for j in failures
    )


def test_baseline_has_six_journeys(full_baseline: CUJBaseline) -> None:
    assert (
        len(full_baseline.journeys) == 6
    ), f"Expected 6 journeys, got {len(full_baseline.journeys)}"


def test_baseline_summary_populated(full_baseline: CUJBaseline) -> None:
    s = full_baseline.summary
    assert s.get("total_journeys") == 6
    assert s.get("passed") == 6
    assert s.get("aggregate_input_tokens", 0) > 0
    assert s.get("aggregate_output_tokens", 0) > 0
    assert s.get("aggregate_savings_pct", 0) > 0


def test_baseline_output_json_serializable(full_baseline: CUJBaseline) -> None:
    raw = asdict(full_baseline)
    serialized = json.dumps(raw, default=str)
    recovered = json.loads(serialized)
    assert recovered["summary"]["total_journeys"] == 6


# ---------------------------------------------------------------------------
# Fixture content sanity checks
# ---------------------------------------------------------------------------


def test_git_diff_fixture_has_sufficient_lines() -> None:
    lines = GIT_DIFF_FIXTURE.splitlines()
    assert len(lines) >= 100, f"Expected >= 100 lines in git diff fixture, got {len(lines)}"


def test_pytest_fixture_has_failures() -> None:
    assert "FAILED" in PYTEST_FIXTURE, "pytest fixture must contain FAILED lines"


def test_npm_fixture_has_warning() -> None:
    assert "warn" in NPM_INSTALL_FIXTURE.lower(), "npm fixture must contain warning lines"
