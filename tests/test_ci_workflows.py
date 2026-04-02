"""Tests for CI workflow definitions related to skill scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
LEGACY_TEST_WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
LEGACY_LINT_WORKFLOW = ROOT / ".github" / "workflows" / "lint.yml"
SKILL_CI = ROOT / ".github" / "workflows" / "skill-ci.yml"
BENCHMARK_GUARD = ROOT / ".github" / "workflows" / "benchmark-guard.yml"
MCP_PROFILE_GUARD = ROOT / ".github" / "workflows" / "mcp-profile-guard.yml"


def test_ci_workflow_exists_and_runs_canonical_validation():
    assert CI_WORKFLOW.exists()
    content = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "name: CI" in content
    assert "workflow_dispatch:" in content
    assert 'python-version: ["3.10", "3.11", "3.12"]' in content
    assert "python -m black --check src tests scripts" in content
    assert "python -m ruff check src tests scripts" in content
    assert (
        'pytest -q -o addopts="" tests/test_ci_workflows.py tests/test_mcp_packaging.py' in content
    )
    assert "python -m pytest tests/ -q --no-cov --ignore=tests/test_performance.py" in content
    assert "python -m build" in content
    assert "python -m twine check dist/*" in content
    assert "token-saver-install-mcp" in content
    assert "--print-config" in content


def test_legacy_test_workflow_is_manual_deprecated_shim():
    assert LEGACY_TEST_WORKFLOW.exists()
    content = LEGACY_TEST_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Legacy Test (Deprecated)" in content
    assert "workflow_dispatch:" in content
    assert "pull_request:" not in content
    assert "push:" not in content
    assert "ci.yml" in content
    assert "deprecated compatibility workflow" in content


def test_legacy_lint_workflow_is_manual_deprecated_shim():
    assert LEGACY_LINT_WORKFLOW.exists()
    content = LEGACY_LINT_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Legacy Lint (Deprecated)" in content
    assert "workflow_dispatch:" in content
    assert "pull_request:" not in content
    assert "push:" not in content
    assert "ci.yml" in content
    assert "deprecated compatibility workflow" in content


def test_skill_ci_workflow_exists():
    assert SKILL_CI.exists()


def test_skill_ci_workflow_has_path_filters_and_commands():
    content = SKILL_CI.read_text(encoding="utf-8")
    assert "name: Skill CI" in content
    assert "pull_request:" in content
    assert "paths:" in content
    assert "scripts/skills/**" in content
    assert "skills/**" in content
    assert "tests/test_skill_scripts.py" in content
    assert "python -m black --check" in content
    assert "python -m ruff check" in content
    assert (
        'pytest -q -o addopts="" tests/test_skill_scripts.py tests/test_help_handlers.py' in content
    )


def test_benchmark_guard_workflow_exists_and_has_regression_gate():
    assert BENCHMARK_GUARD.exists()
    content = BENCHMARK_GUARD.read_text(encoding="utf-8")

    assert "name: Benchmark Guard" in content
    assert "pull_request:" in content
    assert "paths:" in content
    assert "src/benchmark_harness.py" in content
    assert "scripts/benchmarks/**" in content
    assert "tests/fixtures/benchmark_corpus.json" in content
    assert (
        "python scripts/benchmarks/run_benchmarks.py --compare baseline,query_guided,evidence_aware"
        in content
    )
    assert "python scripts/benchmarks/check_benchmark_guard.py" in content
    assert "--strict-case-set" in content
    assert "GITHUB_STEP_SUMMARY" in content


def test_mcp_profile_guard_workflow_exists_and_checks_core_contract():
    assert MCP_PROFILE_GUARD.exists()
    content = MCP_PROFILE_GUARD.read_text(encoding="utf-8")

    assert "name: MCP Profile Guard" in content
    assert "pull_request:" in content
    assert "src/handlers/mcp_core.py" in content
    assert "src/server.py" in content
    assert "tests/test_tool_profiles.py" in content
    assert "tests/test_server_unit.py" in content
    assert (
        'pytest -q -o addopts="" tests/test_tool_profiles.py tests/test_server_unit.py' in content
    )
