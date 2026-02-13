"""Tests for CI workflow definitions related to skill scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_CI = ROOT / ".github" / "workflows" / "skill-ci.yml"


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
