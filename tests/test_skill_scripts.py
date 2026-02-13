"""Smoke tests for Claude skill utility scripts."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_profile_tokens_help():
    result = _run("scripts/skills/profile_tokens.py", "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_compress_context_help():
    result = _run("scripts/skills/compress_context.py", "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_validate_evidence_help():
    result = _run("scripts/skills/validate_evidence.py", "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()
