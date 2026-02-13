"""Smoke and end-to-end tests for Claude skill utility scripts."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "skill_context_sample.txt"


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


def test_skill_workflow_help():
    result = _run("scripts/skills/run_skill_workflow.py", "--help")
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


def test_profile_tokens_json_output():
    result = _run(
        "scripts/skills/profile_tokens.py", "--file", str(FIXTURE), "--file-id", "sample_doc"
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["file_id"] == "sample_doc"
    assert data["original_tokens"] > 0
    assert data["skeleton_tokens"] > 0
    assert data["compression_ratio"] > 0


def test_compress_context_evidence_aware_output():
    result = _run(
        "scripts/skills/compress_context.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--mode",
        "evidence_aware",
        "--query",
        "error handling and retry behavior",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["mode"] == "evidence_aware"
    assert "skeleton_text" in data
    assert "evidence" in data
    assert "sufficient" in data["evidence"]


def test_validate_evidence_exit_codes():
    good = _run(
        "scripts/skills/validate_evidence.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--query",
        "retry behavior",
        "--min-similarity",
        "0.0",
    )
    assert good.returncode == 0
    good_payload = json.loads(good.stdout)
    assert good_payload["sufficient"] is True

    bad = _run(
        "scripts/skills/validate_evidence.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--query",
        "totally unrelated gibberish xylophone quasar",
        "--min-similarity",
        "0.99",
    )
    assert bad.returncode == 1
    bad_payload = json.loads(bad.stdout)
    assert bad_payload["sufficient"] is False


def test_skill_workflow_end_to_end():
    result = _run(
        "scripts/skills/run_skill_workflow.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--query",
        "retry behavior",
        "--mode",
        "evidence_aware",
        "--min-similarity",
        "0.0",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "profile" in data
    assert "compressed" in data
    assert "evidence_validation" in data
    assert data["compressed"]["mode"] == "evidence_aware"
