"""Smoke and end-to-end tests for Claude skill utility scripts."""

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "skill_context_sample.txt"
SKILL_ROOT = ROOT / "skills" / "token-saver-context-compression"
SKILL_SCRIPTS = SKILL_ROOT / "scripts"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _run_skill(*script_and_args: str) -> subprocess.CompletedProcess[str]:
    script = SKILL_SCRIPTS / script_and_args[0]
    extra_args = list(script_and_args[1:])
    return subprocess.run(
        [sys.executable, str(script), *extra_args],
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


def test_portable_skill_compress_context_includes_explainability_fields():
    result = _run_skill(
        "compress_context.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--mode",
        "evidence_aware",
        "--query",
        "error handling and retry behavior",
        "--output-format",
        "json",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["segments"], "Expected selected segments in payload"
    first = data["segments"][0]
    assert "score_components" in first
    assert "selection_reason" in first
    assert "relevance" in first["score_components"]
    assert "position" in first["score_components"]
    assert "final" in first["score_components"]


def test_portable_skill_workflow_includes_explainability_fields():
    result = _run_skill(
        "run_skill_workflow.py",
        "--file",
        str(FIXTURE),
        "--file-id",
        "sample_doc",
        "--mode",
        "evidence_aware",
        "--query",
        "retry behavior",
        "--output-format",
        "json",
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["compressed"]["segments"], "Expected selected segments in workflow payload"
    first = data["compressed"]["segments"][0]
    assert "score_components" in first
    assert "selection_reason" in first


def test_skill_benchmark_toon_vs_json_guard_contract():
    result = _run_skill("benchmark_toon_vs_json.py")
    assert result.returncode == 0

    payload = json.loads(result.stdout)
    assert "benchmarks" in payload
    assert "guard" in payload
    assert payload["guard"]["pass"] is True
    assert payload["guard"]["uniform_auto_should_select"] == "toon"
    assert payload["guard"]["mixed_auto_should_select"] == "json"

    uniform = next(item for item in payload["benchmarks"] if item["dataset"] == "uniform_rows")
    mixed = next(item for item in payload["benchmarks"] if item["dataset"] == "mixed_nested")
    assert uniform["auto_selected_format"] == "toon"
    assert mixed["auto_selected_format"] == "json"


def test_skill_scripts_are_self_contained_without_project_src_imports():
    script_files = sorted(SKILL_SCRIPTS.glob("*.py"))
    assert script_files, "Expected skill scripts to exist"

    for script_path in script_files:
        text = script_path.read_text(encoding="utf-8")
        assert "sys.path.insert(" not in text, f"Unexpected sys.path mutation in {script_path.name}"
        assert "from src." not in text, f"Unexpected project import in {script_path.name}"
        assert "import src." not in text, f"Unexpected project import in {script_path.name}"
