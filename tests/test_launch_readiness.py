"""Launch-readiness documentation tests for Phase 10."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_mentions_multi_tenant_and_web_api_positioning():
    readme = _read("README.md").lower()

    assert "multi-tenant" in readme
    assert "workspace_id" in readme
    assert "user_id" in readme
    assert "http server" in readme or "web api" in readme


def test_saas_multi_tenant_guide_exists_and_covers_scope_fields():
    guide = _read("docs/deployment/SAAS_MULTI_TENANT.md").lower()

    for term in ("workspace_id", "user_id", "agent_id", "session_id", "isolation"):
        assert term in guide


def test_workflow_orchestration_guide_exists_and_covers_core_sequences():
    guide = _read("docs/guides/WORKFLOW_ORCHESTRATION.md").lower()

    for term in (
        "should_compress",
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "modulate_region",
        "advise_context",
    ):
        assert term in guide


def test_claude_workspace_strategy_is_consistent_with_gitignore():
    gitignore = _read(".gitignore")
    claude_guide = _read(".claude/CLAUDE.md").lower()

    assert ".claude/*" in gitignore
    assert "!.claude/CLAUDE.md" in gitignore
    assert "\n.claude/\n" not in gitignore
    assert "local-only by default" in claude_guide
    assert "fresh clones" in claude_guide
    assert "active in this repository" not in claude_guide


def test_competitor_analysis_submodules_are_declared_and_documented():
    gitmodules = _read(".gitmodules")
    competitor_readme = _read("artifacts/competitor-analysis/README.md").lower()

    expected_submodules = {
        "Contextual-Compression": "https://github.com/SrGrace/Contextual-Compression.git",
        "LLMLingua": "https://github.com/microsoft/LLMLingua.git",
        "Selective_Context": "https://github.com/liyucheng09/Selective_Context.git",
        "langchain": "https://github.com/langchain-ai/langchain.git",
    }

    for name, url in expected_submodules.items():
        assert f"path = artifacts/competitor-analysis/{name}" in gitmodules
        assert f"url = {url}" in gitmodules

    assert "git submodule update --init --recursive" in competitor_readme
