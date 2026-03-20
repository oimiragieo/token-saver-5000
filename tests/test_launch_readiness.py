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
