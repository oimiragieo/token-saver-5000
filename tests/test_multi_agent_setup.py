"""Tests for multi-agent MCP configuration (install/uninstall/inspect)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp_install import (
    AGENT_CONFIGS,
    SERVER_KEY,
    _build_agents_md_section,
    _format_all_agents_report,
    detect_agent_config_path,
    inspect_all_agents,
    install_agent_config,
    uninstall_agent_config,
)


class TestDetectAgentConfigPath:
    def test_cursor_path(self, tmp_path: Path):
        path = detect_agent_config_path("cursor", root=tmp_path)
        assert path == tmp_path / ".cursor" / "mcp.json"

    def test_windsurf_path(self, tmp_path: Path):
        path = detect_agent_config_path("windsurf", root=tmp_path)
        assert path == tmp_path / ".windsurf" / "mcp.json"

    def test_cline_path(self, tmp_path: Path):
        path = detect_agent_config_path("cline", root=tmp_path)
        assert path == tmp_path / ".cline" / "mcp.json"

    def test_vscode_copilot_path(self, tmp_path: Path):
        path = detect_agent_config_path("vscode-copilot", root=tmp_path)
        assert path == tmp_path / ".vscode" / "mcp.json"

    def test_codex_path(self, tmp_path: Path):
        path = detect_agent_config_path("codex", root=tmp_path)
        assert path == tmp_path / "AGENTS.md"

    def test_gemini_path(self, tmp_path: Path):
        path = detect_agent_config_path("gemini", root=tmp_path)
        assert path.name == "settings.json"
        assert ".gemini" in str(path)

    def test_unknown_agent_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown agent"):
            detect_agent_config_path("nonexistent", root=tmp_path)


class TestInstallAgentConfig:
    def test_install_cursor_creates_json(self, tmp_path: Path):
        path = install_agent_config("cursor", root=tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert SERVER_KEY in data["mcpServers"]

    def test_install_windsurf_creates_json(self, tmp_path: Path):
        path = install_agent_config("windsurf", root=tmp_path)
        data = json.loads(path.read_text())
        assert SERVER_KEY in data["mcpServers"]

    def test_install_cline_creates_json(self, tmp_path: Path):
        path = install_agent_config("cline", root=tmp_path)
        data = json.loads(path.read_text())
        assert SERVER_KEY in data["mcpServers"]

    def test_install_vscode_copilot_creates_json(self, tmp_path: Path):
        path = install_agent_config("vscode-copilot", root=tmp_path)
        data = json.loads(path.read_text())
        assert SERVER_KEY in data["mcpServers"]

    def test_install_codex_creates_agents_md(self, tmp_path: Path):
        path = install_agent_config("codex", root=tmp_path)
        content = path.read_text()
        assert "## Token Saver MCP Server" in content
        assert "ingest_context" in content

    def test_install_codex_appends_to_existing(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("# My Project\n\nExisting instructions.\n")
        path = install_agent_config("codex", root=tmp_path)
        content = path.read_text()
        assert "Existing instructions." in content
        assert "## Token Saver MCP Server" in content

    def test_install_codex_replaces_existing_section(self, tmp_path: Path):
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text(
            "# My Project\n\n## Token Saver MCP Server\n\nOld config.\n\n## Other Section\n"
        )
        path = install_agent_config("codex", root=tmp_path)
        content = path.read_text()
        assert "Old config." not in content
        assert "## Token Saver MCP Server" in content
        assert "## Other Section" in content

    def test_install_gemini_creates_settings(self, tmp_path: Path):
        config_path = tmp_path / ".gemini" / "settings.json"
        path = install_agent_config("gemini", root=tmp_path, config_path=config_path)
        data = json.loads(path.read_text())
        assert SERVER_KEY in data["mcpServers"]

    def test_install_merges_with_existing_config(self, tmp_path: Path):
        cursor_dir = tmp_path / ".cursor"
        cursor_dir.mkdir()
        config_path = cursor_dir / "mcp.json"
        config_path.write_text(json.dumps({"mcpServers": {"other-server": {"command": "other"}}}))
        install_agent_config("cursor", root=tmp_path)
        data = json.loads(config_path.read_text())
        assert "other-server" in data["mcpServers"]
        assert SERVER_KEY in data["mcpServers"]

    def test_unknown_agent_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Unknown agent"):
            install_agent_config("nonexistent", root=tmp_path)


class TestUninstallAgentConfig:
    def test_uninstall_cursor(self, tmp_path: Path):
        install_agent_config("cursor", root=tmp_path)
        changed = uninstall_agent_config("cursor", root=tmp_path)
        assert changed is True
        path = detect_agent_config_path("cursor", tmp_path)
        # File may be deleted when no servers remain, or have empty mcpServers
        if path.exists():
            data = json.loads(path.read_text())
            assert SERVER_KEY not in data.get("mcpServers", {})
        # If file is gone, the uninstall fully cleaned up — also acceptable

    def test_uninstall_codex(self, tmp_path: Path):
        install_agent_config("codex", root=tmp_path)
        changed = uninstall_agent_config("codex", root=tmp_path)
        assert changed is True
        path = detect_agent_config_path("codex", tmp_path)
        content = path.read_text()
        assert "## Token Saver MCP Server" not in content

    def test_uninstall_nonexistent_returns_false(self, tmp_path: Path):
        changed = uninstall_agent_config("cursor", root=tmp_path)
        assert changed is False


class TestInspectAllAgents:
    def test_returns_all_known_agents(self, tmp_path: Path):
        results = inspect_all_agents(tmp_path)
        for agent in AGENT_CONFIGS:
            assert agent in results
            assert "configured" in results[agent]

    def test_configured_true_when_installed(self, tmp_path: Path):
        install_agent_config("cursor", root=tmp_path)
        results = inspect_all_agents(tmp_path)
        assert results["cursor"]["configured"] is True

    def test_configured_false_when_not_installed(self, tmp_path: Path):
        results = inspect_all_agents(tmp_path)
        assert results["cursor"]["configured"] is False


class TestFormatAllAgentsReport:
    def test_includes_all_agents(self, tmp_path: Path):
        status = inspect_all_agents(tmp_path)
        output = _format_all_agents_report(status)
        assert "Agent Configuration Status" in output
        assert "Cursor" in output
        assert "Windsurf" in output


class TestBuildAgentsMdSection:
    def test_section_contains_tools(self):
        section = _build_agents_md_section()
        assert "ingest_context" in section
        assert "MCP" in section
