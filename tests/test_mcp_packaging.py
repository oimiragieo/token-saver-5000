import json
from pathlib import Path

from src.mcp_install import (
    SERVER_KEY,
    WORKSPACE_FOLDER_VAR,
    build_server_config,
    build_status_report,
    detect_project_mcp_config_path,
    inspect_mcp_installation,
    install_portable_project_mcp,
    install_project_mcp,
    install_mcp,
    remove_mcp_server,
    recommend_setup_target,
    render_mcp_config,
    resolve_project_root_for_cli,
    merge_mcp_server,
    uninstall_mcp,
    uninstall_project_mcp,
)

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_mcp_console_scripts():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "[project.scripts]" in text
    assert 'token-saver-mcp = "src.server:main"' in text
    assert 'token-saver-install-mcp = "src.mcp_install:main"' in text
    assert 'token-saver-setup = "src.setup_cli:main"' in text


def test_claude_desktop_config_example_uses_token_saver_command():
    data = json.loads(
        (ROOT / "config" / "claude_desktop_config.example.json").read_text(encoding="utf-8")
    )

    assert "mcpServers" in data
    assert SERVER_KEY in data["mcpServers"]
    assert data["mcpServers"][SERVER_KEY]["command"] == "token-saver-mcp"
    assert data["mcpServers"][SERVER_KEY]["args"] == []


def test_claude_code_config_example_uses_token_saver_command():
    data = json.loads(
        (ROOT / "config" / "claude_code_mcp.example.json").read_text(encoding="utf-8")
    )

    assert "mcpServers" in data
    assert SERVER_KEY in data["mcpServers"]
    assert data["mcpServers"][SERVER_KEY]["command"] == "token-saver-mcp"
    assert data["mcpServers"][SERVER_KEY]["args"] == []


def test_merge_mcp_server_adds_token_saver_entry():
    merged = merge_mcp_server(
        {"mcpServers": {"other": {"command": "x"}}}, SERVER_KEY, {"command": "y"}
    )

    assert merged["mcpServers"]["other"]["command"] == "x"
    assert merged["mcpServers"][SERVER_KEY]["command"] == "y"


def test_remove_mcp_server_preserves_other_servers():
    updated = remove_mcp_server(
        {
            "mcpServers": {
                SERVER_KEY: {"command": "token-saver-mcp"},
                "other": {"command": "npx"},
            }
        },
        SERVER_KEY,
    )

    assert SERVER_KEY not in updated["mcpServers"]
    assert updated["mcpServers"]["other"]["command"] == "npx"


def test_remove_mcp_server_cleans_empty_mcp_servers_block():
    updated = remove_mcp_server(
        {"mcpServers": {SERVER_KEY: {"command": "token-saver-mcp"}}}, SERVER_KEY
    )

    assert "mcpServers" not in updated


def test_install_mcp_writes_token_saver_entry(tmp_path: Path):
    config_path = tmp_path / "claude_desktop_config.json"
    root = ROOT

    written = install_mcp(config_path=config_path, root=root)
    data = json.loads(written.read_text(encoding="utf-8"))

    assert written == config_path
    assert SERVER_KEY in data["mcpServers"]
    assert data["mcpServers"][SERVER_KEY]["cwd"] == str(root)
    assert "command" in data["mcpServers"][SERVER_KEY]
    assert "args" in data["mcpServers"][SERVER_KEY]


def test_uninstall_mcp_removes_token_saver_and_preserves_other_servers(tmp_path: Path):
    config_path = tmp_path / "claude_desktop_config.json"
    config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    SERVER_KEY: {"command": "token-saver-mcp"},
                    "other": {"command": "npx"},
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    changed = uninstall_mcp(config_path=config_path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    assert changed is True
    assert SERVER_KEY not in data["mcpServers"]
    assert data["mcpServers"]["other"]["command"] == "npx"


def test_build_server_config_sets_pythonpath(tmp_path: Path):
    config = build_server_config(tmp_path)

    assert config["cwd"] == str(tmp_path.resolve())
    assert config["env"]["PYTHONPATH"] == str(tmp_path.resolve())


def test_build_server_config_portable_uses_workspace_folder():
    config = build_server_config(portable=True)

    assert config["command"] == "python"
    assert config["args"] == ["-m", "src.server"]
    assert config["cwd"] == WORKSPACE_FOLDER_VAR
    assert config["env"]["PYTHONPATH"] == WORKSPACE_FOLDER_VAR


def test_detect_project_mcp_config_path_uses_dot_claude(tmp_path: Path):
    path = detect_project_mcp_config_path(tmp_path)

    assert path == tmp_path / ".claude" / ".mcp.json"


def test_render_mcp_config_returns_full_mcp_payload(tmp_path: Path):
    data = render_mcp_config(tmp_path)

    assert list(data) == ["mcpServers"]
    assert SERVER_KEY in data["mcpServers"]
    assert data["mcpServers"][SERVER_KEY]["cwd"] == str(tmp_path.resolve())


def test_render_mcp_config_supports_custom_server_key(tmp_path: Path):
    data = render_mcp_config(tmp_path, server_key="token-saver-local")

    assert "token-saver-local" in data["mcpServers"]
    assert SERVER_KEY not in data["mcpServers"]


def test_render_mcp_config_portable_uses_workspace_folder():
    data = render_mcp_config(portable=True)

    assert data["mcpServers"][SERVER_KEY]["cwd"] == WORKSPACE_FOLDER_VAR
    assert data["mcpServers"][SERVER_KEY]["command"] == "python"


def test_install_project_mcp_writes_repo_scoped_config(tmp_path: Path):
    root = tmp_path / "repo"
    config_path = root / ".claude" / ".mcp.json"

    written = install_project_mcp(root=root)
    data = json.loads(written.read_text(encoding="utf-8"))

    assert written == config_path
    assert data["mcpServers"][SERVER_KEY]["cwd"] == str(root.resolve())


def test_uninstall_project_mcp_deletes_empty_config_file(tmp_path: Path):
    root = tmp_path / "repo"

    written = install_project_mcp(root=root)
    assert written.exists() is True

    changed = uninstall_project_mcp(root=root)

    assert changed is True
    assert written.exists() is False


def test_install_project_mcp_preserves_existing_servers(tmp_path: Path):
    root = tmp_path / "repo"
    config_path = root / ".claude" / ".mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"mcpServers": {"repo": {"command": "npx", "args": ["x"]}}}, indent=2),
        encoding="utf-8",
    )

    written = install_project_mcp(root=root)
    data = json.loads(written.read_text(encoding="utf-8"))

    assert written == config_path
    assert data["mcpServers"]["repo"]["command"] == "npx"
    assert data["mcpServers"][SERVER_KEY]["cwd"] == str(root.resolve())


def test_install_portable_project_mcp_uses_workspace_folder(tmp_path: Path):
    root = tmp_path / "repo"

    written = install_portable_project_mcp(root=root)
    data = json.loads(written.read_text(encoding="utf-8"))

    assert data["mcpServers"][SERVER_KEY]["cwd"] == WORKSPACE_FOLDER_VAR
    assert data["mcpServers"][SERVER_KEY]["env"]["PYTHONPATH"] == WORKSPACE_FOLDER_VAR


def test_install_portable_project_mcp_preserves_existing_servers(tmp_path: Path):
    root = tmp_path / "repo"
    config_path = root / ".claude" / ".mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps({"mcpServers": {"repo": {"command": "npx", "args": ["x"]}}}, indent=2),
        encoding="utf-8",
    )

    written = install_portable_project_mcp(root=root)
    data = json.loads(written.read_text(encoding="utf-8"))

    assert written == config_path
    assert data["mcpServers"]["repo"]["command"] == "npx"
    assert data["mcpServers"][SERVER_KEY]["cwd"] == WORKSPACE_FOLDER_VAR


def test_inspect_mcp_installation_reports_missing_configs(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))

    status = inspect_mcp_installation(root=tmp_path / "repo")

    assert status["server_key"] == SERVER_KEY
    assert status["desktop"]["configured"] is False
    assert status["project"]["configured"] is False
    assert status["project"]["portable"] is False


def test_inspect_mcp_installation_reports_portable_project_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "appdata"))
    root = tmp_path / "repo"

    install_portable_project_mcp(root=root)
    status = inspect_mcp_installation(root=root)

    assert status["project"]["configured"] is True
    assert status["project"]["portable"] is True


def test_inspect_mcp_installation_reports_desktop_config(tmp_path: Path, monkeypatch):
    appdata = tmp_path / "appdata"
    monkeypatch.setenv("APPDATA", str(appdata))
    desktop_path = appdata / "Claude" / "claude_desktop_config.json"

    install_mcp(config_path=desktop_path, root=tmp_path / "repo")
    # Ensure inspect uses the same path (detect_claude_config_path is platform-dependent)
    monkeypatch.setattr("src.mcp_install.detect_claude_config_path", lambda: desktop_path)
    status = inspect_mcp_installation(root=tmp_path / "repo")

    assert status["desktop"]["configured"] is True
    assert status["desktop"]["exists"] is True


def test_resolve_project_root_for_cli_defaults_to_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_project_root_for_cli(None) == tmp_path.resolve()


def test_recommend_setup_target_prefers_portable_project_for_repo_like_directory(tmp_path: Path):
    (tmp_path / ".git").mkdir()

    assert recommend_setup_target(tmp_path) == "portable-project"


def test_recommend_setup_target_defaults_to_desktop_for_plain_directory(tmp_path: Path):
    assert recommend_setup_target(tmp_path) == "desktop"


def test_build_status_report_recommends_follow_up_command():
    report = build_status_report(
        {
            "command_available": True,
            "desktop": {"configured": False, "path": "desktop.json"},
            "project": {
                "configured": False,
                "portable": False,
                "path": ".claude/.mcp.json",
            },
        },
        recommended_target="portable-project",
    )

    assert "Recommended next step" in report
    assert "token-saver-setup --portable-project" in report
