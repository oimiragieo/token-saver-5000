"""Install or render Token Saver 5000 MCP configuration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SERVER_KEY = "token-saver"
WORKSPACE_FOLDER_VAR = "${workspaceFolder}"

# Agent config file locations (relative to project root unless noted)
AGENT_CONFIGS: dict[str, dict[str, Any]] = {
    "claude-desktop": {
        "description": "Claude Desktop (global)",
        "format": "mcp_json",
        "detect": "detect_claude_config_path",
    },
    "claude-project": {
        "description": "Claude Code (project-scoped)",
        "format": "mcp_json",
        "path_relative": ".claude/.mcp.json",
    },
    "cursor": {
        "description": "Cursor",
        "format": "mcp_json",
        "path_relative": ".cursor/mcp.json",
    },
    "windsurf": {
        "description": "Windsurf",
        "format": "mcp_json",
        "path_relative": ".windsurf/mcp.json",
    },
    "cline": {
        "description": "Cline / Roo Code",
        "format": "mcp_json",
        "path_relative": ".cline/mcp.json",
    },
    "vscode-copilot": {
        "description": "VS Code Copilot",
        "format": "mcp_json",
        "path_relative": ".vscode/mcp.json",
    },
    "codex": {
        "description": "Codex / Jules",
        "format": "agents_md",
        "path_relative": "AGENTS.md",
    },
    "gemini": {
        "description": "Gemini CLI",
        "format": "gemini_json",
        "path_global": ".gemini/settings.json",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def detect_claude_config_path() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA is not set; cannot locate Claude Desktop config.")
        return Path(appdata) / "Claude" / "claude_desktop_config.json"
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"


def detect_project_mcp_config_path(root: Path | None = None) -> Path:
    project_root = (root or repo_root()).resolve()
    return project_root / ".claude" / ".mcp.json"


def detect_server_command() -> tuple[str, list[str]]:
    launcher = shutil.which("token-saver-mcp")
    if launcher:
        return launcher, []
    return sys.executable, ["-m", "src.server"]


def resolve_project_root_for_cli(project_root: Path | None) -> Path:
    return (project_root or Path.cwd()).resolve()


def recommend_setup_target(cwd: Path | None = None) -> str:
    candidate = (cwd or Path.cwd()).resolve()
    if (candidate / ".git").exists() or (candidate / ".claude").exists():
        return "portable-project"
    return "desktop"


def build_status_report(status: dict[str, Any], recommended_target: str | None = None) -> str:
    desktop = status["desktop"]
    project = status["project"]
    lines = [
        "Token Saver MCP status",
        f"- command available: {'yes' if status['command_available'] else 'no'}",
        f"- desktop configured: {'yes' if desktop['configured'] else 'no'} ({desktop['path']})",
        (
            "- project configured: "
            f"{'yes' if project['configured'] else 'no'} ({project['path']})"
        ),
    ]
    if project["configured"]:
        lines.append(f"- project mode: {'portable-project' if project['portable'] else 'project'}")
    if recommended_target:
        lines.extend(
            [
                "",
                "Recommended next step:",
                f"- token-saver-setup --{recommended_target}",
            ]
        )
    return "\n".join(lines)


def build_server_config(root: Path | None = None, portable: bool = False) -> dict[str, Any]:
    if portable:
        return {
            "command": "python",
            "args": ["-m", "src.server"],
            "cwd": WORKSPACE_FOLDER_VAR,
            "env": {
                "PYTHONPATH": WORKSPACE_FOLDER_VAR,
            },
        }
    project_root = (root or repo_root()).resolve()
    command, args = detect_server_command()
    return {
        "command": command,
        "args": args,
        "cwd": str(project_root),
        "env": {
            "PYTHONPATH": str(project_root),
        },
    }


def render_mcp_config(
    root: Path | None = None, server_key: str = SERVER_KEY, portable: bool = False
) -> dict[str, Any]:
    return {"mcpServers": {server_key: build_server_config(root, portable=portable)}}


def merge_mcp_server(
    config: dict[str, Any], server_key: str, server_config: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(config)
    existing_servers = merged.get("mcpServers")
    if existing_servers is None:
        merged["mcpServers"] = {server_key: server_config}
        return merged
    if not isinstance(existing_servers, dict):
        raise ValueError("Existing Claude Desktop config has a non-object 'mcpServers' field.")
    merged["mcpServers"] = dict(existing_servers)
    merged["mcpServers"][server_key] = server_config
    return merged


def remove_mcp_server(config: dict[str, Any], server_key: str) -> dict[str, Any]:
    updated = dict(config)
    existing_servers = updated.get("mcpServers")
    if not isinstance(existing_servers, dict) or server_key not in existing_servers:
        return updated

    remaining_servers = dict(existing_servers)
    del remaining_servers[server_key]
    if remaining_servers:
        updated["mcpServers"] = remaining_servers
    else:
        updated.pop("mcpServers", None)
    return updated


def load_existing_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Claude Desktop config: {config_path}") from exc


def inspect_mcp_installation(root: Path | None = None) -> dict[str, Any]:
    project_root = (root or repo_root()).resolve()
    desktop_path = detect_claude_config_path()
    project_path = detect_project_mcp_config_path(project_root)
    launcher = shutil.which("token-saver-mcp")

    desktop_config = load_existing_config(desktop_path) if desktop_path.exists() else {}
    project_config = load_existing_config(project_path) if project_path.exists() else {}

    desktop_server = desktop_config.get("mcpServers", {}).get(SERVER_KEY)
    project_server = project_config.get("mcpServers", {}).get(SERVER_KEY)

    desktop_ok = isinstance(desktop_server, dict)
    project_ok = isinstance(project_server, dict)
    project_portable = (
        project_ok
        and project_server.get("cwd") == WORKSPACE_FOLDER_VAR
        and project_server.get("env", {}).get("PYTHONPATH") == WORKSPACE_FOLDER_VAR
    )

    return {
        "server_key": SERVER_KEY,
        "command_available": launcher is not None,
        "detected_command": launcher or sys.executable,
        "desktop": {
            "path": str(desktop_path),
            "exists": desktop_path.exists(),
            "configured": desktop_ok,
        },
        "project": {
            "path": str(project_path),
            "exists": project_path.exists(),
            "configured": project_ok,
            "portable": project_portable,
        },
    }


def write_merged_mcp_config(
    config_path: Path,
    root: Path | None = None,
    server_key: str = SERVER_KEY,
    portable: bool = False,
) -> Path:
    resolved_config = config_path
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_mcp_server(
        load_existing_config(resolved_config),
        server_key,
        build_server_config(root, portable=portable),
    )
    resolved_config.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return resolved_config


def remove_mcp_server_from_file(config_path: Path, server_key: str = SERVER_KEY) -> bool:
    existing = load_existing_config(config_path)
    updated = remove_mcp_server(existing, server_key)
    if updated == existing:
        return False
    if updated:
        config_path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
    else:
        config_path.unlink(missing_ok=True)
    return True


def install_mcp(config_path: Path | None = None, root: Path | None = None) -> Path:
    return write_merged_mcp_config(config_path or detect_claude_config_path(), root)


def uninstall_mcp(config_path: Path | None = None, server_key: str = SERVER_KEY) -> bool:
    resolved_config = config_path or detect_claude_config_path()
    if not resolved_config.exists():
        return False
    return remove_mcp_server_from_file(resolved_config, server_key)


def install_project_mcp(config_path: Path | None = None, root: Path | None = None) -> Path:
    project_root = root or repo_root()
    return write_merged_mcp_config(
        config_path or detect_project_mcp_config_path(project_root), project_root
    )


def uninstall_project_mcp(
    config_path: Path | None = None,
    root: Path | None = None,
    server_key: str = SERVER_KEY,
) -> bool:
    project_root = root or repo_root()
    resolved_config = config_path or detect_project_mcp_config_path(project_root)
    if not resolved_config.exists():
        return False
    return remove_mcp_server_from_file(resolved_config, server_key)


def install_portable_project_mcp(config_path: Path | None = None, root: Path | None = None) -> Path:
    project_root = root or repo_root()
    return write_merged_mcp_config(
        config_path or detect_project_mcp_config_path(project_root),
        project_root,
        portable=True,
    )


def detect_agent_config_path(agent: str, root: Path | None = None) -> Path:
    """Return the config file path for the given agent."""
    project_root = (root or repo_root()).resolve()
    cfg = AGENT_CONFIGS.get(agent)
    if not cfg:
        raise ValueError(f"Unknown agent: {agent}. Known: {', '.join(AGENT_CONFIGS.keys())}")
    if agent == "claude-desktop":
        return detect_claude_config_path()
    if "path_relative" in cfg:
        return project_root / cfg["path_relative"]
    if "path_global" in cfg:
        return Path.home() / cfg["path_global"]
    raise ValueError(f"No path defined for agent: {agent}")


def _build_agents_md_section(root: Path | None = None) -> str:
    """Generate an AGENTS.md section for Codex/Jules."""
    project_root = (root or repo_root()).resolve()
    command, args = detect_server_command()
    args_str = " ".join(args) if args else ""
    return (
        "\n## Token Saver MCP Server\n\n"
        "This project uses Token Saver for semantic compression of AI context.\n\n"
        f"- **Command:** `{command} {args_str}`\n"
        f"- **Working directory:** `{project_root}`\n"
        "- **Protocol:** MCP (stdio)\n\n"
        "Key tools: `ingest_context`, `query_context`, `compress_text`, "
        "`filter_cli_output`, `get_savings_report`\n"
    )


def _build_gemini_config(root: Path | None = None) -> dict[str, Any]:
    """Generate Gemini CLI settings.json mcpServers block."""
    server_config = build_server_config(root)
    return {"mcpServers": {SERVER_KEY: server_config}}


def install_agent_config(
    agent: str,
    root: Path | None = None,
    config_path: Path | None = None,
    portable: bool = False,
) -> Path:
    """Install Token Saver MCP config for the specified agent.

    For MCP-JSON agents (Cursor, Windsurf, Cline, VS Code Copilot, Gemini):
      Merges server config into the agent's JSON config file.
    For AGENTS.md agents (Codex):
      Appends a section to the project's AGENTS.md file.
    """
    project_root = (root or repo_root()).resolve()
    cfg = AGENT_CONFIGS.get(agent)
    if not cfg:
        raise ValueError(f"Unknown agent: {agent}. Known: {', '.join(AGENT_CONFIGS.keys())}")

    target_path = config_path or detect_agent_config_path(agent, project_root)

    fmt = cfg["format"]
    if fmt == "mcp_json":
        return write_merged_mcp_config(target_path, project_root, portable=portable)

    if fmt == "gemini_json":
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_existing_config(target_path)
        mcp_block = existing.get("mcpServers", {})
        mcp_block[SERVER_KEY] = build_server_config(project_root)
        existing["mcpServers"] = mcp_block
        target_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        return target_path

    if fmt == "agents_md":
        section = _build_agents_md_section(project_root)
        marker = "## Token Saver MCP Server"
        if target_path.exists():
            content = target_path.read_text(encoding="utf-8")
            if marker in content:
                # Replace existing section
                before = content.split(marker)[0].rstrip()
                remaining = content.split(marker, 1)[1]
                after_parts = remaining.split("\n## ", 1)
                after = ("\n## " + after_parts[1]) if len(after_parts) > 1 else ""
                content = before + section + after
            else:
                content = content.rstrip() + "\n" + section
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            content = f"# AI Agent Instructions\n{section}"
        target_path.write_text(content, encoding="utf-8")
        return target_path

    raise ValueError(f"Unsupported format: {fmt}")


def uninstall_agent_config(
    agent: str,
    root: Path | None = None,
    config_path: Path | None = None,
) -> bool:
    """Remove Token Saver config from the specified agent."""
    project_root = (root or repo_root()).resolve()
    cfg = AGENT_CONFIGS.get(agent)
    if not cfg:
        raise ValueError(f"Unknown agent: {agent}")

    target_path = config_path or detect_agent_config_path(agent, project_root)

    fmt = cfg["format"]
    if fmt in ("mcp_json", "gemini_json"):
        if not target_path.exists():
            return False
        return remove_mcp_server_from_file(target_path, SERVER_KEY)

    if fmt == "agents_md":
        if not target_path.exists():
            return False
        content = target_path.read_text(encoding="utf-8")
        marker = "## Token Saver MCP Server"
        if marker not in content:
            return False
        before = content.split(marker)[0].rstrip()
        remaining = content.split(marker, 1)[1]
        after_parts = remaining.split("\n## ", 1)
        after = ("\n## " + after_parts[1]) if len(after_parts) > 1 else ""
        target_path.write_text((before + after).strip() + "\n", encoding="utf-8")
        return True

    return False


def inspect_all_agents(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Check installation status for all known agents."""
    project_root = (root or repo_root()).resolve()
    results: dict[str, dict[str, Any]] = {}
    for agent, cfg in AGENT_CONFIGS.items():
        try:
            path = detect_agent_config_path(agent, project_root)
            exists = path.exists()
            configured = False
            if exists:
                fmt = cfg["format"]
                if fmt in ("mcp_json", "gemini_json"):
                    data = load_existing_config(path)
                    configured = SERVER_KEY in data.get("mcpServers", {})
                elif fmt == "agents_md":
                    content = path.read_text(encoding="utf-8")
                    configured = "## Token Saver MCP Server" in content
            results[agent] = {
                "description": cfg["description"],
                "path": str(path),
                "exists": exists,
                "configured": configured,
            }
        except Exception as exc:
            results[agent] = {
                "description": cfg["description"],
                "path": "unknown",
                "exists": False,
                "configured": False,
                "error": str(exc),
            }
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    agent_names = [k for k in AGENT_CONFIGS.keys() if k not in ("claude-desktop", "claude-project")]
    parser = argparse.ArgumentParser(description="Install or print Token Saver MCP configuration.")
    parser.add_argument(
        "--agent",
        choices=agent_names,
        help=f"Install config for a specific agent: {', '.join(agent_names)}",
    )
    parser.add_argument(
        "--project-config",
        action="store_true",
        help="Write merged config to .claude/.mcp.json in the project root.",
    )
    parser.add_argument(
        "--portable-project-config",
        action="store_true",
        help="Write merged config to .claude/.mcp.json using ${workspaceFolder}.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove Token Saver from the selected desktop or project config.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print JSON config to stdout instead of writing Claude Desktop config.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Print MCP installation status for command, desktop config, and project config.",
    )
    parser.add_argument(
        "--doctor-all",
        action="store_true",
        help="Print MCP installation status for ALL known agents.",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Print doctor output as a human-readable summary instead of JSON.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Override the Claude Desktop config path when writing config.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Override the project root used for cwd/PYTHONPATH in generated config.",
    )
    parser.add_argument(
        "--server-key",
        default=SERVER_KEY,
        help="Override the MCP server key name (default: token-saver).",
    )
    return parser


def _format_all_agents_report(status: dict[str, dict[str, Any]]) -> str:
    """Format multi-agent doctor report as human-readable text."""
    lines = ["Token Saver — Agent Configuration Status", "=" * 45]
    for agent, info in status.items():
        check = "✓" if info["configured"] else "✗"
        lines.append(f"  [{check}] {info['description']:<25} {info['path']}")
    configured = sum(1 for v in status.values() if v["configured"])
    lines.append(f"\n{configured}/{len(status)} agents configured.")
    return "\n".join(lines)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.print_config:
        print(json.dumps(render_mcp_config(args.project_root, args.server_key), indent=2))
        return

    if args.doctor_all:
        project_root = resolve_project_root_for_cli(args.project_root)
        status = inspect_all_agents(project_root)
        if args.human:
            print(_format_all_agents_report(status))
        else:
            print(json.dumps(status, indent=2))
        return

    if args.doctor:
        project_root = resolve_project_root_for_cli(args.project_root)
        status = inspect_mcp_installation(project_root)
        if args.human:
            print(build_status_report(status, recommend_setup_target(project_root)))
        else:
            print(json.dumps(status, indent=2))
        return

    if args.agent:
        project_root = resolve_project_root_for_cli(args.project_root)
        if args.uninstall:
            changed = uninstall_agent_config(args.agent, root=project_root)
            msg = "Removed" if changed else "Did not find"
            desc = AGENT_CONFIGS[args.agent]["description"]
            print(f"{msg} Token Saver config for {desc}.")
        else:
            path = install_agent_config(args.agent, root=project_root)
            desc = AGENT_CONFIGS[args.agent]["description"]
            print(f"Installed Token Saver config for {desc} at: {path}")
        return

    if args.project_config:
        project_root = resolve_project_root_for_cli(args.project_root)
        if args.uninstall:
            changed = uninstall_project_mcp(config_path=args.config_path, root=project_root)
            message = "Removed" if changed else "Did not find"
            print(f"{message} Token Saver project MCP config entry.")
        else:
            path = install_project_mcp(config_path=args.config_path, root=project_root)
            print(f"Installed Token Saver project MCP config at: {path}")
            print("Restart Claude Code or reload project MCP config to load 'token-saver'.")
        return

    if args.portable_project_config:
        project_root = resolve_project_root_for_cli(args.project_root)
        if args.uninstall:
            changed = uninstall_project_mcp(config_path=args.config_path, root=project_root)
            message = "Removed" if changed else "Did not find"
            print(f"{message} Token Saver portable project MCP config entry.")
        else:
            path = install_portable_project_mcp(config_path=args.config_path, root=project_root)
            print(f"Installed portable Token Saver project MCP config at: {path}")
            print("Restart Claude Code or reload project MCP config to load 'token-saver'.")
        return

    if args.uninstall:
        changed = uninstall_mcp(config_path=args.config_path, server_key=args.server_key)
        message = "Removed" if changed else "Did not find"
        print(f"{message} Token Saver Claude Desktop MCP config entry.")
        return

    path = install_mcp(config_path=args.config_path, root=args.project_root)
    print(f"Installed Token Saver MCP config at: {path}")
    print("Restart Claude Desktop to load the 'token-saver' MCP server.")
