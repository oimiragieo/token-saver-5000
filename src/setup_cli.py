"""Guided setup CLI for Token Saver MCP onboarding."""

from __future__ import annotations

import argparse
from pathlib import Path

from .mcp_install import (
    build_status_report,
    inspect_mcp_installation,
    install_mcp,
    install_portable_project_mcp,
    install_project_mcp,
    recommend_setup_target,
    resolve_project_root_for_cli,
    uninstall_mcp,
    uninstall_project_mcp,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guide Token Saver MCP setup and apply the recommended configuration."
    )
    parser.add_argument("--auto", action="store_true", help="Apply the recommended setup target.")
    parser.add_argument("--desktop", action="store_true", help="Install Claude Desktop config.")
    parser.add_argument(
        "--project",
        action="store_true",
        help="Write .claude/.mcp.json for the current workspace using absolute paths.",
    )
    parser.add_argument(
        "--portable-project",
        action="store_true",
        help="Write portable .claude/.mcp.json for the current workspace using ${workspaceFolder}.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Override the workspace root used for project-oriented setup modes.",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove the selected Token Saver MCP config entry instead of installing it.",
    )
    parser.add_argument(
        "--uninstall-all",
        action="store_true",
        help="Remove Token Saver from both Claude Desktop and the current project config.",
    )
    return parser


def _selected_target(args: argparse.Namespace, project_root: Path) -> str | None:
    if args.desktop:
        return "desktop"
    if args.project:
        return "project"
    if args.portable_project:
        return "portable-project"
    if args.auto:
        return recommend_setup_target(project_root)
    return None


def _apply_target(target: str, project_root: Path) -> str:
    if target == "desktop":
        path = install_mcp()
        return f"Installed Claude Desktop config at: {path}"
    if target == "project":
        path = install_project_mcp(root=project_root)
        return f"Installed project MCP config at: {path}"
    path = install_portable_project_mcp(root=project_root)
    return f"Installed portable project MCP config at: {path}"


def _remove_target(target: str, project_root: Path) -> str:
    if target == "desktop":
        changed = uninstall_mcp()
        return (
            "Removed Claude Desktop config entry."
            if changed
            else "Claude Desktop config entry was not present."
        )
    changed = uninstall_project_mcp(root=project_root)
    if target == "project":
        return (
            "Removed project MCP config entry."
            if changed
            else "Project MCP config entry was not present."
        )
    return (
        "Removed portable project MCP config entry."
        if changed
        else "Portable project MCP config entry was not present."
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    project_root = resolve_project_root_for_cli(args.project_root)
    if args.uninstall_all:
        print(_remove_target("desktop", project_root))
        print(_remove_target("portable-project", project_root))
        print("")
        status = inspect_mcp_installation(project_root)
        print(build_status_report(status, recommend_setup_target(project_root)))
        return

    target = _selected_target(args, project_root)

    if target is not None:
        if args.uninstall:
            print(_remove_target(target, project_root))
        else:
            print(_apply_target(target, project_root))
        print("")

    status = inspect_mcp_installation(project_root)
    print(build_status_report(status, recommend_setup_target(project_root)))
