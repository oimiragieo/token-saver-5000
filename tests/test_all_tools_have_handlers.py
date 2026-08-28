"""Verify every registered MCP tool has a dispatch handler."""

from src.handlers.mcp_core import setup_mcp_tools
from src.handlers.mcp_core.dispatch import route_tool_call  # noqa: F401 — import side

import re
from pathlib import Path

_DISPATCH_PATH = Path(__file__).resolve().parents[1] / "src/handlers/mcp_core/dispatch.py"
_ROUTER_KEYS = re.findall(
    r'"([a-z_][a-z0-9_]*)":\s+\w+\.handle',
    _DISPATCH_PATH.read_text(encoding="utf-8"),
)


def test_all_full_profile_tools_have_router_handlers():
    tools = setup_mcp_tools(profile="full")
    tool_names = {t.name for t in tools}
    router_names = set(_ROUTER_KEYS)
    assert tool_names == router_names
    assert len(tool_names) >= 120


def test_critical_tools_registered():
    tools = setup_mcp_tools(profile="full")
    names = {t.name for t in tools}
    for required in (
        "ingest_context",
        "filter_cli_output",
        "compile_knowledge",
        "get_savings_report",
        "check_budget",
        "ace_generate",
        "compress_codebase",
    ):
        assert required in names
