"""Verify every registered MCP tool has a dispatch handler."""

from src.handlers.mcp_core import registered_tool_names, setup_mcp_tools

_ROUTER_KEYS = tuple(sorted(registered_tool_names("full")))


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
