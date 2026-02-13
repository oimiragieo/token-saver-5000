"""Contract tests for MCP tool profile behavior."""

from src.handlers import mcp_core


def test_core_stable_profile_contract():
    tools = mcp_core.setup_mcp_tools(profile="core_stable")
    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "modulate_region",
        "get_stats",
        "list_documents",
        "delete_document",
    }


def test_full_profile_contains_core_stable_tools():
    full_names = {tool.name for tool in mcp_core.setup_mcp_tools(profile="full")}
    core_names = {tool.name for tool in mcp_core.setup_mcp_tools(profile="core_stable")}

    assert core_names.issubset(full_names)
