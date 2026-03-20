"""Contract tests for explicit memory MCP tools."""

import json

import pytest

from src.handlers.help_handlers import handle_tool_help
from src.handlers.mcp_core import route_tool_call, setup_mcp_tools
from src.memory_api import MemoryAPI


MEMORY_TOOL_NAMES = {
    "add_memory",
    "search_memory",
    "list_memories",
    "delete_memory",
    "summarize_user_memory",
    "get_user_profile",
}


def setup_function():
    MemoryAPI.reset_singleton()


def test_memory_tools_are_registered_in_mcp_core():
    tools = {tool.name: tool for tool in setup_mcp_tools()}

    assert MEMORY_TOOL_NAMES.issubset(tools)
    assert {"text", "user_id", "workspace_id"} <= set(tools["add_memory"].inputSchema["properties"])


@pytest.mark.asyncio
async def test_memory_tools_have_help_entries():
    result = await handle_tool_help({}, {"tool_name": "get_user_profile", "verbose": True})
    data = json.loads(result)

    assert data["tool"] == "get_user_profile"
    assert "user_id" in data["parameters"]
    assert "output_fields" in data


@pytest.mark.asyncio
async def test_router_dispatches_memory_tools():
    context = {"memory_api": MemoryAPI()}

    created = json.loads(
        await route_tool_call(
            "add_memory",
            {"text": "Prefer pytest fixtures.", "user_id": "alice", "workspace_id": "acme"},
            context,
        )
    )
    listed = json.loads(
        await route_tool_call(
            "list_memories", {"user_id": "alice", "workspace_id": "acme"}, context
        )
    )

    assert created["status"] == "success"
    assert listed["total_memories"] == 1
