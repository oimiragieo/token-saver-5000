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

KNOWLEDGE_TOOL_NAMES = {
    "ingest_transcript",
    "compile_knowledge",
    "get_knowledge_index",
    "lint_knowledge",
    "search_memory_index",
}


def setup_function():
    MemoryAPI.reset_singleton()


def test_memory_tools_are_registered_in_mcp_core():
    tools = {tool.name: tool for tool in setup_mcp_tools()}

    assert MEMORY_TOOL_NAMES.issubset(tools)
    assert {"text", "user_id", "workspace_id"} <= set(tools["add_memory"].inputSchema["properties"])


def test_knowledge_tools_are_registered_in_mcp_core():
    tools = {tool.name: tool for tool in setup_mcp_tools()}

    assert KNOWLEDGE_TOOL_NAMES.issubset(tools)
    assert "text" in tools["ingest_transcript"].inputSchema["properties"]
    assert "query" in tools["search_memory_index"].inputSchema["properties"]


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


@pytest.mark.asyncio
async def test_router_dispatches_knowledge_tools():
    context = {"memory_api": MemoryAPI()}

    # ingest_transcript
    ingested = json.loads(
        await route_tool_call(
            "ingest_transcript",
            {"text": "We decided to use PostgreSQL for ACID compliance."},
            context,
        )
    )
    assert ingested["status"] == "success"

    # compile_knowledge
    compiled = json.loads(await route_tool_call("compile_knowledge", {}, context))
    assert compiled["status"] == "success"

    # lint_knowledge
    linted = json.loads(await route_tool_call("lint_knowledge", {}, context))
    assert linted["status"] == "success"
    assert "checks_run" in linted

    # search_memory_index
    searched = json.loads(
        await route_tool_call("search_memory_index", {"query": "PostgreSQL"}, context)
    )
    assert searched["status"] == "success"
