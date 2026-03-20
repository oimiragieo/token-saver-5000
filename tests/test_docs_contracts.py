"""Documentation and help-surface contract tests for launch readiness."""

import json

import pytest

from src.handlers.help_handlers import get_tool_help_registry, handle_tool_help
from src.handlers.mcp_core import setup_mcp_tools


def test_help_registry_covers_all_registered_mcp_tools():
    tool_names = {tool.name for tool in setup_mcp_tools()}
    help_names = set(get_tool_help_registry())

    assert help_names == tool_names


@pytest.mark.asyncio
async def test_tool_help_list_reports_real_tool_count():
    result = await handle_tool_help({}, {})
    data = json.loads(result)

    assert data["status"] == "tool_list"
    assert data["total_tools"] == len(setup_mcp_tools())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name",
    [
        "tool_help",
        "list_documents",
        "delete_document",
        "batch_ingest_documents",
        "ace_reflect",
        "afm_get_stats",
        "detect_hallucination",
        "toon_encode",
        "export_graph_json",
    ],
)
async def test_formerly_missing_tools_return_help(tool_name):
    result = await handle_tool_help({}, {"tool_name": tool_name, "verbose": True})
    data = json.loads(result)

    assert data["tool"] == tool_name
    assert data["description"]
    assert isinstance(data.get("parameters", {}), dict)
    assert "related_tools" in data
