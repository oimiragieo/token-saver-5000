import json

import pytest

from src.handlers.help_handlers import handle_tool_help
from src.handlers.mcp_core import setup_mcp_tools


def test_prompt_cache_audit_tool_is_registered():
    tools = {tool.name: tool for tool in setup_mcp_tools()}
    tool = tools["audit_prompt_cacheability"]

    assert {"sections"} <= set(tool.inputSchema["required"])
    assert {"sections"} <= set(tool.inputSchema["properties"])


@pytest.mark.asyncio
async def test_prompt_cache_audit_help_is_documented():
    payload = json.loads(
        await handle_tool_help({}, {"tool_name": "audit_prompt_cacheability", "verbose": True})
    )

    assert payload["category"] == "Prompt Registry"
    assert "sections" in payload["parameters"]
    assert "audit.score" in payload["output_fields"]
    assert "audit.stability_guard.is_stable" in payload["output_fields"]
