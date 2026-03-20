"""Contract tests for prompt registry MCP tools and help entries."""

import json

import pytest

from src.handlers.help_handlers import handle_tool_help
from src.handlers.mcp_core import route_tool_call, setup_mcp_tools
from src.prompt_registry import PromptRegistry


PROMPT_TOOL_NAMES = {
    "create_prompt_template",
    "update_prompt_template",
    "list_prompt_templates",
    "get_prompt_template",
    "deploy_prompt_version",
    "compare_prompt_versions",
    "audit_prompt_cacheability",
    "render_prompt_template",
    "list_prefix_collisions",
}


def setup_function():
    PromptRegistry.reset_singleton()


def test_prompt_tools_are_registered_in_mcp_core():
    tools = {tool.name: tool for tool in setup_mcp_tools()}

    assert PROMPT_TOOL_NAMES.issubset(tools)
    assert {"name", "description", "system_prompt", "user_prompt_template"} <= set(
        tools["create_prompt_template"].inputSchema["properties"]
    )


@pytest.mark.asyncio
async def test_prompt_tools_have_help_entries():
    render_help = json.loads(
        await handle_tool_help({}, {"tool_name": "render_prompt_template", "verbose": True})
    )
    compare_help = json.loads(
        await handle_tool_help({}, {"tool_name": "compare_prompt_versions", "verbose": True})
    )
    deploy_help = json.loads(
        await handle_tool_help({}, {"tool_name": "deploy_prompt_version", "verbose": True})
    )
    collisions_help = json.loads(
        await handle_tool_help({}, {"tool_name": "list_prefix_collisions", "verbose": True})
    )

    assert render_help["tool"] == "render_prompt_template"
    assert "variables" in render_help["parameters"]
    assert "enforce_stability" in render_help["parameters"]
    assert "related_tools" in render_help
    assert "rendered.stability_guard.is_stable" in render_help["output_fields"]
    assert compare_help["tool"] == "compare_prompt_versions"
    assert any("stable prefix" in tip.lower() for tip in compare_help["tips"])
    assert "allow_stable_prefix_change" in deploy_help["parameters"]
    assert collisions_help["tool"] == "list_prefix_collisions"
    assert "collision_count" in collisions_help["output_fields"]
    create_help = json.loads(
        await handle_tool_help({}, {"tool_name": "create_prompt_template", "verbose": True})
    )
    update_help = json.loads(
        await handle_tool_help({}, {"tool_name": "update_prompt_template", "verbose": True})
    )
    assert "stable_prefix_analysis.impact" in create_help["output_fields"]
    assert "stable_prefix_analysis.impact" in update_help["output_fields"]


@pytest.mark.asyncio
async def test_router_dispatches_prompt_registry_tools():
    context = {"prompt_registry": PromptRegistry(seed_defaults=False)}

    created = json.loads(
        await route_tool_call(
            "create_prompt_template",
            {
                "name": "review-default",
                "description": "Review prompt",
                "system_prompt": "You are a reviewer.",
                "user_prompt_template": "Review {diff}",
            },
            context,
        )
    )
    listed = json.loads(await route_tool_call("list_prompt_templates", {}, context))
    rendered = json.loads(
        await route_tool_call(
            "render_prompt_template",
            {
                "name": "review-default",
                "variables": {"diff": "print('hi')"},
            },
            context,
        )
    )

    assert created["status"] == "success"
    assert listed["total_templates"] == 1
    assert rendered["status"] == "success"
    assert isinstance(rendered["rendered"]["prompt_id"], str)
    assert rendered["rendered"]["sections"][-1]["name"] == "user_query"
    assert rendered["rendered"]["stability_guard"]["is_stable"] is True
