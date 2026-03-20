"""Tests for MCP prompt and resource surfaces."""

from __future__ import annotations

import json

import pytest

from src.semantic_modulator.app import mcp_context_surfaces as surfaces


class Tooling:
    def list_tools(self, profile: str):
        return [
            type("Tool", (), {"name": "ingest_context", "description": "Ingest docs"})(),
            type("Tool", (), {"name": "read_skeleton", "description": "Read skeleton"})(),
        ]


@pytest.mark.parametrize(
    "prompt_name,arguments",
    [
        ("document_compression_workflow", {"goal": "Summarize a design doc", "file_id": "doc_1"}),
        ("prompt_cache_review", {"user_prompt": "System: be concise"}),
        ("mcp_setup_assistant", {"target": "portable_project"}),
    ],
)
def test_get_prompt_returns_prompt_messages(prompt_name, arguments):
    result = surfaces.get_prompt(prompt_name, arguments)

    assert result.description
    assert result.messages
    assert result.messages[0].role == "user"
    assert result.messages[0].content.type == "text"

    if prompt_name == "mcp_setup_assistant":
        assert "token-saver-setup" in result.messages[0].content.text


def test_list_prompts_returns_supported_prompt_names():
    prompts = {prompt.name for prompt in surfaces.list_prompts()}

    assert prompts == {
        "document_compression_workflow",
        "prompt_cache_review",
        "mcp_setup_assistant",
    }


def test_list_resources_includes_catalog_and_status():
    resources = {
        str(resource.uri): resource for resource in surfaces.list_resources(Tooling(), "full")
    }

    assert "token-saver://catalog/tools" in resources
    assert "token-saver://status/mcp-installation" in resources
    assert resources["token-saver://catalog/tools"].mimeType == "application/json"


def test_list_resource_templates_includes_tool_help():
    templates = surfaces.list_resource_templates()

    assert len(templates) == 1
    assert templates[0].uriTemplate == "token-saver://tool/{name}/help"


@pytest.mark.asyncio
async def test_read_resource_returns_tool_catalog_json():
    contents = await surfaces.read_resource(
        "token-saver://catalog/tools",
        tooling=Tooling(),
        profile="core_stable",
        context={},
    )

    payload = json.loads(contents[0].text)
    assert payload["profile"] == "core_stable"
    assert payload["total_tools"] == 2
    assert payload["tools"][0]["name"] == "ingest_context"


@pytest.mark.asyncio
async def test_read_resource_returns_tool_help_template_payload(monkeypatch):
    async def fake_tool_help(context, args):
        return json.dumps({"tool": args["tool_name"], "description": "help"})

    monkeypatch.setattr(surfaces, "handle_tool_help", fake_tool_help)

    contents = await surfaces.read_resource(
        "token-saver://tool/ingest_context/help",
        tooling=Tooling(),
        profile="full",
        context={"ok": True},
    )

    payload = json.loads(contents[0].text)
    assert payload["tool"] == "ingest_context"


@pytest.mark.asyncio
async def test_read_resource_returns_installation_status(monkeypatch):
    monkeypatch.setattr(
        surfaces,
        "inspect_mcp_installation",
        lambda: {"desktop": {"configured": False}, "project": {"configured": True}},
    )

    contents = await surfaces.read_resource(
        "token-saver://status/mcp-installation",
        tooling=Tooling(),
        profile="full",
        context={},
    )

    payload = json.loads(contents[0].text)
    assert payload["project"]["configured"] is True


@pytest.mark.asyncio
async def test_read_resource_returns_install_modes_with_guided_setup_command():
    contents = await surfaces.read_resource(
        "token-saver://config/install-modes",
        tooling=Tooling(),
        profile="full",
        context={},
    )

    assert "token-saver-setup" in contents[0].text
    assert "uninstall" in contents[0].text
