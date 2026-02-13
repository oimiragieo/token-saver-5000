"""Contract tests for MCP tooling gateway in enterprise app layer."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest


def test_gateway_reexports_supported_profiles():
    module = importlib.import_module("src.semantic_modulator.app.tooling")
    registry = importlib.import_module("src.semantic_modulator.api.mcp.registry")
    gateway = module.MCPToolingGateway()

    assert gateway.supported_profiles == registry.SUPPORTED_TOOL_PROFILES


def test_gateway_resolve_profile_falls_back_to_full():
    module = importlib.import_module("src.semantic_modulator.app.tooling")
    gateway = module.MCPToolingGateway()
    registry = importlib.import_module("src.semantic_modulator.api.mcp.registry")

    fallback_tools = [Mock(name="ingest_context")]
    fallback_tools[0].name = "ingest_context"

    calls: list[str] = []

    def fake_setup(profile: str = "full"):
        calls.append(profile)
        if profile == "broken":
            raise ValueError("invalid profile")
        if profile == "full":
            return fallback_tools
        raise AssertionError(f"unexpected profile {profile}")

    gateway.resolve_tools_for_profile("broken", fake_setup)

    assert calls == ["broken", "full"]
    assert gateway.profile == "full"
    assert gateway.enabled_tool_names == ["ingest_context"]
    assert gateway.supported_profiles == registry.SUPPORTED_TOOL_PROFILES


@pytest.mark.asyncio
async def test_gateway_route_tool_call_delegates():
    module = importlib.import_module("src.semantic_modulator.app.tooling")
    gateway = module.MCPToolingGateway()
    router = importlib.import_module("src.semantic_modulator.api.mcp.router")

    async def fake_route(name, arguments, context, tool_profile):
        return {"name": name, "arguments": arguments, "tool_profile": tool_profile}

    original = router.route_tool_call
    try:
        router.route_tool_call = fake_route
        result = await gateway.route_tool_call("x", {"k": 1}, {"ctx": True}, tool_profile="full")
    finally:
        router.route_tool_call = original

    assert result["name"] == "x"
    assert result["arguments"] == {"k": 1}
    assert result["tool_profile"] == "full"
