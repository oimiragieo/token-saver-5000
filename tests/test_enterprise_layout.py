"""Contract tests for enterprise package layout scaffolding."""

from __future__ import annotations

import importlib

import pytest


def test_semantic_modulator_package_exposes_version():
    pkg = importlib.import_module("src.semantic_modulator")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)
    assert pkg.__version__


def test_mcp_registry_wrapper_matches_legacy_setup():
    legacy = importlib.import_module("src.handlers.mcp_core")
    wrapper = importlib.import_module("src.semantic_modulator.api.mcp.registry")

    assert wrapper.setup_mcp_tools is not None
    assert [tool.name for tool in wrapper.setup_mcp_tools("core_stable")] == [
        tool.name for tool in legacy.setup_mcp_tools("core_stable")
    ]


@pytest.mark.asyncio
async def test_mcp_router_wrapper_exposes_route_tool_call():
    router = importlib.import_module("src.semantic_modulator.api.mcp.router")
    with pytest.raises(ValueError, match="Unknown tool"):
        await router.route_tool_call("definitely_not_a_tool", {}, {}, "full")


def test_bootstrap_wrapper_exposes_server_factory():
    bootstrap = importlib.import_module("src.semantic_modulator.app.bootstrap")
    server = bootstrap.create_server()
    assert server is not None
    assert server.__class__.__name__ == "SemanticModulatorServer"


def test_server_uses_enterprise_tooling_gateway(monkeypatch):
    server_module = importlib.import_module("src.server")
    called: dict[str, str] = {}

    def fake_resolve(self, configured_profile: str):
        called["profile"] = configured_profile
        return "full", [], False

    monkeypatch.setattr(server_module.MCPToolingGateway, "resolve_tools_for_profile", fake_resolve)
    _ = server_module.SemanticModulatorServer()
    assert called["profile"] == "full"
