"""Application-layer MCP tooling gateway.

Provides a stable app-facing abstraction over MCP registry and router modules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mcp.types import Tool

from src.semantic_modulator.api.mcp import registry as mcp_registry
from src.semantic_modulator.api.mcp import router as mcp_router


class MCPToolingGateway:
    """App-layer facade for tool-profile setup and MCP routing."""

    def __init__(self) -> None:
        self.supported_profiles = mcp_registry.SUPPORTED_TOOL_PROFILES
        self.profile = "full"
        self.enabled_tool_names: list[str] = []

    def resolve_tools_for_profile(
        self,
        configured_profile: str,
        setup_tools: Callable[[str], list[Tool]] | None = None,
    ) -> tuple[str, list[Tool], bool]:
        """Resolve active profile and tools, with fallback-to-full on invalid profile."""
        setup = setup_tools or mcp_registry.setup_mcp_tools
        try:
            tools = setup(configured_profile)
            active_profile = configured_profile
            used_fallback = False
        except ValueError:
            active_profile = "full"
            tools = setup(active_profile)
            used_fallback = True

        self.profile = active_profile
        self.enabled_tool_names = [tool.name for tool in tools]
        return active_profile, tools, used_fallback

    def list_tools(self, profile: str | None = None) -> list[Tool]:
        """List tools for active or explicitly provided profile."""
        selected_profile = profile or self.profile
        tools = mcp_registry.setup_mcp_tools(selected_profile)
        self.profile = selected_profile
        self.enabled_tool_names = [tool.name for tool in tools]
        return tools

    async def route_tool_call(
        self,
        name: str,
        arguments: Any,
        context: Any,
        tool_profile: str,
    ) -> Any:
        """Delegate tool routing to centralized MCP router."""
        route_fn: Callable[[str, Any, Any, str], Awaitable[Any]] = mcp_router.route_tool_call
        return await route_fn(name, arguments, context, tool_profile)
