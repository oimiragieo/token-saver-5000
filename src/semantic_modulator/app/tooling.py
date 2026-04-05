"""Application-layer MCP tooling gateway.

Provides a stable app-facing abstraction over MCP registry and router modules.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from mcp.types import GetPromptResult, Prompt, Resource, ResourceTemplate, Tool

from src.semantic_modulator.api.mcp import registry as mcp_registry
from src.semantic_modulator.api.mcp import router as mcp_router
from src.semantic_modulator.app import mcp_context_surfaces
from src.semantic_modulator.app.contract_validation import (
    contract_key_mismatch_message as _contract_key_mismatch_message,
    validate_contract_keys as _validate_contract_keys,
)


class ProfileState(TypedDict):
    """Gateway profile state envelope."""

    profile: str
    enabled_tool_names: list[str]


class MCPToolingGateway:
    """App-layer facade for tool-profile setup and MCP routing."""

    PROFILE_STATE_KEYS: frozenset[str] = frozenset(ProfileState.__annotations__.keys())

    def __init__(self) -> None:
        self.supported_profiles = mcp_registry.SUPPORTED_TOOL_PROFILES
        self.profile = "full"
        self.enabled_tool_names: list[str] = []

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
        return _contract_key_mismatch_message(
            contract_name=contract_name, missing=missing, extra=extra
        )

    @classmethod
    def validate_contract_keys(
        cls,
        *,
        contract_name: str,
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        _validate_contract_keys(
            contract_name=contract_name, payload=payload, expected_keys=expected_keys
        )

    @classmethod
    def validate_profile_state_map(cls, state: dict[str, Any]) -> ProfileState:
        cls.validate_contract_keys(
            contract_name="profile_state_map",
            payload=state,
            expected_keys=cls.PROFILE_STATE_KEYS,
        )
        return state

    def set_profile_state(self, *, profile: str, tools: list[Tool]) -> ProfileState:
        state = self.validate_profile_state_map(
            {
                "profile": profile,
                "enabled_tool_names": [tool.name for tool in tools],
            }
        )
        self.profile = state["profile"]
        self.enabled_tool_names = state["enabled_tool_names"]
        return state

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

        self.set_profile_state(profile=active_profile, tools=tools)
        return active_profile, tools, used_fallback

    def list_tools(self, profile: str | None = None) -> list[Tool]:
        """List tools for active or explicitly provided profile."""
        selected_profile = profile or self.profile
        tools = mcp_registry.setup_mcp_tools(selected_profile)
        self.set_profile_state(profile=selected_profile, tools=tools)
        return tools

    def list_prompts(self) -> list[Prompt]:
        return mcp_context_surfaces.list_prompts()

    def get_prompt(self, name: str, arguments: Any) -> GetPromptResult:
        return mcp_context_surfaces.get_prompt(name, arguments)

    def list_resources(self, profile: str | None = None) -> list[Resource]:
        selected_profile = profile or self.profile
        return mcp_context_surfaces.list_resources(self, selected_profile)

    def list_resource_templates(self) -> list[ResourceTemplate]:
        return mcp_context_surfaces.list_resource_templates()

    async def read_resource(self, uri: str, context: Any, profile: str | None = None) -> Any:
        selected_profile = profile or self.profile
        return await mcp_context_surfaces.read_resource(
            uri,
            tooling=self,
            profile=selected_profile,
            context=context,
        )

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
