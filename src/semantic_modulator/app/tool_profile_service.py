"""App-layer tool profile bootstrap and diagnostics service."""

from __future__ import annotations

from typing import Any


class ToolProfileBootstrapService:
    """Resolves active MCP tool profile and emits startup diagnostics."""

    @staticmethod
    def bootstrap(*, configured_profile: str, tooling: Any, logger) -> tuple[str, list[str]]:
        active_profile, enabled_tools, used_fallback = tooling.resolve_tools_for_profile(
            configured_profile
        )
        if used_fallback:
            logger.warning(
                "invalid_tool_profile",
                configured_profile=configured_profile,
                fallback_profile="full",
            )

        logger.info(
            "mcp_tool_profile_active",
            profile=active_profile,
            enabled_tools=len(enabled_tools),
            supported_profiles=sorted(tooling.supported_profiles),
        )
        return active_profile, [tool.name for tool in enabled_tools]
