"""MCP tool registry facade.

Phase 1 compatibility layer: delegates to existing central registry.
"""

from typing import Set

from src.handlers.mcp_core import CORE_STABLE_TOOL_NAMES, setup_mcp_tools

__all__ = ["CORE_STABLE_TOOL_NAMES", "setup_mcp_tools", "list_tool_names"]


def list_tool_names(profile: str = "full") -> Set[str]:
    """Return active tool names for the selected profile."""
    return {tool.name for tool in setup_mcp_tools(profile)}
