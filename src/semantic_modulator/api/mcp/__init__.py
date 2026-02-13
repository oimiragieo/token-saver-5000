"""MCP API facade namespace."""

from .registry import CORE_STABLE_TOOL_NAMES, setup_mcp_tools
from .router import route_tool_call

__all__ = ["CORE_STABLE_TOOL_NAMES", "setup_mcp_tools", "route_tool_call"]
