"""MCP router facade.

Phase 1 compatibility layer: delegates routing to the legacy router.
"""

from src.handlers.mcp_core import route_tool_call

__all__ = ["route_tool_call"]
