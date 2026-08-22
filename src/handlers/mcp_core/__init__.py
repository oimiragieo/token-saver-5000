"""
MCP Core Routing Module

This module provides the core MCP tool definitions and routing logic for the
Semantic Modulator server. It maps tool names to their corresponding handler
functions across all handler modules.

Functions:
- setup_mcp_tools: Returns list of all MCP tool schemas
- route_tool_call: Dispatches tool calls to appropriate handlers

Architecture:
- All tool schemas centralized here for maintainability, split across
  schemas_*.py modules grouped by the handler module they route to
- Router (dispatch.py) delegates to handler modules (compression, AFM, file
  sync, visualization, etc.)
- Handlers receive context dict with all necessary server components

Split from a single 3670-line mcp_core.py into this package (N2 slice 2,
2026-08-22) -- see docs/design/2026-08-22-mcp-core-split.md. This __init__
re-exports every symbol the flat module used to expose, including the
underscore-prefixed helpers (nothing outside this file imports them today,
but CORE_STABLE_TOOL_NAMES was "surely internal" until registry.py proved
otherwise -- cheap insurance).
"""

from ._constants import (
    SCOPE_PROPERTIES,
    SUPPORTED_TOOL_PROFILES,
    CORE_STABLE_TOOL_NAMES,
)
from ._profile import (
    _normalize_tool_profile,  # noqa: F401 - intentional re-export, not in __all__
    _enabled_tool_names,  # noqa: F401 - intentional re-export, not in __all__
    _tools_for_profile,  # noqa: F401 - intentional re-export, not in __all__
)
from .setup import setup_mcp_tools
from .dispatch import route_tool_call

__all__ = [
    "SCOPE_PROPERTIES",
    "SUPPORTED_TOOL_PROFILES",
    "CORE_STABLE_TOOL_NAMES",
    "setup_mcp_tools",
    "route_tool_call",
]
