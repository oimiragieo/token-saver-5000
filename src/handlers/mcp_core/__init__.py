"""
MCP Core Routing Module

This module provides the core MCP tool definitions and routing logic for the
Semantic Modulator server. It maps tool names to their corresponding handler
functions across all handler modules.

Functions:
- setup_mcp_tools: Returns list of all MCP tool schemas
- route_tool_call: Dispatches tool calls to appropriate handlers

Architecture:
- The typed registry owns each schema/handler pair; schema literals remain
  split across schemas_*.py modules for maintainability
- setup.py projects schemas and dispatch.py invokes registry handlers
- Handlers receive context dict with all necessary server components

Split from a single 3670-line mcp_core.py into this package (N2 slice 2,
2026-08-22) -- see docs/design/2026-08-22-mcp-core-split.md. This __init__
re-exports every symbol the flat module used to expose, including the
underscore-prefixed helpers (nothing outside this file imports them today,
but CORE_STABLE_TOOL_NAMES was "surely internal" until registry.py proved
otherwise -- cheap insurance).
"""

from ._constants import (
    CORE_STABLE_TOOL_NAMES,
    SCOPE_PROPERTIES,
    SUPPORTED_TOOL_PROFILES,
)
from ._profile import (
    _enabled_tool_names,  # noqa: F401 - intentional re-export, not in __all__
    _normalize_tool_profile,  # noqa: F401 - intentional re-export, not in __all__
    _tools_for_profile,  # noqa: F401 - intentional re-export, not in __all__
)
from .dispatch import route_tool_call
from .registry import (
    REGISTERED_TOOLS,
    RegisteredTool,
    get_registered_tool,
    registered_tool_names,
    registered_tools,
)
from .setup import setup_mcp_tools

__all__ = [
    "CORE_STABLE_TOOL_NAMES",
    "REGISTERED_TOOLS",
    "SCOPE_PROPERTIES",
    "SUPPORTED_TOOL_PROFILES",
    "RegisteredTool",
    "get_registered_tool",
    "registered_tool_names",
    "registered_tools",
    "route_tool_call",
    "setup_mcp_tools",
]
