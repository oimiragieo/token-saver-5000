"""Module-level constants for the MCP core routing package.

Moved verbatim from mcp_core.py (N2 slice 2). `CORE_STABLE_TOOL_NAMES` has
exactly one owner in this package -- nowhere else redefines it.
"""

from typing import Set

SCOPE_PROPERTIES = {
    "workspace_id": {
        "type": "string",
        "description": "Optional workspace scope for multi-tenant isolation",
    },
    "user_id": {
        "type": "string",
        "description": "Optional user scope for multi-tenant isolation",
    },
    "agent_id": {
        "type": "string",
        "description": "Optional agent scope for multi-tenant isolation",
    },
    "session_id": {
        "type": "string",
        "description": "Optional session scope for multi-tenant isolation",
    },
}

CORE_STABLE_TOOL_NAMES: Set[str] = {
    "ingest_context",
    "read_skeleton",
    "search_semantic",
    "modulate_region",
    "get_stats",
    "list_documents",
    "delete_document",
}
SUPPORTED_TOOL_PROFILES = {"full", "core_stable"}
