"""Tool schemas: AFM Dialogue (afm) + Temporal (th). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

AFM_TOOLS: list = [
    Tool(
        name="afm_add_message",
        description=(
            "ADAPTIVE FOCUS MEMORY: Add message to dialogue history. "
            "AFM (Adaptive Focus Memory, arXiv:2511.12712v1) manages multi-turn conversations "
            "by assigning adaptive fidelity to each message based on recency, semantic relevance, "
            "and importance. Messages are automatically classified as CRITICAL (safety-sensitive), "
            "RELEVANT, or TRIVIAL. Use this to build dialogue history before calling afm_build_context."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant", "system"],
                    "description": "Message role (user, assistant, or system)",
                },
                "content": {
                    "type": "string",
                    "description": "Message content",
                },
            },
            "required": ["role", "content"],
        },
    ),
    Tool(
        name="afm_build_context",
        description=(
            "[AFM] ADAPTIVE FOCUS MEMORY: Build optimized context for current query. "
            "Uses semantic similarity + recency weighting + importance classification "
            "to pack dialogue history under strict token budget. Achieves ~66% token reduction "
            "while preserving safety-critical information (e.g., allergies, constraints). "
            "Each message gets adaptive fidelity: FULL (verbatim), COMPRESSED (summary), or "
            "PLACEHOLDER. Messages packed chronologically to preserve conversation flow."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "current_query": {
                    "type": "string",
                    "description": "Current user query to build context for",
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens allowed in context",
                },
                "system_preamble": {
                    "type": "string",
                    "description": "Optional system message to include first",
                },
            },
            "required": ["current_query", "budget_tokens"],
        },
    ),
    Tool(
        name="afm_get_stats",
        description=(
            "[STATS] ADAPTIVE FOCUS MEMORY: Get dialogue statistics. "
            "Returns total messages, current turn index, and importance breakdown "
            "(critical/relevant/trivial counts). Useful for monitoring dialogue state."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="afm_clear_history",
        description=(
            "[DELETE] ADAPTIVE FOCUS MEMORY: Clear dialogue history. "
            "Removes all messages and resets turn counter. Use when starting a new conversation "
            "or when dialogue context is no longer relevant."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="afm_export_history",
        description=(
            "[SAVE] ADAPTIVE FOCUS MEMORY: Export dialogue history to JSON. "
            "Saves current conversation state including all messages, turn counter, "
            "and metadata. Use this to preserve conversations for later resume. "
            "Returns JSON string that can be saved and imported later."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Optional session ID for this export (default: 'default')",
                    "default": "default",
                },
            },
        },
    ),
    Tool(
        name="afm_import_history",
        description=(
            "[LOAD] ADAPTIVE FOCUS MEMORY: Import dialogue history from JSON. "
            "Restores a previously exported conversation state. This replaces "
            "the current dialogue history. Use this to resume saved conversations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Session ID to load (default: 'default')",
                    "default": "default",
                },
            },
        },
    ),
]


TEMPORAL_TOOLS: list = [
    Tool(
        name="get_context_block",
        description=(
            "Build a lifecycle-aware context block with active facts, recent events, "
            "and a cache-stable skeleton prefix."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Document ID"},
                "query": {"type": "string", "description": "Optional query bias"},
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO-8601 or unix timestamp reference time",
                },
                "max_facts": {
                    "type": "integer",
                    "description": "Maximum active facts to include",
                    "default": 5,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum recent events to include",
                    "default": 10,
                },
                "include_invalidated": {
                    "type": "boolean",
                    "description": "Include invalidated facts/events",
                    "default": False,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="search_timeline",
        description="Search lifecycle events across ingests, reads, searches, and invalidations.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Optional timeline text filter"},
                "file_id": {"type": "string", "description": "Optional document ID filter"},
                "fact_id": {"type": "string", "description": "Optional exact fact ID filter"},
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional event type allowlist",
                },
                "since": {"type": "string", "description": "Optional lower time bound"},
                "until": {"type": "string", "description": "Optional upper time bound"},
                "limit": {
                    "type": "integer",
                    "description": "Maximum events to return",
                    "default": 25,
                },
                "include_invalidated": {
                    "type": "boolean",
                    "description": "Include invalidation and supersession events",
                    "default": True,
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="list_fact_history",
        description="List temporal fact versions for a document or exact fact identifier.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Optional document ID filter"},
                "fact_id": {"type": "string", "description": "Optional exact fact ID filter"},
                "as_of": {
                    "type": "string",
                    "description": "Optional reference time for version visibility",
                },
                "include_invalidated": {
                    "type": "boolean",
                    "description": "Include invalidated versions",
                    "default": True,
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="invalidate_fact",
        description="Invalidate a fact so temporal retrieval excludes it by default.",
        inputSchema={
            "type": "object",
            "properties": {
                "fact_id": {"type": "string", "description": "Exact fact ID"},
                "reason": {
                    "type": "string",
                    "description": "Human-readable invalidation reason",
                },
                "timestamp": {
                    "type": "string",
                    "description": "Optional ISO-8601 or unix timestamp",
                },
            },
            "required": ["fact_id", "reason"],
        },
    ),
]
