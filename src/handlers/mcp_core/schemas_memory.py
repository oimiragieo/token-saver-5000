"""Tool schemas: Memory handlers (mh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

MEMORY_TOOLS: list = [
    Tool(
        name="add_memory",
        description=(
            "Store an explicit memory independently of document ingestion. "
            "Useful for user preferences, decisions, gotchas, and persistent workflow hints."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Memory text to store"},
                "category": {
                    "type": "string",
                    "description": "Optional explicit category override",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source tag such as manual, hook, or import",
                },
                "file_id": {
                    "type": "string",
                    "description": "Optional source file or document identifier",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional structured metadata for the memory record",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="search_memory",
        description=(
            "Search explicit memories within the requested scope using lexical overlap "
            "and similarity scoring."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 5)",
                    "default": 5,
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_memories",
        description=(
            "List explicit memories in the requested scope. Supports optional category "
            "filtering and result limits."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter",
                },
                "limit": {
                    "type": "integer",
                    "description": "Optional maximum memories to return",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="delete_memory",
        description="Delete a previously stored explicit memory by ID.",
        inputSchema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "Memory identifier"},
                **SCOPE_PROPERTIES,
            },
            "required": ["memory_id"],
        },
    ),
    Tool(
        name="summarize_user_memory",
        description=(
            "Summarize one user's explicit memories into preferences, topical signals, "
            "and category breakdowns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User identifier"},
                "workspace_id": SCOPE_PROPERTIES["workspace_id"],
                "agent_id": SCOPE_PROPERTIES["agent_id"],
                "session_id": SCOPE_PROPERTIES["session_id"],
            },
            "required": ["user_id"],
        },
    ),
    Tool(
        name="get_user_profile",
        description=(
            "Build a deterministic user profile from explicit stored memories within "
            "the requested scope."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "user_id": {"type": "string", "description": "User identifier"},
                "workspace_id": SCOPE_PROPERTIES["workspace_id"],
                "agent_id": SCOPE_PROPERTIES["agent_id"],
                "session_id": SCOPE_PROPERTIES["session_id"],
            },
            "required": ["user_id"],
        },
    ),
    Tool(
        name="ingest_transcript",
        description=(
            "Extract decisions, lessons, patterns, and gotchas from a conversation "
            "transcript and store them as scoped memories automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw conversation transcript text",
                },
                "mode": {
                    "type": "string",
                    "enum": ["all", "decisions", "patterns"],
                    "description": "Extraction mode (default: all)",
                },
                "source": {
                    "type": "string",
                    "description": "Source tag for stored memories (default: transcript)",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="compile_knowledge",
        description=(
            "Compile flat memories into cross-linked markdown concept articles "
            "with a navigable index. Deduplicates and groups by category."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "write_files": {
                    "type": "boolean",
                    "description": "Persist articles as markdown files (default: false)",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Output directory for compiled files",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="get_knowledge_index",
        description=(
            "Return the compiled knowledge index markdown for index-first retrieval. "
            "Useful for small knowledge bases (<500 entries) where a readable index "
            "beats embedding search."
        ),
        inputSchema={
            "type": "object",
            "properties": {**SCOPE_PROPERTIES},
        },
    ),
    Tool(
        name="lint_knowledge",
        description=(
            "Run quality checks on stored memories: staleness, near-duplicates, "
            "contradictions, and orphan detection. Returns a structured lint report."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stale_days": {
                    "type": "integer",
                    "description": "Days after which a memory is considered stale (default: 30)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="search_memory_index",
        description=(
            "Index-first memory search: compiles an index from stored memories, "
            "then returns matching articles. Best for small corpora (<500 entries) "
            "where an LLM-readable index beats embedding search."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                **SCOPE_PROPERTIES,
            },
            "required": ["query"],
        },
    ),
]
