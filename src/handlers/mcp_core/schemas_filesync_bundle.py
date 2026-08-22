"""Tool schemas: File Sync (fs) + Handoff Bundles (bh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

FILESYNC_TOOLS: list = [
    Tool(
        name="check_file_sync",
        description=(
            "[SYNC] FILE SYNC CHECK: Check if cached document is in sync with source file on disk. "
            "Detects if file was modified after ingestion by comparing modification time and checksums. "
            "Use this before long operations to ensure you're working with current data. "
            "Only works for documents ingested with file_path parameter."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to check sync status",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="diff_cached_file",
        description=(
            "[DIFF] FILE DIFF: Generate unified diff between cached version and current file on disk. "
            "Shows exactly what changed since ingestion (additions, deletions, modifications). "
            "Useful for reviewing changes before refreshing cache or to see what was edited. "
            "Returns unified diff format similar to 'git diff'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to diff",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around changes (default: 3)",
                    "default": 3,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="refresh_document",
        description=(
            "[REFRESH] REFRESH DOCUMENT: Re-ingest document from source file to update cache with latest changes. "
            "Stores old version in history (default: keeps last 10 versions). "
            "Use this when check_file_sync detects staleness or after external edits. "
            "Returns new compression stats and confirms version was saved."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document ID to refresh from disk",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="get_version_history",
        description=(
            "[HISTORY] VERSION HISTORY: Get version timeline for a document. "
            "Shows all cached versions with timestamps, checksums, file paths, and compression stats. "
            "Similar to 'git log' for cached documents. Use this to browse version history "
            "before using diff_cached_file to compare specific versions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID to get version history for",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["doc_id"],
        },
    ),
]


BUNDLE_TOOLS: list = [
    Tool(
        name="create_handoff_bundle",
        description=(
            "Create a structured, auditable handoff bundle from a compressed document, "
            "including distilled skeleton context and optional focused evidence."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "Visible document identifier"},
                "query": {
                    "type": "string",
                    "description": "Optional query to focus bundle distillation",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Focused search result count",
                    "default": 5,
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional handoff metadata for ownership or routing",
                },
                "bundle_id": {
                    "type": "string",
                    "description": "Optional explicit bundle identifier",
                },
                "created_at": {
                    "type": "string",
                    "description": "Optional explicit creation timestamp",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="list_handoff_bundles",
        description="List structured handoff bundles visible to the current scope.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Optional visible document identifier filter",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="get_handoff_bundle",
        description="Fetch one structured handoff bundle including its distilled artifacts.",
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_id": {"type": "string", "description": "Handoff bundle identifier"},
                **SCOPE_PROPERTIES,
            },
            "required": ["bundle_id"],
        },
    ),
    Tool(
        name="replay_handoff_bundle",
        description="Replay a structured handoff bundle as text plus token-efficient artifact payloads.",
        inputSchema={
            "type": "object",
            "properties": {
                "bundle_id": {"type": "string", "description": "Handoff bundle identifier"},
                **SCOPE_PROPERTIES,
            },
            "required": ["bundle_id"],
        },
    ),
]
