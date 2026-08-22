"""Tool schemas: Connector (coh) + Resource (rh) + Detection (dh) + Docs (doch) + Help (hh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

DOCS_TOOLS: list = [
    Tool(
        name="gc_read_doc",
        description=(
            "Read a gotcontext product/API documentation page as markdown. "
            "Recommended: pass a full docs URL returned by gc_search_docs "
            "(e.g. 'https://gotcontext.ai/docs' or 'https://gotcontext.ai/docs/security-scanning'). "
            "A bare slug is resolved best-effort against indexed docs; if it names a "
            "section anchor rather than a standalone doc, the call returns ranked "
            "gc_search_docs suggestions instead of the page. "
            "Returns JSON with markdown, source_url, and length_tokens."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url_or_slug": {
                    "type": "string",
                    "description": (
                        "A full docs URL returned by gc_search_docs (recommended), "
                        "or a docs slug. A slug that names a section anchor rather "
                        "than a standalone doc returns gc_search_docs suggestions — "
                        "call gc_search_docs first to get the exact URL."
                    ),
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["url_or_slug"],
        },
    ),
    Tool(
        name="gc_search_docs",
        description=(
            "Search gotcontext product/API documentation and return ranked docs URLs. "
            "When to use vs gc_lookup: gc_lookup is for company-name disambiguation; "
            "gc_search_docs is for product/API documentation queries. Returns JSON "
            "with results containing title, snippet, url, and score."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language product/API documentation query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of ranked documentation results to return.",
                    "default": 5,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["query"],
        },
    ),
]


DETECTION_TOOLS: list = [
    Tool(
        name="check_blind_spots",
        description=(
            "BLIND SPOT DETECTOR: Analyze if your response missed critical context. "
            "This tool embeds your response and compares it to ALL nodes in the document. "
            "If relevant content was not retrieved, it alerts you and suggests auto-injection. "
            "Use AFTER generating a response to ensure fidelity. "
            "This implements the 'Self-Correcting Context Loop'."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "ai_response": {
                    "type": "string",
                    "description": "The response you generated",
                },
                "file_id": {
                    "type": "string",
                    "description": "Which document was being discussed",
                },
                "retrieved_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Which node IDs you actually retrieved/viewed",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["ai_response", "file_id", "retrieved_nodes"],
        },
    ),
    Tool(
        name="detect_hallucination",
        description=(
            "HALLUCINATION DETECTOR: Check if a response is grounded in source material. "
            "Compares response embedding to document graph. "
            "Flags responses with low similarity to all nodes (possible fabrication). "
            "Use when uncertain about answer accuracy."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "ai_response": {
                    "type": "string",
                    "description": "The response to validate",
                },
                "file_id": {
                    "type": "string",
                    "description": "The source document",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["ai_response", "file_id"],
        },
    ),
]


RESOURCE_TOOLS: list = [
    Tool(
        name="check_resource_health",
        description=(
            "[SAVE] RESOURCE HEALTH: Check resource usage and system health. "
            "Returns storage, memory, and document count metrics with proactive warnings and recommendations. "
            "Use this to monitor resource usage before ingesting large documents or when experiencing slowdowns. "
            "Prevents hitting storage limits unexpectedly."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="check_environment",
        description=(
            "[HEALTH] Check comprehensive environment health: models loaded, memory usage, "
            "cache hit ratio, stale documents, and disk space. "
            "Returns recommendations for optimization. "
            "Use this to understand system state before heavy operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="should_compress",
        description=(
            "[PRE-CHECK] TOKEN-EFFICIENT PRE-CHECK: Estimate token count for a file WITHOUT reading content. "
            "Uses file size heuristics and binary content detection. "
            "CRITICAL: Call this BEFORE reading or ingesting any file. "
            "Detects binary files (PDF, DOCX, images) that need conversion before compression. "
            "Returns recommendation: SKIP (<100 tokens), DIRECT_READ (100-500), COMPRESS (>500), "
            "or CONVERT_THEN_COMPRESS (binary files with MarkItDown suggestion). "
            "Fields: needs_conversion, is_text_readable, conversion_tool, reason."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to assess (checks size + binary detection, minimal read)",
                },
                "content_type": {
                    "type": "string",
                    "enum": ["auto", "prose", "code"],
                    "description": "Content type hint for better estimation (default: auto-detect from extension)",
                    "default": "auto",
                },
            },
            "required": ["file_path"],
        },
    ),
]


HELP_TOOLS: list = [
    Tool(
        name="tool_help",
        description=(
            "[HELP] Get detailed help, examples, and tips for any Semantic Modulator tool. "
            "Returns structured help with parameter descriptions, usage examples, and related tools. "
            "Use without tool_name to see all available tools organized by category. "
            "Set verbose=true for comprehensive examples."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to get help for (omit to see all tools)",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Include full examples (default: false)",
                    "default": False,
                },
            },
            "required": [],
        },
    ),
]


CONNECTOR_TOOLS: list = [
    Tool(
        name="list_connector_types",
        description="List available managed connector types and their purposes.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="create_connector_feed",
        description=(
            "Create a managed connector feed definition for exported web, GitHub, S3, "
            "or Slack payloads."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Connector feed name"},
                "connector_type": {
                    "type": "string",
                    "enum": ["web", "github", "s3", "slack_export"],
                    "description": "Connector type",
                },
                "config": {
                    "type": "object",
                    "description": "Connector-specific feed configuration",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional feed metadata",
                },
            },
            "required": ["name", "connector_type", "config"],
        },
    ),
    Tool(
        name="list_connector_feeds",
        description="List managed connector feeds and their last sync state.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_connector_feed",
        description="Fetch one managed connector feed definition.",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Connector feed name"}},
            "required": ["name"],
        },
    ),
    Tool(
        name="sync_connector_feed",
        description=(
            "Normalize and ingest a managed connector feed through the standard "
            "compression pipeline."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Connector feed name"},
                **SCOPE_PROPERTIES,
            },
            "required": ["name"],
        },
    ),
]
