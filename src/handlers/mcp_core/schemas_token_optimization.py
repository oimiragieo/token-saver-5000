"""Tool schemas: Token Optimization (toh). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool

from ._constants import SCOPE_PROPERTIES

TOKEN_OPTIMIZATION_TOOLS: list = [
    Tool(
        name="estimate_tokens",
        description=(
            "Estimate token count for a given text using multiple methods. "
            "Returns accurate count (tiktoken), fast estimate (bytes/4), "
            "JSON-optimized estimate (bytes/2), and raw byte count. "
            "Useful for budgeting context window usage before ingestion."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to estimate token count for",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="configure_for_client",
        description=(
            "Configure compression parameters for a specific LLM client or model. "
            "Accepts a model identifier (e.g. claude-opus-4-6, gpt-4o) or explicit "
            "context window size. Auto-tunes skeleton ratio, chunk sizes, and "
            "fidelity defaults to maximize token efficiency for the target model."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model identifier (e.g. claude-opus-4-6, gpt-4o, gemini-2.0-pro)",
                },
                "context_window_tokens": {
                    "type": "integer",
                    "description": "Explicit context window size in tokens (overrides model lookup)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="set_compression_profile",
        description=(
            "Set a named compression profile for the session. "
            "Profiles bundle skeleton_ratio, fidelity, and chunk_size into presets: "
            "minimal (max compression), summary (quick overview), balanced (default), "
            "detailed (deep analysis), full (near-original). "
            "Explicit parameters in subsequent tool calls override profile defaults."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "profile_name": {
                    "type": "string",
                    "enum": ["minimal", "summary", "balanced", "detailed", "full"],
                    "description": "Compression profile to activate",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["profile_name"],
        },
    ),
    Tool(
        name="get_compression_profile",
        description=(
            "Get the active compression profile for the session. "
            "Returns the profile name and its parameter values "
            "(skeleton_ratio, fidelity, chunk_size)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="compress_meta_tokens",
        description=(
            "[COMPRESS] Lossless meta-token compression (arXiv 2506.00307). "
            "Finds repeated token subsequences and replaces them with compact "
            "dictionary symbols (§1, §2, …). Fully reversible. "
            "Best for repetitive text with recurring phrases. "
            "Returns compressed_text, dictionary, and token savings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to compress using meta-token substitution",
                },
                "min_length": {
                    "type": "integer",
                    "description": "Minimum n-gram length to substitute (default: 2)",
                    "default": 2,
                    "minimum": 2,
                },
                "min_frequency": {
                    "type": "integer",
                    "description": "Minimum occurrence count for substitution (default: 2)",
                    "default": 2,
                    "minimum": 2,
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum dictionary entries (default: 50)",
                    "default": 50,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="recommend_compression",
        description=(
            "[ADVISE] Recommend the optimal compression profile for a document. "
            "Simulates each profile, predicts quality (entity retention + coverage), "
            "and returns the most compressed profile meeting your quality floor. "
            "Useful before ingesting to choose between minimal/summary/balanced/detailed/full."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to evaluate compression profiles for",
                },
                "quality_floor": {
                    "type": "number",
                    "description": "Minimum acceptable quality (0.0–1.0, default: 0.7)",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "query": {
                    "type": "string",
                    "description": "Optional query for relevance-aware quality scoring",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="recover_session",
        description=(
            "Recover session state after conversation compaction. "
            "Returns a compact summary of all prior ingestions, configurations, "
            "and tool calls for the given session."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID to recover"},
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="filter_cli_output",
        description=(
            "Filter CLI command output to reduce token usage. "
            "Auto-detects command type (git, pytest, npm, lint, etc.) and applies "
            "optimal filtering strategy. Strips ANSI codes, extracts stats, groups "
            "errors, removes progress bars."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Raw CLI output to filter"},
                "command_hint": {
                    "type": "string",
                    "description": (
                        "Optional hint: git_diff, git_status, test_output, "
                        "install_output, lint_output, json_output, ansi_output, "
                        "progress_output, tree_output, log_output"
                    ),
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="get_savings_report",
        description=(
            "Get a detailed report of token savings for this session. "
            "Shows total tokens saved, dollars saved, compression ratios, "
            "per-tool breakdown, monthly projection, and ROI vs the Pro plan. "
            "Use this to justify the value of token compression to your team."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": "Model for cost calculation (default: from session config)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="get_savings_inline",
        description=(
            "Get a compact one-line savings summary. "
            "Embed this in other tool responses to show real-time savings. "
            "Example: 'Saved 3,400 tokens ($0.051) | Session: $2.34 saved (8.1x ROI)'"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="advise_cache_strategy",
        description=(
            "Get the optimal prompt caching strategy for your model. "
            "Each LLM provider handles caching differently -- Anthropic uses explicit markers, "
            "OpenAI is automatic, Gemini has implicit+explicit modes. "
            "Returns specific tips for maximizing cache hits and cost savings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": (
                        "Model identifier (e.g., claude-4-sonnet, gpt-4.1, gemini-2.5-flash)"
                    ),
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["model_id"],
        },
    ),
    Tool(
        name="generate_structural_summary",
        description=(
            "Generate a compact structural outline of a code file. "
            "Extracts imports, class definitions, and function signatures (with type hints). "
            "Replaces function bodies with `...`. Achieves ~80-90% token reduction while "
            "preserving the full API surface. Ideal for codebase exploration, "
            "API discovery, and context-window-efficient code review. "
            "Supports Python (AST-based) and other languages (regex fallback)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Source code text to summarize",
                },
                "file_path": {
                    "type": "string",
                    "description": (
                        "Optional file path (used to detect language from extension "
                        "and to label the output header)"
                    ),
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["text"],
        },
    ),
    Tool(
        name="detect_dead_code",
        description=(
            "Detect Python files in a directory that are never imported by other files. "
            "Uses regex-based import graph analysis to identify unreachable modules. "
            "Entry points (main.py, server.py, __init__.py, test_*.py, etc.) are always "
            "considered live. Returns dead file list with estimated token savings -- "
            "useful for excluding dead code before compression to reduce noise."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to scan for Python files (scans recursively)",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["directory"],
        },
    ),
    Tool(
        name="get_original_output",
        description=(
            "Retrieve the original (pre-compression) content for a tee entry. "
            "When compression is aggressive (>80%), the original is automatically saved. "
            "Use this to recover full output when the compressed version lost important details."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "Tee entry ID (returned in compressed output metadata)",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["entry_id"],
        },
    ),
    Tool(
        name="list_tee_entries",
        description=(
            "List recent tee entries with metadata. "
            "Shows what original content has been preserved for recovery. "
            "Filter by source (cli_optimizer, proxy, compression)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max entries to return (default 20)",
                },
                "source": {
                    "type": "string",
                    "description": "Filter by source: cli_optimizer, proxy, compression",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="tee_store_stats",
        description=(
            "Get tee store statistics: entry count, total size, mode, thresholds. "
            "Use to monitor tee storage usage."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="discover_savings",
        description=(
            "Discover missed token savings opportunities. "
            "Scans a directory or list of text items to estimate what could be compressed. "
            "Returns ranked opportunities with estimated savings per file. "
            "Use before ingesting content to prioritize which files benefit most."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to scan for compressible files",
                },
                "items": {
                    "type": "array",
                    "description": "List of text items to analyze. Each item may be a plain string or an object with a 'text' field.",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string"},
                                    "label": {"type": "string"},
                                    "file_ext": {"type": "string"},
                                },
                                "required": ["text"],
                            },
                        ]
                    },
                },
                "max_files": {
                    "type": "integer",
                    "description": "Max files to scan in directory mode (default 500)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="calculate_roi",
        description=(
            "Calculate ROI of using gotcontext compression vs raw token usage. "
            "Shows monthly cost comparison: without vs with compression, Pro plan cost, "
            "net savings, and ROI multiplier. Powers the website ROI calculator."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": (
                        "Model identifier (e.g., claude-sonnet-4-6, gpt-4o, gemini-2.5-pro)"
                    ),
                },
                "tokens_per_day": {
                    "type": "integer",
                    "description": "Estimated input tokens per day per user (default 500000)",
                },
                "team_size": {
                    "type": "integer",
                    "description": "Number of team members (default 1)",
                },
                "compression_ratio": {
                    "type": "number",
                    "description": "Expected compression ratio 0-1 (default 0.85 = 85%)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="check_budget",
        description=(
            "Check token budget usage against configured limits. "
            "Supports per-session, daily, and monthly budgets. "
            "Returns usage status, alert level, and projected usage. "
            "Schema rejects unknown fields (e.g. legacy 'period' arg) so "
            "MCP agents get explicit validation errors instead of silent "
            "argument-drops."
        ),
        inputSchema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "session_limit": {
                    "type": "integer",
                    "description": "Session token budget limit (0 = unlimited)",
                },
                "daily_limit": {
                    "type": "integer",
                    "description": "Daily token budget limit (0 = unlimited)",
                },
                "monthly_limit": {
                    "type": "integer",
                    "description": "Monthly token budget limit (0 = unlimited)",
                },
                "record_tokens": {
                    "type": "integer",
                    "description": "Record new token usage before checking budget",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Name of tool that consumed the tokens (for tracking)",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="export_team_data",
        description=(
            "Export aggregated team savings data. "
            "Supports JSON, CSV, and Prometheus exposition formats. "
            "Use for team dashboards, monitoring, and cost reporting."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "members": {
                    "type": "array",
                    "description": "Team member stats to aggregate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "sessions": {"type": "integer"},
                            "original_tokens": {"type": "integer"},
                            "compressed_tokens": {"type": "integer"},
                            "operations": {"type": "integer"},
                        },
                        "required": ["user_id"],
                    },
                },
                "format": {
                    "type": "string",
                    "description": "Export format: json (default), csv, prometheus",
                    "enum": ["json", "csv", "prometheus"],
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
]
