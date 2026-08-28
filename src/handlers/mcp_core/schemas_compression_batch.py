"""Compression tool schemas: batch, directory, and extended ops."""

from mcp.types import Tool, ToolAnnotations

from ._constants import SCOPE_PROPERTIES

COMPRESSION_BATCH_TOOLS: list = [
    Tool(
        name="batch_ingest_documents",
        description=(
            "[BATCH] Batch ingest multiple documents concurrently for 4x faster throughput. "
            "Processes documents in parallel with bounded concurrency, progress tracking, "
            "and error isolation. One document failure won't block the entire batch. "
            "Returns detailed results for each document including success status and processing time. "
            "Ideal for enterprise-scale document ingestion."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "documents": {
                    "type": "array",
                    "description": "List of documents to ingest",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_id": {
                                "type": "string",
                                "description": "Unique identifier for this document",
                            },
                            "text": {
                                "type": "string",
                                "description": "Document text content",
                            },
                            "metadata": {
                                "type": "object",
                                "description": "Optional metadata for the document",
                            },
                        },
                        "required": ["file_id", "text"],
                    },
                    "minItems": 1,
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Maximum concurrent ingestions (default: 4, range: 1-8)",
                    "minimum": 1,
                    "maximum": 8,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["documents"],
        },
    ),
    Tool(
        name="ingest_directory",
        description=(
            "[DIR] Bulk ingest code files from a directory using glob patterns. "
            "Scans a directory for matching files and ingests them in parallel. "
            "Uses PathValidator for security (prevents path traversal). "
            "Ideal for quickly ingesting an entire codebase or project directory. "
            "Default patterns: *.py, *.js, *.ts. Default exclusions: node_modules, __pycache__, venv."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to scan for files",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for files to include (default: ['*.py', '*.js', '*.ts'])",
                    "default": ["*.py", "*.js", "*.ts"],
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Patterns to exclude (default: ['**/node_modules/**', '**/__pycache__/**', '**/venv/**'])",
                    "default": ["**/node_modules/**", "**/__pycache__/**", "**/venv/**"],
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum files to ingest (default: 50, range: 1-100)",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                },
                "max_concurrent": {
                    "type": "integer",
                    "description": "Maximum concurrent ingestions (default: 4, range: 1-8)",
                    "minimum": 1,
                    "maximum": 8,
                    "default": 4,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["directory"],
        },
    ),
    Tool(
        name="recommend_fidelity",
        description=(
            "[TIP] Get intelligent recommendation for optimal fidelity level. "
            "Analyzes your use case, number of nodes, token budget, and query complexity "
            "to suggest the best fidelity level (ABSTRACT, OUTLINE, STRUCTURE, DETAILED, or RAW). "
            "Returns recommendation with reasoning, token estimate, and alternatives. "
            "Use this BEFORE modulate_region to make informed decisions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "use_case": {
                    "type": "string",
                    "description": "What you want to do with the content",
                    "enum": [
                        "quick_summary",
                        "topic_overview",
                        "entity_extraction",
                        "question_answering",
                        "detailed_analysis",
                        "exact_quotes",
                        "code_review",
                        "fact_verification",
                    ],
                },
                "num_nodes": {
                    "type": "integer",
                    "description": "Number of nodes you plan to retrieve",
                    "minimum": 1,
                },
                "token_budget": {
                    "type": "integer",
                    "description": "Optional: Maximum tokens available (None = no limit)",
                    "minimum": 10,
                },
                "query_complexity": {
                    "type": "string",
                    "description": "Complexity of your query (default: medium)",
                    "enum": ["simple", "medium", "complex"],
                },
            },
            "required": ["use_case", "num_nodes"],
        },
    ),
    Tool(
        name="diff_reingest",
        description=(
            "Re-ingest a previously ingested document, preserving embeddings for "
            "unchanged chunks. Only recomputes embeddings for changed sections, "
            "saving significant computation time on iterative document updates."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The existing document identifier to update",
                },
                "text": {
                    "type": "string",
                    "description": "The updated document text",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id", "text"],
        },
    ),
    Tool(
        name="find_duplicates",
        description=(
            "Detect near-duplicate content across different ingested documents. "
            "Uses cosine similarity on chunk embeddings to find redundant content "
            "that could be deduplicated to save tokens."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Similarity threshold (0.0-1.0). Default: 0.9",
                    "default": 0.9,
                },
            },
        },
    ),
    Tool(
        name="get_compression_presets",
        description=(
            "List available compression presets (code-review, chat, research, "
            "aggressive, balanced). Each preset maps to optimal skeleton_ratio "
            "and fidelity settings for common use cases."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="check_context_budget",
        description=(
            "Check how much of your LLM context window is being used and get "
            "proactive compression recommendations. Returns usage percentage "
            "and suggests action at 40%/60%/75% thresholds."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "current_tokens": {
                    "type": "integer",
                    "description": "Current token count in context",
                },
                "context_limit": {
                    "type": "integer",
                    "description": "Maximum context window size (default: 200000)",
                    "default": 200000,
                },
            },
            "required": ["current_tokens"],
        },
    ),
    Tool(
        name="prune_by_relevance",
        description=(
            "Prune document nodes by query relevance using attention-guided scoring. "
            "Keeps only the most relevant nodes for a given query, achieving up to 6x "
            "compression with better quality than blind ratio-based pruning."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document ID to prune"},
                "query": {"type": "string", "description": "Query to score relevance against"},
                "keep_ratio": {
                    "type": "number",
                    "description": "Fraction of nodes to keep (0.0-1.0)",
                    "default": 0.5,
                },
            },
            "required": ["doc_id", "query"],
        },
    ),
    Tool(
        name="get_multi_level_skeleton",
        description=(
            "Generate 3-tier skeleton output: headline (top 10%), summary (top 30%), "
            "and full (100%). Client picks the depth needed for their context budget."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "description": "Document ID"},
            },
            "required": ["doc_id"],
        },
    ),
    Tool(
        name="evict_stale",
        description=(
            "Find and list stale documents that have not been accessed within a given time window. "
            "Helps keep context budget tight by identifying candidates for eviction."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "max_age_hours": {
                    "type": "number",
                    "description": "Max hours since last access",
                    "default": 1.0,
                },
            },
        },
    ),
    Tool(
        name="advise_context",
        description=(
            "Analyze all ingested documents and recommend optimal context strategy. "
            "Returns model recommendations, pruning priorities, and compression advice."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_compression_insights",
        description=(
            "Get insights from compression history: best ratios per content type, "
            "average fidelity scores, and data-driven strategy recommendations."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="generate_rewrite_prompt",
        description=(
            "Generate a structured rewrite prompt for client-side LLM compression. "
            "Returns system instructions and user prompt optimized for generative compression."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID (optional if text provided)",
                },
                "text": {
                    "type": "string",
                    "description": "Text to compress (optional if doc_id provided)",
                },
                "target_ratio": {
                    "type": "number",
                    "description": "Target compression ratio",
                    "default": 0.5,
                },
                "preserve_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords to preserve in rewrite",
                },
            },
        },
    ),
    Tool(
        name="compress_codebase",
        description=(
            "Compress a codebase directory into a semantic skeleton. "
            "Uses tensor-grep AST analysis when available for structure extraction. "
            "Falls back to directory scanning without tensor-grep. "
            "Optionally filter by query or file patterns."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Path to codebase directory"},
                "query": {
                    "type": "string",
                    "description": "Optional query to focus on relevant code",
                },
                "max_files": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum files to include",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["directory"],
        },
    ),
    Tool(
        name="search_code",
        description=(
            "Fast regex or literal code search using tensor-grep trigram index. "
            "Returns file paths and matching lines. "
            "Chain with ingest_context for targeted compression of search results. "
            "Falls back gracefully if tensor-grep is not installed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex or literal pattern"},
                "directory": {"type": "string", "description": "Directory to search"},
                **SCOPE_PROPERTIES,
            },
            "required": ["pattern"],
        },
    ),
]
