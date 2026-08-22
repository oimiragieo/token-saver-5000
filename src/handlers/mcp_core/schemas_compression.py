"""Tool schemas: Document Compression (ch). Split from mcp_core.py (N2 slice 2).

Each list is verbatim `Tool(...)` schema literals moved from the original
monolithic `setup_mcp_tools`, unchanged.
"""

from mcp.types import Tool, ToolAnnotations

from ._constants import SCOPE_PROPERTIES

COMPRESSION_TOOLS: list = [
    Tool(
        name="ingest_context",
        description=(
            "Ingest and compress a document into a semantic graph. "
            "This creates a fidelity-preserving encoding that reduces token usage by 80-95%. "
            "The document is analyzed for structure, relationships, and importance. "
            "Returns a compressed skeleton view. "
            "Provide document content via 'text' (inline) or 'file_url' (fetched via HTTPS). "
            "Optionally provide file_path to enable file sync tracking and version history."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The raw document text to ingest (mutually exclusive with file_url)",
                },
                "file_url": {
                    "type": "string",
                    "description": (
                        "HTTPS URL to fetch document content from (mutually exclusive with text). "
                        "Must use https:// scheme. Redirects are not followed. "
                        "Maximum 10 MB. Only text/* and application/json/xml/yaml content types accepted."
                    ),
                },
                "file_id": {
                    "type": "string",
                    "description": "Unique identifier for this document (e.g., 'paper_1', 'manual_v2')",
                },
                "file_path": {
                    "type": "string",
                    "description": "Optional: Path to source file on disk (enables file sync tracking and version history)",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (author, date, source, etc.)",
                    "properties": {
                        "author": {"type": "string"},
                        "date": {"type": "string"},
                        "source": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
                "skeleton_ratio": {
                    "type": ["number", "string"],
                    "description": (
                        "Optional: override the skeleton ratio for THIS document — a "
                        "number in (0.0, 1.0] (fraction of nodes kept as anchors), or "
                        "the string 'auto' for adaptive sizing based on corpus size. "
                        "When omitted, the server's adaptive default applies (~80% of "
                        "nodes kept below 8K tokens, scaling down to ~10% above 100K). "
                        "Applies to this file_id's later read_skeleton calls."
                    ),
                },
                "chunking_strategy": {
                    "type": "string",
                    "enum": ["auto", "fixed", "semantic"],
                    "description": "Chunking strategy: 'auto' (auto-detects structured markdown and uses fixed; otherwise semantic), 'fixed' (paragraph/sentence boundaries), or 'semantic' (embedding-based boundaries). Default: auto",
                    "default": "auto",
                },
                "query": {
                    "type": "string",
                    "description": "Optional query string. When provided, runs ingest + query_guided read_skeleton in one call and returns a query_skeleton field alongside the normal ingest stats. Skipped for very small documents (< 3 nodes).",
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="read_skeleton",
        # F4 (plan v2 §14.4) — readOnlyHint=True signals to MCP clients
        # (e.g. Claude Code's isConcurrencySafe()) that this tool has no
        # side effects and may be dispatched in parallel with other
        # read-only tools.
        annotations=ToolAnnotations(readOnlyHint=True),
        description=(
            "Read the compressed skeleton view of a previously ingested document. "
            "Shows high-importance 'anchor' concepts with summaries, and lists "
            "other sections as expandable nodes. Achieves 80-95% token reduction. "
            "Use this FIRST before requesting specific details. "
            "Selection modes: auto (default, smart detection), baseline, "
            "query_guided, evidence_aware. The response includes "
            "'selection_mode_resolved' indicating which mode was actually used."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The document identifier",
                },
                "selection_mode": {
                    "type": "string",
                    "enum": ["auto", "baseline", "query_guided", "evidence_aware"],
                    "description": (
                        "Anchor selection strategy (default: auto). "
                        "'auto' inspects the document structure and chooses "
                        "'evidence_aware' for structured audit/report docs "
                        "(3+ H2 headings + 3+ numbered findings + verdict keyword) "
                        "or 'baseline' for plain prose. "
                        "The resolved mode is reported in 'selection_mode_resolved'."
                    ),
                    "default": "auto",
                },
                "query": {
                    "type": "string",
                    "description": "Optional query used for query_guided/evidence_aware selection",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Evidence node count for evidence_aware mode",
                    "default": 5,
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Evidence sufficiency threshold for evidence_aware mode",
                    "default": 0.35,
                },
                "anchored_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords that must be preserved in the skeleton output, even if they would otherwise be hidden",
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO-8601 or unix timestamp for temporal filtering",
                },
                "include_invalidated": {
                    "type": "boolean",
                    "description": "Include invalidated facts in skeleton generation",
                    "default": False,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id"],
        },
    ),
    Tool(
        name="modulate_region",
        description=(
            "Retrieve specific sections at a chosen fidelity level. "
            "Use this to 'zoom in' on relevant parts after reading the skeleton. "
            "5 Fidelity levels (JSCCM-inspired adaptive modulation): "
            "'ABSTRACT' (~10 tokens/node) - Quick summary only, "
            "'OUTLINE' (~30 tokens/node) - Summary + section context, "
            "'STRUCTURE' (~50 tokens/node) - Summary + entities + metadata, "
            "'DETAILED' (~100 tokens/node) - Summary + entities + key excerpts, "
            "'RAW' (variable tokens) - Full original content. "
            "This implements adaptive semantic fidelity - choose lower levels when context is tight."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs to retrieve (from skeleton). Use this for multiple regions.",
                },
                "node_id": {
                    "type": "string",
                    "description": "Single node ID convenience (wraps to [node_id]). Use when expanding one region — alternative to node_ids.",
                },
                "fidelity_level": {
                    "type": "string",
                    "enum": [
                        "ABSTRACT",
                        "OUTLINE",
                        "STRUCTURE",
                        "DETAILED",
                        "RAW",
                    ],
                    "description": "Detail level to retrieve (default: RAW for maximum fidelity)",
                    "default": "RAW",
                },
            },
            "anyOf": [
                {"required": ["node_ids"]},
                {"required": ["node_id"]},
            ],
        },
    ),
    Tool(
        name="search_semantic",
        # F4 (plan v2 §14.4) — readOnlyHint=True; vector similarity
        # lookup is pure read.
        annotations=ToolAnnotations(readOnlyHint=True),
        description=(
            "Semantic search across ingested documents. "
            "Uses vector similarity to find relevant sections, "
            "even if exact keywords don't match. "
            "Returns ranked node IDs. "
            "Optionally enables evidence-aware insufficiency detection."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "file_id": {
                    "type": "string",
                    "description": "Optional: limit search to specific document",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5,
                },
                "evidence_aware": {
                    "type": "boolean",
                    "description": "Use insufficiency detection and expanded retrieval when needed",
                    "default": False,
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum best-match similarity for sufficient evidence",
                    "default": 0.35,
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO-8601 or unix timestamp for temporal filtering",
                },
                "include_invalidated": {
                    "type": "boolean",
                    "description": "Include invalidated facts in semantic search results",
                    "default": False,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_stats",
        description=(
            "Get statistics about ingested documents. "
            "Shows token counts, compression ratios, and graph structure. "
            "Useful for understanding the semantic compression efficiency."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Optional: specific file ID, or omit for global stats",
                },
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="list_documents",
        description=(
            "[LIST] LIST DOCUMENTS: Get inventory of all ingested documents. "
            "Returns structured information about each document including file_id, "
            "metadata, node count, token counts, and ingestion time. "
            "Use this to discover what documents are available for querying."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                **SCOPE_PROPERTIES,
            },
        },
    ),
    Tool(
        name="delete_document",
        description=(
            "DELETE DOCUMENT: Permanently delete an ingested document. "
            "Removes the document from memory and persistent storage. "
            "This operation cannot be undone. Use with caution. "
            "Useful for managing storage limits or removing outdated documents."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document identifier to delete",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Confirmation flag (must be true to proceed)",
                    "default": False,
                },
                **SCOPE_PROPERTIES,
            },
            "required": ["file_id", "confirm"],
        },
    ),
    Tool(
        name="adapt_to_context_window",
        description=(
            "ADAPTIVE CONTEXT ALLOCATION (JSCCM-inspired): "
            "Dynamically adjust compression based on available context window. "
            "Low availability (like low SNR in wireless) -> More compression. "
            "High availability -> Less compression, more detail. "
            "Uses learned rate allocator to determine optimal skeleton ratio. "
            "Inspired by JSCCM paper's channel adaptation strategy."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document to generate adaptive skeleton for",
                },
                "available_tokens": {
                    "type": "integer",
                    "description": "How many tokens are currently available in context window",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum context window size (default: 100000)",
                    "default": 100000,
                },
                "query_priority": {
                    "type": "number",
                    "description": "Query importance (0-1, default: 0.5)",
                    "default": 0.5,
                },
            },
            "required": ["file_id", "available_tokens"],
        },
    ),
    Tool(
        name="multilevel_encode",
        description=(
            "MULTI-LEVEL ENCODING (JSCCM-inspired): "
            "Generate skeleton with 3 priority levels: "
            "- Main branch (top 15%, always included) - critical concepts "
            "- Auxiliary branch (next 25%, include if space allows) - important details "
            "- Detail branch (remaining, only if plenty of space) - supplementary content. "
            "Progressively adds levels based on available context window. "
            "Inspired by JSCCM's parallel encoder architecture."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "Document to encode",
                },
                "available_tokens": {
                    "type": "integer",
                    "description": "Available context window tokens",
                },
            },
            "required": ["file_id", "available_tokens"],
        },
    ),
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
