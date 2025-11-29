"""
MCP Core Routing Module

This module provides the core MCP tool definitions and routing logic for the
Semantic Modulator server. It maps tool names to their corresponding handler
functions across all handler modules.

Functions:
- setup_mcp_tools: Returns list of all 39 MCP tool schemas
- route_tool_call: Dispatches tool calls to appropriate handlers

Architecture:
- All tool schemas centralized here for maintainability
- Router delegates to handler modules (compression, AFM, file sync, visualization, etc.)
- Handlers receive context dict with all necessary server components
"""

from typing import Any, Dict, List

from mcp.types import Tool

# Import all handler modules
from . import compression_handlers as ch
from . import afm_handlers as afm
from . import file_sync_handlers as fs
from . import resource_handlers as rh
from . import detection_handlers as dh
from . import ace_handlers as ace
from . import visualization_handlers as vh
from . import help_handlers as hh

# Import structured logging for operation tracking
from ..structured_logging import get_logger

logger = get_logger("semantic-modulator")


def setup_mcp_tools() -> List[Tool]:
    """
    Define all 39 MCP tools available in the Semantic Modulator server.

    Returns:
        List of Tool objects with complete schemas (name, description, inputSchema)

    Tool Categories:
    - Document Compression (9): ingest, read_skeleton, modulate_region, search, stats, list, delete, adapt, multilevel
    - Batch Processing (1): batch_ingest_documents
    - Directory Ingestion (1): ingest_directory
    - Graph Visualization (4): export_graph_json, visualize_graph_html, export_graph_graphml, explain_compression_decision
    - Fidelity Advisor (1): recommend_fidelity
    - Detection (2): check_blind_spots, detect_hallucination
    - AFM Dialogue (6): add_message, build_context, get_stats, clear, export, import
    - File Sync (4): check_sync, diff, refresh, version_history
    - Resource Management (3): check_health, check_environment, should_compress
    - Help & Documentation (1): tool_help
    - ACE Framework (7): ace_generate, ace_reflect, ace_curate, ace_grow, ace_refine, ace_get_playbook, ace_execute_cycle
    """
    return [
        # === DOCUMENT COMPRESSION TOOLS (9) ===
        Tool(
            name="ingest_context",
            description=(
                "Ingest and compress a document into a semantic graph. "
                "This creates a fidelity-preserving encoding that reduces token usage by 80-95%. "
                "The document is analyzed for structure, relationships, and importance. "
                "Returns a compressed skeleton view. "
                "Optionally provide file_path to enable file sync tracking and version history."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The raw document text to ingest",
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
                },
                "required": ["text", "file_id"],
            },
        ),
        Tool(
            name="read_skeleton",
            description=(
                "Read the compressed skeleton view of a previously ingested document. "
                "Shows high-importance 'anchor' concepts with summaries, and lists "
                "other sections as expandable nodes. Achieves 80-95% token reduction. "
                "Use this FIRST before requesting specific details."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "The document identifier",
                    },
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
                        "description": "List of node IDs to retrieve (from skeleton)",
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
                "required": ["node_ids"],
            },
        ),
        Tool(
            name="search_semantic",
            description=(
                "Semantic search across ingested documents. "
                "Uses vector similarity to find relevant sections, "
                "even if exact keywords don't match. "
                "Returns ranked node IDs."
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
                },
            },
        ),
        Tool(
            name="list_documents",
            description=(
                "📚 LIST DOCUMENTS: Get inventory of all ingested documents. "
                "Returns structured information about each document including file_id, "
                "metadata, node count, token counts, and ingestion time. "
                "Use this to discover what documents are available for querying."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
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
                },
                "required": ["file_id", "confirm"],
            },
        ),
        Tool(
            name="adapt_to_context_window",
            description=(
                "ADAPTIVE CONTEXT ALLOCATION (JSCCM-inspired): "
                "Dynamically adjust compression based on available context window. "
                "Low availability (like low SNR in wireless) → More compression. "
                "High availability → Less compression, more detail. "
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
        # === DETECTION TOOLS (2) ===
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
                },
                "required": ["ai_response", "file_id"],
            },
        ),
        # === AFM DIALOGUE TOOLS (6) ===
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
                "PLACEHOLDER (stub). Messages packed chronologically to preserve conversation flow."
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
        # === FILE SYNC TOOLS (4) ===
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
                },
                "required": ["doc_id"],
            },
        ),
        # === RESOURCE MANAGEMENT TOOLS (3) ===
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
                "🔍 Check comprehensive environment health: models loaded, memory usage, "
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
                "⚡ TOKEN-EFFICIENT PRE-CHECK: Estimate token count for a file WITHOUT reading it. "
                "Uses file size heuristics to recommend whether compression is needed. "
                "CRITICAL for token efficiency - call this BEFORE reading large files to avoid wasting tokens. "
                "Returns: estimated tokens, compression recommendation (NO_COMPRESS, RECOMMEND_COMPRESS, "
                "STRONGLY_RECOMMEND, or MUST_COMPRESS), and potential token savings. "
                "This enables intelligent context management without burning tokens to check file size."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to estimate (does NOT read content, only checks size)",
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
        # === HELP & DOCUMENTATION TOOLS (1) ===
        Tool(
            name="tool_help",
            description=(
                "📚 Get detailed help, examples, and tips for any Semantic Modulator tool. "
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
        # === ACE FRAMEWORK TOOLS (7) ===
        Tool(
            name="ace_generate",
            description=(
                "[AFM] ACE GENERATE: Generate reasoning trajectory for a task using ACE playbook. "
                "Produces step-by-step reasoning that applies relevant bullets from the playbook. "
                "Each step includes relevant guidelines, reasoning, and confidence scores. "
                "Use this to guide semantic node selection and compression decisions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or query to reason about",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum trajectory steps (default: 5)",
                    },
                    "top_k_bullets": {
                        "type": "integer",
                        "description": "Bullets to consider per step (default: 5)",
                    },
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="ace_reflect",
            description=(
                "[ANALYZE] ACE REFLECT: Extract insights from a reasoning trajectory. "
                "Analyzes what worked (successes) and what didn't (failures) to formulate new bullets. "
                "Returns insights with confidence scores and reasoning. "
                "Use after completing a task to learn and improve the playbook."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "trajectory": {
                        "type": "array",
                        "description": "Generated reasoning trajectory from ace_generate",
                        "items": {"type": "object"},
                    },
                    "outcome": {
                        "type": "string",
                        "description": "What actually happened (result description)",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the trajectory led to success",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                },
                "required": ["trajectory", "outcome", "success"],
            },
        ),
        Tool(
            name="ace_curate",
            description=(
                "[CURATE] ACE CURATE: Integrate insights into playbook via delta updates. "
                "Applies incremental changes (add/update/remove bullets) with semantic deduplication. "
                "Prevents context collapse through grow-and-refine strategy. "
                "Use after reflecting to evolve the playbook."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "insights": {
                        "type": "array",
                        "description": "Insights from ace_reflect",
                        "items": {"type": "object"},
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                    "max_bullets": {
                        "type": "integer",
                        "description": "Maximum bullets (triggers pruning if exceeded)",
                    },
                },
                "required": ["insights"],
            },
        ),
        Tool(
            name="ace_grow_context",
            description=(
                "[ADD] ACE GROW: Manually add bullets to playbook (grow operation). "
                "Directly insert principles, strategies, tactics, constraints, or preferences. "
                "Use to seed domain-specific knowledge or codify team standards. "
                "Each bullet gets an embedding for semantic operations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bullets": {
                        "type": "array",
                        "description": "Bullets to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "bullet_type": {
                                    "type": "string",
                                    "enum": [
                                        "principle",
                                        "strategy",
                                        "tactic",
                                        "constraint",
                                        "preference",
                                        "learned",
                                    ],
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["text", "bullet_type"],
                        },
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                },
                "required": ["bullets"],
            },
        ),
        Tool(
            name="ace_refine_context",
            description=(
                "🎯 ACE REFINE: Update bullet performance based on feedback (refine operation). "
                "Adjusts confidence scores for specific bullets based on success/failure. "
                "Use to reinforce successful patterns or penalize failed approaches. "
                "Enables continuous improvement of the playbook."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "bullet_ids": {
                        "type": "array",
                        "description": "Bullet IDs to update",
                        "items": {"type": "string"},
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether these bullets led to success",
                    },
                    "confidence_boost": {
                        "type": "number",
                        "description": "Adjustment amount (default: 0.05)",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                },
                "required": ["bullet_ids", "success"],
            },
        ),
        Tool(
            name="ace_get_playbook",
            description=(
                "📖 ACE GET PLAYBOOK: Retrieve current ACE playbook state. "
                "Returns all bullets with performance stats, versioning, and delta history. "
                "Supports filtering by confidence, bullet type, or custom criteria. "
                "Use to inspect the evolved playbook and understand learned patterns."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                    "include_embeddings": {
                        "type": "boolean",
                        "description": "Include bullet embeddings (default: false)",
                    },
                    "min_confidence": {
                        "type": "number",
                        "description": "Filter bullets below this confidence",
                    },
                    "bullet_type": {
                        "type": "string",
                        "description": "Filter by bullet type",
                        "enum": [
                            "principle",
                            "strategy",
                            "tactic",
                            "constraint",
                            "preference",
                            "learned",
                        ],
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="ace_execute_cycle",
            description=(
                "[SYNC] ACE EXECUTE CYCLE: Execute complete ACE cycle (Generate → Reflect → Curate). "
                "Convenience tool that combines the three-step ACE process into one call. "
                "Generates trajectory, reflects on outcome, and curates insights automatically. "
                "Use for rapid iteration and continuous playbook improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or query",
                    },
                    "outcome": {
                        "type": "string",
                        "description": "What actually happened",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task succeeded",
                    },
                    "context_id": {
                        "type": "string",
                        "description": "ACE context identifier (default: 'default')",
                    },
                    "max_trajectory_steps": {
                        "type": "integer",
                        "description": "Maximum trajectory steps (default: 5)",
                    },
                },
                "required": ["task", "outcome", "success"],
            },
        ),
        # === BATCH PROCESSING TOOL (1) ===
        Tool(
            name="batch_ingest_documents",
            description=(
                "🚀 Batch ingest multiple documents concurrently for 4× faster throughput. "
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
                },
                "required": ["documents"],
            },
        ),
        # === DIRECTORY INGESTION TOOL (1) ===
        Tool(
            name="ingest_directory",
            description=(
                "📂 Bulk ingest code files from a directory using glob patterns. "
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
                },
                "required": ["directory"],
            },
        ),
        # === FIDELITY ADVISOR TOOL (1) ===
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
        # === GRAPH VISUALIZATION TOOLS (4) ===
        Tool(
            name="export_graph_json",
            description=(
                "[STATS] Export semantic graph as JSON for programmatic access. "
                "Returns a structured JSON representation of the semantic graph with nodes, edges, "
                "importance scores, and statistics. Perfect for custom analysis or integration with "
                "other tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Document ID to export",
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum nodes to include (default: 50)",
                        "minimum": 1,
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Minimum importance score to include (default: 0.0)",
                        "minimum": 0.0,
                    },
                },
                "required": ["file_id"],
            },
        ),
        Tool(
            name="visualize_graph_html",
            description=(
                "🎨 Generate interactive HTML visualization of the semantic graph. "
                "Creates a beautiful, interactive web page with draggable nodes, zoom/pan, "
                "color-coded importance, and edge weights. Great for exploring and presenting "
                "compression decisions. Requires pyvis library."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Document ID to visualize",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save HTML file (e.g., 'graph.html')",
                    },
                    "max_nodes": {
                        "type": "integer",
                        "description": "Maximum nodes to visualize (default: 50)",
                        "minimum": 1,
                    },
                },
                "required": ["file_id", "output_path"],
            },
        ),
        Tool(
            name="export_graph_graphml",
            description=(
                "📁 Export semantic graph as GraphML for analysis tools. "
                "GraphML is a standard XML format supported by Gephi, Cytoscape, igraph, "
                "and NetworkX. Perfect for advanced network analysis, visualization, "
                "and research workflows."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Document ID to export",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path to save GraphML file (e.g., 'graph.graphml')",
                    },
                },
                "required": ["file_id", "output_path"],
            },
        ),
        Tool(
            name="explain_compression_decision",
            description=(
                "[ANALYZE] Explain why a specific node was kept or dropped during compression. "
                "Provides detailed analysis including importance score ranking, connectivity, "
                "key entities, and relationships with other nodes. Perfect for understanding "
                "and debugging compression decisions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_id": {
                        "type": "string",
                        "description": "Document ID",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "Node ID to explain (e.g., 'quantum_paper_n3')",
                    },
                },
                "required": ["file_id", "node_id"],
            },
        ),
    ]


async def route_tool_call(name: str, args: Dict[str, Any], context: Dict[str, Any]) -> str:
    """
    Route MCP tool calls to appropriate handler functions.

    This function automatically logs:
    - Request ID (correlation)
    - Tool name
    - Execution duration (milliseconds)
    - Memory delta (MB)
    - Success/failure status
    - Error details (if failed)

    Args:
        name: Tool name (e.g., "ingest_context", "afm_add_message")
        args: Tool arguments as dictionary
        context: Server context with all necessary components:
            - compressor: SemanticCompressor instance
            - blind_spot_detector: BlindSpotDetector instance
            - halo_detector: HaloEffectDetector instance
            - context_window_adapter: ContextWindowAdapter instance
            - multilevel_encoder: MultiLevelSemanticEncoder instance
            - focus_manager: FocusManager instance
            - persistence: PersistenceManager instance
            - resource_manager: ResourceManager instance
            - sync_manager: FileSyncManager instance
            - version_manager: VersionManager instance
            - validate_file_id: Validation function
            - validate_node_ids: Validation function
            - validate_token_count: Validation function
            - save_file_sync_metadata: Save function

    Returns:
        Handler result as formatted string

    Raises:
        ValueError: If tool name is unknown
    """
    # Define routing table mapping tool names to handler functions
    router = {
        # Document Compression (9 tools)
        "ingest_context": ch.handle_ingest,
        "read_skeleton": ch.handle_read_skeleton,
        "modulate_region": ch.handle_modulate_region,
        "search_semantic": ch.handle_search_semantic,
        "get_stats": ch.handle_get_stats,
        "list_documents": ch.handle_list_documents,
        "delete_document": ch.handle_delete_document,
        "adapt_to_context_window": ch.handle_adapt_to_context_window,
        "multilevel_encode": ch.handle_multilevel_encode,
        # Batch Processing (1 tool)
        "batch_ingest_documents": ch.handle_batch_ingest,
        # Directory Ingestion (1 tool)
        "ingest_directory": ch.handle_ingest_directory,
        # Fidelity Advisor (1 tool)
        "recommend_fidelity": ch.handle_recommend_fidelity,
        # Detection (2 tools)
        "check_blind_spots": dh.handle_check_blind_spots,
        "detect_hallucination": dh.handle_detect_hallucination,
        # AFM Dialogue (6 tools)
        "afm_add_message": afm.handle_afm_add_message,
        "afm_build_context": afm.handle_afm_build_context,
        "afm_get_stats": afm.handle_afm_get_stats,
        "afm_clear_history": afm.handle_afm_clear_history,
        "afm_export_history": afm.handle_afm_export_history,
        "afm_import_history": afm.handle_afm_import_history,
        # File Sync (4 tools)
        "check_file_sync": fs.handle_check_file_sync,
        "diff_cached_file": fs.handle_diff_cached_file,
        "refresh_document": fs.handle_refresh_document,
        "get_version_history": fs.handle_get_version_history,
        # Resource Management (3 tools)
        "check_resource_health": rh.handle_check_resource_health,
        "check_environment": rh.handle_check_environment,
        "should_compress": rh.handle_should_compress,
        # Help & Documentation (1 tool)
        "tool_help": hh.handle_tool_help,
        # Graph Visualization (4 tools)
        "export_graph_json": vh.handle_export_graph_json,
        "visualize_graph_html": vh.handle_visualize_graph_html,
        "export_graph_graphml": vh.handle_export_graph_graphml,
        "explain_compression_decision": vh.handle_explain_compression_decision,
        # ACE Framework (7 tools) - now using HandlerContext like all other handlers
        "ace_generate": ace.handle_ace_generate,
        "ace_reflect": ace.handle_ace_reflect,
        "ace_curate": ace.handle_ace_curate,
        "ace_grow_context": ace.handle_ace_grow_context,
        "ace_refine_context": ace.handle_ace_refine_context,
        "ace_get_playbook": ace.handle_ace_get_playbook,
        "ace_execute_cycle": ace.handle_ace_execute_cycle,
    }

    # Lookup handler
    if name not in router:
        available_tools = ", ".join(sorted(router.keys()))
        raise ValueError(
            f"Unknown tool: '{name}'\n\n"
            f"Available tools ({len(router)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use list_tools() to see all available tools with descriptions"
        )

    # Route to handler (async)
    handler = router[name]
    handler_module = getattr(handler, "__module__", "unknown")
    handler_name = getattr(handler, "__name__", "unknown")
    logger.info(
        "tool_routing", tool_name=name, handler_module=handler_module, handler_function=handler_name
    )

    return await handler(context, args)
