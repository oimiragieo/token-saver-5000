"""
MCP Core Routing Module

This module provides the core MCP tool definitions and routing logic for the
Semantic Modulator server. It maps tool names to their corresponding handler
functions across all handler modules.

Functions:
- setup_mcp_tools: Returns list of all 54 MCP tool schemas
- route_tool_call: Dispatches tool calls to appropriate handlers

Architecture:
- All tool schemas centralized here for maintainability
- Router delegates to handler modules (compression, AFM, file sync, visualization, etc.)
- Handlers receive context dict with all necessary server components
"""

from typing import Any, Dict, List, Set

from mcp.types import Tool

# Import all handler modules
from . import compression_handlers as ch
from . import afm_handlers as afm
from . import bundle_handlers as bh
from . import file_sync_handlers as fs
from . import resource_handlers as rh
from . import detection_handlers as dh
from . import ace_handlers as ace
from . import visualization_handlers as vh
from . import help_handlers as hh
from . import connector_handlers as coh
from . import model_handlers as moh
from . import multimodal_handlers as mmh
from . import temporal_handlers as th
from . import experimental_handlers as exp
from . import experiment_handlers as eh
from . import memory_handlers as mh
from . import prompt_handlers as ph
from . import token_optimization_handlers as toh

# Import structured logging for operation tracking
from ..structured_logging import get_logger

logger = get_logger("semantic-modulator")

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


def _normalize_tool_profile(profile: str) -> str:
    normalized = (profile or "full").strip().lower()
    if normalized not in SUPPORTED_TOOL_PROFILES:
        raise ValueError(
            f"Unknown tool profile '{profile}'. "
            f"Supported profiles: {sorted(SUPPORTED_TOOL_PROFILES)}"
        )
    return normalized


def _enabled_tool_names(all_names: Set[str], profile: str) -> Set[str]:
    normalized = _normalize_tool_profile(profile)
    if normalized == "full":
        return set(all_names)
    return set(all_names) & CORE_STABLE_TOOL_NAMES


def _tools_for_profile(tools: List[Tool], profile: str) -> List[Tool]:
    enabled_names = _enabled_tool_names({tool.name for tool in tools}, profile)
    return [tool for tool in tools if tool.name in enabled_names]


def setup_mcp_tools(profile: str = "full") -> List[Tool]:
    """
    Define all MCP tools available in the Semantic Modulator server.

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
    - Experimental (9): toon_encode, toon_decode, scar_compress, scar_get_stats, multimodal_ingest,
                        verify_compression, calculate_reward, get_evidence_stats, generate_synthetic_tests
    """
    all_tools = [
        # === DOCUMENT COMPRESSION TOOLS (9) ===
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
                        "description": "Skeleton ratio (0.0-1.0) or 'auto' for adaptive sizing based on corpus size. Default: 0.2",
                        "default": 0.2,
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
        # === HELP & DOCUMENTATION TOOLS (1) ===
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
                "[ACE] ACE REFINE: Update bullet performance based on feedback (refine operation). "
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
                "[ACE] ACE GET PLAYBOOK: Retrieve current ACE playbook state. "
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
                "[SYNC] ACE EXECUTE CYCLE: Execute complete ACE cycle (Generate -> Reflect -> Curate). "
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
                },
                "required": ["documents"],
            },
        ),
        # === DIRECTORY INGESTION TOOL (1) ===
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
                    **SCOPE_PROPERTIES,
                },
                "required": ["file_id"],
            },
        ),
        Tool(
            name="visualize_graph_html",
            description=(
                "[VIZ] Generate interactive HTML visualization of the semantic graph. "
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
                    **SCOPE_PROPERTIES,
                },
                "required": ["file_id", "output_path"],
            },
        ),
        Tool(
            name="export_graph_graphml",
            description=(
                "[VIZ] Export semantic graph as GraphML for analysis tools. "
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
                    **SCOPE_PROPERTIES,
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
                    **SCOPE_PROPERTIES,
                },
                "required": ["file_id", "node_id"],
            },
        ),
        # === EXPERIMENTAL TOOLS (5) - NOT PRODUCTION-READY ===
        Tool(
            name="toon_encode",
            description=(
                "[EXPERIMENTAL] Encode data to TOON format (~40% smaller than JSON). "
                "TOON = Token-Oriented Object Notation. Pure Python, always available. "
                "NOT production-ready. Returns experimental flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "Data to encode (dict or list)",
                    },
                },
                "required": ["data"],
            },
        ),
        Tool(
            name="toon_decode",
            description=(
                "[EXPERIMENTAL] Decode TOON format back to structured data. "
                "TOON is lossy - optimized for LLM consumption, not round-trip serialization. "
                "NOT production-ready. Returns experimental flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "toon_input": {
                        "type": "string",
                        "description": "TOON-formatted string to decode",
                    },
                },
                "required": ["toon_input"],
            },
        ),
        Tool(
            name="scar_compress",
            description=(
                "[EXPERIMENTAL] Compress embeddings using SCAR (learnable compression). "
                "WARNING: Uses UNTRAINED random weights by default. Requires PyTorch. "
                "NOT production-ready without model training. Returns experimental flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to compress embeddings for",
                    },
                    "target_dim": {
                        "type": "integer",
                        "description": "Target embedding dimension (default: 128)",
                        "default": 128,
                    },
                    **SCOPE_PROPERTIES,
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="scar_get_stats",
            description=(
                "[EXPERIMENTAL] Get SCAR compressor statistics and model state. "
                "Shows PyTorch availability and model training status. "
                "NOT production-ready. Returns experimental flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="multimodal_ingest",
            description=(
                "[EXPERIMENTAL] Ingest mixed content (text, code, images). "
                "Requires Pillow for image support. Image paths validated for security. "
                "NOT production-ready. Returns experimental flag."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Unique document identifier",
                    },
                    "text_content": {
                        "type": "string",
                        "description": "Text content to ingest",
                    },
                    "code_content": {
                        "type": "string",
                        "description": "Code content to ingest",
                    },
                    "code_language": {
                        "type": "string",
                        "description": "Code language (default: python)",
                        "default": "python",
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Paths to images (validated for security)",
                    },
                    **SCOPE_PROPERTIES,
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="ingest_multimodal",
            description=(
                "Production-grade multimodal ingestion for text, code, images, audio transcripts, "
                "and document-with-images bundles."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Logical multimodal document identifier",
                    },
                    "text_content": {"type": "string", "description": "Optional text content"},
                    "code_content": {"type": "string", "description": "Optional code content"},
                    "code_language": {
                        "type": "string",
                        "description": "Optional code language label",
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional local image paths validated for security",
                    },
                    "image_captions": {
                        "type": "object",
                        "description": "Optional mapping from submitted image path to caption text",
                    },
                    "image_ocr_text": {
                        "type": "object",
                        "description": "Optional mapping from submitted image path to OCR text",
                    },
                    "audio_items": {
                        "type": "array",
                        "description": "Optional transcript-backed audio payloads",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "path": {"type": "string"},
                                "transcript": {"type": "string"},
                            },
                            "required": ["transcript"],
                        },
                    },
                    "document_items": {
                        "type": "array",
                        "description": "Optional document-with-images bundles",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "text": {"type": "string"},
                                "image_paths": {"type": "array", "items": {"type": "string"}},
                                "image_captions": {"type": "object"},
                                "image_ocr_text": {"type": "object"},
                            },
                        },
                    },
                    "video_items": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Currently unsupported and rejected explicitly",
                    },
                    **SCOPE_PROPERTIES,
                },
                "required": ["doc_id"],
            },
        ),
        Tool(
            name="search_multimodal",
            description="Search a production multimodal project using text, code, or image queries.",
            inputSchema={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Logical multimodal document identifier",
                    },
                    "query": {"type": "string", "description": "Text or code query"},
                    "query_type": {
                        "type": "string",
                        "enum": ["text", "code", "image"],
                        "description": "Query modality",
                        "default": "text",
                    },
                    "image_query_path": {
                        "type": "string",
                        "description": "Required when query_type=image",
                    },
                    "top_k": {"type": "integer", "description": "Result count", "default": 5},
                    "filter_modality": {
                        "type": "string",
                        "enum": ["text", "code", "image"],
                        "description": "Optional result modality filter",
                    },
                    **SCOPE_PROPERTIES,
                },
                "required": ["doc_id"],
            },
        ),
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
        Tool(
            name="get_provider_profile",
            description="Get provider-aware pricing, cache telemetry fields, and prompt-shaping guidance for a model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model identifier"},
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="estimate_model_cost",
            description="Estimate token cost savings for a model using original and compressed token counts.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model identifier"},
                    "original_tokens": {"type": "integer", "description": "Original token count"},
                    "compressed_tokens": {
                        "type": "integer",
                        "description": "Compressed token count",
                    },
                },
                "required": ["model", "original_tokens", "compressed_tokens"],
            },
        ),
        Tool(
            name="optimize_for_model",
            description="Generate provider-aware cost, fidelity, and prompt-shaping recommendations for a target model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model identifier"},
                    "text": {"type": "string", "description": "Representative source text"},
                    "use_case": {
                        "type": "string",
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
                        "description": "Expected retrieval node count",
                    },
                    "token_budget": {
                        "type": "integer",
                        "description": "Optional explicit token budget",
                    },
                    "query_complexity": {
                        "type": "string",
                        "enum": ["simple", "medium", "complex"],
                        "default": "medium",
                    },
                },
                "required": ["model", "text", "use_case", "num_nodes"],
            },
        ),
        Tool(
            name="assess_cache_compatibility",
            description="Assess whether a provider and harness combination exposes enough telemetry to validate prompt cache behavior reliably.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model identifier"},
                    "harness": {
                        "type": "string",
                        "enum": [
                            "anthropic_api",
                            "claude_code",
                            "openai_api",
                            "codex_cli",
                            "gemini_api",
                            "gemini_cli",
                        ],
                        "description": "Provider or CLI surface used to make the request",
                    },
                    "raw_usage_available": {
                        "type": "boolean",
                        "description": "Whether raw provider usage payloads are visible",
                        "default": False,
                    },
                    "cli_stats_available": {
                        "type": "boolean",
                        "description": "Whether CLI-exported cache stats are available",
                        "default": False,
                    },
                },
                "required": ["model", "harness"],
            },
        ),
        Tool(
            name="capture_cache_telemetry",
            description="Normalize provider-side prompt cache telemetry from a real model API response and warn on silent cache misses.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Model identifier"},
                    "api_response": {
                        "type": "object",
                        "description": "Raw provider response object containing usage telemetry",
                    },
                    "file_id": {
                        "type": "string",
                        "description": "Optional document or prompt identifier tied to this request",
                    },
                    "prompt_id": {
                        "type": "string",
                        "description": "Optional prompt identifier returned by render_prompt_template",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session identifier for aggregating multi-turn cache metrics",
                    },
                    "actual_rendered_prefix": {
                        "type": "string",
                        "description": "Optional exact prefix string actually sent to the provider for cache miss diagnosis",
                    },
                    "expected_cache_hit": {
                        "type": "boolean",
                        "description": "Whether this request was expected to reuse a cached prompt prefix",
                        "default": False,
                    },
                },
                "required": ["model", "api_response"],
            },
        ),
        Tool(
            name="diagnose_cache_miss",
            description=(
                "Diagnose why an expected provider cache hit missed by comparing the recorded prompt "
                "expectation with the actual rendered prefix that reached the provider."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt_id": {
                        "type": "string",
                        "description": "Prompt identifier returned by render_prompt_template",
                    },
                    "model": {"type": "string", "description": "Model identifier"},
                    "actual_rendered_prefix": {
                        "type": "string",
                        "description": "Exact rendered prompt prefix actually sent to the provider",
                    },
                    "api_response": {
                        "type": "object",
                        "description": "Raw provider response object containing usage telemetry",
                    },
                },
                "required": ["prompt_id", "model", "actual_rendered_prefix", "api_response"],
            },
        ),
        # === ASG-SI TOOLS (4) - EXPERIMENTAL ===
        Tool(
            name="verify_compression",
            description=(
                "[EXPERIMENTAL] Verify compression operation using ASG-SI contracts. "
                "Checks preconditions (valid input, fidelity level) and postconditions "
                "(compression ratio, skeleton quality). Returns contract violations. "
                "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "document": {
                        "type": "string",
                        "description": "Original document text",
                    },
                    "skeleton_text": {
                        "type": "string",
                        "description": "Compressed skeleton output",
                    },
                    "node_map": {
                        "type": "object",
                        "description": "Node ID to description mapping",
                    },
                    "original_tokens": {
                        "type": "integer",
                        "description": "Original token count",
                    },
                    "skeleton_tokens": {
                        "type": "integer",
                        "description": "Skeleton token count",
                    },
                    "fidelity_level": {
                        "type": "string",
                        "description": "Target fidelity (ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW)",
                    },
                },
                "required": [
                    "document",
                    "skeleton_text",
                    "original_tokens",
                    "skeleton_tokens",
                    "fidelity_level",
                ],
            },
        ),
        Tool(
            name="calculate_reward",
            description=(
                "[EXPERIMENTAL] Calculate decomposed compression reward using ASG-SI system. "
                "Computes 5 reward components: Schema (validation), Semantic (meaning preservation), "
                "Fidelity (ratio adherence), Composition (graph integrity), Memory (efficiency). "
                "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "Original text",
                    },
                    "output_text": {
                        "type": "string",
                        "description": "Compressed output",
                    },
                    "input_tokens": {
                        "type": "integer",
                        "description": "Input token count",
                    },
                    "output_tokens": {
                        "type": "integer",
                        "description": "Output token count",
                    },
                    "fidelity_level": {
                        "type": "string",
                        "description": "Target fidelity level",
                    },
                    "ssim_score": {
                        "type": "number",
                        "description": "Pre-calculated SSIM score (optional)",
                    },
                },
                "required": [
                    "input_text",
                    "output_text",
                    "input_tokens",
                    "output_tokens",
                    "fidelity_level",
                ],
            },
        ),
        Tool(
            name="get_evidence_stats",
            description=(
                "[EXPERIMENTAL] Get evidence store statistics for audit trail. "
                "The store maintains a tamper-evident blockchain-style chain of all "
                "compression operations with cryptographic integrity verification. "
                "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="generate_synthetic_tests",
            description=(
                "[EXPERIMENTAL] Generate synthetic test cases for adversarial testing. "
                "Uses ASG-SI experience synthesis to create boundary cases, adversarial "
                "documents, and stress test scenarios for compression validation. "
                "Based on arxiv.org/abs/2512.23760 Audited Skill-Graph Self-Improvement."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "description": "Test type: boundary, dialogue, ace, or all",
                        "default": "boundary",
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed for reproducibility (optional)",
                    },
                },
            },
        ),
        # --- New tools: diff re-ingestion, dedup, presets ---
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
            name="create_prompt_template",
            description=(
                "Create a managed prompt template with version 1 and optional deployment "
                "label. Use this to make prompts first-class artifacts instead of hard-coded strings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique prompt template name"},
                    "description": {
                        "type": "string",
                        "description": "Human-readable prompt description",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Static system prompt section",
                    },
                    "user_prompt_template": {
                        "type": "string",
                        "description": "User prompt template with optional {variables}",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Template variable names used by the prompt",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional structured prompt metadata",
                    },
                    "deployment_label": {
                        "type": "string",
                        "description": "Optional initial deployment label (for example: production, staging)",
                    },
                },
                "required": ["name", "description", "system_prompt", "user_prompt_template"],
            },
        ),
        Tool(
            name="update_prompt_template",
            description=(
                "Create a new version of an existing prompt template. Supports prompt edits, "
                "variable changes, metadata updates, and change notes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Existing prompt template name"},
                    "description": {
                        "type": "string",
                        "description": "Optional updated template description",
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "Optional replacement system prompt",
                    },
                    "user_prompt_template": {
                        "type": "string",
                        "description": "Optional replacement user prompt template",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional replacement variable list",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata patch merged into the latest version metadata",
                    },
                    "change_note": {
                        "type": "string",
                        "description": "Optional summary of why this version changed",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="list_prompt_templates",
            description=(
                "List managed prompt templates with their latest version and deployment labels. "
                "Optionally include all versions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "include_versions": {
                        "type": "boolean",
                        "description": "Include all versions for each template (default: false)",
                        "default": False,
                    }
                },
            },
        ),
        Tool(
            name="get_prompt_template",
            description=(
                "Get a prompt template and resolve a specific version or deployment label "
                "to the exact prompt content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Prompt template name"},
                    "version": {
                        "type": "integer",
                        "description": "Optional version number to resolve",
                    },
                    "deployment_label": {
                        "type": "string",
                        "description": "Optional deployment label to resolve (mutually exclusive with version)",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="deploy_prompt_version",
            description=(
                "Assign or move a deployment label (production, staging, canary) to a specific "
                "prompt template version."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Prompt template name"},
                    "version": {"type": "integer", "description": "Version to deploy"},
                    "deployment_label": {
                        "type": "string",
                        "description": "Deployment label to assign",
                    },
                    "allow_stable_prefix_change": {
                        "type": "boolean",
                        "description": "Acknowledge and allow deployment if the stable cacheable prefix will change",
                        "default": False,
                    },
                },
                "required": ["name", "version", "deployment_label"],
            },
        ),
        Tool(
            name="compare_prompt_versions",
            description=(
                "Compare two versions of the same prompt template and return changed fields "
                "plus a unified diff."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Prompt template name"},
                    "version_a": {"type": "integer", "description": "Base version"},
                    "version_b": {"type": "integer", "description": "Comparison version"},
                },
                "required": ["name", "version_a", "version_b"],
            },
        ),
        Tool(
            name="render_prompt_template",
            description=(
                "Resolve and render a prompt template into cache-friendly ordered sections "
                "for a provider call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Prompt template name"},
                    "variables": {
                        "type": "object",
                        "description": "Template variables used to render the user prompt",
                    },
                    "version": {
                        "type": "integer",
                        "description": "Optional version number to resolve",
                    },
                    "deployment_label": {
                        "type": "string",
                        "description": "Optional deployment label to resolve",
                    },
                    "tool_definitions": {
                        "type": "string",
                        "description": "Optional serialized tool definitions to pin in the stable prefix",
                    },
                    "rag_context": {
                        "type": "string",
                        "description": "Optional static retrieved context to place before volatile sections",
                    },
                    "few_shot_examples": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional few-shot examples",
                    },
                    "chat_history": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional prior conversation turns",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional dynamic metadata to place in the volatile tail",
                    },
                },
                "required": ["name"],
            },
        ),
        Tool(
            name="list_prefix_collisions",
            description=(
                "List rendered prompt prefixes that collide across templates so shared "
                "provider-cacheable prefixes are visible."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="audit_prompt_cacheability",
            description=(
                "Audit a composed prompt for cache-friendly section ordering and volatile "
                "metadata that can break provider prefix caching."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "sections": {
                        "type": "array",
                        "description": "Ordered prompt sections using canonical names",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": [
                                        "tool_definitions",
                                        "system_instructions",
                                        "rag_context",
                                        "few_shot_examples",
                                        "chat_history",
                                        "metadata",
                                        "user_query",
                                    ],
                                },
                                "content": {"type": "string"},
                            },
                            "required": ["name", "content"],
                        },
                    }
                },
                "required": ["sections"],
            },
        ),
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
        # -- Knowledge Management (Phase 1-4) ------------------------------------
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
        Tool(
            name="create_dataset",
            description=(
                "Create a reusable named benchmark/evaluation dataset from inline cases "
                "or a JSON corpus fixture."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Dataset name"},
                    "description": {"type": "string", "description": "Dataset description"},
                    "cases": {
                        "type": "array",
                        "description": "Optional inline benchmark cases",
                        "items": {
                            "type": "object",
                            "properties": {
                                "case_id": {"type": "string"},
                                "name": {"type": "string"},
                                "text": {"type": "string"},
                                "min_compression_ratio": {"type": "number"},
                                "min_token_savings_pct": {"type": "number"},
                                "query": {"type": "string"},
                            },
                            "required": [
                                "case_id",
                                "name",
                                "text",
                                "min_compression_ratio",
                                "min_token_savings_pct",
                            ],
                        },
                    },
                    "source_path": {
                        "type": "string",
                        "description": "Optional benchmark corpus JSON path",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional structured dataset metadata",
                    },
                },
                "required": ["name", "description"],
            },
        ),
        Tool(
            name="list_datasets",
            description="List named datasets available for experiment runs.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="run_experiment",
            description=(
                "Run a tracked benchmark/evaluation experiment over a named dataset and "
                "store benchmark, verifier, and reward outputs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "dataset_name": {"type": "string", "description": "Dataset name"},
                    "mode": {
                        "type": "string",
                        "enum": ["baseline", "query_guided", "evidence_aware"],
                        "description": "Benchmark mode to execute",
                        "default": "baseline",
                    },
                    "case_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of case IDs to run",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Optional similarity threshold override",
                        "default": 0.75,
                    },
                    "skeleton_ratio": {
                        "type": "number",
                        "description": "Optional skeleton ratio override",
                        "default": 0.2,
                    },
                    "baseline_run_id": {
                        "type": "string",
                        "description": "Optional baseline run identifier",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional structured run metadata",
                    },
                },
                "required": ["dataset_name"],
            },
        ),
        Tool(
            name="get_experiment_run",
            description="Fetch a stored experiment run and its per-case evaluation details.",
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Experiment run identifier"}
                },
                "required": ["run_id"],
            },
        ),
        Tool(
            name="compare_experiment_runs",
            description=(
                "Compare two experiment runs and report deltas in pass counts, compression, "
                "verification, and reward quality."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "run_id_a": {"type": "string", "description": "Base run identifier"},
                    "run_id_b": {"type": "string", "description": "Comparison run identifier"},
                },
                "required": ["run_id_a", "run_id_b"],
            },
        ),
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
        # Phase 5: Research-based features (2025 papers)
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
        # === TOKEN OPTIMIZATION TOOLS (v0.11.0) ===
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
        # === ARXIV PAPER TECHNIQUES (v0.12.0) ===
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
        # === SESSION JOURNAL (v0.13.0) ===
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
        # === TENSOR-GREP INTEGRATION (v0.13.0) ===
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
        # === CLI OUTPUT OPTIMIZER ===
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
        # === SAVINGS TRACKER (v0.14.0) ===
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
        # === CACHE STRATEGY ADVISOR ===
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
        # === STRUCTURAL SUMMARY (v0.15.0) ===
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
        # === DEAD CODE DETECTOR (v0.15.0) ===
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
        # === TEE/RECOVERY (v0.16.0) ===
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
        # === DISCOVER SAVINGS (v0.16.0) ===
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
        # === ROI CALCULATOR (v0.16.0) ===
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
        # === BUDGET MONITORING (v0.16.0) ===
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
        # === TEAM EXPORT (v0.16.0) ===
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
    # Sort tools alphabetically for prompt cache stability (v0.11.0).
    # Claude Code and other MCP clients cache the prompt prefix including tool
    # schemas. Deterministic ordering prevents cache invalidation when tools are
    # added or reordered internally.
    all_tools.sort(key=lambda t: t.name)
    return _tools_for_profile(all_tools, profile)


async def route_tool_call(
    name: str, args: Dict[str, Any], context: Dict[str, Any], tool_profile: str = "full"
) -> str:
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
        # Experimental (5 tools) - NOT production-ready
        "toon_encode": exp.handle_toon_encode,
        "toon_decode": exp.handle_toon_decode,
        "scar_compress": exp.handle_scar_compress,
        "scar_get_stats": exp.handle_scar_get_stats,
        "multimodal_ingest": exp.handle_multimodal_ingest,
        "ingest_multimodal": mmh.handle_ingest_multimodal,
        "search_multimodal": mmh.handle_search_multimodal,
        "create_handoff_bundle": bh.handle_create_handoff_bundle,
        "list_handoff_bundles": bh.handle_list_handoff_bundles,
        "get_handoff_bundle": bh.handle_get_handoff_bundle,
        "replay_handoff_bundle": bh.handle_replay_handoff_bundle,
        "get_provider_profile": moh.handle_get_provider_profile,
        "estimate_model_cost": moh.handle_estimate_model_cost,
        "optimize_for_model": moh.handle_optimize_for_model,
        "assess_cache_compatibility": moh.handle_assess_cache_compatibility,
        "capture_cache_telemetry": moh.handle_capture_cache_telemetry,
        "diagnose_cache_miss": moh.handle_diagnose_cache_miss,
        # ASG-SI (4 tools) - Experimental self-improvement framework
        "verify_compression": exp.handle_verify_compression,
        "calculate_reward": exp.handle_calculate_reward,
        "get_evidence_stats": exp.handle_get_evidence_stats,
        "generate_synthetic_tests": exp.handle_generate_synthetic_tests,
        # New tools: diff re-ingestion, cross-doc dedup, presets
        "diff_reingest": ch.handle_diff_reingest,
        "find_duplicates": ch.handle_find_duplicates,
        "get_compression_presets": ch.handle_get_presets,
        "create_prompt_template": ph.handle_create_prompt_template,
        "update_prompt_template": ph.handle_update_prompt_template,
        "list_prompt_templates": ph.handle_list_prompt_templates,
        "get_prompt_template": ph.handle_get_prompt_template,
        "deploy_prompt_version": ph.handle_deploy_prompt_version,
        "compare_prompt_versions": ph.handle_compare_prompt_versions,
        "render_prompt_template": ph.handle_render_prompt_template,
        "list_prefix_collisions": ph.handle_list_prefix_collisions,
        "audit_prompt_cacheability": ph.handle_audit_prompt_cacheability,
        "add_memory": mh.handle_add_memory,
        "search_memory": mh.handle_search_memory,
        "list_memories": mh.handle_list_memories,
        "delete_memory": mh.handle_delete_memory,
        "summarize_user_memory": mh.handle_summarize_user_memory,
        "get_user_profile": mh.handle_get_user_profile,
        # Knowledge Management (Phase 1-4)
        "ingest_transcript": mh.handle_ingest_transcript,
        "compile_knowledge": mh.handle_compile_knowledge,
        "get_knowledge_index": mh.handle_get_knowledge_index,
        "lint_knowledge": mh.handle_lint_knowledge,
        "search_memory_index": mh.handle_search_memory_index,
        "create_dataset": eh.handle_create_dataset,
        "list_datasets": eh.handle_list_datasets,
        "run_experiment": eh.handle_run_experiment,
        "get_experiment_run": eh.handle_get_experiment_run,
        "compare_experiment_runs": eh.handle_compare_experiment_runs,
        "list_connector_types": coh.handle_list_connector_types,
        "create_connector_feed": coh.handle_create_connector_feed,
        "list_connector_feeds": coh.handle_list_connector_feeds,
        "get_connector_feed": coh.handle_get_connector_feed,
        "sync_connector_feed": coh.handle_sync_connector_feed,
        "get_context_block": th.handle_get_context_block,
        "search_timeline": th.handle_search_timeline,
        "list_fact_history": th.handle_list_fact_history,
        "invalidate_fact": th.handle_invalidate_fact,
        "check_context_budget": ch.handle_check_context_budget,
        # Phase 5: Research-based features (2025 papers)
        "prune_by_relevance": ch.handle_prune_by_relevance,
        "get_multi_level_skeleton": ch.handle_multi_level_skeleton,
        "evict_stale": ch.handle_evict_stale,
        "advise_context": ch.handle_advise_context,
        "get_compression_insights": ch.handle_get_compression_insights,
        "generate_rewrite_prompt": ch.handle_generate_rewrite_prompt,
        # Token Optimization (v0.11.0)
        "estimate_tokens": toh.handle_estimate_tokens,
        "configure_for_client": toh.handle_configure_for_client,
        "set_compression_profile": toh.handle_set_compression_profile,
        "get_compression_profile": toh.handle_get_compression_profile,
        # arXiv paper techniques (v0.12.0)
        "compress_meta_tokens": toh.handle_compress_meta_tokens,
        "recommend_compression": toh.handle_recommend_compression,
        # Session Journal (v0.13.0)
        "recover_session": toh.handle_recover_session,
        # Tensor-Grep Integration (v0.13.0)
        "compress_codebase": ch.handle_compress_codebase,
        "search_code": ch.handle_search_code,
        # CLI Output Optimizer
        "filter_cli_output": toh.handle_filter_cli_output,
        # Savings Tracker (v0.14.0)
        "get_savings_report": toh.handle_get_savings_report,
        "get_savings_inline": toh.handle_get_savings_inline,
        # Cache Strategy Advisor
        "advise_cache_strategy": toh.handle_advise_cache_strategy,
        # Structural Summary + Dead Code Detector (v0.15.0)
        "generate_structural_summary": toh.handle_generate_structural_summary,
        "detect_dead_code": toh.handle_detect_dead_code,
        # Tee/Recovery (v0.16.0)
        "get_original_output": toh.handle_get_original_output,
        "list_tee_entries": toh.handle_list_tee_entries,
        "tee_store_stats": toh.handle_tee_store_stats,
        # Discover Savings (v0.16.0)
        "discover_savings": toh.handle_discover_savings,
        # ROI Calculator (v0.16.0)
        "calculate_roi": toh.handle_calculate_roi,
        # Budget Monitor (v0.16.0)
        "check_budget": toh.handle_check_budget,
        # Team Export (v0.16.0)
        "export_team_data": toh.handle_export_team_data,
    }

    enabled_tools = _enabled_tool_names(set(router.keys()), tool_profile)

    # Lookup handler
    if name not in router:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Unknown tool: '{name}'\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use list_tools() to see all available tools with descriptions"
        )
    if name not in enabled_tools:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Tool '{name}' is not enabled in profile '{_normalize_tool_profile(tool_profile)}'.\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use profile 'full' to enable advanced tools"
        )

    # Pre-execute input validation
    from ..validation_hooks import validate_tool_input

    validation_errors = validate_tool_input(name, args)
    if validation_errors:
        import json

        return json.dumps(
            {
                "error": "Input validation failed",
                "validation_errors": validation_errors,
            },
            indent=2,
        )

    # Route to handler (async)
    handler = router[name]
    handler_module = getattr(handler, "__module__", "unknown")
    handler_name = getattr(handler, "__name__", "unknown")
    logger.info(
        "tool_routing", tool_name=name, handler_module=handler_module, handler_function=handler_name
    )

    return await handler(context, args)
