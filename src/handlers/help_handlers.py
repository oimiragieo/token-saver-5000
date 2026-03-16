"""
Help Handler Module

This module provides handler functions for tool documentation and help:
- handle_tool_help: Provides detailed help, examples, and tips for any MCP tool

New in v0.9.0: Part of the Programmer UX Improvement Plan.
"""

import json
import logging
from typing import Any, Dict

from ..types import HandlerContext
from .compression_handlers import (
    get_ingest_context_output_fields,
    get_read_skeleton_output_fields,
    get_recommend_fidelity_output_fields,
    get_search_semantic_output_fields,
)
from .resource_handlers import get_check_environment_output_fields

logger = logging.getLogger("semantic-modulator")

# Tool documentation registry with examples and tips
TOOL_HELP_REGISTRY: Dict[str, Dict[str, Any]] = {
    # === Document Compression Tools ===
    "ingest_context": {
        "category": "Document Compression",
        "description": "Ingest and compress a document into a semantic graph for efficient retrieval.",
        "parameters": {
            "text": "The raw document text to ingest (required)",
            "file_id": "Unique identifier for this document (required)",
            "file_path": "Optional path to source file for sync tracking",
            "metadata": "Optional metadata dict (author, date, source, tags)",
        },
        "output_fields": get_ingest_context_output_fields(),
        "examples": [
            {
                "description": "Basic text ingestion",
                "args": {"text": "Your document text...", "file_id": "doc_1"},
            },
            {
                "description": "With file path for sync tracking",
                "args": {
                    "text": "Code content...",
                    "file_id": "main.py",
                    "file_path": "/path/to/main.py",
                },
            },
        ],
        "tips": [
            "Use meaningful file_ids (e.g., 'auth_service.py' not 'doc1')",
            "Provide file_path to enable staleness detection",
            "Metadata is preserved and returned in read_skeleton",
        ],
        "related_tools": ["read_skeleton", "search_semantic", "modulate_region"],
    },
    "read_skeleton": {
        "category": "Document Compression",
        "description": "Get a compressed skeleton view of an ingested document.",
        "parameters": {
            "file_id": "ID of the document to read (required)",
            "selection_mode": "Optional: baseline, query_guided, or evidence_aware (default: baseline)",
            "query": "Optional query text. Required for query_guided and evidence_aware modes",
            "top_k": "Optional evidence node count for evidence_aware mode (default: 5)",
            "min_similarity": "Optional sufficiency threshold for evidence_aware mode (default: 0.35)",
        },
        "output_fields": get_read_skeleton_output_fields(),
        "examples": [
            {"description": "Read document skeleton", "args": {"file_id": "my_doc"}},
            {
                "description": "Query-guided skeleton",
                "args": {
                    "file_id": "my_doc",
                    "selection_mode": "query_guided",
                    "query": "error handling strategy",
                },
            },
            {
                "description": "Evidence-aware skeleton",
                "args": {
                    "file_id": "my_doc",
                    "selection_mode": "evidence_aware",
                    "query": "authentication flow",
                    "top_k": 5,
                    "min_similarity": 0.4,
                },
            },
        ],
        "tips": [
            "Skeleton shows high-importance nodes only (~20% of content)",
            "Node IDs in skeleton can be used with modulate_region",
            "Use selection_mode=query_guided to bias anchors toward your task",
            "Use selection_mode=evidence_aware to detect insufficient evidence",
            "Returns staleness warning if source file changed",
        ],
        "related_tools": ["ingest_context", "modulate_region", "check_file_sync"],
    },
    "modulate_region": {
        "category": "Document Compression",
        "description": "Retrieve content at specified fidelity level for specific nodes.",
        "parameters": {
            "node_ids": "List of node IDs to retrieve (required)",
            "fidelity_level": "Detail level: ABSTRACT, OUTLINE, STRUCTURE, DETAILED, RAW",
        },
        "examples": [
            {
                "description": "Get detailed content",
                "args": {"node_ids": ["doc_n0", "doc_n1"], "fidelity_level": "DETAILED"},
            },
            {
                "description": "Quick summary",
                "args": {"node_ids": ["doc_n0"], "fidelity_level": "ABSTRACT"},
            },
        ],
        "tips": [
            "Use recommend_fidelity to choose the best fidelity level",
            "Lower fidelity = fewer tokens, higher fidelity = more detail",
            "Combine with search_semantic to find relevant nodes first",
        ],
        "related_tools": ["read_skeleton", "search_semantic", "recommend_fidelity"],
    },
    "search_semantic": {
        "category": "Document Compression",
        "description": "Find relevant content using semantic similarity search.",
        "parameters": {
            "query": "Search query (required)",
            "file_id": "Optional: limit search to specific document",
            "top_k": "Number of results to return (default: 5)",
            "evidence_aware": "Optional: enable insufficiency detection with expanded retrieval (default: false)",
            "min_similarity": "Optional sufficiency threshold for evidence_aware mode (default: 0.35)",
        },
        "output_fields": get_search_semantic_output_fields(),
        "examples": [
            {"description": "Search all docs", "args": {"query": "authentication logic"}},
            {
                "description": "Search specific doc",
                "args": {"query": "error handling", "file_id": "auth.py", "top_k": 10},
            },
            {
                "description": "Evidence-aware search",
                "args": {
                    "query": "token refresh race condition",
                    "file_id": "auth.py",
                    "evidence_aware": True,
                    "min_similarity": 0.4,
                },
            },
        ],
        "tips": [
            "Returns both similarity (query match) and importance (PageRank) scores",
            "Use file_id to focus search on specific documents",
            "Set evidence_aware=true when correctness matters more than speed",
            "Results are ranked by semantic similarity, not keyword match",
        ],
        "related_tools": ["modulate_region", "read_skeleton", "check_blind_spots"],
    },
    "ingest_directory": {
        "category": "Directory Ingestion",
        "description": "Bulk ingest code files from a directory using glob patterns.",
        "parameters": {
            "directory": "Directory path to scan (required)",
            "patterns": "Glob patterns for files (default: ['*.py', '*.js', '*.ts'])",
            "exclude_patterns": "Patterns to exclude (default: node_modules, __pycache__)",
            "max_files": "Maximum files to ingest (default: 50, max: 100)",
            "max_concurrent": "Concurrent ingestions (default: 4, max: 8)",
        },
        "examples": [
            {
                "description": "Ingest Python files",
                "args": {"directory": "./src", "patterns": ["*.py"]},
            },
            {
                "description": "Ingest with custom exclusions",
                "args": {
                    "directory": "./project",
                    "patterns": ["*.py", "*.js"],
                    "exclude_patterns": ["**/test/**", "**/vendor/**"],
                    "max_files": 30,
                },
            },
        ],
        "tips": [
            "Uses PathValidator for security (prevents path traversal)",
            "Files are processed in parallel for 4x throughput",
            "file_id is derived from relative path (e.g., 'src/main.py')",
        ],
        "related_tools": ["ingest_context", "batch_ingest_documents", "list_documents"],
    },
    # === AFM Dialogue Tools ===
    "afm_add_message": {
        "category": "AFM Dialogue",
        "description": "Add a message to the dialogue history with importance classification.",
        "parameters": {
            "role": "Message role: 'user' or 'assistant' (required)",
            "content": "Message content (required)",
            "importance_override": "Optional: force HIGH, MEDIUM, or LOW importance",
        },
        "examples": [
            {
                "description": "Add user message",
                "args": {"role": "user", "content": "I have a peanut allergy"},
            },
            {
                "description": "Add assistant response",
                "args": {"role": "assistant", "content": "Noted, I'll remember that."},
            },
        ],
        "tips": [
            "Safety messages (allergies, constraints) are auto-classified as HIGH",
            "Messages decay in importance over time (half-life: 12 turns)",
            "Use afm_build_context to get compressed context for LLM",
        ],
        "related_tools": ["afm_build_context", "afm_get_stats", "afm_clear_history"],
    },
    "afm_build_context": {
        "category": "AFM Dialogue",
        "description": "Build compressed dialogue context within a token budget.",
        "parameters": {
            "query": "Current user query (required)",
            "budget_tokens": "Maximum tokens for context (default: 1000)",
        },
        "examples": [
            {
                "description": "Build context",
                "args": {"query": "What can I eat?", "budget_tokens": 500},
            },
        ],
        "tips": [
            "Safety messages are always preserved regardless of budget",
            "Returns both context text and statistics",
            "Use token savings stats to optimize budget allocation",
        ],
        "related_tools": ["afm_add_message", "afm_get_stats"],
    },
    # === ACE Framework Tools ===
    "ace_generate": {
        "category": "ACE Framework",
        "description": "Generate context bullets from task outcome for playbook evolution.",
        "parameters": {
            "task": "Task description (required)",
            "outcome": "Task outcome/result (required)",
            "success": "Whether task succeeded (required)",
            "context_id": "Optional context ID for organizing bullets",
        },
        "examples": [
            {
                "description": "Generate from successful task",
                "args": {"task": "Fix auth bug", "outcome": "Added null check", "success": True},
            },
        ],
        "tips": [
            "Generated bullets are deduplicated at 0.85 similarity threshold",
            "Use ace_execute_cycle for automated Generate->Reflect->Curate",
            "Bullets are ranked by novelty and usefulness",
        ],
        "related_tools": ["ace_reflect", "ace_curate", "ace_execute_cycle"],
    },
    # === File Sync Tools ===
    "check_file_sync": {
        "category": "File Sync",
        "description": "Check if a tracked file has changed since ingestion.",
        "parameters": {
            "file_id": "ID of the document to check (required)",
        },
        "examples": [
            {"description": "Check sync status", "args": {"file_id": "main.py"}},
        ],
        "tips": [
            "Uses mtime + MD5 checksum for change detection",
            "Returns 'stale' if file changed, 'synced' if unchanged",
            "Use refresh_document to re-ingest stale files",
        ],
        "related_tools": ["refresh_document", "diff_cached_file", "get_version_history"],
    },
    "refresh_document": {
        "category": "File Sync",
        "description": "Re-ingest a document from its source file path.",
        "parameters": {
            "file_id": "ID of the document to refresh (required)",
        },
        "examples": [
            {"description": "Refresh stale document", "args": {"file_id": "config.py"}},
        ],
        "tips": [
            "Requires file_path to have been provided during initial ingest",
            "Creates version history entry for the change",
            "Use after check_file_sync reports 'stale' status",
        ],
        "related_tools": ["check_file_sync", "get_version_history", "ingest_context"],
    },
    # === Detection Tools ===
    "check_blind_spots": {
        "category": "Detection",
        "description": "Detect relevant content that may have been missed in retrieval.",
        "parameters": {
            "file_id": "Document to check (required)",
            "retrieved_nodes": "List of node IDs already retrieved (required)",
            "query": "Optional: user's query for context",
        },
        "examples": [
            {
                "description": "Check for blind spots",
                "args": {
                    "file_id": "manual",
                    "retrieved_nodes": ["manual_n0", "manual_n1"],
                    "query": "How to configure auth?",
                },
            },
        ],
        "tips": [
            "Returns urgency score (0-1) for each potential blind spot",
            "High urgency (>0.5) suggests important content was missed",
            "Use after search_semantic to catch gaps in retrieval",
        ],
        "related_tools": ["search_semantic", "detect_hallucination"],
    },
    # === Resource Management ===
    "check_resource_health": {
        "category": "Resource Management",
        "description": "Check storage, memory, and document count metrics.",
        "parameters": {},
        "examples": [
            {"description": "Check health", "args": {}},
        ],
        "tips": [
            "Shows warnings when approaching resource limits",
            "Includes recommendations for optimization",
            "Storage limit: 1GB, Document limit: 1000",
        ],
        "related_tools": ["check_environment", "list_documents", "delete_document"],
    },
    "check_environment": {
        "category": "Resource Management",
        "description": (
            "Check environment health: models, memory, cache, disk space, and MCP tool profile."
        ),
        "parameters": {},
        "output_fields": get_check_environment_output_fields(),
        "examples": [
            {"description": "Check environment", "args": {}},
        ],
        "tips": [
            "Shows which embedding models are loaded",
            "Reports cache hit ratio for performance tuning",
            "Lists any stale documents that need refresh",
            "Includes tool_profile diagnostics (active profile and enabled_tools list)",
        ],
        "related_tools": ["check_resource_health", "check_file_sync"],
    },
    # === Utility Tools ===
    "recommend_fidelity": {
        "category": "Fidelity Advisor",
        "description": "Get recommendation for optimal fidelity level based on use case.",
        "parameters": {
            "use_case": "What you want to do (e.g., 'quick_summary', 'detailed_analysis')",
            "num_nodes": "Number of nodes you plan to retrieve",
            "token_budget": "Optional maximum tokens available",
            "query_complexity": "Optional: 'simple', 'medium', or 'complex'",
        },
        "output_fields": get_recommend_fidelity_output_fields(),
        "examples": [
            {
                "description": "Get recommendation for summary",
                "args": {"use_case": "quick_summary", "num_nodes": 3},
            },
            {
                "description": "With token budget",
                "args": {"use_case": "question_answering", "num_nodes": 5, "token_budget": 1000},
            },
        ],
        "tips": [
            "Use this BEFORE modulate_region to make informed choices",
            "Returns token estimate for each fidelity level",
            "Considers context to suggest alternatives",
        ],
        "related_tools": ["modulate_region", "search_semantic"],
    },
}


async def handle_tool_help(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle tool_help MCP tool (v0.9.0).

    Provides detailed help, examples, and tips for any Semantic Modulator tool.

    Args:
        context: Server context (unused for help)
        args: Tool arguments:
            - tool_name: Name of tool to get help for (required)
            - verbose: Include full examples (default: False)

    Returns:
        JSON string with structured help information
    """
    tool_name = args.get("tool_name", "")
    verbose = args.get("verbose", False)

    if not tool_name:
        # Return list of all tools with categories
        categories: Dict[str, list] = {}
        for name, info in TOOL_HELP_REGISTRY.items():
            cat = info.get("category", "Other")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(
                {
                    "name": name,
                    "description": info.get("description", "")[:80] + "...",
                }
            )

        return json.dumps(
            {
                "status": "tool_list",
                "message": "Specify tool_name to get detailed help",
                "available_tools": categories,
                "total_tools": len(TOOL_HELP_REGISTRY),
                "recommended_workflow": {
                    "description": "Optimal tool sequence for maximum token savings",
                    "steps": [
                        {"step": 1, "tool": "should_compress", "purpose": "Check if compression is worthwhile for your content size"},
                        {"step": 2, "tool": "ingest_context", "purpose": "Ingest document into semantic graph (use chunking_strategy='semantic' for best results)"},
                        {"step": 3, "tool": "read_skeleton", "purpose": "Get compressed view (80-95% token reduction). Use anchored_keywords to preserve critical terms"},
                        {"step": 4, "tool": "search_semantic", "purpose": "Find specific information within compressed docs"},
                        {"step": 5, "tool": "modulate_region", "purpose": "Zoom into specific nodes at chosen fidelity level"},
                        {"step": 6, "tool": "advise_context", "purpose": "Get model-specific optimization recommendations"},
                    ],
                },
                "tool_profiles": {
                    "core_stable": {
                        "tools": ["ingest_context", "read_skeleton", "modulate_region", "search_semantic", "get_stats", "list_documents", "delete_document"],
                        "description": "7 essential tools (~3K tokens). Best for prompt-cache-friendly setups.",
                    },
                    "full": {
                        "tools": "All 58 tools",
                        "description": "Complete toolkit (~16K tokens). Use when context budget allows.",
                    },
                },
            },
            indent=2,
        )

    # Look up specific tool
    if tool_name not in TOOL_HELP_REGISTRY:
        # Suggest similar tools
        suggestions = [
            name
            for name in TOOL_HELP_REGISTRY.keys()
            if tool_name.lower() in name.lower() or name.lower() in tool_name.lower()
        ]

        return json.dumps(
            {
                "status": "not_found",
                "tool_name": tool_name,
                "message": f"Tool '{tool_name}' not found in help registry.",
                "suggestions": suggestions[:5] if suggestions else [],
                "tip": "Use tool_help without tool_name to see all available tools.",
            },
            indent=2,
        )

    # Return help for specific tool
    info = TOOL_HELP_REGISTRY[tool_name]
    result = {
        "tool": tool_name,
        "category": info.get("category", "Other"),
        "description": info.get("description", ""),
        "parameters": info.get("parameters", {}),
        "tips": info.get("tips", []),
        "related_tools": info.get("related_tools", []),
    }
    if "output_fields" in info:
        result["output_fields"] = info.get("output_fields", [])

    if verbose:
        result["examples"] = info.get("examples", [])
    else:
        # Just show first example
        examples = info.get("examples", [])
        if examples:
            result["example"] = examples[0]
            if len(examples) > 1:
                result["more_examples"] = f"Use verbose=true to see {len(examples)} examples"

    return json.dumps(result, indent=2)
