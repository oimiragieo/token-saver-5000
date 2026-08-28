"""
Help Handler Module

This module provides handler functions for tool documentation and help:
- handle_tool_help: Provides detailed help, examples, and tips for any MCP tool

New in v0.9.0: Part of the Programmer UX Improvement Plan.
"""

import json
import logging
from functools import lru_cache
from typing import Any, Dict

from ..types import HandlerContext
from .help_tool_registry import TOOL_HELP_REGISTRY

logger = logging.getLogger("semantic-modulator")

def _infer_tool_category(tool_name: str) -> str:
    for prefix, category in _AUTO_CATEGORY_BY_PREFIX.items():
        if tool_name.startswith(prefix):
            return category
    return _AUTO_CATEGORY_BY_TOOL.get(tool_name, "General")


def _placeholder_value(param_name: str, schema: Dict[str, Any]) -> Any:
    explicit_values = {
        "file_id": "doc_1",
        "doc_id": "doc_1",
        "bundle_id": "bundle_1",
        "memory_id": "mem_1",
        "run_id": "run_1",
        "dataset_name": "release-gate",
        "name": "example-name",
        "query": "authentication flow",
        "text": "Representative context text",
        "directory": ".",
        "file_path": "src\\app.py",
        "content_type": "code",
        "model": "claude-sonnet-4.6",
        "workspace_id": "acme",
        "user_id": "alice",
        "agent_id": "assistant",
        "session_id": "session-1",
        "version": 1,
        "top_k": 5,
        "limit": 10,
        "keep_ratio": 0.5,
        "token_budget": 1200,
        "current_tokens": 4000,
        "context_limit": 8000,
        "num_nodes": 5,
        "original_tokens": 100000,
        "compressed_tokens": 20000,
        "max_concurrent": 4,
        "verbose": True,
    }
    if param_name in explicit_values:
        return explicit_values[param_name]

    schema_type = schema.get("type")
    if schema_type == "integer":
        return 1
    if schema_type == "number":
        return 0.5
    if schema_type == "boolean":
        return True
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}
    return "example"


def _schema_parameters(input_schema: Dict[str, Any]) -> Dict[str, str]:
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))
    parameters: Dict[str, str] = {}
    for name, schema in properties.items():
        description = schema.get("description") or f"{name} parameter"
        if name in required and "(required)" not in description.lower():
            description = f"{description} (required)"
        parameters[name] = description
    return parameters


def _schema_example_args(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    if not required:
        return {}
    return {name: _placeholder_value(name, properties.get(name, {})) for name in required}


def _auto_related_tools(tool_name: str, category: str) -> list[str]:
    related = _AUTO_RELATED_TOOLS.get(tool_name)
    if related:
        return related

    registry = TOOL_HELP_REGISTRY
    same_category = [
        name
        for name, info in registry.items()
        if info.get("category") == category and name != tool_name
    ]
    return same_category[:3]


@lru_cache(maxsize=1)
def get_tool_help_registry() -> Dict[str, Dict[str, Any]]:
    """Return the merged help registry for all registered MCP tools."""
    registry = {name: dict(info) for name, info in TOOL_HELP_REGISTRY.items()}

    from .mcp_core import setup_mcp_tools

    for tool in setup_mcp_tools():
        if tool.name in registry:
            continue

        category = _infer_tool_category(tool.name)
        parameters = _schema_parameters(tool.inputSchema)
        example_args = _schema_example_args(tool.inputSchema)

        registry[tool.name] = {
            "category": category,
            "description": tool.description,
            "parameters": parameters,
            "examples": [
                {
                    "description": f"Basic {tool.name} invocation",
                    "args": example_args,
                }
            ],
            "tips": [
                "Use tool_help with verbose=true to inspect arguments before first use.",
                "Pair this tool with the related tools below when building a longer workflow.",
            ],
            "related_tools": _auto_related_tools(tool.name, category),
        }

    return registry


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
    registry = get_tool_help_registry()

    if not tool_name:
        # Return list of all tools with categories
        categories: Dict[str, list] = {}
        for name, info in registry.items():
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
                "total_tools": len(registry),
                "recommended_workflow": {
                    "description": "Optimal tool sequence for maximum token savings",
                    "steps": [
                        {
                            "step": 1,
                            "tool": "should_compress",
                            "purpose": "Check if compression is worthwhile for your content size",
                        },
                        {
                            "step": 2,
                            "tool": "ingest_context",
                            "purpose": "Ingest document into semantic graph (use chunking_strategy='semantic' for best results)",
                        },
                        {
                            "step": 3,
                            "tool": "read_skeleton",
                            "purpose": "Get compressed view (80-95% token reduction). Use anchored_keywords to preserve critical terms",
                        },
                        {
                            "step": 4,
                            "tool": "search_semantic",
                            "purpose": "Find specific information within compressed docs",
                        },
                        {
                            "step": 5,
                            "tool": "modulate_region",
                            "purpose": "Zoom into specific nodes at chosen fidelity level",
                        },
                        {
                            "step": 6,
                            "tool": "advise_context",
                            "purpose": "Get model-specific optimization recommendations",
                        },
                    ],
                },
                "tool_profiles": {
                    "core_stable": {
                        "tools": [
                            "ingest_context",
                            "read_skeleton",
                            "modulate_region",
                            "search_semantic",
                            "get_stats",
                            "list_documents",
                            "delete_document",
                        ],
                        "description": "7 essential tools (~3K tokens). Best for prompt-cache-friendly setups.",
                    },
                    "full": {
                        "tools": f"All {len(registry)} tools",
                        "description": "Complete toolkit (~16K tokens). Use when context budget allows.",
                    },
                },
            },
            indent=2,
        )

    # Look up specific tool
    if tool_name not in registry:
        # Suggest similar tools
        suggestions = [
            name
            for name in registry.keys()
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
    info = registry[tool_name]
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
