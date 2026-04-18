"""
Input validation hooks for MCP tool calls.

Pre-execute validation that catches common errors before
they reach handlers, providing clearer error messages.
"""

import re
from typing import Any, Dict, List

# Validation rules per tool
_VALIDATORS = {}


def _register(tool_name: str):
    """Decorator to register a validator for a tool."""

    def decorator(func):
        _VALIDATORS[tool_name] = func
        return func

    return decorator


def validate_tool_input(tool_name: str, args: Dict[str, Any]) -> List[str]:
    """Validate tool input before execution.

    Args:
        tool_name: Name of the MCP tool
        args: Tool arguments dict

    Returns:
        List of error messages (empty if valid)
    """
    validator = _VALIDATORS.get(tool_name)
    if validator is None:
        return []
    return validator(args)


@_register("search_semantic")
def _validate_search(args: Dict[str, Any]) -> List[str]:
    errors = []
    query = args.get("query", "")
    if len(query.strip()) < 3:
        errors.append("query must be at least 3 characters for meaningful search")
    top_k = args.get("top_k")
    if top_k is not None and (top_k < 1 or top_k > 100):
        errors.append("top_k must be between 1 and 100")
    return errors


@_register("ingest_context")
def _validate_ingest(args: Dict[str, Any]) -> List[str]:
    errors = []
    text = args.get("text", "")
    if not text or len(text.strip()) == 0:
        errors.append("text cannot be empty or whitespace-only")
    file_id = args.get("file_id", "")
    if file_id:
        if ".." in file_id:
            errors.append("file_id must not contain '..' (parent-directory reference)")
        elif not re.match(r"^[a-zA-Z0-9_\-./]+$", file_id):
            errors.append(
                "file_id must contain only alphanumeric characters, underscores, "
                "hyphens, dots, or forward slashes"
            )
    return errors


@_register("modulate_region")
def _validate_modulate(args: Dict[str, Any]) -> List[str]:
    errors = []
    node_ids = args.get("node_ids", [])
    if not node_ids:
        errors.append("node_ids must not be empty")
    return errors


@_register("delete_document")
def _validate_delete(args: Dict[str, Any]) -> List[str]:
    errors = []
    file_id = args.get("file_id", "")
    if not file_id or len(file_id.strip()) == 0:
        errors.append("file_id is required for deletion")
    return errors


@_register("batch_ingest")
def _validate_batch_ingest(args: Dict[str, Any]) -> List[str]:
    errors = []
    documents = args.get("documents", [])
    if not documents:
        errors.append("documents list cannot be empty")
    if len(documents) > 100:
        errors.append("batch size must not exceed 100 documents")
    return errors
