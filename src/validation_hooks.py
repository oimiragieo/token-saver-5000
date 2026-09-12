"""
Input validation hooks for MCP tool calls.

Pre-execute validation that catches common errors before
they reach handlers, providing clearer error messages.
"""

from typing import Any, Dict, List

from .safe_identifier import validate_safe_storage_id

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
    file_url = args.get("file_url")
    # 'text' is only required when 'file_url' is absent. handle_ingest fetches
    # remote content for file_url and enforces text/file_url mutual exclusivity
    # itself, so requiring non-empty text here would 422 every legitimate
    # file_url-only ingest before the URL is ever fetched (v1.43 dogfood bug).
    if not file_url and (not text or len(text.strip()) == 0):
        errors.append("text cannot be empty or whitespace-only")
    file_id = args.get("file_id", "")
    if file_id:
        error = validate_safe_storage_id(file_id, "file_id", allow_subdirs=True)
        if error:
            errors.append(error)
    return errors


@_register("modulate_region")
def _validate_modulate(args: Dict[str, Any]) -> List[str]:
    errors = []
    # Honor the documented singular `node_id` convenience alias (schema says it
    # wraps to [node_id] and satisfies the requirement on its own). Pre-fix this
    # hook read only `node_ids`, so a caller passing node_id="X" alone was
    # rejected with "node_ids must not be empty" even though the handler accepts
    # it — a broken shortcut on the core zoom-in tool (dogfood 2026-07-04).
    node_ids = args.get("node_ids") or ([args["node_id"]] if args.get("node_id") else [])
    if not node_ids:
        errors.append("node_ids (or node_id) must not be empty")
    return errors


@_register("delete_document")
def _validate_delete(args: Dict[str, Any]) -> List[str]:
    errors = []
    file_id = args.get("file_id", "")
    if not file_id or len(file_id.strip()) == 0:
        errors.append("file_id is required for deletion")
    else:
        error = validate_safe_storage_id(file_id, "file_id", allow_subdirs=True)
        if error:
            errors.append(error)
    return errors


@_register("batch_ingest_documents")
def _validate_batch_ingest(args: Dict[str, Any]) -> List[str]:
    errors = []
    documents = args.get("documents", [])
    if not documents:
        errors.append("documents list cannot be empty")
    if len(documents) > 100:
        errors.append("batch size must not exceed 100 documents")
    for doc in documents:
        if not isinstance(doc, dict):
            continue
        doc_file_id = doc.get("file_id", "")
        if doc_file_id:
            error = validate_safe_storage_id(doc_file_id, "file_id", allow_subdirs=True)
            if error:
                errors.append(f"documents[].{error}")
    return errors
