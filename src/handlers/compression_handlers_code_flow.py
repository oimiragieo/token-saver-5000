"""Compression handlers for codebase search, budget knapsack, and code-flow delta verification."""

from pathlib import Path as _Path
from typing import Any, Dict, List

from .compression_handlers_common import HandlerContext, json, logger


def _apply_codebase_budget(
    *,
    result: Dict[str, Any],
    directory: str,
    query: str | None,
    budget_tokens: int,
    path_validator: Any,
) -> None:
    """Apply knapsack token-budget compression across selected codebase files in-place."""
    from ..code_budget_knapsack import compress_code_with_knapsack
    from ..token_estimation import TokenEstimator

    estimator = TokenEstimator()
    dir_root = _Path(directory)
    selected_files = result.get("selected_files", [])
    readable_entries: List[Dict[str, Any]] = []

    per_file_budget = max(1, budget_tokens // max(1, len(selected_files)))
    for raw_file in selected_files:
        candidate = _Path(raw_file)
        if not candidate.is_absolute():
            candidate = dir_root / candidate
        try:
            validated_file = _Path(path_validator.validate(str(candidate)))
        except ValueError:
            continue
        if not validated_file.is_file():
            continue
        try:
            source_text = validated_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel_label = str(validated_file.relative_to(dir_root)).replace("\\", "/")
        except ValueError:
            rel_label = validated_file.name

        knapsack_res = compress_code_with_knapsack(
            source_text,
            budget_tokens=per_file_budget,
            query=query or "",
        )
        if knapsack_res.reconstructed_code.strip():
            readable_entries.append(
                {
                    "file": rel_label,
                    "code": knapsack_res.reconstructed_code,
                    "value": knapsack_res.total_value,
                    "tokens": knapsack_res.total_tokens,
                }
            )

    def _render_entries(items: List[Dict[str, Any]]) -> str:
        return "\n\n".join(f"# {item['file']}\n{item['code']}" for item in items)

    compressed_code = _render_entries(readable_entries)
    actual_tokens = estimator.count_tokens(compressed_code) if compressed_code else 0
    while readable_entries and actual_tokens > budget_tokens:
        if len(readable_entries) > 1:
            drop_idx = min(
                range(len(readable_entries)),
                key=lambda i: (
                    readable_entries[i]["value"],
                    -readable_entries[i]["tokens"],
                    -i,
                ),
            )
            readable_entries.pop(drop_idx)
            compressed_code = _render_entries(readable_entries)
            actual_tokens = estimator.count_tokens(compressed_code) if compressed_code else 0
        else:
            only_entry = readable_entries[0]
            trimmed = compress_code_with_knapsack(
                only_entry["code"],
                budget_tokens=budget_tokens,
                query=query or "",
            )
            compressed_code = trimmed.reconstructed_code
            actual_tokens = estimator.count_tokens(compressed_code) if compressed_code else 0
            if actual_tokens > budget_tokens:
                readable_entries.clear()
                compressed_code = ""
                actual_tokens = 0
            else:
                only_entry["code"] = compressed_code
                only_entry["tokens"] = actual_tokens
            break

    result["budget_tokens"] = budget_tokens
    result["compressed_tokens"] = actual_tokens
    result["compressed_code"] = compressed_code
    result["compressed_files"] = [
        {"file": item["file"], "tokens": item["tokens"]} for item in readable_entries
    ]
    result["selected_files"] = [item["file"] for item in readable_entries]


async def handle_compress_codebase(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle compress_codebase tool call.

    Uses tensor-grep AST analysis when available; falls back to a plain
    directory glob when tensor-grep is not installed.
    """
    import glob

    from ..tensor_grep_integration import code_search, get_repo_map, is_available

    directory = args.get("directory", ".")
    query = args.get("query")
    raw_max_files = args.get("max_files", 50)
    if (
        isinstance(raw_max_files, bool)
        or not isinstance(raw_max_files, int)
        or not (1 <= raw_max_files <= 100)
    ):
        return json.dumps(
            {
                "status": "error",
                "error": "max_files must be an integer between 1 and 100",
            }
        )
    max_files = raw_max_files
    budget_tokens = args.get("budget_tokens")
    if budget_tokens is not None:
        if (
            isinstance(budget_tokens, bool)
            or not isinstance(budget_tokens, int)
            or budget_tokens < 1
        ):
            return json.dumps(
                {
                    "status": "error",
                    "error": "budget_tokens must be a positive integer",
                }
            )

    path_validator = context.get("path_validator")
    if path_validator is None:
        logger.error("path_validator absent from context; refusing to run unconfined (CWE-22)")
        return json.dumps(
            {
                "status": "error",
                "error": "path_validator missing from server context",
                "message": "directory cannot be confined without a path validator; "
                "refusing rather than enumerating an unconfined path.",
            }
        )
    try:
        directory = path_validator.validate(directory)
    except ValueError as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Invalid directory path: {e}",
                "message": "directory must be within the current working "
                "directory or user home directory.",
            }
        )

    tg_available = is_available()
    result: Dict[str, Any] = {
        "status": "success",
        "tensor_grep_available": tg_available,
        "directory": directory,
    }

    if tg_available:
        repo_map = get_repo_map(directory)
        result["files_found"] = len(repo_map.files)
        result["symbols_found"] = len(repo_map.symbols)
        files = repo_map.files[:max_files]
        if query:
            search = code_search(query, directory)
            matched_files = list({m.get("file", "") for m in search.matches})
            result["query_matched_files"] = len(matched_files)
            files = matched_files[:max_files] if matched_files else files
        result["selected_files"] = files
    else:
        py_files = glob.glob(str(_Path(directory) / "**/*.py"), recursive=True)
        js_files = glob.glob(str(_Path(directory) / "**/*.js"), recursive=True)
        ts_files = glob.glob(str(_Path(directory) / "**/*.ts"), recursive=True)
        files = sorted(set(py_files + js_files + ts_files))[:max_files]
        result["files_found"] = len(files)
        result["selected_files"] = files

    if budget_tokens is not None:
        _apply_codebase_budget(
            result=result,
            directory=directory,
            query=query,
            budget_tokens=budget_tokens,
            path_validator=path_validator,
        )

    result["message"] = (
        f"Found {len(result.get('selected_files', []))} files. "
        "Use ingest_context on individual files to compress them."
    )
    return json.dumps(result, default=str)


async def handle_verify_code_flow_delta(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle verify_code_flow_delta tool call (FlowPlan CFG/DFG delta verifier)."""
    from ..code_flow_graph import verify_code_flow_delta
    from .mcp_core._constants import SCOPE_PROPERTIES

    if not isinstance(args, dict):
        return json.dumps({"status": "error", "error": "Arguments must be an object"})

    allowed_keys = {"before_code", "after_code", "expected_changes"} | set(SCOPE_PROPERTIES.keys())
    unknown_keys = sorted(set(args.keys()) - allowed_keys)
    if unknown_keys:
        return json.dumps(
            {
                "status": "error",
                "error": f"Unknown argument(s) for verify_code_flow_delta: {', '.join(unknown_keys)}",
            }
        )

    before_code = args.get("before_code")
    after_code = args.get("after_code")
    if not isinstance(before_code, str) or not isinstance(after_code, str):
        return json.dumps(
            {
                "status": "error",
                "error": "before_code and after_code must be strings",
            }
        )

    expected_changes = args.get("expected_changes")
    if expected_changes is not None and not isinstance(expected_changes, (list, dict)):
        return json.dumps(
            {
                "status": "error",
                "error": "expected_changes must be a list or object when provided",
            }
        )

    try:
        report = verify_code_flow_delta(
            before_code,
            after_code,
            expected_changes=expected_changes,
        )
    except ValueError as exc:
        return json.dumps({"status": "error", "error": str(exc)})

    return json.dumps({"status": "success", **report.to_dict()})
