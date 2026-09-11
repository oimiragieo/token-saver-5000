"""
Compression-related MCP tool handlers.

This module contains all handlers for document compression operations:
- Ingestion (ingest_context)
- Skeleton reading (read_skeleton)
- Region modulation (modulate_region)
- Semantic search (search_semantic)
- Statistics (get_stats)
- Document listing (list_documents)
- Document deletion (delete_document)
- JSCCM-inspired adaptive operations (adapt_to_context_window, multilevel_encode)
- Fidelity recommendation (recommend_fidelity) - NEW in v0.4.1

Version: 0.7.0 - Added rate limiting, text length validation
"""

import math
import asyncio
import json
import logging
from typing import Any, Dict, List
import hashlib
import inspect
import re

from ..types import HandlerContext  # TypedDict for handler context
from ..semantic_compressor import FidelityLevel, compute_adaptive_ratio
from ..fidelity_advisor import FidelityAdvisor, UseCase
from ..error_helpers import SmartError
from ..compression_advisor import CompressionAdvisor
from ..rate_limiter import RATE_LIMITERS
from ..error_types import RateLimitExceededError
from ..metrics import compute_cost_savings, get_metrics
from ..constants import MAX_TEXT_LENGTH_BYTES, F11_RANKER_PATH
from ..url_fetcher import URLFetchError, fetch_url
from ..identity_scope import (
    compose_scoped_file_id,
    display_file_id,
    parse_scoped_file_id,
    scope_matches,
)
from ..node_identity import collect_file_ids, extract_file_id_from_node
from ..temporal_graph import coerce_timestamp, format_timestamp
from ..compression_pipeline import run_read_skeleton_pipeline

logger = logging.getLogger("semantic-modulator")

_HEADING_RE = re.compile(r"^#{1,3} ", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(\d+\.|- )", re.MULTILINE)


#: Relative-error bands for grading a compression estimate against the actual
#: ratio. Chosen against the two live dogfood cases that exposed the old rule:
#: a 0.8% miss must stay "excellent", a 55% miss must not.
_ESTIMATE_EXCELLENT_MAX_RELATIVE_ERROR = 0.15
_ESTIMATE_GOOD_MAX_RELATIVE_ERROR = 0.40
#: Absorbs float representation error at the band edges so a value that IS the
#: boundary is not demoted by it. Far smaller than any meaningful difference in
#: estimate quality.
_ESTIMATE_BAND_EPSILON = 1e-9


def chunks_for_file(chunks: dict, scoped_file_id: str) -> list:
    """Select a document's chunks, sorted, WITHOUT capturing prefix siblings.

    `nid.startswith(scoped_file_id)` is not a membership test. `"doc-large_n0"`
    starts with `"doc"`, so the shorter document silently absorbed the longer
    one's chunks. Dogfooded live: a no-query `read_skeleton` on the short id
    came back compressed against the sibling's H1, dropping half its nodes
    (80 -> 40) and making the ratio WORSE (2.194 -> 1.706).

    `_node_belongs_to_file` already existed for exactly this collision -- its
    docstring names the same failure and lists the surfaces it broke. It was
    applied inside the compressor and never adopted by the handlers. Routing
    every handler site through this one function is what stops the class
    returning a third time; a sixteenth hand-rolled predicate is how it did.

    Sorted because order carries document structure, and the auto-mode heading
    heuristic depends on the first chunk actually being first.
    """
    from src.semantic_compressor import _node_belongs_to_file

    return sorted(
        (nid, node) for nid, node in chunks.items() if _node_belongs_to_file(nid, scoped_file_id)
    )


def resolve_anchored_node_ids(chunks: dict, scoped_file_id: str, keywords: list) -> set:
    """Boundary-safe anchored-keyword resolution (#420).

    Reuses `chunks_for_file` rather than a bare `node_id.startswith(scoped_file_id)`
    -- the same collision class `chunks_for_file` fixes: `"doc-large_n0"` matches
    a keyword search scoped to `"doc"` under a naive prefix test.
    """
    keywords_lower = [kw.lower() for kw in keywords if kw]
    if not keywords_lower:
        return set()
    return {
        node_id
        for node_id, node in chunks_for_file(chunks, scoped_file_id)
        if any(kw in node.text.lower() for kw in keywords_lower)
    }


def classify_estimate_accuracy(*, actual: float, estimated: float) -> str:
    """Grade an estimated compression ratio against the measured one.

    Uses RELATIVE error. The previous rule compared absolute ratio points
    (`abs(actual - estimated) < 2`), which does not mean the same thing at
    different scales: a gap of 1.5 is catastrophic near ratio 1.0 and
    negligible near ratio 16. It graded a live 2.24x under-prediction
    (estimated 1.25, actual 2.81) as "excellent" purely because the absolute
    gap was under 2.0, on the surface an agent reads to decide whether
    compressing is worth it.

    Symmetric: over- and under-predicting by the same proportion grade alike.

    Never raises. A grader that throws on the compression path would turn a
    cosmetic mislabel into a failed ingest, so degenerate input (zero or
    negative actual ratio, non-finite values) falls through to the most
    conservative grade rather than propagating.
    """
    try:
        if not math.isfinite(actual) or not math.isfinite(estimated) or actual <= 0:
            return "fair"
        relative_error = abs(actual - estimated) / actual
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        # OverflowError is not hypothetical: math.isfinite(10**400) raises
        # "int too large to convert to float" on a Python int. Verified.
        # Catching it here keeps a cosmetic grade from crashing the ingest path.
        return "fair"

    # Compare against a boundary nudged by one ulp-ish epsilon. An exact 40%
    # miss computes as 0.4000000000000001 (verified for actual=3.0,
    # estimated=4.2), so a bare `<=` demotes a value the caller would call
    # exactly "good" purely on floating-point dust.
    if relative_error <= _ESTIMATE_EXCELLENT_MAX_RELATIVE_ERROR + _ESTIMATE_BAND_EPSILON:
        return "excellent"
    if relative_error <= _ESTIMATE_GOOD_MAX_RELATIVE_ERROR + _ESTIMATE_BAND_EPSILON:
        return "good"
    return "fair"


def _is_structured_markdown(text: str) -> bool:
    """Return True when text looks like structured markdown with headings + lists.

    Heuristic: >= 3 ATX headings (H1-H3) AND >= 3 ordered/unordered list items.
    Designed to avoid misfiring on plain prose that has only one or two headings.
    """
    headings = len(_HEADING_RE.findall(text))
    list_items = len(_LIST_ITEM_RE.findall(text))
    return headings >= 3 and list_items >= 3


def _resolve_chunking_strategy(args: Dict[str, Any], text: str) -> tuple[str, str]:
    """Return (effective_chunking_strategy, chunking_strategy_used_label).

    When the caller passes ``chunking_strategy="auto"`` (default), we inspect the
    text and switch to ``"fixed"`` for structured markdown so that heading and
    list boundaries are respected.  Any explicit strategy is passed through
    unchanged with a label that records the caller-supplied value.
    """
    raw = args.get("chunking_strategy", "auto")
    if raw == "auto" and _is_structured_markdown(text):
        return "fixed", "auto-detected: fixed"
    return raw, raw


def _scope_kwargs(args: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "workspace_id": args.get("workspace_id"),
        "user_id": args.get("user_id"),
        "agent_id": args.get("agent_id"),
        "session_id": args.get("session_id"),
    }


def _scoped_file_id(file_id: str, args: Dict[str, Any]) -> str:
    return compose_scoped_file_id(file_id, **_scope_kwargs(args))


def _scope_filtered_results(
    search_results: List[tuple[str, float]], args: Dict[str, Any]
) -> List[tuple[str, float]]:
    scope_kwargs = _scope_kwargs(args)
    return [
        (node_id, similarity)
        for node_id, similarity in search_results
        if scope_matches(extract_file_id_from_node(node_id), **scope_kwargs)
    ]


def _scope_filtered_file_ids(file_ids: List[str], args: Dict[str, Any]) -> List[str]:
    scope_kwargs = _scope_kwargs(args)
    return [file_id for file_id in file_ids if scope_matches(file_id, **scope_kwargs)]


def _has_scope_args(args: Dict[str, Any]) -> bool:
    return any(_scope_kwargs(args).values())


def _scope_filtered_duplicates(
    duplicates: List[Dict[str, Any]], args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Filter find_duplicates() pairs to the caller's tenant scope.

    Keeps a pair only when BOTH nodes' file_ids match the caller's scope
    (mirrors the _scope_filtered_results idiom used by handle_search_semantic).
    The timeout sentinel pair (node_a == node_b == "__timeout__") carries no
    tenant identity and is always preserved so the warning still surfaces.
    """
    scope_kwargs = _scope_kwargs(args)
    filtered = []
    for pair in duplicates:
        node_a = pair.get("node_a")
        node_b = pair.get("node_b")
        if node_a == "__timeout__" or node_b == "__timeout__":
            filtered.append(pair)
            continue
        if scope_matches(extract_file_id_from_node(node_a), **scope_kwargs) and scope_matches(
            extract_file_id_from_node(node_b), **scope_kwargs
        ):
            filtered.append(pair)
    return filtered


def _compressor_temporal_graph(compressor: Any) -> Any:
    sentinel = object()
    static_attr = inspect.getattr_static(compressor, "_temporal_graph", sentinel)
    if static_attr is sentinel:
        return None
    return getattr(compressor, "_temporal_graph", None)


def _temporal_graph(context: HandlerContext) -> Any:
    return _compressor_temporal_graph(context["compressor"])


def _temporal_excluded_node_ids(
    context: HandlerContext, args: Dict[str, Any], scoped_file_id: str | None = None
) -> set[str]:
    temporal_graph = _temporal_graph(context)
    if temporal_graph is None or args.get("include_invalidated", False):
        return set()

    as_of = args.get("as_of")
    if scoped_file_id is not None:
        return temporal_graph.get_invalidated_fact_ids(scoped_file_id, as_of=as_of)

    excluded: set[str] = set()
    for file_id in _scope_filtered_file_ids(list(context["compressor"].graphs.keys()), args):
        excluded.update(temporal_graph.get_invalidated_fact_ids(file_id, as_of=as_of))
    return excluded


def _temporal_filter_search_results(
    context: HandlerContext, search_results: List[tuple[str, float]], args: Dict[str, Any]
) -> List[tuple[str, float]]:
    temporal_graph = _temporal_graph(context)
    if temporal_graph is None or args.get("include_invalidated", False):
        return search_results

    as_of = args.get("as_of")
    return [
        (node_id, similarity)
        for node_id, similarity in search_results
        if temporal_graph.is_fact_active(node_id, as_of=as_of)
    ]


def _scoped_global_stats(context: HandlerContext, args: Dict[str, Any]) -> Dict[str, Any]:
    scoped_file_ids = _scope_filtered_file_ids(list(context["compressor"].graphs.keys()), args)
    if not scoped_file_ids:
        return {"total_files": 0, "total_documents": 0, "total_nodes": 0, "files": []}

    file_stats = [context["compressor"].get_stats(file_id) for file_id in scoped_file_ids]
    return {
        "total_files": len(scoped_file_ids),
        "total_documents": len(scoped_file_ids),
        "total_nodes": sum(stats["total_nodes"] for stats in file_stats),
        "files": [display_file_id(file_id) for file_id in scoped_file_ids],
    }


def _scope_label(scoped_or_raw_file_id: str) -> str | None:
    parsed = parse_scoped_file_id(scoped_or_raw_file_id)
    labels = []
    for field, display_name in (
        ("workspace_id", "workspace"),
        ("user_id", "user"),
        ("agent_id", "agent"),
        ("session_id", "session"),
    ):
        value = parsed.get(field)
        if value is not None:
            labels.append(f"{display_name}={value}")
    return ", ".join(labels) if labels else None


async def _resolve_awaitable(result: Any) -> Any:
    """Await awaitables while leaving synchronous results untouched."""
    if inspect.isawaitable(result):
        return await result
    return result


async def _call_explicit_optional_method(
    obj: Any, attr_name: str, method_name: str, *args: Any, **kwargs: Any
) -> Any:
    """
    Call an explicitly defined optional method, supporting sync and async hooks.

    inspect.getattr_static() avoids Mock/AsyncMock auto-creating attributes that
    would otherwise look present and trigger false-positive hook execution.
    """
    sentinel = object()
    static_attr = inspect.getattr_static(obj, attr_name, sentinel)
    if static_attr is sentinel:
        return None

    target = getattr(obj, attr_name, None)
    if target is None:
        return None

    method = getattr(target, method_name, None)
    if method is None:
        return None

    return await _resolve_awaitable(method(*args, **kwargs))


def _generate_skeleton_with_optional_filters(
    compressor: Any,
    file_id: str,
    *,
    query: str | None = None,
    anchor_node_ids: set[str] | None = None,
    exclude_node_ids: set[str] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if anchor_node_ids is not None:
        kwargs["anchor_node_ids"] = anchor_node_ids

    try:
        params = inspect.signature(compressor._generate_skeleton).parameters
    except (TypeError, ValueError):
        params = {}

    if exclude_node_ids and "exclude_node_ids" in params:
        kwargs["exclude_node_ids"] = exclude_node_ids

    return compressor._generate_skeleton(file_id, **kwargs)


def _flatten_output_fields(schema: Dict[str, Any], prefix: str = "") -> List[str]:
    """Flatten nested dict/list schema keys to dotted output field paths."""
    fields: List[str] = []
    for key, value in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if value:
                fields.extend(_flatten_output_fields(value, prefix=full_key))
            else:
                fields.append(full_key)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            list_prefix = f"{full_key}[]"
            fields.extend(_flatten_output_fields(value[0], prefix=list_prefix))
        else:
            fields.append(full_key)
    return fields


SEARCH_SEMANTIC_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "query": "",
    "file_id": "",
    "evidence_aware": False,
    "total_results": 0,
    "results": [
        {
            "node_id": "",
            "similarity": 0.0,
            "importance": 0.0,
            "summary": "",
            "tokens": 0,
        }
    ],
    "tip": "",
    "score_explanation": {
        "similarity": "",
        "importance": "",
    },
    "temporal_filters": {
        "as_of": "",
        "include_invalidated": False,
    },
    "evidence": {
        "sufficient": True,
        "best_score": 0.0,
        "threshold": 0.0,
        "used_expanded_search": False,
        "message": "",
    },
}


def get_search_semantic_output_fields() -> List[str]:
    """Get canonical output field paths for search_semantic help/docs."""
    return _flatten_output_fields(SEARCH_SEMANTIC_RESPONSE_TEMPLATE)


READ_SKELETON_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "file_id": "",
    "total_nodes": 0,
    "total_tokens": 0,
    "skeleton_tokens": 0,
    "compression_ratio": 0.0,
    "skeleton_text": "",
    # str | None: query-independent skeleton text safe to use as a stable KV-
    # cache prefix across calls with different `query` values. `None` means
    # no query-independent prefix could be produced for this call (e.g. the
    # baseline cache was cold AND the on-demand recompute also failed) --
    # do NOT cache this response's `skeleton_text` as a stable prefix in
    # that case, since it may be query-conditioned.
    "cache_stable_prefix": "",
    "node_map": {},
    "selection_mode": "baseline",
    "query": "",
    "temporal_filters": {
        "as_of": "",
        "include_invalidated": False,
    },
    "evidence": {
        "sufficient": True,
        "best_score": 0.0,
        "threshold": 0.0,
        "used_expanded_search": False,
        "message": "",
        "node_ids": [],
    },
    "staleness_warning": {
        "is_stale": False,
        "reason": "",
        "cached_time": 0,
        "current_time": 0,
        "recommendation": "",
    },
    "pipeline": {
        "final_stage": "baseline",
        "stage_count": 0,
        "stages": [
            {
                "name": "baseline",
                "query": "",
                "anchor_node_count": 0,
                "evidence_used": False,
                "total_nodes": 0,
                "skeleton_tokens": 0,
                "compression_ratio": 0.0,
            }
        ],
    },
}


def get_read_skeleton_output_fields() -> List[str]:
    """Get canonical output field paths for read_skeleton help/docs."""
    return _flatten_output_fields(READ_SKELETON_RESPONSE_TEMPLATE)


# Below this token count the semantic-skeleton's fixed overhead (headers, anchors,
# node IDs) can exceed the savings, so tiny inputs may EXPAND. Mirrors the REST
# guard at api/app/routers/v1/compress.py so MCP and REST give the same honest signal.
# Below this, a non-positive-savings result gets the honesty note. Set to 1000 to
# match the note's own "~1,000+ tokens" guidance (#142 dogfood: a 213-token doc with
# -29% savings slipped past the old 200 threshold and got no "it got bigger" signal).
_SMALL_INPUT_TOKEN_THRESHOLD = 1000


INGEST_CONTEXT_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "status": "success",
    "file_id": "",
    "total_nodes": 0,
    "total_tokens": 0,
    "skeleton_tokens": 0,
    "compression_ratio": 0.0,
    "token_savings": 0,
    "token_savings_percent": 0.0,
    "estimate": {
        "estimated_ratio": 0.0,
        "estimated_compressed": 0,
        "confidence": "",
        "reasoning": "",
        "accuracy": "",
    },
    "note": None,
    "chunking_strategy_used": "",
    "query_skeleton": None,
    "message": "",
    "file_sync_enabled": False,
    "file_path": "",
    "version": 0,
}


def get_ingest_context_output_fields() -> List[str]:
    """Get canonical output field paths for ingest_context help/docs."""
    return _flatten_output_fields(INGEST_CONTEXT_RESPONSE_TEMPLATE)


RECOMMEND_FIDELITY_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "recommended_level": "",
    "confidence": 0.0,
    "reasoning": "",
    "token_estimate": 0,
    "alternatives": [],
    "usage_tip": "",
}


def get_recommend_fidelity_output_fields() -> List[str]:
    """Get canonical output field paths for recommend_fidelity help/docs."""
    return _flatten_output_fields(RECOMMEND_FIDELITY_RESPONSE_TEMPLATE)


# ===========================
# Validation Helpers
# ===========================


def validate_file_id(file_id: str, context: HandlerContext, must_exist: bool = True) -> None:
    """Validate file_id and provide helpful error messages with fuzzy matching (v0.4.1+).

    Args:
        file_id: The file identifier to validate
        context: Server context dict containing compressor instance
        must_exist: If True, check that file_id exists in compressor

    Raises:
        ValueError: If validation fails (with "Did you mean?" suggestions)
    """
    if not file_id:
        raise SmartError.missing_required_field("file_id", "function call")

    if must_exist:
        if file_id not in context["compressor"].graphs:
            # Get list of available file IDs from graphs
            available = list(context["compressor"].graphs.keys())
            if not available:
                raise ValueError(
                    f"Document '{file_id}' not found. No documents ingested yet.\n"
                    "Tip: Use ingest_context() to add documents first."
                )
            # Use SmartError for fuzzy matching
            raise SmartError.file_id_not_found(file_id, available)


def validate_node_ids(node_ids: List[str], context: HandlerContext) -> None:
    """Validate node_ids and provide helpful suggestions with fuzzy matching (v0.4.1+).

    Args:
        node_ids: List of node IDs to validate
        context: Server context dict containing compressor instance

    Raises:
        ValueError: If validation fails (with "Did you mean?" suggestions)
    """
    if not node_ids:
        raise SmartError.missing_required_field("node_ids", "function call")

    invalid_nodes = [nid for nid in node_ids if nid not in context["compressor"].chunks]
    if invalid_nodes:
        # Extract file_id from first node to give better error message.
        # Use shared parser so handlers and server stay in sync.
        file_id = extract_file_id_from_node(node_ids[0])
        valid_nodes = [
            nid
            for nid in context["compressor"].chunks.keys()
            if extract_file_id_from_node(nid) == file_id
        ]

        if not valid_nodes:
            raise ValueError(
                f"Invalid node IDs: {invalid_nodes[:3]}\n"
                f"   No nodes found for '{file_id}'. Document may not be ingested.\n"
                f"Tip: Use ingest_context() to add the document first."
            )

        # Use SmartError for fuzzy matching on first invalid node
        raise SmartError.node_id_not_found(invalid_nodes[0], valid_nodes, file_id)


def validate_token_count(available_tokens: int, max_tokens: int = None) -> None:
    """Validate token counts.

    Args:
        available_tokens: Number of available tokens
        max_tokens: Optional maximum token limit

    Raises:
        ValueError: If validation fails
    """
    if available_tokens < 0:
        raise ValueError(f"available_tokens must be non-negative, got {available_tokens}")

    if available_tokens == 0:
        raise ValueError(
            "available_tokens is 0 - no space for content!\n"
            "Tip: Provide a positive number (e.g., 10000 for 10k tokens available)"
        )

    if max_tokens is not None and available_tokens > max_tokens:
        raise ValueError(
            f"available_tokens ({available_tokens}) exceeds max_tokens ({max_tokens})\n"
            "Tip: available_tokens should be ≤ max_tokens"
        )


# ===========================
# Compression Handlers
# ===========================
