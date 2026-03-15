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

import json
import logging
from typing import Any, Dict, List
import hashlib
import inspect

from ..types import HandlerContext  # TypedDict for handler context
from ..semantic_compressor import FidelityLevel
from ..fidelity_advisor import FidelityAdvisor, UseCase
from ..error_helpers import SmartError
from ..compression_advisor import CompressionAdvisor
from ..rate_limiter import RATE_LIMITERS
from ..error_types import RateLimitExceededError
from ..metrics import compute_cost_savings, get_metrics
from ..constants import MAX_TEXT_LENGTH_BYTES
from ..node_identity import collect_file_ids, extract_file_id_from_node


logger = logging.getLogger("semantic-modulator")


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
    "node_map": {},
    "selection_mode": "baseline",
    "query": "",
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
}


def get_read_skeleton_output_fields() -> List[str]:
    """Get canonical output field paths for read_skeleton help/docs."""
    return _flatten_output_fields(READ_SKELETON_RESPONSE_TEMPLATE)


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
        "accuracy": "",
    },
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


async def handle_ingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle ingest_context tool call.

    Args:
        context: Server context dict with compressor, persistence, resource_manager, etc.
        args: Tool arguments containing text, file_id, optional file_path and metadata

    Returns:
        JSON string with ingestion results including compression stats

    Raises:
        ValueError: If validation fails
        RuntimeError: If ingestion fails
        RateLimitExceededError: If rate limit exceeded (v0.7.0)
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["ingest"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for document ingestion. Please retry in a moment.\n"
            "Tip: The server allows ~10 ingestions/second to prevent resource exhaustion."
        )

    text = args["text"]
    file_id = args["file_id"]
    file_path = args.get("file_path")  # Optional file path for sync tracking
    metadata = args.get("metadata")

    # Text content length validation (v0.7.0 security hardening)
    text_bytes = len(text.encode("utf-8"))
    if text_bytes > MAX_TEXT_LENGTH_BYTES:
        raise ValueError(
            f"Text content too large: {text_bytes:,} bytes (max: {MAX_TEXT_LENGTH_BYTES:,})\n"
            "Tip: Split large documents into smaller chunks before ingestion."
        )

    # SECURITY: Validate file_path to prevent path traversal (CWE-22)
    if file_path:
        try:
            # PathValidator resolves .., symlinks, and validates against allowed directories
            file_path = context["path_validator"].validate(file_path)
            logger.info(f"File path validated: {file_path}")
        except ValueError as e:
            raise ValueError(
                f"Invalid file_path: {str(e)}\n"
                "[TIP] Security: File paths must be within allowed directories to prevent path traversal attacks"
            ) from e

    # Validation
    if not text or len(text.strip()) == 0:
        raise ValueError(
            "text cannot be empty\n"
            "Tip: Provide document content to ingest (minimum ~20 characters recommended)"
        )

    if len(text) < 20:
        raise ValueError(
            f"text is too short ({len(text)} chars)\n"
            "Tip: Provide at least 20 characters for meaningful semantic analysis"
        )

    validate_file_id(file_id, context, must_exist=False)

    # Check resource limits BEFORE ingestion
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    text_size = len(text.encode("utf-8"))
    allowed, error_msg = await context["resource_manager"].check_document_size_async(
        file_id, text_size
    )
    if not allowed:
        raise ValueError(error_msg)

    logger.info(f"Ingesting document: {file_id} ({len(text)} chars, {text_size / 1024:.1f}KB)")

    # NEW v0.4.1: Provide compression estimate before actual compression
    advisor = CompressionAdvisor()
    estimate = advisor.estimate_compression(text, skeleton_ratio=0.2)
    logger.info(
        f"Compression estimate: {estimate.compression_ratio:.1f}× "
        f"({estimate.original_tokens} → ~{estimate.estimated_compressed} tokens)"
    )

    try:
        chunking_strategy = args.get("chunking_strategy", "fixed")
        skeleton = await context["compressor"].ingest_file_async(
            text, file_id, metadata, chunking_strategy=chunking_strategy
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to ingest document: {str(e)}\n"
            "Tip: Check that text is valid and file_id contains only alphanumeric and underscores"
        ) from e

    # Register with resource manager
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    await context["resource_manager"].register_document_async(file_id, text_size)

    # Persist to storage
    try:
        import networkx as nx

        graph_data = nx.node_link_data(context["compressor"].graphs[file_id])
        success = context["persistence"].save_document(
            file_id=file_id,
            chunks={k: v for k, v in context["compressor"].chunks.items() if k.startswith(file_id)},
            graph_data=graph_data,
            metadata=context["compressor"].file_metadata.get(file_id, {}),
        )
        if inspect.isawaitable(success):
            success = await success
        if success:
            logger.info(f"[OK] Persisted document {file_id}")
        else:
            logger.warning(f"[WARN]  Failed to persist {file_id}, will be lost on restart")
    except Exception as e:
        logger.error(f"Failed to persist {file_id}: {e}")

    # NEW: Register with file sync manager and version manager
    checksum = hashlib.md5(text.encode()).hexdigest()
    context["sync_manager"].register_file(file_id, file_path, text)
    try:
        # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
        await context["version_manager"].add_version_async(
            doc_id=file_id,
            content=text,
            checksum=checksum,
            file_path=file_path,
            metadata=metadata or {},
            compression_stats={
                "total_tokens": skeleton.total_tokens,
                "skeleton_tokens": skeleton.skeleton_tokens,
                "compression_ratio": skeleton.compression_ratio,
            },
        )
        logger.info(f"[OK] Registered version history for {file_id}")
    except Exception as e:
        logger.warning(f"[WARN]  Failed to save version history for {file_id}: {e}")

    # Save file sync metadata to persistence
    try:
        metadata_export = context["sync_manager"].export_metadata()
        success = context["persistence"].save_file_sync_metadata(metadata_export)
        if inspect.isawaitable(success):
            success = await success
        if success:
            logger.info(f"[OK] Saved file sync metadata for {len(metadata_export)} documents")
        else:
            logger.warning("[WARN]  Failed to save file sync metadata")
    except Exception as e:
        logger.error(f"Failed to save file sync metadata: {e}")

    # Initialize retrieval history
    context["retrieval_history"][file_id] = []

    # Compare estimate vs actual (v0.4.1+)
    actual_ratio = skeleton.compression_ratio
    estimate_accuracy = (
        "excellent"
        if abs(actual_ratio - estimate.compression_ratio) < 2
        else "good" if abs(actual_ratio - estimate.compression_ratio) < 5 else "fair"
    )

    # Build JSON response
    response = {
        "status": "success",
        "file_id": file_id,
        "total_nodes": skeleton.total_nodes,
        "total_tokens": skeleton.total_tokens,
        "skeleton_tokens": skeleton.skeleton_tokens,
        "compression_ratio": skeleton.compression_ratio,
        "token_savings": skeleton.total_tokens - skeleton.skeleton_tokens,
        "token_savings_percent": round(
            (1 - skeleton.skeleton_tokens / skeleton.total_tokens) * 100, 1
        ) if skeleton.total_tokens > 0 else 0.0,
        "estimate": {"estimated_ratio": estimate.compression_ratio, "accuracy": estimate_accuracy},
        "message": f"Document ingested successfully with {skeleton.total_nodes} semantic nodes",
    }

    try:
        response["cost_savings"] = compute_cost_savings(
            original_tokens=skeleton.total_tokens,
            compressed_tokens=skeleton.skeleton_tokens,
        ).to_dict()
    except Exception:
        response["cost_savings"] = None

    # Record Prometheus metrics for observability
    try:
        metrics = get_metrics()
        fidelity_label = args.get("fidelity_level", "BALANCED")
        metrics.record_compression_ratio(skeleton.compression_ratio, fidelity_label)
        metrics.increment_documents_processed("ingest", fidelity_label, "success")
        metrics.set_active_documents(len(context["compressor"].graphs))
    except Exception:
        pass  # Metrics are best-effort, never block ingestion

    # Phase 5: Record access and compression replay for optimization
    try:
        compressor = context["compressor"]
        if hasattr(compressor, '_access_tracker'):
            compressor._access_tracker.record_access(file_id)
        if hasattr(compressor, '_compression_replay'):
            from ..fidelity_scoring import compute_fidelity_score
            fidelity = 0.0
            try:
                original_text = args.get("text", "")
                if original_text and skeleton.skeleton_text:
                    emb_mgr = compressor.model
                    fidelity = compute_fidelity_score(
                        original_text, skeleton.skeleton_text,
                        lambda texts: emb_mgr.encode(texts)
                    )
            except Exception:
                pass
            content_type = args.get("content_type", "general")
            compressor._compression_replay.record(
                doc_id=file_id,
                content_type=content_type,
                input_tokens=skeleton.total_tokens,
                output_tokens=skeleton.skeleton_tokens,
                ratio=skeleton.compression_ratio,
                fidelity_score=fidelity,
            )
            response["fidelity_score"] = round(fidelity, 4)
    except Exception:
        pass  # Phase 5 features are best-effort

    if file_path:
        response["file_sync_enabled"] = True
        response["file_path"] = file_path
        response["version"] = 1

    return json.dumps(response, indent=2)


async def handle_read_skeleton(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle read_skeleton tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing file_id

    Returns:
        JSON string with skeleton data and optional staleness warning

    Raises:
        RuntimeError: If reading skeleton fails
    """
    file_id = args["file_id"]
    selection_mode = args.get("selection_mode", "baseline")
    query = args.get("query")
    top_k = args.get("top_k", 5)
    min_similarity = args.get("min_similarity", 0.35)

    valid_modes = {"baseline", "query_guided", "evidence_aware"}
    if selection_mode not in valid_modes:
        raise ValueError(
            f"Invalid selection_mode: '{selection_mode}'\n"
            f"[TIP] Valid modes: {sorted(valid_modes)}"
        )
    if selection_mode != "baseline" and not query:
        raise ValueError(
            f"query is required when selection_mode='{selection_mode}'\n"
            "[TIP] Provide a natural-language query to guide anchor selection."
        )

    validate_file_id(file_id, context, must_exist=True)

    logger.info(f"Reading skeleton: {file_id}")

    # NEW: Check file sync status before reading
    staleness_warning = None
    if file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(file_id)
        if not status["in_sync"]:
            staleness_warning = {
                "is_stale": True,
                "reason": status["reason"],
                "cached_time": status.get("cached_mtime"),
                "current_time": status.get("current_mtime"),
                "recommendation": f"Use refresh_document('{file_id}') to update or diff_cached_file('{file_id}') to see changes",
            }

    try:
        evidence_info = None
        compressor = context["compressor"]
        anchored_keywords = args.get("anchored_keywords", [])

        if selection_mode == "query_guided":
            skeleton_response = compressor._generate_skeleton(file_id, query=query)
        elif selection_mode == "evidence_aware":
            evidence = compressor.retrieve_evidence(
                query=query,
                file_id=file_id,
                top_k=top_k,
                min_similarity=min_similarity,
            )
            skeleton_response = compressor._generate_skeleton(
                file_id,
                query=query,
                anchor_node_ids=set(evidence.node_ids),
            )
            evidence_info = {
                "sufficient": evidence.sufficient,
                "best_score": round(evidence.best_score, 3),
                "threshold": evidence.threshold,
                "used_expanded_search": evidence.used_expanded_search,
                "message": evidence.message,
                "node_ids": evidence.node_ids,
            }
        else:
            skeleton_response = compressor._generate_skeleton(file_id)

        # Phase 5: Record access for decay tracking
        try:
            if hasattr(compressor, '_access_tracker'):
                compressor._access_tracker.record_access(file_id)
        except Exception:
            pass

        # Build JSON response
        response = {
            "file_id": skeleton_response.file_id,
            "total_nodes": skeleton_response.total_nodes,
            "total_tokens": skeleton_response.total_tokens,
            "skeleton_tokens": skeleton_response.skeleton_tokens,
            "compression_ratio": skeleton_response.compression_ratio,
            "skeleton_text": skeleton_response.skeleton_text,
            "node_map": skeleton_response.node_map,
            "selection_mode": selection_mode,
        }

        # Phase 5: Apply keyword anchoring if specified
        if anchored_keywords and skeleton_response.node_map:
            try:
                from ..keyword_anchoring import apply_keyword_anchoring
                nodes_for_anchoring = [
                    {"node_id": nid, "text": compressor.chunks[nid].text,
                     "importance": compressor.chunks[nid].importance}
                    for nid in skeleton_response.node_map
                    if nid in compressor.chunks
                ]
                if nodes_for_anchoring:
                    kept = apply_keyword_anchoring(
                        nodes_for_anchoring, anchored_keywords, keep_ratio=1.0
                    )
                    response["anchored_nodes"] = [
                        n["node_id"] for n in kept
                        if any(kw.lower() in n["text"].lower() for kw in anchored_keywords)
                    ]
            except Exception:
                pass
        if query:
            response["query"] = query
        if evidence_info:
            response["evidence"] = evidence_info

        if staleness_warning:
            response["staleness_warning"] = staleness_warning

        return json.dumps(response, indent=2)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read skeleton for '{file_id}': {str(e)}\n"
            f"Tip: Verify the document was ingested successfully with get_stats()"
        ) from e


async def handle_modulate_region(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle modulate_region tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing node_ids and optional fidelity_level

    Returns:
        Modulated content with optional staleness warning

    Raises:
        ValueError: If validation fails
        RuntimeError: If modulation fails
    """
    node_ids = args["node_ids"]
    fidelity_str = args.get("fidelity_level", "RAW")

    # Validation
    validate_node_ids(node_ids, context)

    # NEW: Check file sync status before modulating
    file_id = extract_file_id_from_node(node_ids[0]) if node_ids else None
    warning = ""
    if file_id and file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(file_id)
        if not status["in_sync"]:
            warning = f"""
[WARN]  WARNING: Cache may be stale for '{file_id}'!

{status['reason']}

[TIP] Use refresh_document('{file_id}') to update

Proceeding with cached version...
---

"""

    # Convert string to enum with validation
    try:
        fidelity = FidelityLevel[fidelity_str]
    except KeyError:
        valid_levels = [level.name for level in FidelityLevel]
        raise ValueError(
            f"Invalid fidelity_level: '{fidelity_str}'\n"
            f"[TIP] Valid levels: {valid_levels}\n"
            f"   ABSTRACT: ~10 tokens (summary only)\n"
            f"   OUTLINE: ~30 tokens (summary + section markers)\n"
            f"   STRUCTURE: ~50 tokens (headers + entities)\n"
            f"   DETAILED: ~100 tokens (summary + excerpts)\n"
            f"   RAW: Full original text"
        )

    logger.info(f"Modulating {len(node_ids)} nodes at {fidelity_str} fidelity")

    # Track retrieval for blind spot detection
    for node_id in node_ids:
        file_id = extract_file_id_from_node(node_id)
        if file_id not in context["retrieval_history"]:
            context["retrieval_history"][file_id] = []
        if node_id not in context["retrieval_history"][file_id]:
            context["retrieval_history"][file_id].append(node_id)

    try:
        result = context["compressor"].modulate_region(node_ids, fidelity)
        return warning + result
    except Exception as e:
        raise RuntimeError(
            f"Failed to modulate region: {str(e)}\n"
            f"Tip: Verify node IDs are valid with read_skeleton()"
        ) from e


async def handle_search_semantic(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle search_semantic tool call.

    v0.9.0: Now returns similarity scores alongside importance (PageRank) scores.
    - similarity: How well the node matches the search query (cosine similarity)
    - importance: How central the node is in the document graph (PageRank)

    Args:
        context: Server context dict
        args: Tool arguments containing query, optional file_id and top_k

    Returns:
        JSON string with search results including similarity scores
    """
    query = args["query"]
    file_id = args.get("file_id")
    top_k = args.get("top_k", 5)
    evidence_aware = args.get("evidence_aware", False)
    min_similarity = args.get("min_similarity", 0.35)

    logger.info(f"Semantic search: '{query}' in {file_id or 'all files'}")

    if evidence_aware:
        evidence = context["compressor"].retrieve_evidence(
            query=query,
            file_id=file_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        search_results = evidence.scores[:top_k]
    else:
        evidence = None
        # Use search_semantic_with_scores to get both node IDs and similarity scores
        search_results = context["compressor"].search_semantic_with_scores(query, file_id, top_k)

    # Build structured results with both similarity and importance
    results = []
    for node_id, similarity_score in search_results:
        node = context["compressor"].chunks[node_id]
        summary = context["compressor"]._generate_summary(node.text, max_length=100)
        results.append(
            {
                "node_id": node_id,
                "similarity": round(similarity_score, 3),  # Query match score
                "importance": round(node.importance, 3),  # PageRank centrality
                "summary": summary,
                "tokens": node.metadata.get("tokens", 0),
            }
        )

    # Build JSON response
    response = {
        "query": query,
        "file_id": file_id,
        "evidence_aware": evidence_aware,
        "total_results": len(results),
        "results": results,
        "tip": "Use modulate_region() with node_ids to retrieve full content",
        "score_explanation": {
            "similarity": "Semantic match to query (higher = better match)",
            "importance": "PageRank centrality in document graph (higher = more central)",
        },
    }
    if evidence is not None:
        response["evidence"] = {
            "sufficient": evidence.sufficient,
            "best_score": round(evidence.best_score, 3),
            "threshold": evidence.threshold,
            "used_expanded_search": evidence.used_expanded_search,
            "message": evidence.message,
        }

    # Phase 5: Record access for decay tracking
    try:
        compressor = context["compressor"]
        if hasattr(compressor, '_access_tracker'):
            accessed_files = set()
            for r in results:
                fid = extract_file_id_from_node(r["node_id"])
                if fid:
                    accessed_files.add(fid)
            for fid in accessed_files:
                compressor._access_tracker.record_access(fid)
    except Exception:
        pass

    return json.dumps(response, indent=2)


async def handle_get_stats(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle get_stats tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing optional file_id

    Returns:
        Formatted statistics
    """
    file_id = args.get("file_id")

    stats = context["compressor"].get_stats(file_id)

    if file_id:
        result = f"""
[STATS] Document Statistics: {file_id}

Total nodes: {stats['total_nodes']}
Total edges: {stats['total_edges']}
Original tokens: {stats['total_tokens']:,}
Skeleton tokens: {stats['skeleton_tokens']:,}
Compression ratio: {stats['compression_ratio']:.1f}x

Token savings: {stats['total_tokens'] - stats['skeleton_tokens']:,} ({(1 - stats['skeleton_tokens'] / stats['total_tokens']) * 100:.1f}%)

Metadata: {json.dumps(stats['metadata'], indent=2)}
"""
    else:
        result = f"""
[STATS] Global Statistics

Total files ingested: {stats['total_files']}
Total nodes: {stats['total_nodes']}

Files: {', '.join(stats['files'])}
"""

    return result


async def handle_list_documents(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle list_documents tool call.

    Args:
        context: Server context dict
        args: Tool arguments (none required)

    Returns:
        Formatted document inventory
    """
    logger.info("Listing all ingested documents")

    # Get all unique file_ids from chunks (supports both text and code node formats)
    file_ids = sorted(collect_file_ids(context["compressor"].chunks.keys()))

    if not file_ids:
        return """
[DOC] Document Inventory

No documents ingested yet.

[TIP] Use ingest_context(text, file_id) to add documents.
"""

    # Build structured inventory
    documents = []
    for file_id in sorted(file_ids):
        stats = context["compressor"].get_stats(file_id)
        metadata = stats.get("metadata", {})

        doc_info = {
            "file_id": file_id,
            "title": metadata.get("title", file_id),
            "total_nodes": stats["total_nodes"],
            "total_tokens": stats["total_tokens"],
            "skeleton_tokens": stats["skeleton_tokens"],
            "compression_ratio": stats["compression_ratio"],
            "metadata": metadata,
        }
        documents.append(doc_info)

    # Format output
    result_lines = ["[DOC] Document Inventory\n"]
    result_lines.append(f"Total documents: {len(documents)}\n")

    for i, doc in enumerate(documents, 1):
        result_lines.append(f"{i}. [{doc['file_id']}]")
        if doc["title"] != doc["file_id"]:
            result_lines.append(f"   Title: {doc['title']}")
        result_lines.append(f"   Nodes: {doc['total_nodes']}")
        result_lines.append(
            f"   Tokens: {doc['total_tokens']:,} → {doc['skeleton_tokens']:,} ({doc['compression_ratio']:.1f}x compression)"
        )

        # Include relevant metadata
        if "author" in doc["metadata"]:
            result_lines.append(f"   Author: {doc['metadata']['author']}")
        if "date" in doc["metadata"]:
            result_lines.append(f"   Date: {doc['metadata']['date']}")
        if "tags" in doc["metadata"]:
            tags = ", ".join(doc["metadata"]["tags"][:3])
            result_lines.append(f"   Tags: {tags}")

        result_lines.append("")  # Blank line between documents

    result_lines.append("[TIP] Next steps:")
    result_lines.append("  - read_skeleton(file_id) - View compressed structure")
    result_lines.append("  - search_semantic(query) - Find relevant content")
    result_lines.append("  - get_stats(file_id) - Detailed statistics")

    return "\n".join(result_lines)


async def handle_delete_document(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle delete_document tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing file_id and confirm flag

    Returns:
        Confirmation prompt or deletion success message

    Raises:
        RuntimeError: If deletion fails
    """
    file_id = args["file_id"]
    confirm = args.get("confirm", False)

    # Validation
    validate_file_id(file_id, context, must_exist=True)

    if not confirm:
        return f"""
[WARN]  DELETE CONFIRMATION REQUIRED

You are about to delete document: {file_id}

This will:
  -Remove all {len([k for k in context['compressor'].chunks.keys() if k.startswith(file_id)])} semantic nodes from memory
  -Delete persistent storage (cannot be undone)
  -Clear retrieval history for this document

To proceed, call again with confirm=true:
  delete_document(file_id="{file_id}", confirm=true)

Tip: Use list_documents() to see all available documents first
"""

    logger.info(f"Deleting document: {file_id}")

    # Get stats before deletion
    stats = context["compressor"].get_stats(file_id)
    node_count = stats["total_nodes"]

    # Delete from memory
    try:
        # Remove chunks
        chunks_to_delete = [k for k in context["compressor"].chunks.keys() if k.startswith(file_id)]
        for chunk_id in chunks_to_delete:
            del context["compressor"].chunks[chunk_id]

        # Remove graph
        if file_id in context["compressor"].graphs:
            del context["compressor"].graphs[file_id]

        # Remove metadata
        if file_id in context["compressor"].file_metadata:
            del context["compressor"].file_metadata[file_id]

        # Remove retrieval history
        if file_id in context["retrieval_history"]:
            del context["retrieval_history"][file_id]

        logger.info(f"[OK] Removed {file_id} from memory ({node_count} nodes)")

    except Exception as e:
        logger.error(f"Failed to delete {file_id} from memory: {e}")
        raise RuntimeError(f"Failed to delete from memory: {e}")

    # Delete from persistent storage
    try:
        success = context["persistence"].delete_document(file_id)
        if success:
            logger.info(f"[OK] Deleted {file_id} from persistent storage")
        else:
            logger.warning(f"[WARN]  Failed to delete {file_id} from persistent storage")
    except Exception as e:
        logger.error(f"Failed to delete {file_id} from storage: {e}")

    # Unregister from resource manager
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    try:
        await context["resource_manager"].unregister_document_async(file_id)
    except Exception as e:
        logger.warning(f"Failed to unregister {file_id} from resource manager: {e}")

    # NEW: Clean up file sync metadata and version history
    try:
        context["sync_manager"].remove_metadata(file_id)
        # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
        await context["version_manager"].delete_versions_async(file_id)
        # Save metadata after removal
        metadata_export = context["sync_manager"].export_metadata()
        context["persistence"].save_file_sync_metadata(metadata_export)
        logger.info(f"[OK] Cleaned up file sync metadata and version history for {file_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up file sync data for {file_id}: {e}")

    return f"""
[DELETE] Document Deleted Successfully

File ID: {file_id}
Nodes removed: {node_count}
Memory freed: ~{node_count * 2}KB (estimated)

[OK] Document has been permanently deleted from:
   -Memory (semantic graph, chunks, metadata)
   -Persistent storage (ChromaDB/JSON)
   -Resource tracking

[TIP] Remaining documents: {len(collect_file_ids(context["compressor"].chunks.keys()))}
   Use list_documents() to see what's left.
"""


async def handle_adapt_to_context_window(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle adapt_to_context_window tool call (JSCCM-inspired).

    Args:
        context: Server context dict
        args: Tool arguments containing file_id, available_tokens, optional max_tokens and query_priority

    Returns:
        Adapted skeleton text

    Raises:
        ValueError: If validation fails
        RuntimeError: If adaptation fails
    """
    file_id = args["file_id"]
    available_tokens = args["available_tokens"]
    max_tokens = args.get("max_tokens", 100000)
    query_priority = args.get("query_priority", 0.5)

    # Validation
    validate_file_id(file_id, context, must_exist=True)
    validate_token_count(available_tokens, max_tokens)

    if not 0.0 <= query_priority <= 1.0:
        raise ValueError(
            f"query_priority must be between 0.0 and 1.0, got {query_priority}\n"
            "Tip: 0.0 = low priority, 0.5 = medium, 1.0 = high priority"
        )

    logger.info(
        f"Adapting skeleton for {file_id}: {available_tokens}/{max_tokens} tokens available"
    )

    try:
        result = context["context_window_adapter"].adapt_to_context_window(
            file_id=file_id,
            available_tokens=available_tokens,
            max_tokens=max_tokens,
            query_priority=query_priority,
        )
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to adapt to context window: {str(e)}\n"
            "Tip: This is a JSCCM-inspired feature. Check that the document exists and token counts are valid."
        ) from e


async def handle_multilevel_encode(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle multilevel_encode tool call (JSCCM-inspired).

    Args:
        context: Server context dict
        args: Tool arguments containing file_id and available_tokens

    Returns:
        Multi-level encoded skeleton

    Raises:
        ValueError: If validation fails
        RuntimeError: If encoding fails
    """
    file_id = args["file_id"]
    available_tokens = args["available_tokens"]

    # Validation
    validate_file_id(file_id, context, must_exist=True)
    validate_token_count(available_tokens)

    logger.info(
        f"Generating multi-level encoding for {file_id}: {available_tokens} tokens available"
    )

    try:
        result = context["multilevel_encoder"].generate_adaptive_skeleton(file_id, available_tokens)
        return result
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate multi-level encoding: {str(e)}\n"
            "Tip: This JSCCM-inspired feature requires Main + Auxiliary + Detail branches.\n"
            "   Try with at least 1000 tokens available for meaningful output."
        ) from e


async def handle_recommend_fidelity(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle recommend_fidelity tool call (NEW in v0.4.1).

    Args:
        context: Server context dict (not used but required for handler signature)
        args: Tool arguments containing use_case, num_nodes, optional token_budget and query_complexity

    Returns:
        JSON string with fidelity recommendation, reasoning, token estimate, and alternatives

    Raises:
        ValueError: If validation fails

    Example:
        >>> handle_recommend_fidelity(
        ...     {"use_case": "question_answering", "num_nodes": 3, "token_budget": 200},
        ...     context
        ... )
        '{"recommended_level": "STRUCTURE", "confidence": 0.9, ...}'
    """
    use_case_str = args["use_case"]
    num_nodes = args["num_nodes"]
    token_budget = args.get("token_budget")
    query_complexity = args.get("query_complexity", "medium")

    # Validation
    if num_nodes < 1:
        raise ValueError(
            f"num_nodes must be at least 1, got {num_nodes}\n"
            "Tip: Specify how many nodes you plan to retrieve"
        )

    if num_nodes > 1000:
        raise ValueError(
            f"num_nodes is very high ({num_nodes})\n"
            "Tip: Consider retrieving fewer nodes for better token efficiency.\n"
            "   Most queries work well with 3-10 nodes."
        )

    if token_budget is not None:
        if token_budget < 10:
            raise ValueError(
                f"token_budget is too low ({token_budget})\n"
                "Tip: Even ABSTRACT fidelity needs ~10 tokens per node.\n"
                f"   For {num_nodes} nodes, minimum budget: {num_nodes * 10} tokens"
            )

        if token_budget > 1_000_000:
            raise ValueError(
                f"token_budget is very high ({token_budget:,})\n"
                "Tip: Most use cases work well with 100-10,000 token budgets."
            )

    if query_complexity not in ["simple", "medium", "complex"]:
        raise ValueError(
            f"query_complexity must be 'simple', 'medium', or 'complex', got '{query_complexity}'\n"
            "Tip: Use 'medium' if unsure"
        )

    # Convert string to enum
    try:
        use_case_enum = UseCase(use_case_str)
    except ValueError:
        valid_cases = [case.value for case in UseCase]
        raise ValueError(
            f"Unknown use_case: '{use_case_str}'\n" f"[TIP] Valid options: {', '.join(valid_cases)}"
        )

    # Get recommendation
    logger.info(
        f"Recommending fidelity for use_case={use_case_str}, "
        f"num_nodes={num_nodes}, budget={token_budget}, complexity={query_complexity}"
    )

    advisor = FidelityAdvisor()
    rec = advisor.recommend(use_case_enum, num_nodes, token_budget, query_complexity)

    # Format response
    response = {
        "recommended_level": rec.recommended_level.value,
        "confidence": rec.confidence,
        "reasoning": rec.reasoning,
        "token_estimate": rec.token_estimate,
        "alternatives": rec.alternatives,
        "usage_tip": (
            f"Use modulate_region with fidelity_level='{rec.recommended_level.value}' "
            f"to retrieve {num_nodes} nodes (~{rec.token_estimate} tokens)"
        ),
    }

    return json.dumps(response, indent=2)


# ===========================
# Batch Processing Handler
# ===========================


async def handle_batch_ingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle batch_ingest_documents MCP tool (v0.6.0, rate limiting v0.7.0).

    Ingests multiple documents concurrently with bounded parallelism,
    progress tracking, and error isolation.

    Args:
        context: Server context dict
        args: Tool arguments:
            - documents: List of {file_id, text, metadata} objects
            - max_concurrent: Optional max concurrent ingestions (default 4)

    Returns:
        JSON string with batch results:
        {
            "total": 10,
            "successful": 9,
            "failed": 1,
            "results": [
                {"file_id": "doc1", "success": true, "processing_time": 1.2},
                {"file_id": "doc2", "success": false, "error": "ValueError: text too short"}
            ],
            "summary": "Batch complete: 9/10 succeeded in 12.5s"
        }

    Raises:
        ValueError: If validation fails
        RateLimitExceededError: If rate limit exceeded (v0.7.0)
    """
    # Rate limiting for batch operations (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["batch_ingest"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for batch ingestion. Please retry in a moment.\n"
            "Tip: The server allows ~2 batch operations/second to prevent resource exhaustion."
        )

    from ..batch_manager import BatchCompressionManager, BatchDocument

    # Extract arguments
    documents_list = args.get("documents", [])
    max_concurrent = args.get("max_concurrent", 4)

    # Validate arguments
    if not documents_list:
        raise SmartError.missing_required_field("documents", "batch_ingest_documents")

    if not isinstance(documents_list, list):
        raise ValueError(
            f"documents must be a list, got {type(documents_list).__name__}\n"
            "Tip: Provide a list of objects with 'file_id' and 'text' fields"
        )

    if not (1 <= max_concurrent <= 8):
        raise ValueError(
            f"max_concurrent must be between 1 and 8, got {max_concurrent}\n"
            "Tip: Use 4 for balanced performance, 1 for sequential, 8 for maximum throughput"
        )

    # Validate each document
    batch_documents = []
    for i, doc in enumerate(documents_list):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Document {i} must be an object, got {type(doc).__name__}\n"
                "Tip: Each document should have 'file_id' and 'text' fields"
            )

        file_id = doc.get("file_id")
        text = doc.get("text")
        metadata = doc.get("metadata", {})

        # Validate file_id
        if not file_id:
            raise SmartError.missing_required_field(
                f"documents[{i}].file_id", "batch_ingest_documents"
            )

        if not isinstance(file_id, str):
            raise ValueError(
                f"documents[{i}].file_id must be a string, got {type(file_id).__name__}"
            )

        # Validate text exists (allow empty strings for batch error isolation)
        if text is None:
            raise SmartError.missing_required_field(
                f"documents[{i}].text", "batch_ingest_documents"
            )

        if not isinstance(text, str):
            raise ValueError(f"documents[{i}].text must be a string, got {type(text).__name__}")

        # Create BatchDocument
        batch_documents.append(BatchDocument(file_id=file_id, text=text, metadata=metadata))

    # Log batch operation
    logger.info(
        f"Starting batch ingestion: {len(batch_documents)} documents, "
        f"max_concurrent={max_concurrent}"
    )

    # Create batch manager and execute
    manager = BatchCompressionManager(
        compressor=context["compressor"], max_concurrent=max_concurrent
    )

    import time

    start_time = time.time()
    results = await manager.compress_batch(batch_documents)
    total_time = time.time() - start_time

    # Format results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    result_list = []
    for result in results:
        entry = {
            "file_id": result.file_id,
            "success": result.success,
            "processing_time": round(result.processing_time, 2),
        }

        if result.success:
            # Include skeleton summary
            if result.result:
                entry["skeleton_preview"] = result.result.skeleton_text[:200] + "..."
                entry["compression_ratio"] = result.result.compression_ratio
        else:
            entry["error"] = result.error

        result_list.append(entry)

    # Create response
    response = {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "max_concurrent": max_concurrent,
        "total_time": round(total_time, 2),
        "avg_time_per_doc": round(total_time / len(results), 2) if results else 0.0,
        "results": result_list,
        "summary": (
            f"Batch complete: {successful}/{len(results)} succeeded in {total_time:.1f}s "
            f"(avg {total_time/len(results):.2f}s/doc)"
        ),
    }

    if failed > 0:
        failed_ids = [r.file_id for r in results if not r.success]
        response["failed_file_ids"] = failed_ids
        response["tip"] = (
            f"[WARN] {failed} documents failed. Check error messages for each failed document."
        )

    logger.info(response["summary"])

    return json.dumps(response, indent=2)


async def handle_ingest_directory(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle ingest_directory MCP tool (v0.9.0).

    Bulk ingest code files from a directory using glob patterns.
    Uses PathValidator for security (prevents path traversal attacks).
    Leverages BatchCompressionManager for concurrent processing.

    Args:
        context: Server context dict with path_validator and compressor
        args: Tool arguments:
            - directory: Directory path to scan
            - patterns: Glob patterns for files to include (default: ['*.py', '*.js', '*.ts'])
            - exclude_patterns: Patterns to exclude (default: node_modules, __pycache__, venv)
            - max_files: Maximum files to ingest (default: 50, max: 100)
            - max_concurrent: Maximum concurrent ingestions (default: 4, max: 8)

    Returns:
        JSON string with directory ingestion results

    Raises:
        ValueError: If directory is invalid or path traversal detected
    """
    import os
    import time
    from pathlib import Path
    from ..batch_manager import BatchCompressionManager, BatchDocument

    # Extract arguments with defaults
    directory = args.get("directory", "")
    patterns = args.get("patterns", ["*.py", "*.js", "*.ts"])
    exclude_patterns = args.get(
        "exclude_patterns",
        [
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/venv/**",
            "**/.venv/**",
            "**/.git/**",
        ],
    )
    max_files = args.get("max_files", 50)
    max_concurrent = args.get("max_concurrent", 4)

    # Validate directory is provided
    if not directory:
        raise SmartError.missing_required_field("directory", "ingest_directory")

    # Validate and normalize directory path using PathValidator (CWE-22 protection)
    path_validator = context["path_validator"]
    try:
        validated_dir = path_validator.validate(directory)
    except ValueError as e:
        raise ValueError(
            f"Invalid directory path: {e}\n"
            "Tip: Directory must be within the current working directory or user home directory."
        )

    # Check directory exists
    if not os.path.isdir(validated_dir):
        raise ValueError(
            f"Directory not found: {validated_dir}\n" "Tip: Provide an existing directory path."
        )

    # Validate parameters
    if not (1 <= max_files <= 100):
        raise ValueError(
            f"max_files must be between 1 and 100, got {max_files}\n"
            "Tip: Use smaller values for faster ingestion."
        )

    if not (1 <= max_concurrent <= 8):
        raise ValueError(
            f"max_concurrent must be between 1 and 8, got {max_concurrent}\n"
            "Tip: Use 4 for balanced performance."
        )

    # Scan directory for matching files
    logger.info(f"Scanning directory: {validated_dir}")
    logger.info(f"Patterns: {patterns}, Excludes: {exclude_patterns}")

    matched_files = []
    dir_path = Path(validated_dir)

    for pattern in patterns:
        # Use recursive glob for patterns with ** and non-recursive otherwise
        if "**" in pattern:
            for file_path in dir_path.rglob(pattern.replace("**/", "")):
                if file_path.is_file():
                    matched_files.append(file_path)
        else:
            # Check in immediate directory and subdirectories
            for file_path in dir_path.rglob(pattern):
                if file_path.is_file():
                    matched_files.append(file_path)

    # Remove duplicates
    matched_files = list(set(matched_files))

    # Apply exclusions
    # P2-1 fix: Use PurePath.match() for proper glob pattern support
    from pathlib import PurePath

    def is_excluded(path: Path) -> bool:
        """Check if path matches any exclusion pattern using proper glob matching.

        PurePath.match() handles both simple patterns (*.pyc) and ** patterns natively.
        Simple patterns like '*.pyc' match from the right, so 'src/file.pyc' matches '*.pyc'.
        """
        path_obj = PurePath(str(path))

        for exclude in exclude_patterns:
            try:
                # PurePath.match() supports ** patterns and simple patterns natively
                # e.g., '*.pyc' matches 'src/file.pyc', '**/node_modules/**' matches nested dirs
                if path_obj.match(exclude):
                    return True
            except ValueError:
                # Invalid pattern - log and skip
                logger.warning(f"Invalid exclude pattern: {exclude}")
                continue
        return False

    filtered_files = [f for f in matched_files if not is_excluded(f)]

    # Validate each file path
    validated_files = []
    for file_path in filtered_files:
        try:
            validated_path = path_validator.validate(str(file_path))
            validated_files.append(Path(validated_path))
        except ValueError:
            logger.warning(f"Skipping file outside allowed directories: {file_path}")
            continue

    # Limit files
    if len(validated_files) > max_files:
        logger.info(f"Limiting from {len(validated_files)} to {max_files} files")
        validated_files = validated_files[:max_files]

    if not validated_files:
        return json.dumps(
            {
                "status": "no_files",
                "directory": validated_dir,
                "patterns": patterns,
                "message": "No matching files found in directory.",
                "tip": "Try different patterns or check the directory path.",
            },
            indent=2,
        )

    # Read files and prepare batch documents
    batch_documents = []
    skipped_files = []

    for file_path in validated_files:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Generate file_id from relative path
            try:
                rel_path = file_path.relative_to(dir_path)
                file_id = str(rel_path).replace(os.sep, "/")
            except ValueError:
                file_id = file_path.name

            batch_documents.append(
                BatchDocument(
                    file_id=file_id,
                    text=text,
                    metadata={
                        "source_path": str(file_path),
                        "file_size": len(text),
                    },
                    file_path=str(file_path),  # Enable file sync tracking
                )
            )
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            skipped_files.append({"path": str(file_path), "error": str(e)})

    if not batch_documents:
        return json.dumps(
            {
                "status": "read_failed",
                "directory": validated_dir,
                "skipped_files": skipped_files,
                "message": "Could not read any files.",
            },
            indent=2,
        )

    # Log operation
    logger.info(
        f"Starting directory ingestion: {len(batch_documents)} files, "
        f"max_concurrent={max_concurrent}"
    )

    # Create batch manager and execute
    manager = BatchCompressionManager(
        compressor=context["compressor"], max_concurrent=max_concurrent
    )

    start_time = time.time()
    results = await manager.compress_batch(batch_documents)
    total_time = time.time() - start_time

    # Format results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    result_list = []
    for result in results:
        entry = {
            "file_id": result.file_id,
            "success": result.success,
            "processing_time": round(result.processing_time, 2),
        }

        if result.success:
            if result.result:
                entry["compression_ratio"] = result.result.compression_ratio
                entry["node_count"] = result.result.total_nodes
        else:
            entry["error"] = result.error

        result_list.append(entry)

    # Create response
    response = {
        "status": "complete",
        "directory": validated_dir,
        "patterns": patterns,
        "total_files_found": len(matched_files),
        "files_after_exclusion": len(filtered_files),
        "files_ingested": len(batch_documents),
        "successful": successful,
        "failed": failed,
        "skipped_read_errors": len(skipped_files),
        "total_time": round(total_time, 2),
        "avg_time_per_file": round(total_time / len(results), 2) if results else 0.0,
        "results": result_list,
        "summary": (
            f"Directory ingestion complete: {successful}/{len(batch_documents)} files "
            f"succeeded in {total_time:.1f}s"
        ),
    }

    if skipped_files:
        response["skipped_files"] = skipped_files

    if failed > 0:
        failed_ids = [r.file_id for r in results if not r.success]
        response["failed_file_ids"] = failed_ids

    logger.info(response["summary"])

    return json.dumps(response, indent=2)


# =========================================================================
# Diff Re-ingestion, Cross-doc Dedup, Preset Tools
# =========================================================================


async def handle_diff_reingest(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Re-ingest a document preserving unchanged chunk embeddings."""
    file_id = args.get("file_id")
    text = args.get("text")

    if not file_id or not text:
        return json.dumps({"error": "Both 'file_id' and 'text' are required"}, indent=2)

    try:
        compressor = context["compressor"]
        result = await compressor.diff_reingest_async(file_id, text)

        # Persist updated document to disk (same as handle_ingest)
        try:
            import networkx as nx
            graph_data = nx.node_link_data(compressor.graphs[file_id])
            success = context["persistence"].save_document(
                file_id=file_id,
                chunks={k: v for k, v in compressor.chunks.items() if k.startswith(file_id)},
                graph_data=graph_data,
                metadata=compressor.file_metadata.get(file_id, {}),
            )
            if inspect.isawaitable(success):
                success = await success
            if success:
                logger.info(f"[OK] Persisted diff-reingested document {file_id}")
            else:
                logger.warning(f"[WARN] Failed to persist diff-reingested {file_id}")
        except Exception as e:
            logger.error(f"Failed to persist diff-reingested {file_id}: {e}")

        # Save version history
        try:
            checksum = hashlib.md5(text.encode()).hexdigest()
            await context["version_manager"].add_version_async(
                doc_id=file_id,
                content=text,
                checksum=checksum,
                metadata={},
                compression_stats={
                    "chunks_unchanged": result.chunks_unchanged,
                    "chunks_updated": result.chunks_updated,
                    "chunks_added": result.chunks_added,
                    "chunks_removed": result.chunks_removed,
                },
            )
            logger.info(f"[OK] Registered version for diff-reingested {file_id}")
        except Exception as e:
            logger.warning(f"[WARN] Failed to save version for diff-reingested {file_id}: {e}")

        return json.dumps({
            "status": "success",
            "file_id": result.file_id,
            "chunks_unchanged": result.chunks_unchanged,
            "chunks_updated": result.chunks_updated,
            "chunks_added": result.chunks_added,
            "chunks_removed": result.chunks_removed,
            "message": (
                f"Diff re-ingestion complete: {result.chunks_unchanged} unchanged, "
                f"{result.chunks_updated} updated, {result.chunks_added} added, "
                f"{result.chunks_removed} removed"
            ),
        }, indent=2)
    except ValueError as e:
        return json.dumps({"error": str(e)}, indent=2)
    except Exception as e:
        logger.error(f"Diff re-ingestion failed: {e}")
        return json.dumps({"error": f"Diff re-ingestion failed: {e}"}, indent=2)


async def handle_find_duplicates(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Find near-duplicate chunks across different documents."""
    threshold = args.get("threshold", 0.9)
    timeout_seconds = args.get("timeout_seconds", 30.0)

    try:
        compressor = context["compressor"]
        duplicates = compressor.find_duplicates(threshold=threshold, timeout_seconds=timeout_seconds)
        return json.dumps({
            "status": "success",
            "duplicate_count": len(duplicates),
            "threshold": threshold,
            "duplicates": duplicates[:50],
            "message": f"Found {len(duplicates)} duplicate pairs above {threshold} similarity",
        }, indent=2)
    except Exception as e:
        logger.error(f"Duplicate detection failed: {e}")
        return json.dumps({"error": f"Duplicate detection failed: {e}"}, indent=2)


async def handle_get_presets(context: HandlerContext, args: Dict[str, Any]) -> str:
    """List available compression presets."""
    from ..compression_presets import list_presets
    presets = list_presets()
    return json.dumps({
        "status": "success",
        "presets": [p.to_dict() for p in presets],
        "message": f"{len(presets)} compression presets available",
    }, indent=2)


async def handle_check_context_budget(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Check context budget usage and recommend compression action."""
    current_tokens = args.get("current_tokens")
    if current_tokens is None:
        return json.dumps({"error": "'current_tokens' is required"}, indent=2)
    context_limit = args.get("context_limit", 200_000)

    from ..token_threshold import check_context_budget
    result = check_context_budget(current_tokens, context_limit)
    return json.dumps(result.to_dict(), indent=2)


# =============================================================================
# Phase 5 Handlers — Research-based features (2025 papers)
# =============================================================================


async def handle_prune_by_relevance(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Prune document nodes by query relevance (AttentionRAG)."""
    from ..attention_pruning import prune_by_relevance

    doc_id = args.get("doc_id", "")
    query = args.get("query", "")
    keep_ratio = args.get("keep_ratio", 0.5)

    compressor = context["compressor"]
    if doc_id not in compressor.graphs:
        return json.dumps({"error": f"Document '{doc_id}' not found"})

    # Collect node embeddings for this doc
    node_embeddings = {}
    for nid, node in compressor.chunks.items():
        if nid.startswith(doc_id):
            node_embeddings[nid] = node.embedding

    if not node_embeddings:
        return json.dumps({"error": "No nodes found for document"})

    # Encode query
    from ..embeddings import EmbeddingManager
    emb_mgr = EmbeddingManager()
    query_emb = emb_mgr.encode([query])[0]

    kept_ids = prune_by_relevance(node_embeddings, query_emb, keep_ratio)

    return json.dumps({
        "doc_id": doc_id,
        "total_nodes": len(node_embeddings),
        "kept_nodes": len(kept_ids),
        "kept_node_ids": kept_ids,
        "compression_ratio": round(len(kept_ids) / len(node_embeddings), 3),
    }, indent=2)


async def handle_multi_level_skeleton(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Generate 3-tier skeleton: headline, summary, full (Squeezed Attention)."""
    from ..multi_level_skeleton import generate_multi_level_skeleton

    doc_id = args.get("doc_id", "")
    compressor = context["compressor"]

    if doc_id not in compressor.graphs:
        return json.dumps({"error": f"Document '{doc_id}' not found"})

    # Build node list with importance scores
    nodes = []
    for nid, node in compressor.chunks.items():
        if nid.startswith(doc_id):
            nodes.append({
                "node_id": nid,
                "text": node.text,
                "importance": node.importance,
            })

    result = generate_multi_level_skeleton(nodes)
    return json.dumps({
        "doc_id": doc_id,
        "levels": result,
    }, indent=2)


async def handle_evict_stale(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Evict stale documents based on access recency (DynamicKV/ACON)."""
    max_age_hours = args.get("max_age_hours", 1.0)
    max_age_seconds = max_age_hours * 3600

    compressor = context["compressor"]
    tracker = getattr(compressor, '_access_tracker', None)

    if tracker is None:
        return json.dumps({
            "evicted": [],
            "message": "Access tracking not enabled. Access documents first.",
        })

    stale_ids = tracker.find_stale(max_age_seconds=max_age_seconds)

    evicted = []
    for doc_id in stale_ids:
        if doc_id in compressor.graphs:
            evicted.append(doc_id)

    return json.dumps({
        "stale_documents": stale_ids,
        "evictable": evicted,
        "max_age_hours": max_age_hours,
    }, indent=2)


async def handle_advise_context(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Analyze context and recommend optimal strategy (MCP Best Practices)."""
    from ..context_advisor import advise_context

    compressor = context["compressor"]

    doc_stats = []
    for file_id, graph in compressor.graphs.items():
        total_tokens = sum(
            compressor.chunks[nid].metadata.get("tokens", 0)
            for nid in graph.nodes
            if nid in compressor.chunks
        )
        avg_importance = 0.0
        if graph.nodes:
            avg_importance = sum(
                compressor.chunks[nid].importance
                for nid in graph.nodes
                if nid in compressor.chunks
            ) / len(graph.nodes)

        doc_stats.append({
            "doc_id": file_id,
            "tokens": total_tokens,
            "importance": round(avg_importance, 3),
        })

    advice = advise_context(doc_stats)
    return json.dumps(advice, indent=2)


async def handle_get_compression_insights(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Get compression replay insights (ACON)."""
    compressor = context["compressor"]
    replay_log = getattr(compressor, '_compression_replay', None)

    if replay_log is None:
        return json.dumps({"insights": {}, "message": "No compression history recorded yet."})

    insights = replay_log.get_insights()
    return json.dumps({"insights": insights}, indent=2)


async def handle_generate_rewrite_prompt(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Generate rewrite prompt for client-side LLM compression (SCOPE)."""
    from ..generative_rewrite import generate_rewrite_prompt

    text = args.get("text", "")
    target_ratio = args.get("target_ratio", 0.5)
    preserve_keywords = args.get("preserve_keywords", [])

    if not text:
        doc_id = args.get("doc_id", "")
        compressor = context["compressor"]
        if doc_id in compressor.graphs:
            text = " ".join(
                compressor.chunks[nid].text
                for nid in compressor.graphs[doc_id].nodes
                if nid in compressor.chunks
            )

    prompt = generate_rewrite_prompt(text, target_ratio, preserve_keywords or None)
    return json.dumps(prompt, indent=2)
