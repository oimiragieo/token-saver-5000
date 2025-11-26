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

Version: 0.4.1
"""

import json
import logging
from typing import Any, Dict, List
import hashlib

from ..types import HandlerContext  # TypedDict for handler context
from ..semantic_compressor import FidelityLevel
from ..fidelity_advisor import FidelityAdvisor, UseCase
from ..error_helpers import SmartError
from ..compression_advisor import CompressionAdvisor


logger = logging.getLogger("semantic-modulator")


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
        if file_id not in context["compressor"].chunks:
            # Extract unique file IDs from all node IDs
            available = list(
                set([nid.split("_n")[0] for nid in context["compressor"].chunks.keys()])
            )
            if not available:
                raise ValueError(
                    f"Document '{file_id}' not found. No documents ingested yet.\n"
                    "💡 Tip: Use ingest_context() to add documents first."
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
        # Extract file_id from first node to give better error message
        file_id = node_ids[0].rsplit("_n", 1)[0] if "_n" in node_ids[0] else "unknown"
        valid_nodes = [
            nid for nid in context["compressor"].chunks.keys() if nid.startswith(file_id)
        ]

        if not valid_nodes:
            raise ValueError(
                f"Invalid node IDs: {invalid_nodes[:3]}\n"
                f"   No nodes found for '{file_id}'. Document may not be ingested.\n"
                f"💡 Tip: Use ingest_context() to add the document first."
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
            "💡 Tip: Provide a positive number (e.g., 10000 for 10k tokens available)"
        )

    if max_tokens is not None and available_tokens > max_tokens:
        raise ValueError(
            f"available_tokens ({available_tokens}) exceeds max_tokens ({max_tokens})\n"
            "💡 Tip: available_tokens should be ≤ max_tokens"
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
        Success message with compression stats

    Raises:
        ValueError: If validation fails
        RuntimeError: If ingestion fails
    """
    text = args["text"]
    file_id = args["file_id"]
    file_path = args.get("file_path")  # Optional file path for sync tracking
    metadata = args.get("metadata")

    # SECURITY: Validate file_path to prevent path traversal (CWE-22)
    if file_path:
        try:
            # PathValidator resolves .., symlinks, and validates against allowed directories
            file_path = context["path_validator"].validate(file_path)
            logger.info(f"File path validated: {file_path}")
        except ValueError as e:
            raise ValueError(
                f"Invalid file_path: {str(e)}\n"
                "💡 Security: File paths must be within allowed directories to prevent path traversal attacks"
            ) from e

    # Validation
    if not text or len(text.strip()) == 0:
        raise ValueError(
            "text cannot be empty\n"
            "💡 Tip: Provide document content to ingest (minimum ~20 characters recommended)"
        )

    if len(text) < 20:
        raise ValueError(
            f"text is too short ({len(text)} chars)\n"
            "💡 Tip: Provide at least 20 characters for meaningful semantic analysis"
        )

    validate_file_id(file_id, context, must_exist=False)

    # Check resource limits BEFORE ingestion
    text_size = len(text.encode("utf-8"))
    allowed, error_msg = context["resource_manager"].check_document_size(file_id, text_size)
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
        skeleton = await context["compressor"].ingest_file_async(text, file_id, metadata)
    except Exception as e:
        raise RuntimeError(
            f"Failed to ingest document: {str(e)}\n"
            "💡 Tip: Check that text is valid and file_id contains only alphanumeric and underscores"
        ) from e

    # Register with resource manager
    context["resource_manager"].register_document(file_id, text_size)

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
        if success:
            logger.info(f"✅ Persisted document {file_id}")
        else:
            logger.warning(f"⚠️  Failed to persist {file_id}, will be lost on restart")
    except Exception as e:
        logger.error(f"Failed to persist {file_id}: {e}")

    # NEW: Register with file sync manager and version manager
    checksum = hashlib.md5(text.encode()).hexdigest()
    context["sync_manager"].register_file(file_id, file_path, text)
    context["version_manager"].add_version(
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
    logger.info(f"✅ Registered file sync tracking for {file_id}")

    # Save file sync metadata to persistence
    try:
        metadata_export = context["sync_manager"].export_metadata()
        success = context["persistence"].save_file_sync_metadata(metadata_export)
        if success:
            logger.info(f"✅ Saved file sync metadata for {len(metadata_export)} documents")
        else:
            logger.warning("⚠️  Failed to save file sync metadata")
    except Exception as e:
        logger.error(f"Failed to save file sync metadata: {e}")

    # Initialize retrieval history
    context["retrieval_history"][file_id] = []

    file_sync_note = ""
    if file_path:
        file_sync_note = f"\n🔄 File sync enabled: Tracking {file_path}\n   Version 1 stored. Use check_file_sync('{file_id}') to monitor changes."

    # Compare estimate vs actual (v0.4.1+)
    actual_ratio = skeleton.compression_ratio
    estimate_accuracy = (
        "excellent"
        if abs(actual_ratio - estimate.compression_ratio) < 2
        else "good" if abs(actual_ratio - estimate.compression_ratio) < 5 else "fair"
    )

    result = f"""
✅ Document ingested successfully!

File ID: {file_id} containing {skeleton.total_nodes} semantic nodes
Original tokens: {skeleton.total_tokens:,}
Skeleton tokens: {skeleton.skeleton_tokens:,}
Compression ratio: {skeleton.compression_ratio:.1f}x (estimated: {estimate.compression_ratio:.1f}x - {estimate_accuracy} prediction)

📊 Token savings: {skeleton.total_tokens - skeleton.skeleton_tokens:,} tokens ({(1 - skeleton.skeleton_tokens / skeleton.total_tokens) * 100:.1f}%){file_sync_note}

💡 IMPORTANT: Use read_skeleton('{file_id}') to view the semantic map BEFORE requesting specific details.
   This "map before territory" approach ensures you understand the document structure.

Next steps:
1. read_skeleton('{file_id}') - View the compressed structure (recommended first step)
2. modulate_region() - Retrieve specific sections at chosen fidelity
3. search_semantic() - Find relevant content via vector similarity
4. check_blind_spots() - Verify response completeness after generating answers

{skeleton.skeleton_text[:500]}...
(Use read_skeleton to see full structure)
"""
    return result


async def handle_read_skeleton(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle read_skeleton tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing file_id

    Returns:
        Skeleton text with optional staleness warning

    Raises:
        RuntimeError: If reading skeleton fails
    """
    file_id = args["file_id"]
    validate_file_id(file_id, context, must_exist=True)

    logger.info(f"Reading skeleton: {file_id}")

    # NEW: Check file sync status before reading
    warning = ""
    if file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(file_id)
        if not status["in_sync"]:
            time_info = ""
            if "current_mtime" in status and "cached_mtime" in status:
                from datetime import datetime

                cached_time = datetime.fromtimestamp(status["cached_mtime"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                current_time = datetime.fromtimestamp(status["current_mtime"]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                time_info = f"\nCached: {cached_time}\nCurrent: {current_time}"

            warning = f"""
⚠️  WARNING: Cache may be stale!

{status['reason']}{time_info}

💡 Use refresh_document('{file_id}') to update
💡 Use diff_cached_file('{file_id}') to see changes

Proceeding with cached version...
---

"""

    try:
        skeleton_text = context["compressor"].read_skeleton(file_id)
        return warning + skeleton_text
    except Exception as e:
        raise RuntimeError(
            f"Failed to read skeleton for '{file_id}': {str(e)}\n"
            f"💡 Tip: Verify the document was ingested successfully with get_stats()"
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
    # Extract file_id from first node (format: file_id_n123)
    file_id = "_".join(node_ids[0].split("_")[:-1]) if node_ids else None
    warning = ""
    if file_id and file_id in context["sync_manager"].file_metadata:
        status = context["sync_manager"].check_file_sync(file_id)
        if not status["in_sync"]:
            warning = f"""
⚠️  WARNING: Cache may be stale for '{file_id}'!

{status['reason']}

💡 Use refresh_document('{file_id}') to update

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
            f"💡 Valid levels: {valid_levels}\n"
            f"   ABSTRACT: ~10 tokens (summary only)\n"
            f"   OUTLINE: ~30 tokens (summary + section markers)\n"
            f"   STRUCTURE: ~50 tokens (headers + entities)\n"
            f"   DETAILED: ~100 tokens (summary + excerpts)\n"
            f"   RAW: Full original text"
        )

    logger.info(f"Modulating {len(node_ids)} nodes at {fidelity_str} fidelity")

    # Track retrieval for blind spot detection
    for node_id in node_ids:
        # Extract file_id from node_id (format: file_id_n123)
        file_id = "_".join(node_id.split("_")[:-1])
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
            f"💡 Tip: Verify node IDs are valid with read_skeleton()"
        ) from e


async def handle_search_semantic(context: HandlerContext, args: Dict[str, Any]) -> str:
    """Handle search_semantic tool call.

    Args:
        context: Server context dict
        args: Tool arguments containing query, optional file_id and top_k

    Returns:
        Formatted search results
    """
    query = args["query"]
    file_id = args.get("file_id")
    top_k = args.get("top_k", 5)

    logger.info(f"Semantic search: '{query}' in {file_id or 'all files'}")

    node_ids = context["compressor"].search_semantic(query, file_id, top_k)

    result_lines = [f"🔍 Semantic Search Results for: '{query}'"]
    result_lines.append(f"Found {len(node_ids)} relevant nodes:\n")

    for i, node_id in enumerate(node_ids, 1):
        node = context["compressor"].chunks[node_id]
        summary = context["compressor"]._generate_summary(node.text, max_length=100)
        result_lines.append(f"{i}. [{node_id}]")
        result_lines.append(f"   Importance: {node.importance:.3f}")
        result_lines.append(f"   Summary: {summary}\n")

    result_lines.append(f"💡 Tip: Use modulate_region({node_ids[:3]}) to retrieve full content")

    return "\n".join(result_lines)


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
📊 Document Statistics: {file_id}

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
📊 Global Statistics

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

    # Get all unique file_ids from chunks
    file_ids = list(set([nid.split("_n")[0] for nid in context["compressor"].chunks.keys()]))

    if not file_ids:
        return """
📚 Document Inventory

No documents ingested yet.

💡 Use ingest_context(text, file_id) to add documents.
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
    result_lines = ["📚 Document Inventory\n"]
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

    result_lines.append("💡 Next steps:")
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
⚠️  DELETE CONFIRMATION REQUIRED

You are about to delete document: {file_id}

This will:
  • Remove all {len([k for k in context['compressor'].chunks.keys() if k.startswith(file_id)])} semantic nodes from memory
  • Delete persistent storage (cannot be undone)
  • Clear retrieval history for this document

To proceed, call again with confirm=true:
  delete_document(file_id="{file_id}", confirm=true)

💡 Tip: Use list_documents() to see all available documents first
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

        logger.info(f"✅ Removed {file_id} from memory ({node_count} nodes)")

    except Exception as e:
        logger.error(f"Failed to delete {file_id} from memory: {e}")
        raise RuntimeError(f"Failed to delete from memory: {e}")

    # Delete from persistent storage
    try:
        success = context["persistence"].delete_document(file_id)
        if success:
            logger.info(f"✅ Deleted {file_id} from persistent storage")
        else:
            logger.warning(f"⚠️  Failed to delete {file_id} from persistent storage")
    except Exception as e:
        logger.error(f"Failed to delete {file_id} from storage: {e}")

    # Unregister from resource manager
    try:
        context["resource_manager"].unregister_document(file_id)
    except Exception as e:
        logger.warning(f"Failed to unregister {file_id} from resource manager: {e}")

    # NEW: Clean up file sync metadata and version history
    try:
        context["sync_manager"].remove_metadata(file_id)
        context["version_manager"].delete_versions(file_id)
        # Save metadata after removal
        metadata_export = context["sync_manager"].export_metadata()
        context["persistence"].save_file_sync_metadata(metadata_export)
        logger.info(f"✅ Cleaned up file sync metadata and version history for {file_id}")
    except Exception as e:
        logger.warning(f"Failed to clean up file sync data for {file_id}: {e}")

    return f"""
🗑️ Document Deleted Successfully

File ID: {file_id}
Nodes removed: {node_count}
Memory freed: ~{node_count * 2}KB (estimated)

✅ Document has been permanently deleted from:
   • Memory (semantic graph, chunks, metadata)
   • Persistent storage (ChromaDB/JSON)
   • Resource tracking

💡 Remaining documents: {len(set([nid.split('_n')[0] for nid in context['compressor'].chunks.keys()]))}
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
            "💡 Tip: 0.0 = low priority, 0.5 = medium, 1.0 = high priority"
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
            "💡 Tip: This is a JSCCM-inspired feature. Check that the document exists and token counts are valid."
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
            "💡 Tip: This JSCCM-inspired feature requires Main + Auxiliary + Detail branches.\n"
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
            "💡 Tip: Specify how many nodes you plan to retrieve"
        )

    if num_nodes > 1000:
        raise ValueError(
            f"num_nodes is very high ({num_nodes})\n"
            "💡 Tip: Consider retrieving fewer nodes for better token efficiency.\n"
            "   Most queries work well with 3-10 nodes."
        )

    if token_budget is not None:
        if token_budget < 10:
            raise ValueError(
                f"token_budget is too low ({token_budget})\n"
                "💡 Tip: Even ABSTRACT fidelity needs ~10 tokens per node.\n"
                f"   For {num_nodes} nodes, minimum budget: {num_nodes * 10} tokens"
            )

        if token_budget > 1_000_000:
            raise ValueError(
                f"token_budget is very high ({token_budget:,})\n"
                "💡 Tip: Most use cases work well with 100-10,000 token budgets."
            )

    if query_complexity not in ["simple", "medium", "complex"]:
        raise ValueError(
            f"query_complexity must be 'simple', 'medium', or 'complex', got '{query_complexity}'\n"
            "💡 Tip: Use 'medium' if unsure"
        )

    # Convert string to enum
    try:
        use_case_enum = UseCase(use_case_str)
    except ValueError:
        valid_cases = [case.value for case in UseCase]
        raise ValueError(
            f"Unknown use_case: '{use_case_str}'\n" f"💡 Valid options: {', '.join(valid_cases)}"
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
    Handle batch_ingest_documents MCP tool (v0.6.0).

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
    """
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
            "💡 Tip: Provide a list of objects with 'file_id' and 'text' fields"
        )

    if not (1 <= max_concurrent <= 8):
        raise ValueError(
            f"max_concurrent must be between 1 and 8, got {max_concurrent}\n"
            "💡 Tip: Use 4 for balanced performance, 1 for sequential, 8 for maximum throughput"
        )

    # Validate each document
    batch_documents = []
    for i, doc in enumerate(documents_list):
        if not isinstance(doc, dict):
            raise ValueError(
                f"Document {i} must be an object, got {type(doc).__name__}\n"
                "💡 Tip: Each document should have 'file_id' and 'text' fields"
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
            f"⚠️ {failed} documents failed. Check error messages for each failed document."
        )

    logger.info(response["summary"])

    return json.dumps(response, indent=2)
