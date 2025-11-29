"""
File Sync Handler Functions

This module contains handler functions for file sync and version management:
- check_file_sync: Check if cached document is in sync with source file
- diff_cached_file: Generate unified diff between cached and current file
- refresh_document: Re-ingest document from source file to update cache
- get_version_history: Get version timeline for a document

These handlers implement real-time staleness detection and version history
tracking introduced in v0.4.0.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


async def handle_check_file_sync(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_file_sync tool call.

    Checks if cached document is in sync with source file on disk by
    comparing modification time and checksums.

    Args:
        context: Server context containing sync_manager and validation functions
        args: Tool arguments with 'file_id'

    Returns:
        JSON string with sync status

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    file_id = args["file_id"]
    sync_manager = context["sync_manager"]

    # Validation - check if file exists in sync manager
    if file_id not in sync_manager.file_metadata:
        raise ValueError(f"Document '{file_id}' not found in file sync manager")

    logger.info(f"Checking file sync for: {file_id}")

    status = sync_manager.check_file_sync(file_id)

    # Build JSON response
    response = {"file_id": file_id, "in_sync": status["in_sync"], "reason": status["reason"]}

    # Add optional fields if present
    if "file_path" in status:
        response["file_path"] = status["file_path"]

    if "cached_checksum" in status:
        response["cached_checksum"] = status["cached_checksum"]

    if "current_checksum" in status:
        response["current_checksum"] = status["current_checksum"]

    if "cached_mtime" in status:
        response["cached_mtime"] = status["cached_mtime"]
        response["cached_time"] = datetime.fromtimestamp(status["cached_mtime"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    if "current_mtime" in status:
        response["current_mtime"] = status["current_mtime"]
        response["current_time"] = datetime.fromtimestamp(status["current_mtime"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return json.dumps(response, indent=2)


async def handle_diff_cached_file(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle diff_cached_file tool call.

    Generates unified diff between cached version and current file on disk.
    Shows exactly what changed since ingestion (additions, deletions, modifications).

    v0.8.0 audit fix: Improved UX with detailed error messages from version_manager.

    Args:
        context: Server context containing sync_manager, version_manager, and validation
        args: Tool arguments with 'file_id' and optional 'context_lines'

    Returns:
        Unified diff in git-style format or detailed error message

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    file_id = args["file_id"]
    context_lines = args.get("context_lines", 3)
    sync_manager = context["sync_manager"]
    version_manager = context["version_manager"]

    # Validation - check sync manager registration
    # v0.8.0: Use ASCII-only output for enterprise compatibility
    if file_id not in sync_manager.file_metadata:
        return (
            f"[ERROR] Cannot diff '{file_id}'\n"
            f"\n"
            f"File sync not enabled for this document (no source file registered).\n"
            f"\n"
            f"Tip: Re-ingest with file_path parameter:\n"
            f'  ingest_context(text=..., file_id="{file_id}", file_path="/path/to/file")'
        )

    logger.info(f"Generating diff for: {file_id}")

    # Use version manager to generate diff (v0.8.0: use async version to avoid blocking event loop)
    # Version manager now returns detailed error messages instead of None
    diff = await version_manager.diff_with_current_file_async(file_id, context_lines=context_lines)

    # v0.8.0: version_manager now always returns a string (error message or diff)
    # None only returned for truly unexpected cases
    if not diff:
        return f"[ERROR] Unexpected error generating diff for '{file_id}'"

    return diff


async def handle_refresh_document(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle refresh_document tool call.

    Re-ingests document from source file to update cache with latest changes.
    Stores old version in history (default: keeps last 10 versions).

    Args:
        context: Server context containing all necessary managers and functions
        args: Tool arguments with 'file_id'

    Returns:
        Refresh confirmation with new stats or error message

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    file_id = args["file_id"]
    sync_manager = context["sync_manager"]
    version_manager = context["version_manager"]
    compressor = context["compressor"]
    persistence = context["persistence"]

    # Validation - check if file exists in compressor
    if file_id not in compressor.graphs:
        raise ValueError(f"Document '{file_id}' not found")

    # v0.8.0: Use ASCII-only output for enterprise compatibility
    if file_id not in sync_manager.file_metadata:
        return (
            f"[ERROR] Cannot refresh '{file_id}'\n"
            f"\n"
            f"File sync not enabled for this document (no source file registered).\n"
            f"\n"
            f"Tip: Documents must be ingested with file_path to enable refresh:\n"
            f'  ingest_context(text=..., file_id="{file_id}", file_path="/path/to/file")'
        )

    metadata = sync_manager.file_metadata[file_id]

    if not metadata.file_path:
        return f"[ERROR] Document '{file_id}' has no source file (text-only ingestion)"

    logger.info(f"Refreshing document from disk: {file_id} <- {metadata.file_path}")

    # Read current file
    try:
        with open(metadata.file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return (
            f"[ERROR] Error reading source file\n"
            f"\n"
            f"Path: {metadata.file_path}\n"
            f"Error: {str(e)}\n"
            f"\n"
            f"Tip: File may have been moved or deleted. Check the path and try again."
        )

    # Re-ingest the document
    try:
        skeleton = await compressor.ingest_file_async(
            text=content,
            file_id=file_id,
            metadata={"refreshed_at": __import__("datetime").datetime.now().isoformat()},
        )
    except Exception as e:
        return (
            f"[ERROR] Error re-ingesting document\n"
            f"\n"
            f"Error: {str(e)}\n"
            f"\n"
            f"Tip: Check that the file contains valid content."
        )

    # Update sync metadata
    checksum = hashlib.md5(content.encode()).hexdigest()
    sync_manager.update_metadata(file_id, metadata.file_path, content)

    # Persist file sync metadata
    try:
        metadata_export = sync_manager.export_metadata()
        persistence.save_file_sync_metadata(metadata_export)
    except Exception as e:
        logger.warning(f"Failed to save file sync metadata: {e}")

    # Store new version (v0.8.0: use async version to avoid blocking event loop)
    await version_manager.add_version_async(
        doc_id=file_id,
        content=content,
        checksum=checksum,
        file_path=metadata.file_path,
        compression_stats={
            "total_tokens": skeleton.total_tokens,
            "skeleton_tokens": skeleton.skeleton_tokens,
            "compression_ratio": skeleton.compression_ratio,
        },
    )

    # Update persistence
    try:
        import networkx as nx

        graph_data = nx.node_link_data(compressor.graphs[file_id])
        persistence.save_document(
            file_id=file_id,
            chunks={k: v for k, v in compressor.chunks.items() if k.startswith(file_id)},
            graph_data=graph_data,
            metadata=compressor.file_metadata.get(file_id, {}),
        )
    except Exception as e:
        logger.error(f"Failed to persist refreshed {file_id}: {e}")

    # v0.8.0 audit fix: Use async wrapper to avoid blocking event loop
    history = await version_manager.get_version_history_async(file_id)
    version_count = len(history)

    # v0.8.0: Use ASCII-only output for enterprise compatibility
    return (
        f"[OK] Refreshed '{file_id}' from {metadata.file_path}\n"
        f"\n"
        f"New stats:\n"
        f"  - Total tokens: {skeleton.total_tokens:,}\n"
        f"  - Skeleton tokens: {skeleton.skeleton_tokens:,}\n"
        f"  - Compression: {skeleton.compression_ratio:.1f}x\n"
        f"  - Checksum: {checksum[:8]}...\n"
        f"\n"
        f"Version history: {version_count} versions stored\n"
        f"\n"
        f"Tip: Use diff_cached_file('{file_id}') to see what changed from previous version"
    )


async def handle_get_version_history(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle get_version_history tool call.

    Gets version timeline for a document showing all cached versions with
    timestamps, checksums, file paths, and compression stats. Similar to
    'git log' for cached documents.

    Args:
        context: Server context containing version_manager
        args: Tool arguments with 'doc_id'

    Returns:
        JSON string with version history
    """
    doc_id = args["doc_id"]
    version_manager = context["version_manager"]

    # Get version history (v0.8.0 audit fix: use async wrapper)
    history = await version_manager.get_version_history_async(doc_id)

    # Build JSON response
    versions = []
    for version in history:
        timestamp_raw = version.get("timestamp", None)

        # Format timestamp (convert float to datetime string, remove microseconds)
        if timestamp_raw and isinstance(timestamp_raw, (int, float)):
            timestamp_str = datetime.fromtimestamp(timestamp_raw).isoformat().split(".")[0]
        else:
            timestamp_str = str(timestamp_raw) if timestamp_raw else "Unknown"

        version_info = {
            "version_id": version.get("version_id", "?"),
            "timestamp": timestamp_str,
            "timestamp_raw": timestamp_raw,
            "file_path": version.get("file_path", "N/A"),
            "checksum": version.get("checksum", "N/A"),
            "size_bytes": version.get("size_bytes", 0),
        }

        # Add compression stats if available
        stats = version.get("compression_stats", {})
        if stats:
            version_info["compression_stats"] = {
                "total_tokens": stats.get("total_tokens", 0),
                "skeleton_tokens": stats.get("skeleton_tokens", 0),
                "compression_ratio": stats.get("compression_ratio", 0),
            }

        versions.append(version_info)

    response = {"doc_id": doc_id, "total_versions": len(versions), "versions": versions}

    return json.dumps(response, indent=2)
