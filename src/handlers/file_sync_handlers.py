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
import logging
from datetime import datetime
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


def handle_check_file_sync(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_file_sync tool call.

    Checks if cached document is in sync with source file on disk by
    comparing modification time and checksums.

    Args:
        context: Server context containing sync_manager and validation functions
        args: Tool arguments with 'file_id'

    Returns:
        Sync status report with recommendations

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    file_id = args["file_id"]
    sync_manager = context["sync_manager"]
    validate_file_id = context["validate_file_id"]

    # Validation
    validate_file_id(file_id, must_exist=True)

    logger.info(f"Checking file sync for: {file_id}")

    status = sync_manager.check_file_sync(file_id)

    if status["in_sync"]:
        return f"""
✅ {file_id} is in sync with source file

{status['reason']}
"""

    # Out of sync
    result = [f"⚠️  {file_id} is OUT OF SYNC\n"]
    result.append(f"Reason: {status['reason']}")

    if "file_path" in status:
        result.append(f"Source: {status['file_path']}")

    if "cached_checksum" in status and "current_checksum" in status:
        result.append(f"\nCached checksum:  {status['cached_checksum'][:8]}...")
        result.append(f"Current checksum: {status['current_checksum'][:8]}...")

    if "cached_mtime" in status and "current_mtime" in status:
        cached_time = datetime.fromtimestamp(status["cached_mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        current_time = datetime.fromtimestamp(status["current_mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        result.append(f"\nCached: {cached_time}")
        result.append(f"Current: {current_time}")

    result.append(f"\n💡 Use refresh_document('{file_id}') to update cache")
    result.append(f"💡 Use diff_cached_file('{file_id}') to see changes")

    return "\n".join(result)


def handle_diff_cached_file(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle diff_cached_file tool call.

    Generates unified diff between cached version and current file on disk.
    Shows exactly what changed since ingestion (additions, deletions, modifications).

    Args:
        context: Server context containing sync_manager, version_manager, and validation
        args: Tool arguments with 'file_id' and optional 'context_lines'

    Returns:
        Unified diff in git-style format or error message

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    file_id = args["file_id"]
    context_lines = args.get("context_lines", 3)
    sync_manager = context["sync_manager"]
    version_manager = context["version_manager"]
    validate_file_id = context["validate_file_id"]

    # Validation
    validate_file_id(file_id, must_exist=True)

    if file_id not in sync_manager.file_metadata:
        return f"""
❌ Cannot diff {file_id}

File sync not enabled for this document (no source file registered).

💡 Tip: Re-ingest with file_path parameter:
   ingest_context(text=..., file_id="{file_id}", file_path="/path/to/file")
"""

    logger.info(f"Generating diff for: {file_id}")

    # Use version manager to generate diff
    diff = version_manager.diff_with_current_file(file_id, context_lines=context_lines)

    if not diff:
        return f"❌ Cannot generate diff for {file_id} (no source file or version history)"

    return diff


def handle_refresh_document(context: HandlerContext, args: Dict[str, Any]) -> str:
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
    validate_file_id = context["validate_file_id"]
    save_file_sync_metadata = context["save_file_sync_metadata"]

    # Validation
    validate_file_id(file_id, must_exist=True)

    if file_id not in sync_manager.file_metadata:
        return f"""
❌ Cannot refresh {file_id}

File sync not enabled for this document (no source file registered).

💡 Tip: Documents must be ingested with file_path to enable refresh:
   ingest_context(text=..., file_id="{file_id}", file_path="/path/to/file")
"""

    metadata = sync_manager.file_metadata[file_id]

    if not metadata.file_path:
        return f"❌ Document {file_id} has no source file (text-only ingestion)"

    logger.info(f"Refreshing document from disk: {file_id} <- {metadata.file_path}")

    # Read current file
    try:
        with open(metadata.file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"""
❌ Error reading source file

Path: {metadata.file_path}
Error: {str(e)}

💡 File may have been moved or deleted. Check the path and try again.
"""

    # Re-ingest the document
    try:
        skeleton = compressor.ingest_file(
            text=content,
            file_id=file_id,
            metadata={"refreshed_at": __import__("datetime").datetime.now().isoformat()},
        )
    except Exception as e:
        return f"""
❌ Error re-ingesting document

Error: {str(e)}

💡 Check that the file contains valid content.
"""

    # Update sync metadata
    checksum = hashlib.md5(content.encode()).hexdigest()
    sync_manager.update_metadata(file_id, metadata.file_path, content)
    save_file_sync_metadata()  # Persist the update

    # Store new version
    version_manager.add_version(
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

    version_count = len(version_manager.get_version_history(file_id))

    return f"""
✅ Refreshed {file_id} from {metadata.file_path}

New stats:
  • Total tokens: {skeleton.total_tokens:,}
  • Skeleton tokens: {skeleton.skeleton_tokens:,}
  • Compression: {skeleton.compression_ratio:.1f}x
  • Checksum: {checksum[:8]}...

Version history: {version_count} versions stored

💡 Use diff_cached_file('{file_id}') to see what changed from previous version
"""


def handle_get_version_history(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle get_version_history tool call.

    Gets version timeline for a document showing all cached versions with
    timestamps, checksums, file paths, and compression stats. Similar to
    'git log' for cached documents.

    Args:
        context: Server context containing version_manager
        args: Tool arguments with 'doc_id'

    Returns:
        Formatted version history timeline
    """
    doc_id = args["doc_id"]
    version_manager = context["version_manager"]

    # Get version history
    history = version_manager.get_version_history(doc_id)

    if not history:
        return f"""
📜 No version history for '{doc_id}'

This document either:
  • Has not been ingested yet
  • Was ingested without file_path (text-only mode)
  • Has no saved versions

💡 Tip: Documents ingested with file_path automatically track versions when refreshed.
"""

    # Format timeline view
    lines = [f"📜 Version History: {doc_id}", "=" * 70, ""]

    for version in history:
        timestamp_raw = version.get("timestamp", None)
        version_id = version.get("version_id", "?")
        file_path = version.get("file_path", "N/A")
        checksum = version.get("checksum", "N/A")
        size = version.get("size_bytes", 0)
        stats = version.get("compression_stats", {})

        # Format timestamp (convert float to datetime string, remove microseconds)
        if timestamp_raw and isinstance(timestamp_raw, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp_raw).isoformat().split(".")[0]
        else:
            timestamp = str(timestamp_raw) if timestamp_raw else "Unknown"

        lines.append(f"Version {version_id} - {timestamp}")
        lines.append(f"  File: {file_path}")
        lines.append(f"  Checksum: {checksum[:16]}...")
        lines.append(f"  Size: {size:,} bytes")

        if stats:
            total_tokens = stats.get("total_tokens", 0)
            skeleton_tokens = stats.get("skeleton_tokens", 0)
            compression = stats.get("compression_ratio", 0)
            lines.append(
                f"  Compression: {total_tokens:,} → {skeleton_tokens:,} tokens ({compression:.1f}x)"
            )

        lines.append("")

    lines.append("=" * 70)
    lines.append(f"Total versions: {len(history)}")
    lines.append("")
    lines.append("💡 Use diff_cached_file() to compare cached vs current file")

    return "\n".join(lines)
