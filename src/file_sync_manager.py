"""
File Sync Manager for Semantic Modulator

Tracks file changes and detects when cached documents are out of sync
with their source files on disk.

Features:
- File modification time tracking
- Content hash (MD5) verification
- Automatic staleness detection
- Diff generation (cached vs current)
- Sync/refresh capabilities
- Automatic LRU eviction when max entries exceeded (v0.4.2)
"""

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

from .constants import MAX_FILE_SYNC_ENTRIES

logger = logging.getLogger("file_sync")


@dataclass
class FileMetadata:
    """Metadata about a cached file"""

    file_path: Optional[str]  # Path to source file (None if text-only)
    mtime: Optional[float]  # Last modification time
    checksum: Optional[str]  # MD5 hash of content
    ingestion_time: float  # When it was ingested
    size_bytes: int  # Original file size
    doc_id: str  # Document ID in MCP
    source_type: str = "file"  # file, web, github, s3, slack_export, etc.
    source_id: Optional[str] = None  # Connector/source identifier


class FileSyncManager:
    """
    Manages synchronization between cached documents and source files.

    Detects when files have changed on disk since ingestion and provides
    tools to diff, refresh, and track file versions.

    **Automatic LRU Eviction (v0.4.2):**
    - Keeps only the last N file metadata entries (configurable via max_entries)
    - Eviction based on ingestion_time (least recently ingested removed first)
    - Prevents unbounded memory growth on long-running servers

    Args:
        max_entries: Maximum file metadata entries to track (0 = unlimited)
                     Default: MAX_FILE_SYNC_ENTRIES (1000 entries)
    """

    def __init__(self, max_entries: int = MAX_FILE_SYNC_ENTRIES):
        """
        Initialize file sync manager.

        Args:
            max_entries: Maximum metadata entries to track (0 = unlimited)
        """
        self.file_metadata: Dict[str, FileMetadata] = {}  # doc_id -> metadata
        self._lock = threading.Lock()  # Thread safety for concurrent access
        self.max_entries = max_entries  # LRU eviction limit
        logger.info(
            f"FileSyncManager initialized "
            f"(max_entries={max_entries if max_entries > 0 else 'unlimited'})"
        )

    def register_file(
        self,
        doc_id: str,
        file_path: Optional[str],
        content: str,
        *,
        source_type: str = "file",
        source_id: Optional[str] = None,
    ) -> FileMetadata:
        """
        Register a file when it's ingested with automatic LRU eviction (v0.4.2).

        **Security (v0.6.1):**
        - file_path MUST be an absolute path (validated by PathValidator at entry point)
        - This is a defense-in-depth check to prevent path traversal attacks (CWE-22)

        **Automatic LRU Eviction:**
        - If max_entries > 0 and entry count exceeds limit, evicts oldest entry
        - Eviction based on ingestion_time (least recently ingested removed first)
        - Eviction happens automatically after adding new entry

        Args:
            doc_id: Document ID in MCP
            file_path: Path to source file (None if text-only, MUST be absolute if provided)
            content: The content that was ingested

        Returns:
            FileMetadata object

        Raises:
            ValueError: If file_path is not absolute (security misconfiguration)
        """
        # SECURITY CHECK: Verify file_path is absolute (defense-in-depth)
        # Path validation should happen at entry point (handle_ingest), but we verify here
        # to prevent bugs if file_sync_manager is called directly
        if file_path and not os.path.isabs(file_path):
            raise ValueError(
                f"Security violation: file_path must be absolute (got: {file_path})\n"
                "This indicates a security misconfiguration. File paths must be validated "
                "by PathValidator before being passed to FileSyncManager."
            )

        # Calculate checksum (outside lock - no shared state)
        checksum = self._calculate_checksum(content)

        # Get modification time if file exists (outside lock - I/O operation)
        mtime = None
        size_bytes = len(content.encode("utf-8"))

        if file_path and os.path.exists(file_path):
            stat = os.stat(file_path)
            mtime = stat.st_mtime
            size_bytes = stat.st_size

        metadata = FileMetadata(
            file_path=file_path,
            mtime=mtime,
            checksum=checksum,
            ingestion_time=datetime.now().timestamp(),
            size_bytes=size_bytes,
            doc_id=doc_id,
            source_type=source_type,
            source_id=source_id,
        )

        # Thread-safe: protect dictionary write and eviction
        evicted_doc_id = None
        with self._lock:
            self.file_metadata[doc_id] = metadata

            # Automatic LRU eviction (v0.4.2)
            if self.max_entries > 0 and len(self.file_metadata) > self.max_entries:
                # Find oldest entry by ingestion_time
                oldest_doc_id = min(
                    self.file_metadata.keys(),
                    key=lambda k: self.file_metadata[k].ingestion_time,
                )
                del self.file_metadata[oldest_doc_id]
                evicted_doc_id = oldest_doc_id

        log_msg = f"Registered file: {doc_id} (path={file_path}, checksum={checksum[:8]}...)"
        if evicted_doc_id:
            log_msg += f", evicted oldest: {evicted_doc_id}"
        logger.info(log_msg)

        return metadata

    def check_file_sync(self, doc_id: str) -> Dict[str, any]:
        """
        Check if cached document is in sync with source file.

        Args:
            doc_id: Document ID to check

        Returns:
            Dictionary with sync status:
            - in_sync: bool
            - reason: str (if not in sync)
            - current_mtime: float
            - cached_mtime: float
            - current_checksum: str
            - cached_checksum: str
        """
        # Thread-safe: lock to read metadata
        with self._lock:
            if doc_id not in self.file_metadata:
                return {
                    "in_sync": False,
                    "reason": "Document not registered (no metadata)",
                    "has_source_file": False,
                }

            metadata = self.file_metadata[doc_id]

            # If no source file, can't check sync
            if not metadata.file_path:
                return {
                    "in_sync": True,
                    "reason": (
                        "No source file (text-only document)"
                        if metadata.source_type == "file"
                        else f"External source registered via {metadata.source_type}"
                    ),
                    "has_source_file": False,
                    "source_type": metadata.source_type,
                    "source_id": metadata.source_id,
                }

            # Copy values we need for I/O (outside lock)
            file_path = metadata.file_path
            cached_mtime = metadata.mtime
            cached_checksum = metadata.checksum

        # I/O operations outside lock (don't block other threads)
        # Check if file still exists
        if not os.path.exists(file_path):
            return {
                "in_sync": False,
                "reason": f"Source file deleted: {file_path}",
                "has_source_file": False,
            }

        # Get current file stats
        current_stat = os.stat(file_path)
        current_mtime = current_stat.st_mtime

        # Quick check: modification time
        if current_mtime != cached_mtime:
            # File modified, but check content hash to be sure
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    current_content = f.read()
                current_checksum = self._calculate_checksum(current_content)

                if current_checksum != cached_checksum:
                    return {
                        "in_sync": False,
                        "reason": "File content changed on disk",
                        "has_source_file": True,
                        "current_mtime": current_mtime,
                        "cached_mtime": cached_mtime,
                        "current_checksum": current_checksum,
                        "cached_checksum": cached_checksum,
                        "file_path": file_path,
                    }
                else:
                    # mtime changed but content same (e.g., touch command)
                    # Update mtime but consider in sync
                    # Thread-safe: lock to modify metadata
                    with self._lock:
                        if doc_id in self.file_metadata:
                            self.file_metadata[doc_id].mtime = current_mtime
                    return {
                        "in_sync": True,
                        "reason": "Content unchanged (mtime diff only)",
                        "has_source_file": True,
                    }
            except Exception as e:
                return {
                    "in_sync": False,
                    "reason": f"Error reading file: {e}",
                    "has_source_file": True,
                }

        # File unchanged
        return {
            "in_sync": True,
            "reason": "File unchanged since ingestion",
            "has_source_file": True,
            "mtime": cached_mtime,
            "checksum": cached_checksum,
        }

    def get_file_diff(
        self, doc_id: str, context_lines: int = 3, version_manager=None
    ) -> Optional[str]:
        """
        Generate a diff between cached content and current file.

        **v0.8.0 Update (Issue 6 fix):**
        - Now properly integrates with VersionManager for cached content retrieval
        - Pass version_manager parameter to enable full diff functionality
        - Without version_manager, returns deprecation notice

        Args:
            doc_id: Document ID
            context_lines: Number of context lines in diff
            version_manager: Optional VersionManager instance for retrieving cached content

        Returns:
            Unified diff string, or None if no diff/error

        Example:
            >>> diff = sync_manager.get_file_diff("my_doc", version_manager=version_manager)
        """
        # Thread-safe: lock to read metadata
        with self._lock:
            if doc_id not in self.file_metadata:
                return None

            metadata = self.file_metadata[doc_id]
            file_path = metadata.file_path

        # Check outside lock
        if not file_path or not os.path.exists(file_path):
            return None

        try:
            # v0.8.0: Use VersionManager to get cached content (Issue 6 fix)
            if version_manager is not None:
                # Delegate to version_manager which has full content stored
                diff = version_manager.diff_with_current_file(doc_id, context_lines=context_lines)
                return diff

            # Legacy path: No version_manager provided
            # Return deprecation notice with instructions
            return (
                "[WARN]  Diff requires VersionManager integration.\n\n"
                "To generate diffs, use one of these approaches:\n\n"
                "1. Pass version_manager parameter:\n"
                f"   sync_manager.get_file_diff('{doc_id}', version_manager=vm)\n\n"
                "2. Use version_manager directly:\n"
                f"   version_manager.diff_with_current_file('{doc_id}')\n\n"
                "3. Use the MCP tool:\n"
                f"   diff_cached_file(file_id='{doc_id}')\n"
            )

        except Exception as e:
            logger.error(f"Error generating diff for {doc_id}: {e}")
            return None

    def get_stale_documents(self) -> List[str]:
        """
        Get list of document IDs that are out of sync.

        Returns:
            List of doc_ids with stale cache
        """
        # Thread-safe: copy keys while holding lock to avoid iteration errors
        with self._lock:
            doc_ids = list(self.file_metadata.keys())

        # Check sync outside lock (calls check_file_sync which has its own locking)
        stale = []
        for doc_id in doc_ids:
            sync_status = self.check_file_sync(doc_id)
            if not sync_status["in_sync"]:
                stale.append(doc_id)

        return stale

    def get_sync_summary(self) -> Dict[str, any]:
        """
        Get summary of all document sync statuses.

        Returns:
            Summary dictionary with counts and details
        """
        # Thread-safe: copy items while holding lock
        with self._lock:
            total = len(self.file_metadata)
            items = list(self.file_metadata.items())

        # Process outside lock (calls check_file_sync which has its own locking)
        in_sync = 0
        out_of_sync = 0
        no_source = 0
        details = []

        for doc_id, metadata in items:
            status = self.check_file_sync(doc_id)

            detail = {
                "doc_id": doc_id,
                "file_path": metadata.file_path,
                "in_sync": status["in_sync"],
                "reason": status.get("reason", ""),
                "source_type": metadata.source_type,
                "source_id": metadata.source_id,
            }
            details.append(detail)

            if not status.get("has_source_file"):
                no_source += 1
            elif status["in_sync"]:
                in_sync += 1
            else:
                out_of_sync += 1

        return {
            "total_documents": total,
            "in_sync": in_sync,
            "out_of_sync": out_of_sync,
            "no_source_file": no_source,
            "details": details,
        }

    def _calculate_checksum(self, content: str) -> str:
        """
        Calculate MD5 checksum of content.

        Args:
            content: Text content

        Returns:
            MD5 hex digest
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def compute_checksum(self, content: str) -> str:
        """
        Public API: Calculate MD5 checksum of content.

        This is a public alias for _calculate_checksum() for external use.

        Args:
            content: Text content

        Returns:
            MD5 hex digest
        """
        return self._calculate_checksum(content)

    def update_metadata(self, doc_id: str, file_path: str, content: str):
        """
        Update metadata after re-ingestion.

        Args:
            doc_id: Document ID
            file_path: Updated file path
            content: Updated content
        """
        self.register_file(doc_id, file_path, content)
        logger.info(f"Updated metadata for {doc_id}")

    def remove_metadata(self, doc_id: str):
        """
        Remove metadata when document is deleted.

        Args:
            doc_id: Document ID to remove
        """
        # Thread-safe: lock for dictionary deletion
        with self._lock:
            if doc_id in self.file_metadata:
                del self.file_metadata[doc_id]
                logger.info(f"Removed metadata for {doc_id}")

    def export_metadata(self) -> Dict[str, Dict]:
        """
        Export all metadata for persistence.

        Returns:
            Dictionary of doc_id -> metadata dict
        """
        # Thread-safe: lock while iterating and copying metadata
        with self._lock:
            return {
                doc_id: {
                    "file_path": meta.file_path,
                    "mtime": meta.mtime,
                    "checksum": meta.checksum,
                    "ingestion_time": meta.ingestion_time,
                    "size_bytes": meta.size_bytes,
                    "doc_id": meta.doc_id,
                    "source_type": meta.source_type,
                    "source_id": meta.source_id,
                }
                for doc_id, meta in self.file_metadata.items()
            }

    def import_metadata(self, metadata_dict: Dict[str, Dict]):
        """
        Import metadata from persistence.

        Args:
            metadata_dict: Dictionary of metadata to import
        """
        # Thread-safe: lock while modifying dictionary
        with self._lock:
            for doc_id, meta_dict in metadata_dict.items():
                self.file_metadata[doc_id] = FileMetadata(**meta_dict)

        logger.info(f"Imported metadata for {len(metadata_dict)} documents")

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about file sync metadata with LRU eviction info (v0.4.2).

        Returns:
            Dictionary with stats including max_entries configuration
        """
        with self._lock:
            total_entries = len(self.file_metadata)
            total_size_bytes = sum(meta.size_bytes for meta in self.file_metadata.values())

            # Find oldest and newest entries
            if self.file_metadata:
                oldest_time = min(meta.ingestion_time for meta in self.file_metadata.values())
                newest_time = max(meta.ingestion_time for meta in self.file_metadata.values())
            else:
                oldest_time = None
                newest_time = None

            # Count files with/without source paths
            with_source = sum(1 for meta in self.file_metadata.values() if meta.file_path)
            text_only = total_entries - with_source

        return {
            "total_entries": total_entries,
            "max_entries_limit": self.max_entries if self.max_entries > 0 else "unlimited",
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "entries_with_source_file": with_source,
            "text_only_entries": text_only,
            "oldest_entry_time": oldest_time,
            "newest_entry_time": newest_time,
            "approaching_limit": (
                total_entries / self.max_entries > 0.9 if self.max_entries > 0 else False
            ),
        }
