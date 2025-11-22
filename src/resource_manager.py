"""
Resource Management for Semantic Modulator

Implements resource limits and monitoring to prevent memory exhaustion.

Limits:
- Max document size: 100MB (configurable)
- Max total storage: 1GB (configurable)
- Max concurrent documents: 1000 (configurable)
"""

import logging
import sys
from typing import Dict, Optional
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, memory monitoring limited")


logger = logging.getLogger("resource_manager")


@dataclass
class ResourceLimits:
    """Resource limit configuration"""
    max_document_size_mb: float = 100.0  # Max size per document
    max_total_storage_mb: float = 1024.0  # Max total storage (1GB)
    max_documents: int = 1000  # Max number of documents
    max_memory_mb: float = 2048.0  # Max RAM usage (2GB)
    warn_threshold: float = 0.8  # Warn at 80% of limit


class ResourceManager:
    """
    Manages resource limits and monitoring.

    Prevents:
    - Memory exhaustion from large documents
    - Unbounded storage growth
    - System-wide resource issues
    """

    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize resource manager.

        Args:
            limits: Resource limit configuration
        """
        self.limits = limits or ResourceLimits()
        self.document_sizes: Dict[str, float] = {}  # file_id -> size in MB
        logger.info(f"Resource manager initialized: {self.limits}")

    def check_document_size(self, file_id: str, size_bytes: int) -> tuple[bool, Optional[str]]:
        """
        Check if document size is within limits.

        Args:
            file_id: Document identifier
            size_bytes: Document size in bytes

        Returns:
            (allowed, error_message)
        """
        size_mb = size_bytes / (1024 * 1024)

        # Check individual document size
        if size_mb > self.limits.max_document_size_mb:
            return False, (
                f"Document too large: {size_mb:.1f}MB exceeds limit of {self.limits.max_document_size_mb:.1f}MB\n"
                f"💡 Tip: Split document into smaller sections or increase max_document_size_mb"
            )

        # Check total storage
        total_size_mb = sum(self.document_sizes.values()) + size_mb
        if total_size_mb > self.limits.max_total_storage_mb:
            return False, (
                f"Total storage limit exceeded: {total_size_mb:.1f}MB > {self.limits.max_total_storage_mb:.1f}MB\n"
                f"💡 Tip: Delete old documents or increase max_total_storage_mb\n"
                f"   Current documents: {len(self.document_sizes)}, Total size: {sum(self.document_sizes.values()):.1f}MB"
            )

        # Check document count
        if len(self.document_sizes) >= self.limits.max_documents:
            return False, (
                f"Document count limit exceeded: {len(self.document_sizes)} >= {self.limits.max_documents}\n"
                f"💡 Tip: Delete old documents or increase max_documents"
            )

        # Warn if approaching limits
        if size_mb > self.limits.max_document_size_mb * self.limits.warn_threshold:
            logger.warning(
                f"⚠️  Document {file_id} is large: {size_mb:.1f}MB "
                f"({size_mb/self.limits.max_document_size_mb*100:.0f}% of limit)"
            )

        if total_size_mb > self.limits.max_total_storage_mb * self.limits.warn_threshold:
            logger.warning(
                f"⚠️  Total storage approaching limit: {total_size_mb:.1f}MB "
                f"({total_size_mb/self.limits.max_total_storage_mb*100:.0f}% of limit)"
            )

        return True, None

    def register_document(self, file_id: str, size_bytes: int):
        """
        Register a document after successful ingestion.

        Args:
            file_id: Document identifier
            size_bytes: Document size in bytes
        """
        size_mb = size_bytes / (1024 * 1024)
        self.document_sizes[file_id] = size_mb
        logger.info(f"Registered document {file_id}: {size_mb:.2f}MB")

    def unregister_document(self, file_id: str):
        """
        Unregister a document after deletion.

        Args:
            file_id: Document identifier
        """
        if file_id in self.document_sizes:
            size_mb = self.document_sizes.pop(file_id)
            logger.info(f"Unregistered document {file_id}: {size_mb:.2f}MB")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory stats in MB
        """
        stats = {
            "documents_count": len(self.document_sizes),
            "total_documents_mb": sum(self.document_sizes.values()),
            "limit_documents_mb": self.limits.max_total_storage_mb,
            "utilization_pct": (sum(self.document_sizes.values()) / self.limits.max_total_storage_mb * 100)
                if self.limits.max_total_storage_mb > 0 else 0,
        }

        # Add system memory if psutil available
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                stats["process_memory_mb"] = memory_info.rss / (1024 * 1024)
                stats["process_memory_pct"] = process.memory_percent()

                system_memory = psutil.virtual_memory()
                stats["system_memory_available_mb"] = system_memory.available / (1024 * 1024)
                stats["system_memory_used_pct"] = system_memory.percent

            except Exception as e:
                logger.debug(f"Failed to get system memory: {e}")

        return stats

    def check_memory_health(self) -> tuple[bool, Optional[str]]:
        """
        Check if memory usage is healthy.

        Returns:
            (healthy, warning_message)
        """
        stats = self.get_memory_usage()

        # Check document storage
        if stats["utilization_pct"] > 100:
            return False, (
                f"⚠️  Storage limit exceeded: {stats['total_documents_mb']:.1f}MB / {stats['limit_documents_mb']:.1f}MB\n"
                f"   Delete documents to free space"
            )

        if stats["utilization_pct"] > self.limits.warn_threshold * 100:
            return True, (
                f"⚠️  Storage approaching limit: {stats['utilization_pct']:.0f}% used\n"
                f"   Consider deleting old documents"
            )

        # Check system memory if available
        if PSUTIL_AVAILABLE and "process_memory_mb" in stats:
            if stats["process_memory_mb"] > self.limits.max_memory_mb:
                return False, (
                    f"⚠️  Memory limit exceeded: {stats['process_memory_mb']:.1f}MB > {self.limits.max_memory_mb:.1f}MB\n"
                    f"   Restart server or delete documents"
                )

            if stats["process_memory_mb"] > self.limits.max_memory_mb * self.limits.warn_threshold:
                return True, (
                    f"⚠️  Memory usage high: {stats['process_memory_mb']:.1f}MB "
                    f"({stats['process_memory_mb']/self.limits.max_memory_mb*100:.0f}% of limit)"
                )

        return True, None

    def get_stats(self) -> Dict[str, any]:
        """
        Get comprehensive resource stats.

        Returns:
            Dictionary with resource statistics
        """
        stats = self.get_memory_usage()

        stats.update({
            "limits": {
                "max_document_size_mb": self.limits.max_document_size_mb,
                "max_total_storage_mb": self.limits.max_total_storage_mb,
                "max_documents": self.limits.max_documents,
                "max_memory_mb": self.limits.max_memory_mb,
            },
            "documents": list(self.document_sizes.keys()),
            "largest_documents": sorted(
                [(k, v) for k, v in self.document_sizes.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],  # Top 5 largest
        })

        # Health check
        healthy, warning = self.check_memory_health()
        stats["healthy"] = healthy
        stats["warning"] = warning

        return stats

    def suggest_cleanup(self) -> Optional[str]:
        """
        Suggest documents to delete based on size.

        Returns:
            Cleanup suggestion or None
        """
        if not self.document_sizes:
            return None

        stats = self.get_stats()

        if stats["utilization_pct"] < self.limits.warn_threshold * 100:
            return None

        # Find largest documents
        largest = stats["largest_documents"]

        suggestion = [
            f"💡 Cleanup Suggestions:",
            f"   Current storage: {stats['total_documents_mb']:.1f}MB / {stats['limit_documents_mb']:.1f}MB ({stats['utilization_pct']:.0f}%)",
            f"",
            f"   Consider deleting large documents:",
        ]

        for file_id, size_mb in largest[:3]:
            suggestion.append(f"   - {file_id}: {size_mb:.1f}MB")

        return "\n".join(suggestion)
