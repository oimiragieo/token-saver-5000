"""
Resource Management for Semantic Modulator

Implements resource limits and monitoring to prevent memory exhaustion.

Limits:
- Max document size: 100MB (configurable)
- Max total storage: 1GB (configurable)
- Max concurrent documents: 1000 (configurable)
"""

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any
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
    max_connector_documents_per_feed: int = 100
    max_connector_payload_mb: float = 250.0
    max_multimodal_assets_per_document: int = 128
    max_multimodal_payload_mb: float = 100.0
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

        # Concurrency protection (v0.8.0 audit fix - Issue 3)
        # Uses threading.Lock since operations are fast in-memory dict updates (<1ms).
        #
        # NOTE: ResourceManager uses threading.Lock + ThreadPoolExecutor pattern:
        # - All operations are pure in-memory dict read/writes (no disk I/O)
        # - Lock hold time is <1ms, acceptable for single-client MCP server
        # - psutil calls (in check_health) take ~10ms max
        # - check_health_async() uses run_in_executor to avoid blocking event loop
        self._lock = threading.Lock()

        # Thread pool for async wrappers (v0.8.0 audit fix)
        # Allows sync methods with locks to be called from async handlers
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="resource_mgr")

        logger.info(f"Resource manager initialized: {self.limits}")

    def check_document_size(self, file_id: str, size_bytes: int) -> tuple[bool, Optional[str]]:
        """
        Check if document size is within limits.

        Handles re-ingest correctly by computing candidate total that accounts
        for replacing existing document (v0.8.0 audit fix for Issue 5).

        Args:
            file_id: Document identifier
            size_bytes: Document size in bytes

        Returns:
            (allowed, error_message)
        """
        size_mb = size_bytes / (1024 * 1024)

        # Thread-safe: read shared state under lock to prevent TOCTOU races
        # (v0.8.0 audit fix - Issue 3)
        with self._lock:
            # Get old size if this is a re-ingest (0 if new document)
            # This prevents double-counting: candidate = total - old + new
            old_size_mb = self.document_sizes.get(file_id, 0)
            current_total_mb = sum(self.document_sizes.values())
            doc_count = len(self.document_sizes)
            is_reingest = file_id in self.document_sizes

        candidate_total_mb = current_total_mb - old_size_mb + size_mb

        # Check total storage first (system-wide limit)
        if candidate_total_mb > self.limits.max_total_storage_mb:
            return False, (
                f"Total storage limit exceeded: {candidate_total_mb:.1f}MB > {self.limits.max_total_storage_mb:.1f}MB\n"
                f"Tip: Delete old documents or increase max_total_storage_mb\n"
                f"   Current documents: {doc_count}, Total size: {current_total_mb:.1f}MB"
            )

        # Check document count (skip for re-ingest - not adding a new document)
        # Note: is_reingest and doc_count captured under lock above
        if not is_reingest and doc_count >= self.limits.max_documents:
            return False, (
                f"Too many documents: {doc_count} >= {self.limits.max_documents}\n"
                f"Tip: Delete old documents or increase max_documents"
            )

        # Check individual document size
        if size_mb > self.limits.max_document_size_mb:
            return False, (
                f"Document too large: {size_mb:.1f}MB exceeds limit of {self.limits.max_document_size_mb:.1f}MB\n"
                f"Tip: Split document into smaller sections or increase max_document_size_mb"
            )

        # Warn if approaching limits
        if size_mb > self.limits.max_document_size_mb * self.limits.warn_threshold:
            logger.warning(
                f"[WARN] Document {file_id} is large: {size_mb:.1f}MB "
                f"({size_mb/self.limits.max_document_size_mb*100:.0f}% of limit)"
            )

        if candidate_total_mb > self.limits.max_total_storage_mb * self.limits.warn_threshold:
            logger.warning(
                f"[WARN] Total storage approaching limit: {candidate_total_mb:.1f}MB "
                f"({candidate_total_mb/self.limits.max_total_storage_mb*100:.0f}% of limit)"
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
        # Thread-safe: protect mutations (v0.8.0 audit fix - Issue 3)
        with self._lock:
            self.document_sizes[file_id] = size_mb
        logger.info(f"Registered document {file_id}: {size_mb:.2f}MB")

    def unregister_document(self, file_id: str):
        """
        Unregister a document after deletion.

        Args:
            file_id: Document identifier
        """
        # Thread-safe: protect mutations (v0.8.0 audit fix - Issue 3)
        with self._lock:
            if file_id in self.document_sizes:
                size_mb = self.document_sizes.pop(file_id)
                logger.info(f"Unregistered document {file_id}: {size_mb:.2f}MB")

    def check_health(self) -> Dict[str, Any]:
        """
        Proactive health check for resource usage.

        Returns comprehensive health status including warnings
        when approaching limits.

        Thread-safe: Uses internal lock to protect shared state access
        (v0.8.0 audit fix - Issue 3).

        Returns:
            Dictionary with health status:
            - healthy: bool - Overall health status
            - warnings: List[str] - List of warning messages
            - metrics: Dict - Current resource metrics
            - recommendations: List[str] - Suggested actions
        """
        warnings = []
        recommendations = []

        # Thread-safe: protect reads from concurrent modifications
        with self._lock:
            total_size_mb = sum(self.document_sizes.values())
            total_docs = len(self.document_sizes)

        # Storage usage
        storage_pct = total_size_mb / self.limits.max_total_storage_mb
        if storage_pct >= 1.0:
            warnings.append(
                f"Storage at capacity: {total_size_mb:.1f}MB / {self.limits.max_total_storage_mb:.1f}MB"
            )
            recommendations.append("Delete unused documents with delete_document()")
        elif storage_pct >= self.limits.warn_threshold:
            warnings.append(f"Storage at {storage_pct:.1%} capacity")
            recommendations.append("Consider cleaning up old documents")

        # Document count
        doc_pct = total_docs / self.limits.max_documents
        if doc_pct >= 1.0:
            warnings.append(f"Document count at limit: {total_docs} / {self.limits.max_documents}")
            recommendations.append("Delete unused documents to free up slots")
        elif doc_pct >= self.limits.warn_threshold:
            warnings.append(f"Document count at {doc_pct:.1%} of limit")

        # Memory usage (if psutil available)
        memory_warning = None
        memory_mb = 0.0
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                memory_pct = memory_mb / self.limits.max_memory_mb

                if memory_pct >= 1.0:
                    memory_warning = (
                        f"Memory usage high: {memory_mb:.1f}MB / {self.limits.max_memory_mb:.1f}MB"
                    )
                    warnings.append(memory_warning)
                    recommendations.append("Restart server to reclaim memory")
                elif memory_pct >= self.limits.warn_threshold:
                    warnings.append(f"Memory at {memory_pct:.1%} of limit")
            except Exception as e:
                logger.debug(f"Failed to get memory usage: {e}")

        # Determine overall health
        healthy = len(warnings) == 0

        return {
            "healthy": healthy,
            "warnings": warnings,
            "recommendations": recommendations,
            "metrics": {
                "storage_mb": total_size_mb,
                "storage_limit_mb": self.limits.max_total_storage_mb,
                "storage_usage_pct": storage_pct * 100,
                "document_count": total_docs,
                "document_limit": self.limits.max_documents,
                "document_usage_pct": doc_pct * 100,
                "memory_mb": memory_mb if PSUTIL_AVAILABLE else None,
                "memory_limit_mb": self.limits.max_memory_mb,
            },
        }

    async def check_health_async(self) -> Dict[str, Any]:
        """
        Async wrapper for check_health (v0.8.0 audit fix - Issue 3).

        Runs check_health in thread pool to avoid blocking event loop
        when called from async handlers. This is necessary because:
        - check_health uses threading.Lock (blocks event loop if awaited directly)
        - psutil calls may take ~10ms

        Returns:
            Dictionary with health status (same as check_health)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.check_health)

    async def check_document_size_async(
        self, file_id: str, size_bytes: int
    ) -> tuple[bool, Optional[str]]:
        """
        Async wrapper for check_document_size (v0.8.0 audit fix - Issue 3).

        Runs check_document_size in thread pool to avoid blocking event loop
        when called from async handlers.

        Args:
            file_id: Document identifier
            size_bytes: Document size in bytes

        Returns:
            (allowed, error_message)
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, lambda: self.check_document_size(file_id, size_bytes)
        )

    async def register_document_async(self, file_id: str, size_bytes: int) -> None:
        """
        Async wrapper for register_document (v0.8.0 audit fix - Issue 3).

        Runs register_document in thread pool to avoid blocking event loop
        when called from async handlers.

        Args:
            file_id: Document identifier
            size_bytes: Document size in bytes
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor, lambda: self.register_document(file_id, size_bytes)
        )

    async def unregister_document_async(self, file_id: str) -> None:
        """
        Async wrapper for unregister_document (v0.8.0 audit fix - Issue 3).

        Runs unregister_document in thread pool to avoid blocking event loop
        when called from async handlers.

        Args:
            file_id: Document identifier
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, lambda: self.unregister_document(file_id))

    def check_connector_batch(
        self, feed_name: str, document_count: int, total_size_bytes: int
    ) -> tuple[bool, Optional[str]]:
        """Validate connector sync batch size against feed-level quotas."""
        total_size_mb = total_size_bytes / (1024 * 1024)
        if document_count <= 0:
            return False, f"Connector feed '{feed_name}' produced no documents to ingest"
        if document_count > self.limits.max_connector_documents_per_feed:
            return False, (
                f"Connector feed '{feed_name}' exceeds document limit: "
                f"{document_count} > {self.limits.max_connector_documents_per_feed}"
            )
        if total_size_mb > self.limits.max_connector_payload_mb:
            return False, (
                f"Connector feed '{feed_name}' exceeds payload limit: "
                f"{total_size_mb:.1f}MB > {self.limits.max_connector_payload_mb:.1f}MB"
            )
        return True, None

    def check_multimodal_batch(
        self, doc_id: str, asset_count: int, total_size_bytes: int
    ) -> tuple[bool, Optional[str]]:
        total_size_mb = total_size_bytes / (1024 * 1024)
        if asset_count > self.limits.max_multimodal_assets_per_document:
            return False, (
                f"Multimodal asset limit exceeded for {doc_id}: "
                f"{asset_count} > {self.limits.max_multimodal_assets_per_document}"
            )
        if total_size_mb > self.limits.max_multimodal_payload_mb:
            return False, (
                f"Multimodal payload too large for {doc_id}: "
                f"{total_size_mb:.1f}MB > {self.limits.max_multimodal_payload_mb:.1f}MB"
            )
        return True, None

    async def check_connector_batch_async(
        self, feed_name: str, document_count: int, total_size_bytes: int
    ) -> tuple[bool, Optional[str]]:
        """Async wrapper for connector batch quota checks."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.check_connector_batch(feed_name, document_count, total_size_bytes),
        )

    async def check_multimodal_batch_async(
        self, doc_id: str, asset_count: int, total_size_bytes: int
    ) -> tuple[bool, Optional[str]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.check_multimodal_batch(doc_id, asset_count, total_size_bytes),
        )

    def get_usage_summary(self) -> str:
        """
        Get human-readable usage summary.

        Returns:
            Formatted string with current resource usage
        """
        health = self.check_health()
        metrics = health["metrics"]

        lines = []
        lines.append("Resource Usage Summary")
        lines.append("-" * 50)

        # Storage
        lines.append(
            f"Storage: {metrics['storage_mb']:.1f}MB / {metrics['storage_limit_mb']:.1f}MB ({metrics['storage_usage_pct']:.1f}%)"
        )

        # Documents
        lines.append(
            f"Documents: {metrics['document_count']} / {metrics['document_limit']} ({metrics['document_usage_pct']:.1f}%)"
        )

        # Memory (if available)
        if metrics["memory_mb"] is not None:
            memory_pct = (metrics["memory_mb"] / metrics["memory_limit_mb"]) * 100
            lines.append(
                f"Memory: {metrics['memory_mb']:.1f}MB / {metrics['memory_limit_mb']:.1f}MB ({memory_pct:.1f}%)"
            )

        # Status
        lines.append("-" * 50)
        if health["healthy"]:
            lines.append("Status: [OK] Healthy")
        else:
            lines.append("Status: [WARN] Warnings detected")

        # Warnings
        if health["warnings"]:
            lines.append("\n[WARN] Warnings:")
            for warning in health["warnings"]:
                lines.append(f"  - {warning}")

        # Recommendations
        if health["recommendations"]:
            lines.append("\nRecommendations:")
            for rec in health["recommendations"]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

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
            "utilization_pct": (
                (sum(self.document_sizes.values()) / self.limits.max_total_storage_mb * 100)
                if self.limits.max_total_storage_mb > 0
                else 0
            ),
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
                f"[ERROR] Storage limit exceeded: {stats['total_documents_mb']:.1f}MB / {stats['limit_documents_mb']:.1f}MB\n"
                f"   Delete documents to free space"
            )

        if stats["utilization_pct"] > self.limits.warn_threshold * 100:
            return True, (
                f"[WARN] Storage approaching limit: {stats['utilization_pct']:.0f}% used\n"
                f"   Consider deleting old documents"
            )

        # Check system memory if available
        if PSUTIL_AVAILABLE and "process_memory_mb" in stats:
            if stats["process_memory_mb"] > self.limits.max_memory_mb:
                return False, (
                    f"[ERROR] Memory limit exceeded: {stats['process_memory_mb']:.1f}MB > {self.limits.max_memory_mb:.1f}MB\n"
                    f"   Restart server or delete documents"
                )

            if stats["process_memory_mb"] > self.limits.max_memory_mb * self.limits.warn_threshold:
                return True, (
                    f"[WARN] Memory usage high: {stats['process_memory_mb']:.1f}MB "
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

        stats.update(
            {
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
                    reverse=True,
                )[
                    :5
                ],  # Top 5 largest
            }
        )

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
            "Cleanup Suggestions:",
            f"   Current storage: {stats['total_documents_mb']:.1f}MB / {stats['limit_documents_mb']:.1f}MB ({stats['utilization_pct']:.0f}%)",
            "",
            "   Consider deleting large documents:",
        ]

        for file_id, size_mb in largest[:3]:
            suggestion.append(f"   - {file_id}: {size_mb:.1f}MB")

        return "\n".join(suggestion)
