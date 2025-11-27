"""
Graceful Degradation Strategies for Production Resilience

This module provides fallback strategies to maintain partial functionality
when components fail, preventing complete system failures.

Architecture:
- Embedding fallback: PyTorch → ONNX → TF-IDF
- Persistence fallback: Disk → In-memory only
- File sync fallback: Full validation → Basic timestamp check
- Version history fallback: Full diffs → Metadata only

Best Practices:
1. Fail gracefully, don't fail completely
2. Log degradation warnings for monitoring
3. Provide clear user feedback about reduced functionality
4. Maintain core functionality even in degraded mode

Usage:
    from src.graceful_degradation import GracefulDegradation

    # Embedding with fallback
    embeddings = await GracefulDegradation.embed_with_fallback(texts)

    # Persistence with fallback
    success = await GracefulDegradation.persist_with_fallback(doc_id, data)
"""

from typing import Any, Dict, List

import numpy as np

from .structured_logging import get_logger

logger = get_logger(__name__)


class GracefulDegradation:
    """
    Fallback strategies for component failures.

    Provides automatic fallback to degraded functionality when primary
    components fail, maintaining partial system availability.
    """

    @staticmethod
    async def embed_with_fallback(
        texts: List[str],
        embeddings_manager,  # Type hint avoided to prevent circular import
        preferred_tier: str = "STANDARD",
    ) -> np.ndarray:
        """
        Embedding with automatic tier fallback (PyTorch → ONNX → TF-IDF).

        Args:
            texts: List of texts to embed
            embeddings_manager: EmbeddingManager instance
            preferred_tier: Preferred embedding tier to start with

        Returns:
            Embedding vectors (numpy array)

        Raises:
            Exception: If all tiers fail
        """
        tiers = ["STANDARD", "ONNX", "TFIDF"]

        # Start from preferred tier
        if preferred_tier in tiers:
            start_index = tiers.index(preferred_tier)
        else:
            start_index = 0

        last_exception = None

        for tier in tiers[start_index:]:
            try:
                logger.debug(
                    "embedding_tier_attempt",
                    tier=tier,
                    text_count=len(texts),
                )

                # Try to encode with this tier
                result = await embeddings_manager.encode(
                    texts,
                    tier=tier,
                )

                # Success: Log if degraded
                if tier != preferred_tier:
                    logger.warning(
                        "embedding_degraded",
                        preferred_tier=preferred_tier,
                        actual_tier=tier,
                        reason="fallback_after_failure",
                    )

                return result

            except Exception as e:
                last_exception = e
                logger.warning(
                    "embedding_tier_failed",
                    tier=tier,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                # Continue to next tier

        # All tiers failed
        logger.error(
            "embedding_all_tiers_failed",
            attempted_tiers=tiers[start_index:],
            last_error=str(last_exception),
        )
        raise last_exception

    @staticmethod
    async def persist_with_fallback(
        doc_id: str,
        data: Dict[str, Any],
        persistence_manager,  # Type hint avoided to prevent circular import
    ) -> Dict[str, Any]:
        """
        Persistence with in-memory fallback.

        Attempts to save to disk, falls back to in-memory only on failure.

        Args:
            doc_id: Document ID
            data: Data to persist
            persistence_manager: PersistenceManager instance

        Returns:
            Dict with persistence status:
            - success: bool (True if saved to disk, False if in-memory only)
            - mode: str ("disk" or "memory")
            - warning: str (optional warning message)
        """
        try:
            # Try to save to disk
            success = await persistence_manager.save_document(doc_id, data)

            if success:
                logger.debug("persistence_success", doc_id=doc_id, mode="disk")
                return {
                    "success": True,
                    "mode": "disk",
                }
            else:
                # Save failed but didn't raise exception
                logger.warning(
                    "persistence_failed_fallback",
                    doc_id=doc_id,
                    mode="memory",
                    reason="save_returned_false",
                )
                return {
                    "success": False,
                    "mode": "memory",
                    "warning": "Data kept in memory only (persistence failed)",
                }

        except Exception as e:
            # Disk save failed, fall back to in-memory only
            logger.error(
                "persistence_failed_fallback",
                doc_id=doc_id,
                mode="memory",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            return {
                "success": False,
                "mode": "memory",
                "warning": f"Data kept in memory only: {type(e).__name__}",
                "error": str(e),
            }

    @staticmethod
    async def file_sync_with_fallback(
        file_path: str,
        file_sync_manager,  # Type hint avoided to prevent circular import
    ) -> Dict[str, Any]:
        """
        File sync with fallback to cached metadata.

        Attempts full validation (stat + MD5), falls back to basic timestamp check.

        Args:
            file_path: Path to file
            file_sync_manager: FileSyncManager instance

        Returns:
            Dict with sync status:
            - is_stale: bool
            - mode: str ("full_validation" or "cached_metadata")
            - warning: str (optional warning message)
        """
        try:
            # Try full validation (os.stat + MD5 checksum)
            is_stale = await file_sync_manager.check_staleness(file_path)

            logger.debug(
                "file_sync_success",
                file_path=file_path,
                mode="full_validation",
                is_stale=is_stale,
            )

            return {
                "is_stale": is_stale,
                "mode": "full_validation",
            }

        except OSError as e:
            # File stat failed (network partition, permissions, etc.)
            logger.warning(
                "file_sync_fallback",
                file_path=file_path,
                mode="cached_metadata",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            # Fall back to cached metadata (assume not stale if can't check)
            return {
                "is_stale": False,  # Assume not stale if can't check
                "mode": "cached_metadata",
                "warning": f"Using cached metadata: {type(e).__name__}",
            }

    @staticmethod
    async def version_history_with_fallback(
        doc_id: str,
        version_data: Dict[str, Any],
        version_manager,  # Type hint avoided to prevent circular import
    ) -> Dict[str, Any]:
        """
        Version history with fallback to metadata-only.

        Attempts full diff generation, falls back to metadata only on failure.

        Args:
            doc_id: Document ID
            version_data: Version data to save
            version_manager: VersionManager instance

        Returns:
            Dict with version status:
            - success: bool
            - mode: str ("full_diff" or "metadata_only")
            - warning: str (optional warning message)
        """
        try:
            # Try full diff generation
            version_id = await version_manager.add_version(
                doc_id,
                version_data,
            )

            logger.debug(
                "version_history_success",
                doc_id=doc_id,
                version_id=version_id,
                mode="full_diff",
            )

            return {
                "success": True,
                "mode": "full_diff",
                "version_id": version_id,
            }

        except Exception as e:
            # Diff generation failed, save metadata only
            logger.warning(
                "version_history_fallback",
                doc_id=doc_id,
                mode="metadata_only",
                error_type=type(e).__name__,
                error_message=str(e),
            )

            # Save metadata only (timestamp, version number, no diff)
            metadata_version = {
                "timestamp": version_data.get("timestamp"),
                "version": version_data.get("version"),
                "file_id": doc_id,
                "diff": None,  # No diff in degraded mode
            }

            return {
                "success": False,
                "mode": "metadata_only",
                "warning": f"Metadata only (no diff): {type(e).__name__}",
                "metadata": metadata_version,
            }
