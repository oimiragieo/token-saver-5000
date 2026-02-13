"""App-layer lifecycle service for startup and shutdown orchestration."""

from __future__ import annotations

from typing import Callable


class ServerLifecycleService:
    """Coordinates startup/shutdown sequencing around persistence and sync state."""

    @staticmethod
    def startup(
        *,
        load_persisted_documents: Callable[[], None],
        load_file_sync_metadata: Callable[[], None],
        logger,
    ) -> None:
        logger.info("server_startup", phase="initializing")
        load_persisted_documents()
        load_file_sync_metadata()
        logger.info("server_startup", phase="complete")

    @staticmethod
    def shutdown(*, save_file_sync_metadata: Callable[[], None], logger) -> bool:
        logger.info("server_shutdown", phase="started")
        try:
            save_file_sync_metadata()
            logger.info("server_shutdown", phase="state_persisted")
        except Exception as e:  # pragma: no cover - exercised via server integration tests
            logger.error("server_shutdown_error", error=str(e), exc_info=True)

        logger.info("server_shutdown", phase="complete")
        return False
