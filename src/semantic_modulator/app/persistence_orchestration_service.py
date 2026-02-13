"""App-layer persistence/sync orchestration service for server lifecycle."""

from __future__ import annotations

from typing import Any


class PersistenceOrchestrationService:
    """Handles persisted document restore and file-sync metadata load/save flows."""

    @staticmethod
    def load_persisted_documents(
        *, compressor: Any, persistence: Any, resource_manager: Any, logger
    ) -> None:
        try:
            file_ids = persistence.list_documents()
            if not file_ids:
                logger.info(
                    "persistence_load", status="empty", message="No persisted documents found"
                )
                return

            logger.info("persistence_load_started", document_count=len(file_ids))
            loaded_count = 0

            for file_id in file_ids:
                try:
                    data = persistence.load_document(file_id)
                    if data:
                        compressor.chunks.update(data["chunks"])
                        compressor.graphs[file_id] = data["graph_data"]
                        compressor.file_metadata[file_id] = data["metadata"]

                        total_size = sum(len(chunk.text) for chunk in data["chunks"].values())
                        resource_manager.register_document(file_id, total_size)

                        loaded_count += 1
                        logger.info(
                            "document_loaded", file_id=file_id, node_count=len(data["chunks"])
                        )
                except Exception as e:
                    logger.error("document_load_failed", file_id=file_id, error=str(e))

            logger.info(
                "persistence_load_complete",
                loaded=loaded_count,
                total=len(file_ids),
                success_rate=f"{loaded_count}/{len(file_ids)}",
            )

        except Exception as e:
            logger.error("persistence_load_error", error=str(e), exc_info=True)

    @staticmethod
    def load_file_sync_metadata(*, persistence: Any, sync_manager: Any, logger) -> None:
        try:
            metadata = persistence.load_file_sync_metadata()
            if not metadata:
                logger.info(
                    "file_sync_load",
                    status="empty",
                    message="No file sync metadata found (first run or no tracked files)",
                )
                return

            sync_manager.import_metadata(metadata)
            logger.info("file_sync_load_complete", document_count=len(metadata))

        except Exception as e:
            logger.error("file_sync_load_error", error=str(e), exc_info=True)

    @staticmethod
    def save_file_sync_metadata(*, sync_manager: Any, persistence: Any, logger) -> None:
        try:
            metadata = sync_manager.export_metadata()
            success = persistence.save_file_sync_metadata(metadata)
            if success:
                logger.info("file_sync_save_complete", document_count=len(metadata))
            else:
                logger.warning("file_sync_save_warning", message="No file sync metadata to save")
        except Exception as e:
            logger.error("file_sync_save_error", error=str(e))
