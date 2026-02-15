"""App-layer persistence/sync orchestration service for server lifecycle."""

from __future__ import annotations

from typing import Any, TypedDict


class LoadPersistedRequest(TypedDict):
    """Persisted document loading request envelope."""

    compressor: Any
    persistence: Any
    resource_manager: Any
    logger: Any


class LoadSyncRequest(TypedDict):
    """File sync metadata load request envelope."""

    persistence: Any
    sync_manager: Any
    logger: Any


class SaveSyncRequest(TypedDict):
    """File sync metadata save request envelope."""

    sync_manager: Any
    persistence: Any
    logger: Any


class PersistenceOrchestrationService:
    """Handles persisted document restore and file-sync metadata load/save flows."""

    LOAD_PERSISTED_REQUEST_KEYS: frozenset[str] = frozenset(
        LoadPersistedRequest.__annotations__.keys()
    )
    LOAD_SYNC_REQUEST_KEYS: frozenset[str] = frozenset(LoadSyncRequest.__annotations__.keys())
    SAVE_SYNC_REQUEST_KEYS: frozenset[str] = frozenset(SaveSyncRequest.__annotations__.keys())

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
        return f"{contract_name} keys mismatch: missing={missing} extra={extra}"

    @classmethod
    def validate_contract_keys(
        cls,
        *,
        contract_name: str,
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        actual_keys = set(payload.keys())
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            raise ValueError(
                cls.contract_key_mismatch_message(
                    contract_name=contract_name,
                    missing=missing,
                    extra=extra,
                )
            )

    @classmethod
    def validate_load_persisted_request_map(cls, request: dict[str, Any]) -> LoadPersistedRequest:
        cls.validate_contract_keys(
            contract_name="load_persisted_request_map",
            payload=request,
            expected_keys=cls.LOAD_PERSISTED_REQUEST_KEYS,
        )
        return request

    @classmethod
    def validate_load_sync_request_map(cls, request: dict[str, Any]) -> LoadSyncRequest:
        cls.validate_contract_keys(
            contract_name="load_sync_request_map",
            payload=request,
            expected_keys=cls.LOAD_SYNC_REQUEST_KEYS,
        )
        return request

    @classmethod
    def validate_save_sync_request_map(cls, request: dict[str, Any]) -> SaveSyncRequest:
        cls.validate_contract_keys(
            contract_name="save_sync_request_map",
            payload=request,
            expected_keys=cls.SAVE_SYNC_REQUEST_KEYS,
        )
        return request

    @classmethod
    def load_persisted_documents(
        cls, *, compressor: Any, persistence: Any, resource_manager: Any, logger
    ) -> None:
        request = cls.validate_load_persisted_request_map(
            {
                "compressor": compressor,
                "persistence": persistence,
                "resource_manager": resource_manager,
                "logger": logger,
            }
        )
        compressor = request["compressor"]
        persistence = request["persistence"]
        resource_manager = request["resource_manager"]
        logger = request["logger"]
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

    @classmethod
    def load_file_sync_metadata(cls, *, persistence: Any, sync_manager: Any, logger) -> None:
        request = cls.validate_load_sync_request_map(
            {"persistence": persistence, "sync_manager": sync_manager, "logger": logger}
        )
        persistence = request["persistence"]
        sync_manager = request["sync_manager"]
        logger = request["logger"]
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

    @classmethod
    def save_file_sync_metadata(cls, *, sync_manager: Any, persistence: Any, logger) -> None:
        request = cls.validate_save_sync_request_map(
            {"sync_manager": sync_manager, "persistence": persistence, "logger": logger}
        )
        sync_manager = request["sync_manager"]
        persistence = request["persistence"]
        logger = request["logger"]
        try:
            metadata = sync_manager.export_metadata()
            success = persistence.save_file_sync_metadata(metadata)
            if success:
                logger.info("file_sync_save_complete", document_count=len(metadata))
            else:
                logger.warning("file_sync_save_warning", message="No file sync metadata to save")
        except Exception as e:
            logger.error("file_sync_save_error", error=str(e))
