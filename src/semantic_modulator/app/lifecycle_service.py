"""App-layer lifecycle service for startup and shutdown orchestration."""

from __future__ import annotations

from typing import Any, Callable, TypedDict


class StartupRequest(TypedDict):
    """Startup request envelope for lifecycle orchestration."""

    load_persisted_documents: Callable[[], None]
    load_file_sync_metadata: Callable[[], None]
    logger: Any


class ShutdownRequest(TypedDict):
    """Shutdown request envelope for lifecycle orchestration."""

    save_file_sync_metadata: Callable[[], None]
    logger: Any


class ServerLifecycleService:
    """Coordinates startup/shutdown sequencing around persistence and sync state."""

    STARTUP_REQUEST_KEYS: frozenset[str] = frozenset(StartupRequest.__annotations__.keys())
    SHUTDOWN_REQUEST_KEYS: frozenset[str] = frozenset(ShutdownRequest.__annotations__.keys())

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
    def validate_startup_request_map(cls, request: dict[str, Any]) -> StartupRequest:
        cls.validate_contract_keys(
            contract_name="startup_request_map",
            payload=request,
            expected_keys=cls.STARTUP_REQUEST_KEYS,
        )
        return request

    @classmethod
    def validate_shutdown_request_map(cls, request: dict[str, Any]) -> ShutdownRequest:
        cls.validate_contract_keys(
            contract_name="shutdown_request_map",
            payload=request,
            expected_keys=cls.SHUTDOWN_REQUEST_KEYS,
        )
        return request

    @classmethod
    def startup(
        cls,
        *,
        load_persisted_documents: Callable[[], None],
        load_file_sync_metadata: Callable[[], None],
        logger,
    ) -> None:
        request = cls.validate_startup_request_map(
            {
                "load_persisted_documents": load_persisted_documents,
                "load_file_sync_metadata": load_file_sync_metadata,
                "logger": logger,
            }
        )
        load_persisted_documents = request["load_persisted_documents"]
        load_file_sync_metadata = request["load_file_sync_metadata"]
        logger = request["logger"]
        logger.info("server_startup", phase="initializing")
        load_persisted_documents()
        load_file_sync_metadata()
        logger.info("server_startup", phase="complete")

    @classmethod
    def shutdown(cls, *, save_file_sync_metadata: Callable[[], None], logger) -> bool:
        request = cls.validate_shutdown_request_map(
            {"save_file_sync_metadata": save_file_sync_metadata, "logger": logger}
        )
        save_file_sync_metadata = request["save_file_sync_metadata"]
        logger = request["logger"]
        logger.info("server_shutdown", phase="started")
        try:
            save_file_sync_metadata()
            logger.info("server_shutdown", phase="state_persisted")
        except Exception as e:  # pragma: no cover - exercised via server integration tests
            logger.error("server_shutdown_error", error=str(e), exc_info=True)

        logger.info("server_shutdown", phase="complete")
        return False
