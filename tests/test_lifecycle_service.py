"""Contract tests for app-layer server lifecycle service."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest


def test_lifecycle_startup_invokes_load_steps_in_order():
    module = importlib.import_module("src.semantic_modulator.app.lifecycle_service")
    service = module.ServerLifecycleService()
    logger = Mock()
    calls: list[str] = []

    def load_docs():
        calls.append("docs")

    def load_sync():
        calls.append("sync")

    service.startup(
        load_persisted_documents=load_docs, load_file_sync_metadata=load_sync, logger=logger
    )

    assert calls == ["docs", "sync"]
    logger.info.assert_any_call("server_startup", phase="initializing")
    logger.info.assert_any_call("server_startup", phase="complete")


def test_lifecycle_shutdown_calls_save_and_never_suppresses_exceptions():
    module = importlib.import_module("src.semantic_modulator.app.lifecycle_service")
    service = module.ServerLifecycleService()
    logger = Mock()
    calls: list[str] = []

    def save_sync():
        calls.append("save")

    result = service.shutdown(save_file_sync_metadata=save_sync, logger=logger)

    assert calls == ["save"]
    assert result is False
    logger.info.assert_any_call("server_shutdown", phase="started")
    logger.info.assert_any_call("server_shutdown", phase="state_persisted")
    logger.info.assert_any_call("server_shutdown", phase="complete")


def test_lifecycle_service_request_contracts_declared():
    module = importlib.import_module("src.semantic_modulator.app.lifecycle_service")
    service = module.ServerLifecycleService
    assert service.STARTUP_REQUEST_KEYS == frozenset(
        {"load_persisted_documents", "load_file_sync_metadata", "logger"}
    )
    assert service.SHUTDOWN_REQUEST_KEYS == frozenset({"save_file_sync_metadata", "logger"})


def test_lifecycle_service_validate_startup_request_map_rejects_extra_key():
    module = importlib.import_module("src.semantic_modulator.app.lifecycle_service")
    service = module.ServerLifecycleService
    with pytest.raises(ValueError, match="startup_request_map keys mismatch"):
        service.validate_startup_request_map(
            {
                "load_persisted_documents": lambda: None,
                "load_file_sync_metadata": lambda: None,
                "logger": Mock(),
                "extra": True,
            }
        )
