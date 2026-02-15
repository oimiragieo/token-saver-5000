"""Contract tests for app-layer persistence orchestration service."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def test_load_persisted_documents_restores_graph_and_registers_resource():
    module = importlib.import_module("src.semantic_modulator.app.persistence_orchestration_service")
    service = module.PersistenceOrchestrationService()

    compressor = SimpleNamespace(chunks={}, graphs={}, file_metadata={})
    persistence = Mock()
    persistence.list_documents.return_value = ["doc1"]
    persistence.load_document.return_value = {
        "chunks": {"doc1_n0": SimpleNamespace(text="hello")},
        "graph_data": {"nodes": ["doc1_n0"], "edges": []},
        "metadata": {"file_id": "doc1"},
    }
    resource_manager = Mock()
    logger = Mock()

    service.load_persisted_documents(
        compressor=compressor,
        persistence=persistence,
        resource_manager=resource_manager,
        logger=logger,
    )

    assert "doc1_n0" in compressor.chunks
    assert "doc1" in compressor.graphs
    assert "doc1" in compressor.file_metadata
    resource_manager.register_document.assert_called_once()


def test_load_file_sync_metadata_imports_when_present():
    module = importlib.import_module("src.semantic_modulator.app.persistence_orchestration_service")
    service = module.PersistenceOrchestrationService()

    persistence = Mock()
    persistence.load_file_sync_metadata.return_value = {"doc1": {"checksum": "abc"}}
    sync_manager = Mock()
    logger = Mock()

    service.load_file_sync_metadata(
        persistence=persistence,
        sync_manager=sync_manager,
        logger=logger,
    )

    sync_manager.import_metadata.assert_called_once_with({"doc1": {"checksum": "abc"}})


def test_save_file_sync_metadata_warns_when_not_saved():
    module = importlib.import_module("src.semantic_modulator.app.persistence_orchestration_service")
    service = module.PersistenceOrchestrationService()

    sync_manager = Mock()
    sync_manager.export_metadata.return_value = {}
    persistence = Mock()
    persistence.save_file_sync_metadata.return_value = False
    logger = Mock()

    service.save_file_sync_metadata(
        sync_manager=sync_manager,
        persistence=persistence,
        logger=logger,
    )

    logger.warning.assert_called_once_with(
        "file_sync_save_warning", message="No file sync metadata to save"
    )


def test_persistence_service_contract_keysets_declared():
    module = importlib.import_module("src.semantic_modulator.app.persistence_orchestration_service")
    service = module.PersistenceOrchestrationService
    assert service.LOAD_PERSISTED_REQUEST_KEYS == frozenset(
        {"compressor", "persistence", "resource_manager", "logger"}
    )
    assert service.LOAD_SYNC_REQUEST_KEYS == frozenset({"persistence", "sync_manager", "logger"})
    assert service.SAVE_SYNC_REQUEST_KEYS == frozenset({"sync_manager", "persistence", "logger"})


def test_persistence_service_validate_load_sync_request_map_rejects_extra_key():
    module = importlib.import_module("src.semantic_modulator.app.persistence_orchestration_service")
    service = module.PersistenceOrchestrationService
    with pytest.raises(ValueError, match="load_sync_request_map keys mismatch"):
        service.validate_load_sync_request_map(
            {"persistence": Mock(), "sync_manager": Mock(), "logger": Mock(), "extra": True}
        )
