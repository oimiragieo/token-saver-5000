"""Contract tests for app-layer server service adapter."""

from __future__ import annotations

import importlib
from unittest.mock import Mock


def test_adapter_delegates_persistence_load_calls_with_expected_args():
    module = importlib.import_module("src.semantic_modulator.app.server_service_adapter")
    persistence_service = Mock()
    context_service = Mock()
    progress_service = Mock()
    adapter = module.ServerServiceAdapter(
        persistence_service=persistence_service,
        context_service=context_service,
        progress_service=progress_service,
        logger=Mock(),
    )

    compressor = Mock()
    persistence = Mock()
    resource_manager = Mock()
    adapter.load_persisted_documents(
        compressor=compressor,
        persistence=persistence,
        resource_manager=resource_manager,
    )

    persistence_service.load_persisted_documents.assert_called_once_with(
        compressor=compressor,
        persistence=persistence,
        resource_manager=resource_manager,
        logger=adapter.logger,
    )


def test_adapter_delegates_context_and_progress_helpers():
    module = importlib.import_module("src.semantic_modulator.app.server_service_adapter")
    persistence_service = Mock()
    context_service = Mock()
    progress_service = Mock()
    adapter = module.ServerServiceAdapter(
        persistence_service=persistence_service,
        context_service=context_service,
        progress_service=progress_service,
        logger=Mock(),
    )

    context_service.extract_file_id_from_node.return_value = "doc"
    progress_service.create_progress_bar.return_value = "bar"

    assert adapter.extract_file_id_from_node("doc_n1") == "doc"
    assert adapter.create_progress_bar(50.0, 40) == "bar"
