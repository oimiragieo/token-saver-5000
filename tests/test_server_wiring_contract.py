"""Integration contract tests for server wiring from factory output."""

from __future__ import annotations

from unittest.mock import Mock, patch

from src.server import SemanticModulatorServer


def test_server_constructor_relies_on_factory_for_runtime_and_adapter():
    fake_factory_output = {
        "compressor": Mock(),
        "blind_spot_detector": Mock(),
        "halo_detector": Mock(),
        "context_window_adapter": Mock(),
        "multilevel_encoder": Mock(),
        "focus_manager": Mock(),
        "persistence": Mock(),
        "resource_manager": Mock(),
        "sync_manager": Mock(),
        "version_manager": Mock(),
        "path_validator": Mock(),
        "ace_framework": Mock(),
        "ace_contexts": Mock(),
        "tooling": Mock(),
        "context_service": Mock(),
        "lifecycle_service": Mock(),
        "progress_service": Mock(),
        "persistence_service": Mock(),
        "tool_profile_service": Mock(),
        "runtime_service": Mock(),
        "service_adapter": Mock(),
        "context_window_monitor": {"max_tokens": 100000, "used_tokens": 0, "history": []},
        "retrieval_history": {},
    }
    fake_factory_output["tool_profile_service"].bootstrap.return_value = (
        "full",
        ["ingest_context"],
    )

    with (
        patch("src.server.ServerFactoryService.build", return_value=fake_factory_output),
        patch.object(SemanticModulatorServer, "_setup_handlers"),
    ):
        server = SemanticModulatorServer()

    assert server.runtime_service is fake_factory_output["runtime_service"]
    assert server.service_adapter is fake_factory_output["service_adapter"]
    assert server.enabled_tool_names == ["ingest_context"]
