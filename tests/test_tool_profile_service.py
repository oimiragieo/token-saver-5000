"""Contract tests for app-layer tool profile bootstrap service."""

from __future__ import annotations

import importlib
from unittest.mock import Mock


def test_profile_bootstrap_returns_active_profile_and_names():
    module = importlib.import_module("src.semantic_modulator.app.tool_profile_service")
    service = module.ToolProfileBootstrapService()

    tooling = Mock()
    tooling.supported_profiles = {"full", "core_stable"}
    tooling.resolve_tools_for_profile.return_value = (
        "core_stable",
        [Mock(name="ingest_context")],
        False,
    )
    tooling.resolve_tools_for_profile.return_value[1][0].name = "ingest_context"
    logger = Mock()

    profile, names = service.bootstrap(
        configured_profile="core_stable", tooling=tooling, logger=logger
    )

    assert profile == "core_stable"
    assert names == ["ingest_context"]
    logger.info.assert_any_call(
        "mcp_tool_profile_active",
        profile="core_stable",
        enabled_tools=1,
        supported_profiles=["core_stable", "full"],
    )


def test_profile_bootstrap_logs_fallback_warning():
    module = importlib.import_module("src.semantic_modulator.app.tool_profile_service")
    service = module.ToolProfileBootstrapService()

    tooling = Mock()
    tooling.supported_profiles = {"full", "core_stable"}
    tooling.resolve_tools_for_profile.return_value = ("full", [Mock()], True)
    logger = Mock()

    service.bootstrap(configured_profile="bad_profile", tooling=tooling, logger=logger)

    logger.warning.assert_called_once_with(
        "invalid_tool_profile",
        configured_profile="bad_profile",
        fallback_profile="full",
    )
