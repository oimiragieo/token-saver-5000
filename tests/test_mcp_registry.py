"""Parity and characterization tests for the typed MCP engine registry."""

from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest

from src.handlers import mcp_core
from src.handlers.mcp_core.registry import (
    _HANDLER_SPECS,
    REGISTERED_TOOLS,
    RegisteredTool,
    _build_registry,
)


def test_registry_is_the_schema_and_handler_join():
    """Every exposed schema has exactly one typed handler metadata record."""
    entries = mcp_core.registered_tools("full")
    assert entries
    assert len(entries) == len({entry.name for entry in entries})
    assert all(isinstance(entry, RegisteredTool) for entry in entries)
    assert [entry.name for entry in entries] == sorted(entry.name for entry in entries)
    assert {entry.name for entry in entries} == {tool.name for tool in mcp_core.setup_mcp_tools()}
    assert all(callable(entry.handler) for entry in entries)


def test_setup_is_a_projection_of_registry_and_profiles_are_unchanged():
    full = mcp_core.setup_mcp_tools("full")
    core = mcp_core.setup_mcp_tools("core_stable")
    assert full == [entry.schema for entry in REGISTERED_TOOLS]
    assert core == [entry.schema for entry in mcp_core.registered_tools("core_stable")]
    assert {entry.name for entry in mcp_core.registered_tools("core_stable")} == {
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "modulate_region",
        "get_stats",
        "list_documents",
        "delete_document",
    }


def test_execution_metadata_has_typed_policy_homes_and_independent_flags():
    for entry in REGISTERED_TOOLS:
        assert entry.timeout is None or isinstance(entry.timeout, float)
        assert isinstance(entry.deps, tuple)
        assert entry.fixture_id
        assert entry.test_policy
        assert isinstance(entry.fixture_id, str)
        assert isinstance(entry.test_policy, str)
        for flag in (
            entry.read_only,
            entry.mutates_state,
            entry.destructive,
            entry.idempotent,
            entry.open_world,
            entry.experimental,
            entry.requires_model,
            entry.requires_context,
            entry.requires_auth,
        ):
            assert isinstance(flag, bool)
        assert not (entry.read_only and entry.destructive)


def test_every_handler_proxy_resolves_to_a_real_module_attribute():
    missing = []
    for entry in REGISTERED_TOOLS:
        module = import_module(entry.handler.__module__)
        if not hasattr(module, entry.handler.__name__):
            missing.append(f"{entry.name}:{entry.handler.__module__}.{entry.handler.__name__}")
    assert missing == []


def test_registry_build_rejects_an_unresolved_handler(monkeypatch):
    module, _ = _HANDLER_SPECS["ingest_context"]
    monkeypatch.setitem(
        _HANDLER_SPECS,
        "ingest_context",
        (module, "handle_missing_for_registry_test"),
    )

    with pytest.raises(
        RuntimeError,
        match="ingest_context:src.handlers.compression_handlers.handle_missing_for_registry_test",
    ):
        _build_registry()


@pytest.mark.asyncio
async def test_registry_handler_proxy_observes_runtime_handler_patches():
    """The refactor keeps the old patch/reload behavior of route dispatch."""
    with patch(
        "src.handlers.compression_handlers.handle_ingest",
        new_callable=AsyncMock,
        return_value="patched",
    ) as handler:
        result = await mcp_core.route_tool_call(
            "ingest_context", {"text": "x"}, {"compressor": object()}
        )
    handler.assert_awaited_once()
    assert result == "patched"


@pytest.mark.asyncio
async def test_get_multi_level_skeleton_routes_to_the_live_handler():
    context = {"compressor": object()}
    with patch(
        "src.handlers.compression_handlers.handle_multi_level_skeleton",
        new_callable=AsyncMock,
        return_value="skeleton",
    ) as handler:
        result = await mcp_core.route_tool_call(
            "get_multi_level_skeleton", {"doc_id": "fixture"}, context
        )
    handler.assert_awaited_once_with(context, {"doc_id": "fixture"})
    assert result == "skeleton"
