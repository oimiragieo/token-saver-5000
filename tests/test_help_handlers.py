"""Tests for help handler documentation registry and responses."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.handlers.help_handlers import handle_tool_help
from src.handlers.resource_handlers import handle_check_environment


@pytest.mark.asyncio
async def test_read_skeleton_help_includes_query_guided_params():
    result = await handle_tool_help({}, {"tool_name": "read_skeleton", "verbose": True})
    data = json.loads(result)

    params = data["parameters"]
    assert "selection_mode" in params
    assert "query" in params
    assert "top_k" in params
    assert "min_similarity" in params


@pytest.mark.asyncio
async def test_search_semantic_help_includes_evidence_aware_params():
    result = await handle_tool_help({}, {"tool_name": "search_semantic", "verbose": True})
    data = json.loads(result)

    params = data["parameters"]
    assert "evidence_aware" in params
    assert "min_similarity" in params


@pytest.mark.asyncio
async def test_tool_list_response_contains_total_tools():
    result = await handle_tool_help({}, {})
    data = json.loads(result)
    assert data["status"] == "tool_list"
    assert data["total_tools"] > 0


@pytest.mark.asyncio
async def test_check_environment_help_mentions_tool_profile_diagnostics():
    result = await handle_tool_help({}, {"tool_name": "check_environment", "verbose": True})
    data = json.loads(result)

    tips_text = " ".join(data.get("tips", [])).lower()
    description_text = data.get("description", "").lower()

    assert "tool_profile" in tips_text or "tool profile" in tips_text
    assert "enabled_tools" in tips_text or "enabled tool" in tips_text
    assert "environment" in description_text


@pytest.mark.asyncio
async def test_check_environment_help_output_fields_match_runtime_keys():
    help_result = await handle_tool_help({}, {"tool_name": "check_environment", "verbose": True})
    help_data = json.loads(help_result)
    documented_fields = help_data.get("output_fields", [])

    compressor = SimpleNamespace(graphs={}, chunks={})
    sync_manager = Mock()
    sync_manager.export_metadata.return_value = {}
    context = {
        "compressor": compressor,
        "sync_manager": sync_manager,
        "tool_profile": "core_stable",
        "enabled_tool_names": ["ingest_context"],
    }

    runtime_result = await handle_check_environment(context, {})
    runtime_data = json.loads(runtime_result)
    runtime_profile = runtime_data.get("tool_profile", {})

    assert "tool_profile.profile" in documented_fields
    assert "tool_profile.enabled_tool_count" in documented_fields
    assert "tool_profile.enabled_tools" in documented_fields
    assert runtime_profile.get("profile") == "core_stable"
    assert runtime_profile.get("enabled_tool_count") == 1
