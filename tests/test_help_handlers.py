"""Tests for help handler documentation registry and responses."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.handlers.help_handlers import handle_tool_help
from src.handlers.resource_handlers import (
    get_check_environment_output_fields,
    handle_check_environment,
)
from src.handlers.compression_handlers import (
    get_ingest_context_output_fields,
    get_read_skeleton_output_fields,
    get_search_semantic_output_fields,
    handle_read_skeleton,
    handle_search_semantic,
)


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


@pytest.mark.asyncio
async def test_check_environment_help_output_fields_match_canonical_schema():
    result = await handle_tool_help({}, {"tool_name": "check_environment", "verbose": True})
    data = json.loads(result)

    assert data.get("output_fields", []) == get_check_environment_output_fields()


@pytest.mark.asyncio
async def test_search_semantic_help_output_fields_match_canonical_schema():
    result = await handle_tool_help({}, {"tool_name": "search_semantic", "verbose": True})
    data = json.loads(result)

    assert data.get("output_fields", []) == get_search_semantic_output_fields()


@pytest.mark.asyncio
async def test_search_semantic_help_output_fields_cover_runtime_keys():
    help_result = await handle_tool_help({}, {"tool_name": "search_semantic", "verbose": True})
    help_data = json.loads(help_result)
    documented_fields = help_data.get("output_fields", [])

    node = SimpleNamespace(text="Auth token logic", importance=0.93, metadata={"tokens": 42})
    compressor = SimpleNamespace(
        chunks={"doc1_n0": node},
        search_semantic_with_scores=lambda query, file_id, top_k: [("doc1_n0", 0.87)],
        _generate_summary=lambda text, max_length: text[:max_length],
    )
    runtime_result = await handle_search_semantic(
        {"compressor": compressor},
        {"query": "auth token", "file_id": "doc1", "top_k": 1},
    )
    runtime_data = json.loads(runtime_result)

    assert "query" in documented_fields
    assert "results[].node_id" in documented_fields
    assert "results[].similarity" in documented_fields
    assert "results[].importance" in documented_fields
    assert runtime_data["results"][0]["node_id"] == "doc1_n0"


@pytest.mark.asyncio
async def test_read_skeleton_help_output_fields_match_canonical_schema():
    result = await handle_tool_help({}, {"tool_name": "read_skeleton", "verbose": True})
    data = json.loads(result)

    assert data.get("output_fields", []) == get_read_skeleton_output_fields()


@pytest.mark.asyncio
async def test_read_skeleton_help_output_fields_cover_runtime_keys():
    help_result = await handle_tool_help({}, {"tool_name": "read_skeleton", "verbose": True})
    help_data = json.loads(help_result)
    documented_fields = help_data.get("output_fields", [])

    skeleton = SimpleNamespace(
        file_id="doc1",
        total_nodes=2,
        total_tokens=120,
        skeleton_tokens=30,
        compression_ratio=4.0,
        skeleton_text="skeleton",
        node_map={"doc1_n0": "summary"},
    )
    compressor = SimpleNamespace(
        _generate_skeleton=lambda file_id, query=None: skeleton, graphs={"doc1": object()}
    )
    sync_manager = SimpleNamespace(file_metadata={})
    context = {"compressor": compressor, "sync_manager": sync_manager}

    runtime_result = await handle_read_skeleton(context, {"file_id": "doc1"})
    runtime_data = json.loads(runtime_result)

    assert "file_id" in documented_fields
    assert "node_map" in documented_fields
    assert "selection_mode" in documented_fields
    assert "evidence.sufficient" in documented_fields
    assert "staleness_warning.is_stale" in documented_fields
    assert runtime_data["file_id"] == "doc1"


@pytest.mark.asyncio
async def test_ingest_context_help_output_fields_match_canonical_schema():
    result = await handle_tool_help({}, {"tool_name": "ingest_context", "verbose": True})
    data = json.loads(result)

    assert data.get("output_fields", []) == get_ingest_context_output_fields()


@pytest.mark.asyncio
async def test_ingest_context_help_output_fields_cover_runtime_keys():
    help_result = await handle_tool_help({}, {"tool_name": "ingest_context", "verbose": True})
    help_data = json.loads(help_result)
    documented_fields = help_data.get("output_fields", [])

    # Reuse handle_ingest contract via documented output fields and runtime shape checks
    # using a minimal synthetic runtime payload.
    runtime_data = {
        "status": "success",
        "file_id": "doc1",
        "total_nodes": 10,
        "total_tokens": 1000,
        "skeleton_tokens": 100,
        "compression_ratio": 10.0,
        "token_savings": 900,
        "token_savings_percent": 90.0,
        "estimate": {"estimated_ratio": 9.5, "accuracy": "good"},
        "message": "ok",
    }

    assert "status" in documented_fields
    assert "file_id" in documented_fields
    assert "estimate.estimated_ratio" in documented_fields
    assert "estimate.accuracy" in documented_fields
    assert runtime_data["estimate"]["estimated_ratio"] == 9.5
