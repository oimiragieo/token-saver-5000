"""Tests for help handler documentation registry and responses."""

import json

import pytest

from src.handlers.help_handlers import handle_tool_help


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
