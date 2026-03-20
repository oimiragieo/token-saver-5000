"""Tests for explicit memory MCP handlers."""

import json

import pytest

from src.memory_api import MemoryAPI


@pytest.fixture
def memory_context():
    MemoryAPI.reset_singleton()
    return {"memory_api": MemoryAPI()}


@pytest.mark.asyncio
async def test_add_search_list_and_delete_memory(memory_context):
    from src.handlers.memory_handlers import (
        handle_add_memory,
        handle_delete_memory,
        handle_list_memories,
        handle_search_memory,
    )

    created = json.loads(
        await handle_add_memory(
            memory_context,
            {
                "text": "Prefer pytest fixtures for isolation.",
                "user_id": "alice",
                "workspace_id": "acme",
            },
        )
    )
    searched = json.loads(
        await handle_search_memory(
            memory_context,
            {"query": "pytest fixtures", "user_id": "alice", "workspace_id": "acme"},
        )
    )
    listed = json.loads(
        await handle_list_memories(memory_context, {"user_id": "alice", "workspace_id": "acme"})
    )
    deleted = json.loads(
        await handle_delete_memory(
            memory_context,
            {
                "memory_id": created["memory"]["memory_id"],
                "user_id": "alice",
                "workspace_id": "acme",
            },
        )
    )

    assert created["status"] == "success"
    assert searched["results"][0]["memory_id"] == created["memory"]["memory_id"]
    assert listed["total_memories"] == 1
    assert deleted["deleted_memory"]["memory_id"] == created["memory"]["memory_id"]


@pytest.mark.asyncio
async def test_summarize_and_profile_handlers(memory_context):
    from src.handlers.memory_handlers import (
        handle_add_memory,
        handle_get_user_profile,
        handle_summarize_user_memory,
    )

    await handle_add_memory(
        memory_context,
        {
            "text": "Prefer concise bullet summaries.",
            "user_id": "alice",
            "workspace_id": "acme",
        },
    )
    await handle_add_memory(
        memory_context,
        {
            "text": "Always use black before commits.",
            "user_id": "alice",
            "workspace_id": "acme",
        },
    )

    summary = json.loads(
        await handle_summarize_user_memory(
            memory_context, {"user_id": "alice", "workspace_id": "acme"}
        )
    )
    profile = json.loads(
        await handle_get_user_profile(memory_context, {"user_id": "alice", "workspace_id": "acme"})
    )

    assert summary["summary"]["memory_count"] == 2
    assert profile["profile"]["user_id"] == "alice"
    assert len(profile["profile"]["preferences"]) >= 1
