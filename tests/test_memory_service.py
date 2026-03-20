"""Tests for explicit memory service behavior."""

from src.memory_api import MemoryAPI


def setup_function():
    MemoryAPI.reset_singleton()


def test_add_list_search_and_delete_memory():
    api = MemoryAPI.get_api()
    created = api.add_memory(
        text="Prefer pytest fixtures for state isolation.",
        user_id="alice",
        workspace_id="acme",
    )

    listed = api.list_memories(user_id="alice", workspace_id="acme")
    searched = api.search_memory(query="pytest fixtures", user_id="alice", workspace_id="acme")
    deleted = api.delete_memory(created["memory_id"], user_id="alice", workspace_id="acme")

    assert listed[0]["memory_id"] == created["memory_id"]
    assert searched[0]["memory_id"] == created["memory_id"]
    assert searched[0]["score"] > 0
    assert deleted["memory_id"] == created["memory_id"]
    assert api.list_memories(user_id="alice", workspace_id="acme") == []


def test_scope_isolation_matches_phase_one_tenancy_rules():
    api = MemoryAPI.get_api()
    api.add_memory(text="Workspace A prefers concise outputs.", user_id="alice", workspace_id="a")
    api.add_memory(text="Workspace B prefers detailed outputs.", user_id="alice", workspace_id="b")
    api.add_memory(text="Global unscoped note.")

    assert len(api.list_memories(user_id="alice", workspace_id="a")) == 1
    assert len(api.list_memories(user_id="alice", workspace_id="b")) == 1
    assert len(api.list_memories()) == 1


def test_user_summary_and_profile_are_derived_from_explicit_memories():
    api = MemoryAPI.get_api()
    api.add_memory(
        text="Prefer pytest fixtures for isolation.", user_id="alice", workspace_id="acme"
    )
    api.add_memory(
        text="Always use black and ruff before commits.",
        category="pattern",
        user_id="alice",
        workspace_id="acme",
    )

    summary = api.summarize_user_memory(user_id="alice", workspace_id="acme")
    profile = api.get_user_profile(user_id="alice", workspace_id="acme")

    assert summary["memory_count"] == 2
    assert "pytest fixtures for isolation" in " ".join(profile["preferences"]).lower()
    assert profile["user_id"] == "alice"
    assert profile["category_breakdown"]["pattern"] >= 1
