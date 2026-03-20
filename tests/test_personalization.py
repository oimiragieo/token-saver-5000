"""Tests for personalization synthesis from explicit memories."""

from src.personalization import build_user_profile, summarize_user_memories


def test_build_user_profile_extracts_preferences_and_topics():
    memories = [
        {
            "memory_id": "mem_2",
            "text": "Prefer pytest fixtures for state isolation.",
            "category": "pattern",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "memory_id": "mem_1",
            "text": "Always use black and ruff before commits.",
            "category": "decision",
            "created_at": "2026-01-02T00:00:00Z",
        },
    ]

    profile = build_user_profile(memories, "alice")

    assert profile["user_id"] == "alice"
    assert profile["memory_count"] == 2
    assert any("pytest fixtures" in preference.lower() for preference in profile["preferences"])
    assert "pytest" in profile["recurring_topics"] or "fixtures" in profile["recurring_topics"]


def test_summarize_user_memories_returns_compact_summary_view():
    memories = [
        {
            "memory_id": "mem_1",
            "text": "Prefer concise bullet summaries.",
            "category": "general",
            "created_at": "2026-01-01T00:00:00Z",
        }
    ]

    summary = summarize_user_memories(memories, "alice")

    assert summary["user_id"] == "alice"
    assert summary["memory_count"] == 1
    assert "explicit memories" in summary["summary"].lower()
