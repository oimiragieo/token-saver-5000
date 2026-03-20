"""Golden regression tests for explicit memory profile synthesis."""

from src.personalization import build_user_profile


def test_profile_output_shape_is_stable():
    profile = build_user_profile(
        [
            {
                "memory_id": "mem_2",
                "text": "Prefer concise bullet summaries.",
                "category": "general",
                "created_at": "2026-01-02T00:00:00Z",
            },
            {
                "memory_id": "mem_1",
                "text": "Always use black before commits.",
                "category": "pattern",
                "created_at": "2026-01-01T00:00:00Z",
            },
        ],
        "alice",
    )

    assert profile == {
        "user_id": "alice",
        "memory_count": 2,
        "category_breakdown": {"general": 1, "pattern": 1},
        "preferences": profile["preferences"],
        "recurring_topics": profile["recurring_topics"],
        "recent_memories": profile["recent_memories"],
        "profile_summary": profile["profile_summary"],
    }
    assert any("concise bullet summaries" in item.lower() for item in profile["preferences"])
