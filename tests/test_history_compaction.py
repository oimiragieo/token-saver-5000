from src.history_compaction import compact_conversation_history


def test_history_compaction_preserves_recent_turns_at_tail():
    messages = [
        {"role": "system", "content": "You are precise and terse."},
        {"role": "user", "content": "We are discussing prompt caching and stable prefixes."},
        {"role": "assistant", "content": "Stable prefixes should appear before volatile metadata."},
        {"role": "user", "content": "Now compare Claude, Gemini, and OpenAI cache telemetry."},
        {
            "role": "assistant",
            "content": "Claude uses cache_read_input_tokens; OpenAI uses cached_tokens.",
        },
    ]

    result = compact_conversation_history(
        messages,
        budget_tokens=80,
        preserve_recent_turns=2,
    )

    assert result["preserved_recent_turns"] == 2
    assert "summarized_history" in result["stable_prefix"]
    assert "Now compare Claude, Gemini, and OpenAI cache telemetry." in result["recent_tail"]
    assert "Claude uses cache_read_input_tokens" in result["recent_tail"]
    assert result["stable_prefix_tokens"] + result["recent_tail_tokens"] <= 80
