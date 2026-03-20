"""User preference extraction and profile synthesis for explicit memory APIs."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

_PREFERENCE_PATTERNS = (
    re.compile(r"\bprefer\s+(?P<value>[^.!\n]+)", re.IGNORECASE),
    re.compile(r"\balways\s+use\s+(?P<value>[^.!\n]+)", re.IGNORECASE),
    re.compile(r"\bdo\s+not\s+use\s+(?P<value>[^.!\n]+)", re.IGNORECASE),
    re.compile(r"\bdon'?t\s+use\s+(?P<value>[^.!\n]+)", re.IGNORECASE),
    re.compile(r"\blikes?\s+(?P<value>[^.!\n]+)", re.IGNORECASE),
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]+")
_STOPWORDS = {
    "about",
    "after",
    "again",
    "always",
    "because",
    "before",
    "being",
    "between",
    "build",
    "cache",
    "caching",
    "code",
    "context",
    "default",
    "deploy",
    "details",
    "document",
    "dont",
    "from",
    "have",
    "into",
    "like",
    "make",
    "memory",
    "more",
    "never",
    "only",
    "prefer",
    "production",
    "prompt",
    "should",
    "that",
    "their",
    "them",
    "then",
    "they",
    "this",
    "tool",
    "use",
    "user",
    "using",
    "with",
}


def _extract_preferences(text: str) -> list[str]:
    preferences: list[str] = []
    for pattern in _PREFERENCE_PATTERNS:
        for match in pattern.finditer(text):
            value = " ".join(match.group("value").strip().split())
            if value:
                preferences.append(value.rstrip("."))
    return preferences


def _extract_topics(texts: list[str], limit: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        for word in _WORD_RE.findall(text.lower()):
            if len(word) < 4 or word in _STOPWORDS:
                continue
            counts[word] += 1
    return [topic for topic, _ in counts.most_common(limit)]


def build_user_profile(memories: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
    """Build a deterministic user profile from explicit memory entries."""
    category_breakdown = Counter(memory.get("category", "general") for memory in memories)

    preferences: list[str] = []
    seen_preferences: set[str] = set()
    texts = [str(memory.get("text", "")) for memory in memories]
    for text in texts:
        for preference in _extract_preferences(text):
            key = preference.lower()
            if key in seen_preferences:
                continue
            seen_preferences.add(key)
            preferences.append(preference)

    recurring_topics = _extract_topics(texts)
    recent_memories = [
        {
            "memory_id": memory["memory_id"],
            "text": memory["text"],
            "category": memory["category"],
            "created_at": memory["created_at"],
        }
        for memory in memories[:5]
    ]

    summary_parts = [f"Profile for user '{user_id}' is based on {len(memories)} explicit memories."]
    if preferences:
        summary_parts.append(f"Observed preferences: {', '.join(preferences[:3])}.")
    if recurring_topics:
        summary_parts.append(f"Recurring topics: {', '.join(recurring_topics[:3])}.")

    return {
        "user_id": user_id,
        "memory_count": len(memories),
        "category_breakdown": dict(sorted(category_breakdown.items())),
        "preferences": preferences,
        "recurring_topics": recurring_topics,
        "recent_memories": recent_memories,
        "profile_summary": " ".join(summary_parts),
    }


def summarize_user_memories(memories: list[dict[str, Any]], user_id: str) -> dict[str, Any]:
    """Create a lightweight summary view over a user's explicit memories."""
    profile = build_user_profile(memories, user_id)
    return {
        "user_id": user_id,
        "memory_count": profile["memory_count"],
        "category_breakdown": profile["category_breakdown"],
        "preferences": profile["preferences"],
        "recurring_topics": profile["recurring_topics"],
        "summary": profile["profile_summary"],
    }
