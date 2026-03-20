"""Stable-prefix-preserving conversation history compaction."""

from __future__ import annotations

from typing import Any

import tiktoken

from .extractive_baseline import ExtractiveCompressor


def _count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _render_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = str(message.get("content", "")).strip()
        if content:
            rendered.append(f"[{role}] {content}")
    return "\n".join(rendered)


def compact_conversation_history(
    messages: list[dict[str, Any]],
    *,
    budget_tokens: int,
    preserve_recent_turns: int = 2,
) -> dict[str, Any]:
    """Compact older conversation turns while preserving recent tail turns verbatim."""
    if budget_tokens <= 0:
        raise ValueError("compact_conversation_history requires 'budget_tokens' > 0")

    system_messages = [message for message in messages if message.get("role") == "system"]
    non_system = [message for message in messages if message.get("role") != "system"]
    recent = non_system[-preserve_recent_turns:] if preserve_recent_turns > 0 else []
    older = non_system[: max(0, len(non_system) - len(recent))]

    recent_tail = _render_messages(recent)
    recent_tail_tokens = _count_tokens(recent_tail)
    prefix_budget = max(budget_tokens - recent_tail_tokens, 0)

    stable_parts: list[str] = []
    system_prefix = _render_messages(system_messages)
    if system_prefix:
        stable_parts.append(system_prefix)
        prefix_budget = max(prefix_budget - _count_tokens(system_prefix), 0)

    older_rendered = _render_messages(older)
    if older_rendered:
        summary_target = max(prefix_budget, 12)
        summary = ExtractiveCompressor().compress_text(
            older_rendered,
            query="conversation summary important facts",
            target_tokens=summary_target,
        )["compressed_text"]
        stable_parts.append(f"[summarized_history]\n{summary}")

    stable_prefix = "\n\n".join(part for part in stable_parts if part.strip())
    stable_prefix_tokens = _count_tokens(stable_prefix)

    if stable_prefix_tokens + recent_tail_tokens > budget_tokens and stable_prefix:
        overflow = stable_prefix_tokens + recent_tail_tokens - budget_tokens
        trimmed_target = max(stable_prefix_tokens - overflow, 12)
        summarized_history = ExtractiveCompressor().compress_text(
            stable_prefix,
            query="stable conversation summary",
            target_tokens=trimmed_target,
        )["compressed_text"]
        stable_prefix = f"[summarized_history]\n{summarized_history}"
        stable_prefix_tokens = _count_tokens(stable_prefix)

    return {
        "stable_prefix": stable_prefix,
        "recent_tail": recent_tail,
        "stable_prefix_tokens": stable_prefix_tokens,
        "recent_tail_tokens": recent_tail_tokens,
        "preserved_recent_turns": len(recent),
    }
