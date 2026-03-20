"""Prompt cacheability auditing helpers for stable-prefix prompt design."""

from __future__ import annotations

import re
from typing import Any

CANONICAL_SECTION_ORDER = [
    "tool_definitions",
    "system_instructions",
    "rag_context",
    "few_shot_examples",
    "chat_history",
    "metadata",
    "user_query",
]

_SECTION_INDEX = {name: index for index, name in enumerate(CANONICAL_SECTION_ORDER)}
_VOLATILE_SECTIONS = {"metadata", "user_query"}
_VOLATILE_PATTERNS = [
    (
        "uuid_like_token",
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
        ),
        "UUID-like identifiers in stable prompt sections usually destroy provider prefix cache reuse.",
    ),
    (
        "iso_timestamp",
        re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"),
        "Timestamps should stay in metadata or user-query tails, not the stable prefix.",
    ),
    (
        "volatile_key_name",
        re.compile(
            r"\b(?:request_id|session_id|trace_id|timestamp|current_time|uuid)\b", re.IGNORECASE
        ),
        "Volatile runtime identifiers belong at the tail of the prompt, not the cached prefix.",
    ),
]


def _normalize_sections(sections: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not isinstance(sections, list) or not sections:
        raise ValueError("'sections' must be a non-empty list")

    normalized: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for item in sections:
        if not isinstance(item, dict):
            raise ValueError("Each section must be an object with 'name' and 'content'")
        name = item.get("name")
        content = item.get("content")
        if name not in _SECTION_INDEX:
            raise ValueError(
                f"Unknown section name '{name}'. Expected one of: {', '.join(CANONICAL_SECTION_ORDER)}"
            )
        if name in seen_names:
            raise ValueError(f"Duplicate section name '{name}' is not allowed")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Section '{name}' requires non-empty string 'content'")
        seen_names.add(name)
        normalized.append({"name": name, "content": content})
    return normalized


def _render_sections(sections: list[dict[str, str]]) -> str:
    return "\n\n".join(f"[{item['name']}]\n{item['content']}" for item in sections)


def audit_prompt_cacheability(sections: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = _normalize_sections(sections)
    issues: list[dict[str, str]] = []
    score = 100

    indices = [_SECTION_INDEX[item["name"]] for item in normalized]
    if indices != sorted(indices):
        issues.append(
            {
                "code": "section_order_violation",
                "message": (
                    "Prompt sections are out of cache-friendly order. Keep tool defs, system instructions, "
                    "docs, and examples ahead of chat history, metadata, and the immediate query."
                ),
            }
        )
        score -= 25

    for item in normalized:
        if item["name"] in _VOLATILE_SECTIONS:
            continue
        for code, pattern, message in _VOLATILE_PATTERNS:
            if pattern.search(item["content"]):
                issues.append(
                    {
                        "code": "volatile_content_in_stable_section",
                        "message": f"{message} Found in '{item['name']}' via {code}.",
                    }
                )
                score -= 20
                break

    first_volatile_index = next(
        (index for index, item in enumerate(normalized) if item["name"] in _VOLATILE_SECTIONS),
        len(normalized),
    )
    stable_sections = normalized[:first_volatile_index]
    volatile_sections = normalized[first_volatile_index:]
    score = max(score, 0)

    return {
        "score": score,
        "is_cache_friendly": score >= 80 and not issues,
        "issues": issues,
        "recommended_order": list(CANONICAL_SECTION_ORDER),
        "present_order": [item["name"] for item in normalized],
        "cacheable_prefix": _render_sections(stable_sections) if stable_sections else "",
        "volatile_suffix": _render_sections(volatile_sections) if volatile_sections else "",
    }
