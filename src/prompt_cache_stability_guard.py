"""Preventive stable-prefix guard for cache-friendly prompt assembly."""

from __future__ import annotations

import hashlib
from typing import Any

from .prompt_cache_audit import _VOLATILE_PATTERNS, audit_prompt_cacheability


def evaluate_prompt_stability(
    sections: list[dict[str, Any]],
    *,
    max_stable_prefix_chars: int = 5000,
) -> dict[str, Any]:
    """Return a guard verdict for the stable prompt prefix."""
    if not isinstance(max_stable_prefix_chars, int) or max_stable_prefix_chars <= 0:
        raise ValueError("'max_stable_prefix_chars' must be an integer > 0")

    audit = audit_prompt_cacheability(sections)
    stable_prefix = audit["cacheable_prefix"]
    inspected_prefix = stable_prefix[:max_stable_prefix_chars]
    violations: list[dict[str, str | int]] = []

    for pattern_code, pattern, message in _VOLATILE_PATTERNS:
        match = pattern.search(inspected_prefix)
        if match is None:
            continue
        violations.append(
            {
                "code": "volatile_content_in_stable_prefix",
                "pattern": pattern_code,
                "message": message,
                "match": match.group(0),
                "position": match.start(),
            }
        )

    if any(issue["code"] == "section_order_violation" for issue in audit["issues"]):
        violations.append(
            {
                "code": "section_order_violation",
                "message": (
                    "Stable prompt sections must stay in canonical order before volatile metadata "
                    "and the user query."
                ),
                "match": "",
                "position": 0,
            }
        )

    warning = None
    if violations:
        warning = (
            "Prompt stability guard found cache-breaking content in the stable prefix. "
            "Move volatile data to metadata or the user-query tail before sending this prompt."
        )

    return {
        "is_stable": len(violations) == 0,
        "order_valid": not any(item["code"] == "section_order_violation" for item in violations),
        "stable_prefix": stable_prefix,
        "stable_prefix_hash": hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest(),
        "stable_prefix_length": len(stable_prefix),
        "inspected_prefix_length": min(len(stable_prefix), max_stable_prefix_chars),
        "max_stable_prefix_chars": max_stable_prefix_chars,
        "violations": violations,
        "warning": warning,
    }
