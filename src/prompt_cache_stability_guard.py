"""Preventive stable-prefix guard for cache-friendly prompt assembly."""

from __future__ import annotations

import hashlib
from typing import Any

from .prompt_cache_audit import _VOLATILE_PATTERNS, audit_prompt_cacheability

_VOLATILITY_RANK: dict[str, int] = {"global": 0, "session": 1, "none": 2}
_ANCHOR_NAMES: frozenset[str] = frozenset({"identity", "constraints"})
_DEFAULT_VOLATILITY_BY_NAME: dict[str, str] = {
    "identity": "global",
    "constraints": "global",
    "tool_definitions": "global",
    "system_instructions": "global",
    "rag_context": "session",
    "few_shot_examples": "session",
    "chat_history": "session",
    "metadata": "none",
    "user_query": "none",
}


def _resolve_volatility(section: dict[str, Any]) -> str:
    vol = section.get("volatility")
    if isinstance(vol, str) and vol in _VOLATILITY_RANK:
        return vol
    return _DEFAULT_VOLATILITY_BY_NAME.get(str(section.get("name", "")), "none")


def cache_align_sections(
    sections: list[dict[str, Any]],
    max_reorder_moves: int = 3,
) -> dict[str, Any]:
    """Deterministically align prompt sections by volatility while pinning anchors.

    Volatility order: ``global < session < none``. Sections named ``identity`` or
    ``constraints`` (or with ``pinned=True``) remain fixed at their original indices.
    Non-pinned sections are moved toward sorted order via adjacent transpositions,
    bounded by ``max_reorder_moves``.
    """
    if not isinstance(sections, list) or not sections:
        raise ValueError("'sections' must be a non-empty list")
    if not isinstance(max_reorder_moves, int) or max_reorder_moves < 0:
        raise ValueError("'max_reorder_moves' must be an integer >= 0")

    pinned_indices: set[int] = set()
    movable_slots: list[int] = []
    movable_items: list[tuple[int, int, dict[str, Any]]] = []

    for idx, sec in enumerate(sections):
        if not isinstance(sec, dict) or "name" not in sec or "content" not in sec:
            raise ValueError("Each section must be a dict with 'name' and 'content'")
        is_pinned = sec.get("name") in _ANCHOR_NAMES or bool(sec.get("pinned", False))
        if is_pinned:
            pinned_indices.add(idx)
        else:
            rank = _VOLATILITY_RANK[_resolve_volatility(sec)]
            movable_slots.append(idx)
            movable_items.append((rank, len(movable_items), dict(sec)))

    # Bound adjacent transpositions to max_reorder_moves using insertion-style swaps.
    moves_used = 0
    for i in range(1, len(movable_items)):
        j = i
        while j > 0 and moves_used < max_reorder_moves:
            if movable_items[j - 1][0] > movable_items[j][0]:
                movable_items[j - 1], movable_items[j] = movable_items[j], movable_items[j - 1]
                moves_used += 1
                j -= 1
            else:
                break

    aligned_sections: list[dict[str, Any]] = [dict(sec) for sec in sections]
    for slot_idx, (_, _, sec) in zip(movable_slots, movable_items):
        aligned_sections[slot_idx] = sec

    has_volatile_prefix_leak = False
    aligned_vols = [_resolve_volatility(sec) for sec in aligned_sections]
    for idx, vol in enumerate(aligned_vols):
        if vol in ("session", "none") and any(v == "global" for v in aligned_vols[idx + 1 :]):
            has_volatile_prefix_leak = True
            break

    cache_breakpoints: list[dict[str, Any]] = []
    max_rank_seen = 0
    for idx, sec in enumerate(aligned_sections):
        vol = aligned_vols[idx]
        rank = _VOLATILITY_RANK[vol]
        if rank > max_rank_seen:
            max_rank_seen = rank
        # Never emit a breakpoint once a more-volatile section has appeared earlier.
        if vol in ("global", "session") and rank >= max_rank_seen:
            next_vol = aligned_vols[idx + 1] if idx + 1 < len(aligned_sections) else None
            if next_vol != vol:
                cache_breakpoints.append({"index": idx, "name": sec["name"], "volatility": vol})

    return {
        "aligned_sections": aligned_sections,
        "moves_used": moves_used,
        "cache_breakpoints": cache_breakpoints,
        "has_volatile_prefix_leak": has_volatile_prefix_leak,
    }


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
