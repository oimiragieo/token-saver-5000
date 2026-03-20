"""Context block synthesis from temporal facts and recent access history."""

from __future__ import annotations

from typing import Any

from .temporal_graph import format_timestamp


def build_context_block(
    *,
    doc_id: str,
    active_facts: list[dict[str, Any]],
    recent_events: list[dict[str, Any]],
    access_info: dict[str, Any] | None,
    compression_history: list[dict[str, Any]],
    skeleton_text: str,
    max_facts: int = 5,
) -> dict[str, Any]:
    """Build a compact, lifecycle-aware context block."""
    visible_facts = active_facts[:max_facts]
    latest_ratio = compression_history[-1]["ratio"] if compression_history else None
    last_accessed = access_info.get("last_accessed") if access_info else None
    access_summary = (
        {
            "first_accessed": format_timestamp(access_info["first_accessed"]),
            "last_accessed": format_timestamp(access_info["last_accessed"]),
            "access_count": access_info["access_count"],
        }
        if access_info
        else None
    )
    summary_parts = [
        f"{len(active_facts)} active facts",
        f"{len(recent_events)} recent events",
    ]
    if last_accessed is not None:
        summary_parts.append(f"last accessed {format_timestamp(last_accessed)}")
    if latest_ratio is not None:
        summary_parts.append(f"latest compression ratio {latest_ratio:.2f}x")

    return {
        "doc_id": doc_id,
        "summary": "; ".join(summary_parts),
        "active_fact_count": len(active_facts),
        "active_facts": visible_facts,
        "recent_events": recent_events,
        "access": access_summary,
        "compression": {
            "history_count": len(compression_history),
            "latest_ratio": latest_ratio,
        },
        "cache_stable_prefix": skeleton_text,
    }
