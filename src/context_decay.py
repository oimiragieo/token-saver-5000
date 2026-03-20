"""
Context decay and eviction for stale document management.

Based on DynamicKV (ICLR 2025) and ACON (2025) — tracks document access
recency and auto-decays relevance scores over time, enabling automatic
eviction of stale context to keep the context budget tight.
"""

import math
import time
from typing import Any, Dict, List, Optional


class AccessTracker:
    """Tracks document access timestamps and counts."""

    def __init__(self):
        self._access_log: Dict[str, dict] = {}
        self._events: List[dict[str, Any]] = []

    def record_access(
        self,
        doc_id: str,
        timestamp: float | None = None,
        access_type: str = "access",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an access event for a document."""
        now = time.time() if timestamp is None else float(timestamp)
        if doc_id not in self._access_log:
            self._access_log[doc_id] = {
                "first_accessed": now,
                "last_accessed": now,
                "access_count": 1,
            }
        else:
            self._access_log[doc_id]["last_accessed"] = now
            self._access_log[doc_id]["access_count"] += 1
        self._events.append(
            {
                "doc_id": doc_id,
                "event_type": access_type,
                "timestamp": now,
                "metadata": dict(metadata or {}),
            }
        )

    def get_access_info(self, doc_id: str) -> Optional[dict]:
        """Get access info for a document, or None if not tracked."""
        return self._access_log.get(doc_id)

    def find_stale(self, max_age_seconds: float = 3600) -> List[str]:
        """Find documents not accessed within max_age_seconds.

        Args:
            max_age_seconds: Maximum age in seconds before considered stale

        Returns:
            List of stale document IDs
        """
        now = time.time()
        stale = []
        for doc_id, info in self._access_log.items():
            age = now - info["last_accessed"]
            if age > max_age_seconds:
                stale.append(doc_id)
        return stale

    def get_all_stats(self) -> Dict[str, dict]:
        """Get access stats for all tracked documents."""
        return dict(self._access_log)

    def get_access_timeline(
        self,
        doc_id: str | None = None,
        *,
        since: float | None = None,
        until: float | None = None,
        access_type: str | None = None,
        limit: int = 50,
    ) -> List[dict[str, Any]]:
        """Return individual access events in reverse chronological order."""
        events: List[dict[str, Any]] = []
        for event in reversed(self._events):
            if doc_id is not None and event["doc_id"] != doc_id:
                continue
            if access_type is not None and event["event_type"] != access_type:
                continue
            if since is not None and event["timestamp"] < since:
                continue
            if until is not None and event["timestamp"] > until:
                continue
            events.append(dict(event))
            if len(events) >= limit:
                break
        return events


def compute_decay_score(
    last_accessed: float,
    access_count: int,
    base_importance: float,
    half_life_seconds: float = 3600,
) -> float:
    """Compute a decayed importance score based on access recency.

    Uses exponential decay: score decreases by half every half_life_seconds.
    Access frequency provides a small boost.

    Args:
        last_accessed: Unix timestamp of last access
        access_count: Number of times accessed
        base_importance: Original importance score (0-1)
        half_life_seconds: Time for score to halve (default: 1 hour)

    Returns:
        Decayed score in [0, 1]
    """
    age = time.time() - last_accessed
    decay_factor = math.pow(0.5, age / half_life_seconds)

    # Small frequency boost (log-scaled, capped at 0.2)
    freq_boost = min(0.2, math.log1p(access_count) * 0.05)

    score = base_importance * decay_factor + freq_boost
    return max(0.0, min(1.0, score))
