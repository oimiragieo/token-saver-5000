"""Reusable cache for segment-level compression artifacts."""

from __future__ import annotations

import hashlib
from typing import Any, Callable


def _hash_value(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class SegmentCompressionCache:
    """Small in-memory cache keyed by segment text, method, and query."""

    def __init__(self) -> None:
        self._entries: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    def _build_key(self, *, segment_text: str, method: str, query: str | None) -> str:
        normalized_query = query or ""
        return f"{method}:{_hash_value(segment_text)}:{_hash_value(normalized_query)}"

    def get_or_compute(
        self,
        *,
        segment_text: str,
        method: str,
        query: str | None,
        compute: Callable[[], Any],
    ) -> Any:
        key = self._build_key(segment_text=segment_text, method=method, query=query)
        if key in self._entries:
            self._hits += 1
            return self._entries[key]
        self._misses += 1
        value = compute()
        self._entries[key] = value
        return value

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
        }
