"""
Response formatter for MCP tool results.

Enforces TOOL_RESULT_SOFT_LIMIT_CHARS / TOOL_RESULT_HARD_LIMIT_CHARS to keep
responses within Claude Code's tool-result size caps. Adds token estimates and
pagination metadata to every response.
"""

from __future__ import annotations

import hashlib
import json

from src.constants import (
    TOOL_RESULT_HARD_LIMIT_CHARS,
    TOOL_RESULT_PREVIEW_CHARS,
    TOOL_RESULT_SOFT_LIMIT_CHARS,
)
from src.token_estimation import TokenEstimator

# Keys stripped when a response exceeds the soft limit.
METADATA_KEYS_TO_STRIP = ["processing_time_ms", "cache_stats", "debug_info", "timing", "_internal"]

# Keys that are stable across repeated calls (good for KV cache prefix matching).
STABLE_KEYS = {"status", "file_id", "tool_name", "schema_version", "_header", "error"}

# Keys that change on every call and should appear last to avoid cache invalidation.
VOLATILE_KEYS = {
    "processing_time_ms",
    "timestamp",
    "cache_stats",
    "debug_info",
    "_internal",
    "timing",
    "_token_estimates",
    "_truncated",
    "_continuation_token",
}


def cache_stable_format(data: dict, provider: str) -> dict:
    """Re-order *data* keys to maximise KV-cache prefix stability.

    For ``anthropic`` and ``google`` providers the ordering is:
      stable keys -> content keys -> volatile keys

    For ``openai`` providers the same ordering is applied, and then a
    ``_stable_summary`` key is appended at the tail so that middle-truncation
    still exposes both ``status`` and ``file_id`` at the end of the response.

    For unknown or ``None`` providers the original dict is returned unchanged
    (backward compatibility guarantee).

    Args:
        data: The response dict to re-order.
        provider: One of ``"anthropic"``, ``"google"``, ``"openai"``, or any
            other value to opt out of ordering.

    Returns:
        A new dict with keys in the provider-appropriate order.
        If *provider* is unknown/None, the same *data* object is returned
        without copying.
    """
    if provider not in ("anthropic", "google", "openai"):
        return data

    stable: dict = {}
    content: dict = {}
    volatile: dict = {}

    for key, value in data.items():
        if key in STABLE_KEYS:
            stable[key] = value
        elif key in VOLATILE_KEYS:
            volatile[key] = value
        else:
            content[key] = value

    ordered: dict = {}
    ordered.update(stable)
    ordered.update(content)
    ordered.update(volatile)

    if provider == "openai":
        # Mirror stable keys at the tail so middle-truncation preserves both ends.
        summary: dict = {}
        if "status" in stable:
            summary["status"] = stable["status"]
        if "file_id" in stable:
            summary["file_id"] = stable["file_id"]
        if summary:
            ordered["_stable_summary"] = summary

    return ordered


_estimator = TokenEstimator()


def _serialized_size(data: dict) -> int:
    """Return the character length of the JSON serialization of *data*."""
    return len(json.dumps(data, separators=(",", ":")))


def _make_continuation_token(data: dict) -> str:
    """Create a deterministic continuation token based on content hash.

    Args:
        data: The original (pre-pagination) response dict.

    Returns:
        Hex digest string that uniquely identifies this content.
    """
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


_FIDELITY_ORDER = ["RAW", "DETAILED", "STRUCTURE", "OUTLINE", "ABSTRACT"]
"""Fidelity levels ordered from least to most compact (index 0 = least compressed)."""


def _is_more_compact_fidelity(a: str, b: str) -> bool:
    """Return True if fidelity *a* is more compact (higher compression) than *b*.

    Args:
        a: Fidelity level string to compare.
        b: Reference fidelity level string.

    Returns:
        True if *a* appears later in the fidelity order (more compressed) than *b*.
    """
    try:
        return _FIDELITY_ORDER.index(a) > _FIDELITY_ORDER.index(b)
    except ValueError:
        return False


class ResponseFormatter:
    """Formats MCP tool responses to stay within size limits.

    Behaviour:
    - Responses <= soft_limit: pass through with metadata appended.
    - Responses > soft_limit and <= hard_limit: strip internal/debug keys, re-check.
    - Responses > hard_limit after stripping: apply truncation_strategy.

    Truncation strategies:
    - "paginate" (default): preview + continuation token (existing behaviour).
    - "proportional": 20% head + ellipsis + 80% tail with [Truncated by Token Saver] prefix.
    - "head": keep first hard_limit chars of serialized JSON.

    Args:
        soft_limit: Character threshold at which metadata stripping is attempted.
        hard_limit: Character threshold at which truncation is applied.
        preview_chars: Max characters included in a paginated preview.
        truncation_strategy: One of "paginate", "proportional", or "head".
    """

    def __init__(
        self,
        soft_limit: int = TOOL_RESULT_SOFT_LIMIT_CHARS,
        hard_limit: int = TOOL_RESULT_HARD_LIMIT_CHARS,
        preview_chars: int = TOOL_RESULT_PREVIEW_CHARS,
        truncation_strategy: str = "paginate",
    ) -> None:
        self.soft_limit = soft_limit
        self.hard_limit = hard_limit
        self.preview_chars = preview_chars
        self.truncation_strategy = truncation_strategy

    def format_response(
        self,
        data: dict,
        tool_name: str | None = None,
        provider: str | None = None,
    ) -> dict:
        """Format *data* for safe delivery as an MCP tool result.

        Always adds _token_estimates, _truncated, and _continuation_token.
        Never mutates the original *data* dict.

        Args:
            data: The raw tool result dict to format.
            tool_name: Optional tool name. When provided, a _header key is inserted
                as the first key in the returned dict.
            provider: Optional provider hint (``"anthropic"``, ``"google"``, ``"openai"``).
                When set, ``cache_stable_format`` is applied to the data BEFORE
                truncation so that KV-cache prefix matching is maximised.
                Defaults to ``None`` (no reordering — backward compatible).

        Returns:
            A new dict with formatting metadata attached.
        """
        # Apply cache-stable ordering before any size measurement so the
        # ordering is preserved even if the response fits within limits.
        if provider is not None:
            data = cache_stable_format(data, provider)

        # Work on a shallow copy so we never mutate the caller's dict.
        working = dict(data)

        serialized = json.dumps(working, separators=(",", ":"))
        size = len(serialized)

        # --- soft-limit: strip heavyweight metadata keys ---
        if size > self.soft_limit:
            for key in METADATA_KEYS_TO_STRIP:
                working.pop(key, None)
            serialized = json.dumps(working, separators=(",", ":"))
            size = len(serialized)

        # --- hard-limit: apply truncation strategy ---
        if size > self.hard_limit:
            result: dict = self._apply_truncation(data, serialized)
            result["_token_estimates"] = _estimator.estimate_all(serialized)
            return self._prepend_header(result, tool_name, serialized)

        # --- under hard limit: attach metadata and return ---
        result = working
        result["_truncated"] = False
        result["_continuation_token"] = None
        result["_token_estimates"] = _estimator.estimate_all(serialized)
        return self._prepend_header(result, tool_name, serialized)

    def _apply_truncation(self, original_data: dict, serialized: str) -> dict:
        """Apply the configured truncation strategy.

        Args:
            original_data: The original (pre-stripping) response dict.
            serialized: The current JSON serialization of the working dict.

        Returns:
            A new dict with truncation metadata set.
        """
        if self.truncation_strategy == "proportional":
            return self._truncate_proportional(serialized)
        if self.truncation_strategy == "head":
            return self._truncate_head(serialized)
        # Default: "paginate"
        return self._truncate_paginate(original_data, serialized)

    def _truncate_paginate(self, original_data: dict, serialized: str) -> dict:
        """Paginate: preview + deterministic continuation token."""
        token = _make_continuation_token(original_data)
        preview_str = serialized[: self.preview_chars]
        return {
            "_preview": preview_str,
            "_truncated": True,
            "_continuation_token": token,
        }

    def _truncate_proportional(self, serialized: str) -> dict:
        """Proportional: 20% head + ellipsis + 80% tail with [Truncated] prefix."""
        original_len = len(serialized)
        head_end = max(1, original_len // 5)  # 20%
        tail_start = original_len - max(1, int(original_len * 0.80))  # 80% tail
        head = serialized[:head_end]
        tail = serialized[tail_start:]
        original_tokens = _estimator.estimate_fast(serialized)
        truncated_chars = head_end + len(tail)
        prefix = f"[Truncated by Token Saver: {original_tokens} tokens -> {truncated_chars} chars]"
        content = f"{prefix}\n{head}\n...\n{tail}"
        return {
            "_preview": content,
            "_truncated": True,
            "_continuation_token": None,
        }

    def _truncate_head(self, serialized: str) -> dict:
        """Head: keep first hard_limit chars of serialized JSON."""
        head = serialized[: self.hard_limit]
        return {
            "_preview": head,
            "_truncated": True,
            "_continuation_token": None,
        }

    def _prepend_header(self, result: dict, tool_name: str | None, serialized: str) -> dict:
        """Optionally prepend a _header key as the first entry in the result dict.

        Args:
            result: The formatted response dict.
            tool_name: Optional tool name string.
            serialized: The JSON serialization used for key stats.

        Returns:
            The result dict, possibly with _header prepended.
        """
        if tool_name is None:
            return result

        token_count = _estimator.estimate_fast(serialized)
        key_stats = f"{len(result)} keys, ~{token_count} tokens"
        raw_header = f"Token Saver | {tool_name} | {key_stats}"
        header = raw_header[:199]  # Enforce < 200 chars

        # Build a new ordered dict with _header first
        ordered: dict = {"_header": header}
        ordered.update(result)
        return ordered
