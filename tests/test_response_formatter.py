"""
Tests for src/response_formatter.py (ResponseFormatter class).

TDD test suite for v0.11.0 token optimization.

ResponseFormatter:
- Measures response size against TOOL_RESULT_SOFT_LIMIT_CHARS / TOOL_RESULT_HARD_LIMIT_CHARS
- Strips metadata keys (processing_time_ms, cache_stats, debug_info) when over soft limit
- Paginates (returns preview + continuation token) when over hard limit
- Adds _token_estimates and _truncated fields to every response
- Continuation token is deterministic for the same input
"""

import json
import pytest

from src.response_formatter import ResponseFormatter
from src.constants import TOOL_RESULT_SOFT_LIMIT_CHARS, TOOL_RESULT_HARD_LIMIT_CHARS

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def formatter():
    """Default ResponseFormatter with production limits."""
    return ResponseFormatter()


@pytest.fixture
def small_formatter():
    """ResponseFormatter with small limits for deterministic testing."""
    # soft=200, hard=400 — easy to exceed with test data
    return ResponseFormatter(soft_limit=200, hard_limit=400)


def _make_response(extra_chars: int = 0, include_metadata: bool = False) -> dict:
    """Build a response dict of controlled size."""
    payload = {"result": "x" * extra_chars, "status": "ok"}
    if include_metadata:
        payload["processing_time_ms"] = 42
        payload["cache_stats"] = {"hits": 1, "misses": 0}
        payload["debug_info"] = {"trace": "abc"}
    return payload


# ============================================================================
# Basic pass-through tests
# ============================================================================


def test_format_response_under_limit_passes_through(formatter):
    """Response well under soft limit is returned with data keys intact."""
    data = {"result": "hello", "status": "ok"}
    out = formatter.format_response(data)

    assert out["result"] == "hello"
    assert out["status"] == "ok"


def test_format_response_preserves_original_data_keys(formatter):
    """Original payload keys are preserved in formatted output."""
    data = {"alpha": 1, "beta": [1, 2, 3], "gamma": {"nested": True}}
    out = formatter.format_response(data)

    assert out["alpha"] == 1
    assert out["beta"] == [1, 2, 3]
    assert out["gamma"] == {"nested": True}


# ============================================================================
# Token estimates in every response
# ============================================================================


def test_token_estimates_included_in_every_response(formatter):
    """Every formatted response includes a _token_estimates field."""
    out = formatter.format_response({"msg": "test"})

    assert "_token_estimates" in out
    estimates = out["_token_estimates"]
    assert "accurate" in estimates or "estimated" in estimates
    assert "bytes" in estimates


def test_token_estimates_have_numeric_values(formatter):
    """All _token_estimates values are non-negative numbers."""
    out = formatter.format_response({"msg": "hello world"})
    estimates = out["_token_estimates"]

    for key, value in estimates.items():
        assert isinstance(value, (int, float)), f"_token_estimates[{key!r}] is not numeric"
        assert value >= 0, f"_token_estimates[{key!r}] is negative"


# ============================================================================
# Truncated flag
# ============================================================================


def test_truncated_flag_false_for_small_response(formatter):
    """Responses under hard limit have _truncated=False."""
    out = formatter.format_response({"msg": "small"})

    assert "_truncated" in out
    assert out["_truncated"] is False


def test_truncated_flag_set_when_paginated(small_formatter):
    """Responses over hard limit have _truncated=True."""
    # Build something bigger than hard_limit=400
    big_data = {"content": "A" * 500}
    out = small_formatter.format_response(big_data)

    assert out["_truncated"] is True


# ============================================================================
# Soft limit: strip metadata
# ============================================================================


def test_format_response_at_soft_limit_strips_metadata(small_formatter):
    """When over soft limit, processing_time_ms / cache_stats / debug_info are stripped."""
    # Build a response that exceeds soft_limit=200 but stays under hard_limit=400
    large_content = "B" * 180  # base payload plus metadata puts it over 200
    data = {
        "result": large_content,
        "processing_time_ms": 99,
        "cache_stats": {"hits": 10},
        "debug_info": {"stack": "trace"},
    }
    out = small_formatter.format_response(data)

    serialized = json.dumps(data)
    if len(serialized) > small_formatter.soft_limit:
        assert "processing_time_ms" not in out or out.get("_truncated") is True
        # At minimum, metadata stripping or pagination must have occurred
        assert out.get("_truncated") is True or "processing_time_ms" not in out


def test_metadata_keys_stripped_when_serialized_size_exceeds_soft_limit():
    """Metadata keys are stripped when JSON size exceeds soft limit."""
    # Use a tiny soft limit to ensure stripping happens
    fmt = ResponseFormatter(soft_limit=50, hard_limit=200)
    data = {
        "result": "ok",
        "processing_time_ms": 123,
        "cache_stats": {"hits": 5, "misses": 2},
        "debug_info": {"level": "verbose"},
    }
    out = fmt.format_response(data)

    # The serialized size of data exceeds 50 chars, so metadata must be stripped or truncated
    serialized = json.dumps(data)
    if len(serialized) > fmt.soft_limit:
        metadata_keys = {"processing_time_ms", "cache_stats", "debug_info"}
        remaining = metadata_keys & set(out.keys())
        # Either metadata is stripped or response is paginated (truncated)
        assert len(remaining) == 0 or out.get("_truncated") is True


# ============================================================================
# Hard limit: pagination
# ============================================================================


def test_format_response_at_hard_limit_paginates(small_formatter):
    """Response exceeding hard limit returns a preview with a continuation token."""
    big_data = {"content": "Z" * 500}
    out = small_formatter.format_response(big_data)

    assert out["_truncated"] is True
    assert "_continuation_token" in out
    assert out["_continuation_token"] is not None


def test_paginated_response_contains_preview():
    """Paginated responses include a non-empty _preview or truncated content."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100)
    big_data = {"content": "Y" * 200}
    out = fmt.format_response(big_data)

    assert out["_truncated"] is True
    # The response should not be completely empty — some indicator of content
    assert "_continuation_token" in out or "_preview" in out


def test_continuation_token_is_none_when_not_paginated(formatter):
    """Non-paginated responses have _continuation_token=None."""
    out = formatter.format_response({"msg": "tiny"})

    assert out.get("_continuation_token") is None


# ============================================================================
# Continuation token determinism
# ============================================================================


def test_pagination_continuation_token_is_deterministic():
    """Same input data produces the same continuation token."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100)
    data = {"content": "W" * 200}

    out1 = fmt.format_response(data)
    out2 = fmt.format_response(data)

    assert out1["_continuation_token"] == out2["_continuation_token"]


def test_different_data_produces_different_tokens():
    """Different input data produces different continuation tokens."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100)
    data_a = {"content": "A" * 200}
    data_b = {"content": "B" * 200}

    out_a = fmt.format_response(data_a)
    out_b = fmt.format_response(data_b)

    assert out_a["_continuation_token"] != out_b["_continuation_token"]


# ============================================================================
# Custom limits via constructor
# ============================================================================


def test_custom_limits_via_constructor():
    """ResponseFormatter respects custom soft_limit and hard_limit."""
    fmt = ResponseFormatter(soft_limit=10, hard_limit=20)
    # Both limits are tiny — anything non-trivial should trigger truncation
    data = {"content": "Hello World, this is some content!"}
    out = fmt.format_response(data)

    # With such small limits the response must be truncated
    assert out["_truncated"] is True


def test_default_limits_match_constants():
    """Default ResponseFormatter limits match TOOL_RESULT_SOFT/HARD_LIMIT_CHARS."""
    fmt = ResponseFormatter()
    assert fmt.soft_limit == TOOL_RESULT_SOFT_LIMIT_CHARS
    assert fmt.hard_limit == TOOL_RESULT_HARD_LIMIT_CHARS


# ============================================================================
# Edge cases
# ============================================================================


def test_empty_response_handling(formatter):
    """Empty dict is handled gracefully and returns valid structure."""
    out = formatter.format_response({})

    assert isinstance(out, dict)
    assert "_token_estimates" in out
    assert "_truncated" in out


def test_nested_json_response_measurement(formatter):
    """Deeply nested JSON response is measured correctly (not by key count)."""
    nested = {"level1": {"level2": {"level3": {"level4": "deep value " * 10}}}}
    out = formatter.format_response(nested)

    # Measurement should reflect full serialized size, not just top-level keys
    estimates = out["_token_estimates"]
    assert estimates["bytes"] > 0


def test_format_response_does_not_mutate_input(formatter):
    """format_response does not modify the original input dict."""
    data = {"result": "original", "processing_time_ms": 5}
    original_keys = set(data.keys())
    original_result = data["result"]

    formatter.format_response(data)

    assert set(data.keys()) == original_keys
    assert data["result"] == original_result


def test_response_with_list_value(formatter):
    """Response containing list values is handled correctly."""
    data = {"items": list(range(50)), "count": 50}
    out = formatter.format_response(data)

    assert "_token_estimates" in out
    assert "_truncated" in out
