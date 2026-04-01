"""
Tests for src/token_estimation.py (TokenEstimator class).

TDD test suite for v0.11.0 token optimization.

TokenEstimator:
- estimate_all(text) -> dict with {accurate, estimated, json_estimated, bytes}
- count_tokens(text)  -> int  (tiktoken cl100k_base, falls back to len/4)
- estimate_fast(text) -> int  (len(text) // 4)
- estimate_json(text) -> int  (len(text) // 2)
- count_bytes(text)   -> int  (exact UTF-8 byte count)
- Gracefully handles tiktoken import failure via fallback
"""

import pytest
from unittest.mock import patch


from src.token_estimation import TokenEstimator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def estimator():
    """Fresh TokenEstimator for each test."""
    return TokenEstimator()


# ============================================================================
# estimate_fast — len(text) // 4
# ============================================================================


def test_fast_estimate_uses_len_div_4(estimator):
    """estimate_fast returns len(text) // 4."""
    text = "A" * 100
    result = estimator.estimate_fast(text)
    assert result == 100 // 4


def test_fast_estimate_on_short_text(estimator):
    """estimate_fast works on text shorter than 4 characters."""
    assert estimator.estimate_fast("hi") == 0
    assert estimator.estimate_fast("hey!") == 1


def test_fast_estimate_on_empty_string(estimator):
    """estimate_fast returns 0 for empty string."""
    assert estimator.estimate_fast("") == 0


# ============================================================================
# estimate_json — len(text) // 2
# ============================================================================


def test_json_estimate_uses_len_div_2(estimator):
    """estimate_json returns len(text) // 2."""
    text = "B" * 200
    result = estimator.estimate_json(text)
    assert result == 200 // 2


def test_json_estimate_on_empty_string(estimator):
    """estimate_json returns 0 for empty string."""
    assert estimator.estimate_json("") == 0


# ============================================================================
# count_bytes — exact UTF-8 byte count
# ============================================================================


def test_byte_count_is_exact(estimator):
    """count_bytes returns exact byte length of UTF-8 encoding."""
    text = "Hello"
    assert estimator.count_bytes(text) == len("Hello".encode("utf-8"))


def test_byte_count_on_empty_string(estimator):
    """count_bytes returns 0 for empty string."""
    assert estimator.count_bytes("") == 0


def test_unicode_content_byte_difference(estimator):
    """Unicode chars: byte count > len(text) for multi-byte characters."""
    text = "\u00e9\u00e0\u00fc"  # é, à, ü — each is 2 bytes in UTF-8
    char_len = len(text)
    byte_len = estimator.count_bytes(text)
    assert byte_len > char_len


def test_ascii_bytes_equal_char_count(estimator):
    """For pure ASCII, byte count equals character count."""
    text = "hello world 123"
    assert estimator.count_bytes(text) == len(text)


# ============================================================================
# count_tokens — tiktoken or fallback
# ============================================================================


def test_accurate_count_is_positive_for_nonempty(estimator):
    """count_tokens returns a positive integer for non-empty text."""
    result = estimator.count_tokens("Hello, world!")
    assert isinstance(result, int)
    assert result > 0


def test_accurate_count_zero_for_empty(estimator):
    """count_tokens returns 0 for empty string."""
    assert estimator.count_tokens("") == 0


def test_accurate_count_matches_tiktoken():
    """When tiktoken is available, count_tokens uses cl100k_base encoding."""
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")
    est = TokenEstimator()
    text = "The quick brown fox jumps over the lazy dog."
    expected = len(enc.encode(text))
    assert est.count_tokens(text) == expected


def test_fallback_when_tiktoken_unavailable():
    """When tiktoken import fails, count_tokens falls back to len(text) // 4."""
    with patch.dict("sys.modules", {"tiktoken": None}):
        # Re-instantiate so it re-evaluates tiktoken availability
        est = TokenEstimator()
        text = "A" * 80
        result = est.count_tokens(text)
        # Fallback is len // 4, so should equal 20
        assert result == 80 // 4


# ============================================================================
# estimate_all — combined dict
# ============================================================================


def test_estimate_all_returns_dict(estimator):
    """estimate_all(text) returns a dict."""
    result = estimator.estimate_all("some text")
    assert isinstance(result, dict)


def test_estimate_all_has_required_keys(estimator):
    """estimate_all dict contains accurate, estimated, json_estimated, bytes."""
    result = estimator.estimate_all("some text")
    assert "accurate" in result
    assert "estimated" in result
    assert "json_estimated" in result
    assert "bytes" in result


def test_empty_string_returns_zeros(estimator):
    """All estimate_all values are 0 for empty string."""
    result = estimator.estimate_all("")
    assert result["estimated"] == 0
    assert result["json_estimated"] == 0
    assert result["bytes"] == 0
    assert result["accurate"] == 0


def test_estimate_all_values_are_non_negative(estimator):
    """All estimate_all values are >= 0."""
    result = estimator.estimate_all("Hello there!")
    for key, value in result.items():
        assert value >= 0, f"estimate_all[{key!r}] is negative"


def test_estimate_all_json_estimate_gte_fast_estimate(estimator):
    """json_estimated >= estimated for the same text (json is denser)."""
    text = "word " * 50
    result = estimator.estimate_all(text)
    # json_estimated = len // 2, estimated = len // 4, so json > estimated
    assert result["json_estimated"] >= result["estimated"]


# ============================================================================
# Code content
# ============================================================================


def test_code_content_estimation(estimator):
    """Code content (Python) is estimated correctly — all fields positive."""
    code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
"""
    result = estimator.estimate_all(code)
    assert result["bytes"] > 0
    assert result["estimated"] > 0
    assert result["json_estimated"] > 0


def test_code_content_byte_count_matches_utf8(estimator):
    """Byte count for code matches manual UTF-8 encoding."""
    code = "def foo(): pass\n"
    assert estimator.count_bytes(code) == len(code.encode("utf-8"))


# ============================================================================
# Consistency checks
# ============================================================================


def test_bytes_field_matches_count_bytes(estimator):
    """estimate_all['bytes'] equals count_bytes(text)."""
    text = "Testing consistency 123"
    result = estimator.estimate_all(text)
    assert result["bytes"] == estimator.count_bytes(text)


def test_estimated_field_matches_estimate_fast(estimator):
    """estimate_all['estimated'] equals estimate_fast(text)."""
    text = "Testing fast estimate consistency"
    result = estimator.estimate_all(text)
    assert result["estimated"] == estimator.estimate_fast(text)


def test_json_estimated_field_matches_estimate_json(estimator):
    """estimate_all['json_estimated'] equals estimate_json(text)."""
    text = "Testing json estimate consistency"
    result = estimator.estimate_all(text)
    assert result["json_estimated"] == estimator.estimate_json(text)
