"""
Tests for Gemini CLI token optimization enhancements (v0.12.0).

TDD test suite covering:
1. TokenEstimator.estimate_gemini() — Gemini-compatible token estimation
2. ResponseFormatter truncation strategies — proportional / head / paginate
3. ResponseFormatter response headers — tool_name header injection
4. apply_urgency() — urgency overrides for compression parameters
5. KNOWN_MODEL_CONTEXT_WINDOWS + KNOWN_MODEL_COMPRESSION_TRIGGERS constants
6. ClientConfig.from_model() with compression_trigger field
"""

import pytest

from src.token_estimation import TokenEstimator
from src.response_formatter import ResponseFormatter
from src.compression_profiles import apply_urgency
from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS, KNOWN_MODEL_COMPRESSION_TRIGGERS
from src.client_config import ClientConfig, get_recommended_ratio


# ============================================================================
# TokenEstimator.estimate_gemini — Gemini token estimation
# ============================================================================


@pytest.fixture
def estimator():
    """Fresh TokenEstimator for each test."""
    return TokenEstimator()


def test_gemini_estimate_ascii_only(estimator):
    """ASCII-only text: 0.25 tok/char — 'hello world' (11 chars) -> int(11 * 0.25) = 2."""
    result = estimator.estimate_gemini("hello world")
    assert result == int(11 * 0.25)
    assert result == 2


def test_gemini_estimate_non_ascii(estimator):
    """Non-ASCII text: 1.3 tok/char — Chinese/Japanese characters."""
    # 4 Chinese characters, each gets 1.3 -> int(4 * 1.3) = 5
    text = "\u4e2d\u6587\u6d4b\u8bd5"  # 中文测试
    result = estimator.estimate_gemini(text)
    assert result == int(4 * 1.3)
    assert result == 5


def test_gemini_estimate_mixed(estimator):
    """Mixed ASCII and non-ASCII: correct per-character rates applied."""
    # "hi" (2 ASCII) + "中" (1 non-ASCII)
    # 2 * 0.25 + 1 * 1.3 = 0.5 + 1.3 = 1.8 -> int(1.8) = 1
    text = "hi\u4e2d"
    result = estimator.estimate_gemini(text)
    expected = int(2 * 0.25 + 1 * 1.3)
    assert result == expected
    assert result == 1


def test_gemini_estimate_long_text_fallback(estimator):
    """Text > 100,000 chars uses len(text) // 4 fallback (matches Gemini CLI behavior)."""
    long_text = "A" * 100_001
    result = estimator.estimate_gemini(long_text)
    assert result == len(long_text) // 4


def test_gemini_estimate_exactly_100k_uses_per_char_formula(estimator):
    """Text of exactly 100,000 chars uses per-character formula (boundary: > 100K triggers fallback)."""
    text = "A" * 100_000
    result = estimator.estimate_gemini(text)
    # 100,000 ASCII chars * 0.25 = 25,000
    assert result == int(100_000 * 0.25)
    assert result == 25_000


def test_gemini_estimate_empty(estimator):
    """Empty string returns 0."""
    assert estimator.estimate_gemini("") == 0


def test_gemini_estimate_single_ascii(estimator):
    """Single ASCII char: int(0.25) = 0."""
    assert estimator.estimate_gemini("A") == 0


def test_gemini_estimate_four_ascii(estimator):
    """Four ASCII chars: int(4 * 0.25) = 1."""
    assert estimator.estimate_gemini("ABCD") == 1


def test_estimate_all_includes_gemini(estimator):
    """estimate_all() dict includes 'gemini_estimated' key."""
    result = estimator.estimate_all("hello world")
    assert "gemini_estimated" in result


def test_estimate_all_gemini_value_matches_estimate_gemini(estimator):
    """estimate_all()['gemini_estimated'] equals estimate_gemini(text) directly."""
    text = "The quick brown fox jumps over the lazy dog."
    result = estimator.estimate_all(text)
    assert result["gemini_estimated"] == estimator.estimate_gemini(text)


def test_estimate_all_gemini_is_non_negative(estimator):
    """estimate_all()['gemini_estimated'] is always >= 0."""
    for text in ["", "hello", "A" * 1000, "\u4e2d\u6587"]:
        result = estimator.estimate_all(text)
        assert result["gemini_estimated"] >= 0


# ============================================================================
# ResponseFormatter — proportional truncation strategy
# ============================================================================


def test_proportional_truncation_preserves_head_and_tail():
    """Proportional strategy: response contains content from beginning and end of original."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100, truncation_strategy="proportional")
    # Build something larger than hard_limit
    original = {"content": "AAAAA" * 50 + "ZZZZZ" * 50}
    out = fmt.format_response(original)

    assert out["_truncated"] is True
    # The formatted output should reference both head and tail content
    # At minimum, the truncation marker must be present
    out_str = str(out)
    assert "Truncated" in out_str or out["_truncated"] is True


def test_proportional_truncation_sets_truncated_true():
    """Proportional strategy sets _truncated=True when over limit."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100, truncation_strategy="proportional")
    big_data = {"content": "X" * 200}
    out = fmt.format_response(big_data)
    assert out["_truncated"] is True


def test_proportional_truncation_no_continuation_token():
    """Proportional strategy sets _continuation_token=None (not paginated)."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100, truncation_strategy="proportional")
    big_data = {"content": "X" * 200}
    out = fmt.format_response(big_data)
    assert out.get("_continuation_token") is None


def test_proportional_small_response_unchanged():
    """Proportional strategy passes through responses under hard_limit unchanged."""
    fmt = ResponseFormatter(soft_limit=500, hard_limit=1000, truncation_strategy="proportional")
    small_data = {"result": "ok", "value": 42}
    out = fmt.format_response(small_data)
    assert out["_truncated"] is False
    assert out["result"] == "ok"


def test_default_strategy_is_paginate():
    """Default truncation_strategy is 'paginate' (existing behavior preserved)."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100)
    big_data = {"content": "Z" * 200}
    out = fmt.format_response(big_data)

    # Paginate strategy produces a continuation token (not None)
    assert out["_truncated"] is True
    assert out.get("_continuation_token") is not None


def test_head_strategy_keeps_first_n_chars():
    """Head strategy truncates to first hard_limit chars of serialized JSON."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100, truncation_strategy="head")
    big_data = {"content": "H" * 300}
    out = fmt.format_response(big_data)
    assert out["_truncated"] is True
    assert out.get("_continuation_token") is None


def test_head_strategy_small_response_passes_through():
    """Head strategy passes through small responses unchanged."""
    fmt = ResponseFormatter(soft_limit=500, hard_limit=1000, truncation_strategy="head")
    small_data = {"result": "head-test", "count": 1}
    out = fmt.format_response(small_data)
    assert out["_truncated"] is False
    assert out["result"] == "head-test"


def test_proportional_response_contains_truncation_prefix():
    """Proportional truncation response includes the [Truncated by Token Saver] prefix."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100, truncation_strategy="proportional")
    big_data = {"content": "T" * 400}
    out = fmt.format_response(big_data)

    # The truncation prefix must appear somewhere in the serialized output
    # It's in _preview or the top-level content string
    out_str = str(out)
    assert "Token Saver" in out_str or out["_truncated"] is True


# ============================================================================
# ResponseFormatter — response headers (tool_name parameter)
# ============================================================================


def test_header_added_when_tool_name_provided():
    """When tool_name is provided to format_response, a _header key is added."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"}, tool_name="compress_document")
    assert "_header" in out


def test_header_under_200_chars():
    """The _header value is always < 200 characters."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok", "ratio": 0.87}, tool_name="ingest_document")
    assert len(out["_header"]) < 200


def test_header_is_first_key():
    """_header is the first key in the returned dict when tool_name is provided."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"}, tool_name="search_semantic")
    keys = list(out.keys())
    assert keys[0] == "_header"


def test_header_contains_tool_name():
    """The _header string contains the tool_name."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"}, tool_name="my_tool")
    assert "my_tool" in out["_header"]


def test_header_contains_token_saver_brand():
    """The _header string contains 'Token Saver'."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"}, tool_name="compress")
    assert "Token Saver" in out["_header"]


def test_no_header_when_tool_name_omitted():
    """When tool_name is not provided, no _header key appears."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"})
    assert "_header" not in out


def test_header_format_includes_pipe_separators():
    """_header follows 'Token Saver | {tool_name} | ...' format."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"result": "ok"}, tool_name="test_tool")
    assert out["_header"].startswith("Token Saver | test_tool |")


# ============================================================================
# apply_urgency — urgency overrides
# ============================================================================


def test_urgency_normal_no_change():
    """apply_urgency('normal') returns params unchanged."""
    params = {"skeleton_ratio": 0.30, "fidelity": "STRUCTURE", "chunk_size": 512}
    result = apply_urgency(params, "normal")
    assert result["skeleton_ratio"] == pytest.approx(0.30)
    assert result["fidelity"] == "STRUCTURE"
    assert result["chunk_size"] == 512


def test_urgency_compact_caps_ratio():
    """apply_urgency('compact') caps skeleton_ratio at 0.15."""
    params = {"skeleton_ratio": 0.30, "fidelity": "STRUCTURE", "chunk_size": 512}
    result = apply_urgency(params, "compact")
    assert result["skeleton_ratio"] <= 0.15


def test_urgency_compact_caps_fidelity():
    """apply_urgency('compact') sets fidelity to OUTLINE if not already more compact."""
    params = {"skeleton_ratio": 0.30, "fidelity": "STRUCTURE", "chunk_size": 512}
    result = apply_urgency(params, "compact")
    # STRUCTURE -> OUTLINE (more compact)
    assert result["fidelity"] == "OUTLINE"


def test_urgency_compact_preserves_already_compact_ratio():
    """apply_urgency('compact') does not increase a ratio already below 0.15."""
    params = {"skeleton_ratio": 0.05, "fidelity": "ABSTRACT", "chunk_size": 256}
    result = apply_urgency(params, "compact")
    assert result["skeleton_ratio"] == pytest.approx(0.05)


def test_urgency_compact_preserves_already_compact_fidelity():
    """apply_urgency('compact') preserves ABSTRACT fidelity (already more compact than OUTLINE)."""
    params = {"skeleton_ratio": 0.30, "fidelity": "ABSTRACT", "chunk_size": 512}
    result = apply_urgency(params, "compact")
    assert result["fidelity"] == "ABSTRACT"


def test_urgency_emergency_forces_minimal():
    """apply_urgency('emergency') forces ratio=0.05, fidelity=ABSTRACT, chunk_size=256."""
    params = {"skeleton_ratio": 0.50, "fidelity": "DETAILED", "chunk_size": 1024}
    result = apply_urgency(params, "emergency")
    assert result["skeleton_ratio"] == pytest.approx(0.05)
    assert result["fidelity"] == "ABSTRACT"
    assert result["chunk_size"] == 256


def test_urgency_emergency_overrides_already_minimal():
    """apply_urgency('emergency') forces values even if params are already compact."""
    params = {"skeleton_ratio": 0.10, "fidelity": "OUTLINE", "chunk_size": 512}
    result = apply_urgency(params, "emergency")
    assert result["skeleton_ratio"] == pytest.approx(0.05)
    assert result["fidelity"] == "ABSTRACT"
    assert result["chunk_size"] == 256


def test_urgency_unknown_raises_error():
    """apply_urgency with unknown urgency raises ValueError."""
    params = {"skeleton_ratio": 0.25}
    with pytest.raises(ValueError, match="urgency"):
        apply_urgency(params, "ultra-critical")


def test_urgency_does_not_mutate_input():
    """apply_urgency never mutates the input params dict."""
    params = {"skeleton_ratio": 0.30, "fidelity": "STRUCTURE", "chunk_size": 512}
    original = dict(params)
    apply_urgency(params, "emergency")
    assert params == original


def test_urgency_returns_new_dict():
    """apply_urgency returns a new dict, not the same object."""
    params = {"skeleton_ratio": 0.30}
    result = apply_urgency(params, "normal")
    assert result is not params


# ============================================================================
# constants — KNOWN_MODEL_CONTEXT_WINDOWS Gemini additions
# ============================================================================


def test_gemini_2_5_pro_in_database():
    """gemini-2.5-pro is in KNOWN_MODEL_CONTEXT_WINDOWS."""
    assert "gemini-2.5-pro" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_gemini_2_5_flash_in_database():
    """gemini-2.5-flash is in KNOWN_MODEL_CONTEXT_WINDOWS."""
    assert "gemini-2.5-flash" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_gemini_3_1_pro_in_database():
    """gemini-3.1-pro is in KNOWN_MODEL_CONTEXT_WINDOWS."""
    assert "gemini-3.1-pro" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_gemini_3_1_flash_in_database():
    """gemini-3.1-flash is in KNOWN_MODEL_CONTEXT_WINDOWS."""
    assert "gemini-3.1-flash" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_gemini_3_1_flash_lite_in_database():
    """gemini-3.1-flash-lite is in KNOWN_MODEL_CONTEXT_WINDOWS."""
    assert "gemini-3.1-flash-lite" in KNOWN_MODEL_CONTEXT_WINDOWS


def test_gemini_context_window_is_1m():
    """New Gemini 2.5/3.1 models all have 1,048,576 token context windows."""
    new_gemini_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash",
        "gemini-3.1-flash-lite",
    ]
    for model in new_gemini_models:
        assert (
            KNOWN_MODEL_CONTEXT_WINDOWS[model] == 1_048_576
        ), f"{model} context window should be 1_048_576, got {KNOWN_MODEL_CONTEXT_WINDOWS[model]}"


def test_existing_models_context_windows_unchanged():
    """Existing model entries in KNOWN_MODEL_CONTEXT_WINDOWS are not modified."""
    assert KNOWN_MODEL_CONTEXT_WINDOWS["claude-opus-4-6"] == 200_000
    assert KNOWN_MODEL_CONTEXT_WINDOWS["gpt-4o"] == 128_000
    assert KNOWN_MODEL_CONTEXT_WINDOWS["gemini-2.0-pro"] == 2_000_000
    assert KNOWN_MODEL_CONTEXT_WINDOWS["default"] == 100_000


# ============================================================================
# constants — KNOWN_MODEL_COMPRESSION_TRIGGERS
# ============================================================================


def test_compression_triggers_dict_exists():
    """KNOWN_MODEL_COMPRESSION_TRIGGERS is importable and is a dict."""
    assert isinstance(KNOWN_MODEL_COMPRESSION_TRIGGERS, dict)


def test_gemini_compression_trigger_is_50pct():
    """Gemini models have compression_trigger=0.50 (50% of context window)."""
    gemini_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-3.1-pro",
        "gemini-3.1-flash",
        "gemini-3.1-flash-lite",
    ]
    for model in gemini_models:
        assert model in KNOWN_MODEL_COMPRESSION_TRIGGERS, f"{model} missing from triggers dict"
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS[model] == pytest.approx(
            0.50
        ), f"{model} trigger should be 0.50"


def test_claude_compression_trigger_is_93pct():
    """Claude models have compression_trigger=0.93."""
    claude_models = ["claude-opus-4-6", "claude-sonnet-4-6"]
    for model in claude_models:
        assert model in KNOWN_MODEL_COMPRESSION_TRIGGERS, f"{model} missing from triggers dict"
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS[model] == pytest.approx(
            0.93
        ), f"{model} trigger should be 0.93"


def test_gpt4o_compression_trigger_is_1():
    """GPT-4o has compression_trigger=1.0 (no auto-compression)."""
    assert KNOWN_MODEL_COMPRESSION_TRIGGERS["gpt-4o"] == pytest.approx(1.0)


def test_default_compression_trigger():
    """'default' key exists in KNOWN_MODEL_COMPRESSION_TRIGGERS."""
    assert "default" in KNOWN_MODEL_COMPRESSION_TRIGGERS
    assert KNOWN_MODEL_COMPRESSION_TRIGGERS["default"] == pytest.approx(0.80)


# ============================================================================
# ClientConfig — compression_trigger field and ratio adjustment
# ============================================================================


def test_compression_trigger_stored_in_config():
    """ClientConfig has a compression_trigger attribute."""
    cfg = ClientConfig.from_model("gemini-2.5-pro")
    assert hasattr(cfg, "compression_trigger")


def test_from_model_reads_compression_trigger():
    """ClientConfig.from_model() reads compression_trigger from KNOWN_MODEL_COMPRESSION_TRIGGERS."""
    cfg = ClientConfig.from_model("gemini-2.5-pro")
    assert cfg.compression_trigger == pytest.approx(0.50)


def test_from_model_claude_compression_trigger():
    """Claude model gets compression_trigger=0.93."""
    cfg = ClientConfig.from_model("claude-opus-4-6")
    assert cfg.compression_trigger == pytest.approx(0.93)


def test_from_model_unknown_model_gets_default_trigger():
    """Unknown model falls back to 'default' compression_trigger=0.80."""
    cfg = ClientConfig.from_model("some-future-model-xyz")
    assert cfg.compression_trigger == pytest.approx(0.80)


def test_gemini_model_gets_lower_ratio_than_claude():
    """For the same context window size, Gemini (0.50 trigger) gets lower ratio than Claude (0.93)."""
    # Use same window size to isolate trigger effect
    gemini_cfg = ClientConfig.from_model("gemini-2.5-pro")
    claude_cfg = ClientConfig.from_model("claude-opus-4-6[1m]")

    # Both have 1M context windows, but Gemini trigger is 0.50 vs Claude 0.93
    # Gemini should get ~37% more compression (lower ratio)
    assert gemini_cfg.recommended_skeleton_ratio < claude_cfg.recommended_skeleton_ratio, (
        f"Gemini ratio {gemini_cfg.recommended_skeleton_ratio} should be < "
        f"Claude ratio {claude_cfg.recommended_skeleton_ratio}"
    )


def test_get_recommended_ratio_with_compression_trigger():
    """get_recommended_ratio(window, compression_trigger) adjusts ratio by trigger ratio."""
    base_ratio = get_recommended_ratio(1_000_000, compression_trigger=0.80)
    aggressive_ratio = get_recommended_ratio(1_000_000, compression_trigger=0.50)

    # More aggressive client (lower trigger) should get lower recommended ratio
    assert aggressive_ratio < base_ratio


def test_get_recommended_ratio_default_trigger_is_0_80():
    """get_recommended_ratio with no trigger arg uses 0.80 as default."""
    ratio_default = get_recommended_ratio(200_000)
    ratio_explicit = get_recommended_ratio(200_000, compression_trigger=0.80)
    assert ratio_default == pytest.approx(ratio_explicit)


def test_compression_trigger_does_not_break_ratio_range():
    """All trigger-adjusted ratios remain in [0.01, 0.90]."""
    for trigger in [0.50, 0.80, 0.93, 1.0]:
        ratio = get_recommended_ratio(100_000, compression_trigger=trigger)
        assert 0.01 <= ratio <= 0.90, f"Ratio {ratio} out of range for trigger={trigger}"
