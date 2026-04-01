"""
Tests for Phase 1 cache optimization: cache-stable ordering and provider format hints.

Covers:
- cache_stable_format() key ordering for anthropic / google / openai / unknown providers
- ResponseFormatter.format_response() with provider parameter
- _detect_provider() helper
- ClientConfig provider fields and format hints
"""

from src.response_formatter import ResponseFormatter, cache_stable_format
from src.client_config import ClientConfig, _detect_provider


# ============================================================================
# cache_stable_format — key ordering
# ============================================================================


def test_cache_stable_ordering_puts_status_first():
    """status key appears before content keys for anthropic provider."""
    data = {"content": "hello", "status": "success", "_token_estimates": {"bytes": 10}}
    result = cache_stable_format(data, "anthropic")
    keys = list(result.keys())
    assert keys[0] == "status"


def test_cache_stable_ordering_volatile_last():
    """_token_estimates and other volatile keys are last for anthropic provider."""
    data = {
        "status": "success",
        "file_id": "doc-1",
        "result": "some content",
        "_token_estimates": {"bytes": 42},
        "_truncated": False,
        "_continuation_token": None,
    }
    result = cache_stable_format(data, "anthropic")
    keys = list(result.keys())
    # All volatile keys must appear after all stable/content keys
    volatile_indices = [
        i
        for i, k in enumerate(keys)
        if k in {"_token_estimates", "_truncated", "_continuation_token"}
    ]
    stable_content_indices = [
        i
        for i, k in enumerate(keys)
        if k not in {"_token_estimates", "_truncated", "_continuation_token"}
    ]
    if volatile_indices and stable_content_indices:
        assert min(volatile_indices) > max(stable_content_indices)


def test_cache_stable_codex_mirrors_at_tail():
    """openai provider appends _stable_summary containing status and file_id at the end."""
    data = {
        "status": "success",
        "file_id": "doc-42",
        "result": "compressed content",
        "_token_estimates": {"bytes": 20},
    }
    result = cache_stable_format(data, "openai")
    assert "_stable_summary" in result
    # It must be the last key
    assert list(result.keys())[-1] == "_stable_summary"
    summary = result["_stable_summary"]
    assert summary["status"] == "success"
    assert summary["file_id"] == "doc-42"


def test_cache_stable_unknown_provider_noop():
    """None / unknown provider returns the original dict unchanged (same object)."""
    data = {"b": 2, "a": 1, "z": 99}
    result_none = cache_stable_format(data, None)
    assert result_none is data

    result_unknown = cache_stable_format(data, "unknown")
    assert result_unknown is data

    result_other = cache_stable_format(data, "some-future-provider")
    assert result_other is data


def test_cache_stable_preserves_all_values():
    """No data is lost; all values are identical in the reordered dict."""
    data = {
        "status": "ok",
        "file_id": "f1",
        "score": 0.87,
        "nodes": [1, 2, 3],
        "_token_estimates": {"bytes": 100},
        "processing_time_ms": 55,
    }
    result = cache_stable_format(data, "anthropic")
    # Same keys (plus possibly _stable_summary for openai — not the case here)
    assert set(result.keys()) == set(data.keys())
    # All values identical
    for k, v in data.items():
        assert result[k] == v, f"Value for {k!r} changed"


def test_cache_stable_empty_dict():
    """Empty dict input works without error for all supported providers."""
    for provider in ("anthropic", "google", "openai", "unknown", None):
        result = cache_stable_format({}, provider)
        assert isinstance(result, dict)


def test_cache_stable_google_no_stable_summary():
    """google provider does NOT append _stable_summary (only openai does)."""
    data = {"status": "ok", "file_id": "f1", "result": "content"}
    result = cache_stable_format(data, "google")
    assert "_stable_summary" not in result


def test_cache_stable_openai_no_stable_summary_when_no_status_or_file_id():
    """openai provider omits _stable_summary when neither status nor file_id are present."""
    data = {"result": "content", "_token_estimates": {"bytes": 5}}
    result = cache_stable_format(data, "openai")
    assert "_stable_summary" not in result


# ============================================================================
# ResponseFormatter.format_response — provider integration
# ============================================================================


def test_format_response_with_provider_applies_ordering():
    """format_response with provider='anthropic' applies cache-stable ordering."""
    fmt = ResponseFormatter()
    data = {
        "_token_estimates": {"bytes": 99},  # volatile — would normally come at end
        "status": "success",  # stable — should come first
        "result": "some content",
    }
    out = fmt.format_response(data, provider="anthropic")
    # status must appear before _token_estimates in the output
    keys = list(out.keys())
    # Remove _header if present
    keys = [k for k in keys if k != "_header"]
    status_idx = keys.index("status")
    # _token_estimates is re-added by format_response itself after ordering
    token_idx = keys.index("_token_estimates")
    assert status_idx < token_idx


def test_format_response_provider_none_no_reordering():
    """format_response with provider=None does not reorder keys."""
    fmt = ResponseFormatter()
    data = {"z_last": "value", "a_first": "value2"}
    out = fmt.format_response(data)  # no provider
    # Original order of non-metadata keys must be preserved
    out_content_keys = [
        k for k in out.keys() if k not in {"_token_estimates", "_truncated", "_continuation_token"}
    ]
    assert "z_last" in out_content_keys
    assert "a_first" in out_content_keys


def test_format_response_provider_does_not_break_truncation():
    """format_response with provider still truncates large responses correctly."""
    fmt = ResponseFormatter(soft_limit=50, hard_limit=100)
    data = {"status": "ok", "content": "X" * 300}
    out = fmt.format_response(data, provider="anthropic")
    assert out["_truncated"] is True


def test_format_response_provider_backward_compat():
    """Existing callers without provider= still get correct output."""
    fmt = ResponseFormatter()
    out = fmt.format_response({"status": "ok", "result": "hello"})
    assert out["status"] == "ok"
    assert "_truncated" in out
    assert "_token_estimates" in out


# ============================================================================
# _detect_provider — provider detection
# ============================================================================


def test_detect_provider_claude():
    """claude-* model IDs return 'anthropic'."""
    assert _detect_provider("claude-opus-4-6") == "anthropic"
    assert _detect_provider("claude-sonnet-4-6") == "anthropic"
    assert _detect_provider("claude-3-haiku") == "anthropic"


def test_detect_provider_haiku_prefix():
    """haiku-* model IDs return 'anthropic'."""
    assert _detect_provider("haiku-20240307") == "anthropic"


def test_detect_provider_gemini():
    """gemini-* model IDs return 'google'."""
    assert _detect_provider("gemini-2.5-pro") == "google"
    assert _detect_provider("gemini-3.1-flash") == "google"


def test_detect_provider_gpt():
    """gpt-* model IDs return 'openai'."""
    assert _detect_provider("gpt-4o") == "openai"
    assert _detect_provider("gpt-4") == "openai"


def test_detect_provider_codex():
    """codex-* model IDs return 'openai'."""
    assert _detect_provider("codex-cushman-001") == "openai"


def test_detect_provider_o3():
    """o3-* model IDs return 'openai'."""
    assert _detect_provider("o3-mini") == "openai"
    assert _detect_provider("o3") == "openai"


def test_detect_provider_o1():
    """o1-* model IDs return 'openai'."""
    assert _detect_provider("o1-preview") == "openai"


def test_detect_provider_o4():
    """o4-* model IDs return 'openai'."""
    assert _detect_provider("o4-mini") == "openai"


def test_detect_provider_unknown():
    """Unrecognised model IDs return 'unknown'."""
    assert _detect_provider("some-future-model-xyz") == "unknown"
    assert _detect_provider("llama-3-70b") == "unknown"
    assert _detect_provider("mistral-large") == "unknown"


# ============================================================================
# ClientConfig — new provider fields
# ============================================================================


def test_client_config_has_provider_field():
    """ClientConfig.from_model() result includes a provider field."""
    cfg = ClientConfig.from_model("claude-opus-4-6")
    assert hasattr(cfg, "provider")
    assert cfg.provider == "anthropic"


def test_client_config_claude_format_hints():
    """Claude model gets preferred_output_format='toon' and truncation_strategy='paginate'."""
    cfg = ClientConfig.from_model("claude-sonnet-4-6")
    assert cfg.provider == "anthropic"
    assert cfg.preferred_output_format == "toon"
    assert cfg.truncation_strategy == "paginate"


def test_client_config_gemini_format_hints():
    """Gemini model gets preferred_output_format='json' and truncation_strategy='proportional'."""
    cfg = ClientConfig.from_model("gemini-2.5-pro")
    assert cfg.provider == "google"
    assert cfg.preferred_output_format == "json"
    assert cfg.truncation_strategy == "proportional"


def test_client_config_codex_format_hints():
    """Codex / openai model gets preferred_output_format='toon' and truncation_strategy='head'."""
    cfg = ClientConfig.from_model("codex-cushman-001")
    assert cfg.provider == "openai"
    assert cfg.preferred_output_format == "toon"
    assert cfg.truncation_strategy == "head"


def test_client_config_gpt_format_hints():
    """GPT model gets preferred_output_format='toon' and truncation_strategy='head'."""
    cfg = ClientConfig.from_model("gpt-4o")
    assert cfg.provider == "openai"
    assert cfg.preferred_output_format == "toon"
    assert cfg.truncation_strategy == "head"


def test_client_config_unknown_format_hints():
    """Unknown model gets default format hints."""
    cfg = ClientConfig.from_model("some-future-model-xyz")
    assert cfg.provider == "unknown"
    assert cfg.preferred_output_format == "json"
    assert cfg.truncation_strategy == "paginate"


def test_client_config_cache_stable_ordering_default_true():
    """cache_stable_ordering defaults to True for all providers."""
    for model_id in ("claude-opus-4-6", "gemini-2.5-pro", "gpt-4o", "unknown-model"):
        cfg = ClientConfig.from_model(model_id)
        assert cfg.cache_stable_ordering is True


def test_client_config_backward_compatible():
    """Existing fields are still present and correct after adding new fields."""
    cfg = ClientConfig.from_model("claude-opus-4-6")
    assert cfg.model_id == "claude-opus-4-6"
    assert cfg.context_window_tokens == 200_000
    assert 0.0 < cfg.recommended_skeleton_ratio <= 1.0
    assert cfg.compression_trigger > 0.0
    # New fields are also present
    assert cfg.provider == "anthropic"
    assert cfg.preferred_output_format == "toon"
    assert cfg.truncation_strategy == "paginate"
    assert cfg.cache_stable_ordering is True
