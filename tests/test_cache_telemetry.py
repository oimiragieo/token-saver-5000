import json

import pytest

from src.model_optimizer import summarize_provider_cache_usage
from src.provider_telemetry_normalizer import normalize_cache_telemetry


def test_summarize_anthropic_cache_hit_with_warning_free_result():
    telemetry = summarize_provider_cache_usage(
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_creation_input_tokens": 450,
                "cache_read_input_tokens": 400,
            }
        },
        file_id="doc-a",
        expected_cache_hit=True,
    )

    assert telemetry["provider"] == "anthropic"
    assert telemetry["cache_read_field"] == "cache_read_input_tokens"
    assert telemetry["cache_read_input_tokens"] == 400
    assert telemetry["cache_hit_detected"] is True
    assert telemetry["cache_hit_ratio"] == 0.8
    assert telemetry["estimated_cache_read_cost_usd"] > 0
    assert "warning" not in telemetry


def test_summarize_openai_cached_tokens_from_nested_usage_details():
    telemetry = summarize_provider_cache_usage(
        model="gpt-5.4",
        api_response={
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 300,
                },
            }
        },
        expected_cache_hit=False,
    )

    assert telemetry["provider"] == "openai"
    assert telemetry["cached_tokens"] == 300
    assert telemetry["cache_hit_detected"] is True
    assert telemetry["cache_hit_ratio"] == 0.6


def test_summarize_gemini_cache_miss_emits_warning_when_hit_expected():
    telemetry = summarize_provider_cache_usage(
        model="gemini-3.1-pro-preview",
        api_response={
            "usageMetadata": {
                "promptTokenCount": 600,
                "cachedContentTokenCount": 0,
                "candidatesTokenCount": 80,
            }
        },
        file_id="doc-cache",
        expected_cache_hit=True,
    )

    assert telemetry["provider"] == "google"
    assert telemetry["cachedContentTokenCount"] == 0
    assert telemetry["cache_hit_detected"] is False
    assert telemetry["cache_hit_ratio"] == 0.0
    assert "warning" in telemetry
    assert "cache miss" in telemetry["warning"].lower()


def test_summarize_gemini_usage_metadata_snake_case_payload():
    telemetry = summarize_provider_cache_usage(
        model="gemini-3.1-pro-preview",
        api_response={
            "usage_metadata": {
                "prompt_token_count": 640,
                "cached_content_token_count": 320,
                "candidates_token_count": 80,
            }
        },
        expected_cache_hit=False,
    )

    assert telemetry["provider"] == "google"
    assert telemetry["cachedContentTokenCount"] == 320
    assert telemetry["cache_hit_detected"] is True
    assert telemetry["cache_hit_ratio"] == 0.5


def test_summarize_gemini_cli_stats_payload_with_camel_case_totals():
    telemetry = summarize_provider_cache_usage(
        model="gemini-3.1-pro-preview",
        api_response={
            "stats": {
                "session": {
                    "inputTokens": 720,
                    "outputTokens": 90,
                    "cachedTokens": 360,
                }
            }
        },
        expected_cache_hit=False,
    )

    assert telemetry["provider"] == "google"
    assert telemetry["cachedContentTokenCount"] == 360
    assert telemetry["cache_hit_detected"] is True
    assert telemetry["cache_hit_ratio"] == 0.5


def test_summarize_missing_provider_field_raises_clear_error():
    with pytest.raises(ValueError) as exc_info:
        summarize_provider_cache_usage(
            model="claude-sonnet-4.6",
            api_response={"usage": {"input_tokens": 100}},
        )

    assert "cache_read_input_tokens" in str(exc_info.value)


def test_normalize_cache_telemetry_flattens_validation_and_session_state():
    normalized = normalize_cache_telemetry(
        {
            "model": "gpt-5.4",
            "provider": "openai",
            "cache_read_field": "cached_tokens",
            "cached_tokens": 300,
            "total_input_tokens": 500,
            "total_output_tokens": 100,
            "cache_creation_input_tokens": 50,
            "cache_hit_detected": True,
            "cache_hit_ratio": 0.6,
            "estimated_cache_savings_usd": 0.0042,
            "validation": {
                "status": "validated",
                "prefix_integrity": {"prefix_changed": True},
                "cache_creation_churn": {"churn_detected": True},
            },
            "session_metrics": {"cache_hit_ratio": 0.75},
            "cache_health": [
                {"label": "production", "degraded": True, "coherence": {"skew_detected": True}}
            ],
        }
    )

    assert normalized["provider"] == "openai"
    assert normalized["cached_input_tokens"] == 300
    assert normalized["validation_status"] == "validated"
    assert normalized["prefix_changed"] is True
    assert normalized["cache_creation_churn_detected"] is True
    assert normalized["session_cache_hit_ratio"] == 0.75
    assert normalized["degraded_label_count"] == 1
    assert normalized["cache_health_skew_detected"] is True


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_normalizes_result():
    from src.handlers.model_handlers import handle_capture_cache_telemetry

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {"metrics": None},
            {
                "model": "gpt-5.4",
                "api_response": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 300},
                    }
                },
                "file_id": "doc-openai",
                "expected_cache_hit": True,
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["provider"] == "openai"
    assert payload["telemetry"]["cache_hit_detected"] is True
    assert payload["telemetry"]["cached_tokens"] == 300
    assert payload["telemetry"]["file_id"] == "doc-openai"
