import json
from unittest.mock import Mock, patch

import pytest

from src.handlers.model_handlers import (
    handle_assess_cache_compatibility,
    handle_capture_cache_telemetry,
    handle_diagnose_cache_miss,
    handle_estimate_model_cost,
    handle_get_provider_profile,
    handle_optimize_for_model,
)


@pytest.mark.asyncio
async def test_get_provider_profile_handler():
    payload = json.loads(await handle_get_provider_profile({}, {"model": "claude-sonnet-4.6"}))

    assert payload["status"] == "success"
    assert payload["profile"]["provider"] == "anthropic"


@pytest.mark.asyncio
async def test_estimate_model_cost_handler():
    payload = json.loads(
        await handle_estimate_model_cost(
            {},
            {
                "model": "gpt-5.4",
                "original_tokens": 100_000,
                "compressed_tokens": 20_000,
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["saved_tokens"] == 80_000
    assert payload["telemetry"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_optimize_for_model_handler():
    payload = json.loads(
        await handle_optimize_for_model(
            {},
            {
                "model": "gemini-3.1-pro-preview",
                "text": "Authentication and rollout context. " * 80,
                "use_case": "detailed_analysis",
                "num_nodes": 6,
                "query_complexity": "complex",
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["plan"]["provider_profile"]["provider"] == "google"
    assert payload["plan"]["recommended_fidelity"]["recommended_level"] in {"DETAILED", "RAW"}


@pytest.mark.asyncio
async def test_assess_cache_compatibility_handler():
    payload = json.loads(
        await handle_assess_cache_compatibility(
            {},
            {
                "model": "gpt-5.3-codex",
                "harness": "codex_cli",
                "raw_usage_available": False,
                "cli_stats_available": True,
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["assessment"]["provider"] == "openai"
    assert payload["assessment"]["support_level"] == "supported"
    assert "cached_tokens" in payload["assessment"]["telemetry"]["primary_fields"]


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_warns_on_expected_cache_miss():
    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
                "expected_cache_hit": True,
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["provider"] == "anthropic"
    assert payload["telemetry"]["cache_hit_detected"] is False
    assert "warning" in payload["telemetry"]


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_emits_observability_payload():
    observe = Mock()
    observe.trace.return_value.__enter__ = Mock(return_value=Mock())
    observe.trace.return_value.__exit__ = Mock(return_value=None)
    metrics = Mock()

    with (
        patch("src.handlers.model_handlers.get_observability", return_value=observe),
        patch("src.handlers.model_handlers.get_metrics", return_value=metrics),
    ):
        payload = json.loads(
            await handle_capture_cache_telemetry(
                {},
                {
                    "model": "gpt-5.4",
                    "api_response": {
                        "usage": {
                            "prompt_tokens": 500,
                            "completion_tokens": 100,
                            "prompt_tokens_details": {"cached_tokens": 300},
                        }
                    },
                    "expected_cache_hit": True,
                },
            )
        )

    assert payload["status"] == "success"
    assert payload["telemetry"]["observability"]["cached_input_tokens"] == 300
    assert payload["telemetry"]["observability"]["validation_status"] == "unvalidated"
    observe.record_cache_telemetry.assert_called_once()
    metrics.record_provider_cache_telemetry.assert_called_once()


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_validates_prompt_expectation():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": '[metadata]\n{"request_id":"req-1"}',
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["status"] == "validated"
    assert payload["telemetry"]["validation"]["prompt_id"] == prompt_id
    assert "warning" in payload["telemetry"]["validation"]


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_stale_expectation_warning():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": '[metadata]\n{"request_id":"req-1"}',
        },
    )
    PromptCacheMiddleware.invalidate_template_expectations(
        template_name="review-default",
        label="production",
        previous_version=1,
        new_version=2,
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 250,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["status"] == "validated_against_stale_expectation"
    assert "stale expectation" in payload["telemetry"]["warning"].lower()
    assert "re-render" in payload["telemetry"]["warning"].lower()


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_stale_sibling_coherence():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    PromptCacheMiddleware.record_expectation(
        "review-security",
        {
            "template": {"name": "review-security"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["staging"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    PromptCacheMiddleware.invalidate_template_expectations(
        template_name="review-security",
        label="staging",
        previous_version=1,
        new_version=2,
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 250,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["sibling_coherence"]["coherence_valid"] is False
    assert (
        payload["telemetry"]["validation"]["sibling_coherence"]["stale_siblings"][0][
            "template_name"
        ]
        == "review-security"
    )


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_adds_diagnostic_with_actual_prefix():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "gpt-5.4",
                "prompt_id": prompt_id,
                "actual_rendered_prefix": (
                    '[system_instructions]\nBe accurate.\n{"id":"550e8400-e29b-41d4-a716-446655440000"}'
                ),
                "api_response": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["diagnostic"]["probable_cause"] == (
        "framework_uuid_injection"
    )
    assert (
        "disable dynamic id injection"
        in payload["telemetry"]["validation"]["diagnostic"]["suggested_remediation"].lower()
    )


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_section_interleaving():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {
                "score": 99,
                "is_cache_friendly": True,
                "present_order": ["system_instructions", "rag_context", "user_query"],
            },
            "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
            "volatile_suffix": "[user_query]\nWhat changed?",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "actual_rendered_prefix": (
                    "[system_instructions]\nBe accurate.\n\n"
                    "[rag_context]\nStatic docs.\n\n"
                    "[few_shot_examples]\nQ: hi\nA: hello\n\n"
                    "[user_query]\nWhat changed?"
                ),
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["section_interleaving"]["layout_changed"] is True
    assert payload["telemetry"]["validation"]["diagnostic"]["probable_cause"] == (
        "section_interleaving"
    )
    assert "section layout drift" in payload["telemetry"]["warning"].lower()


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_cache_creation_churn():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    first_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    await handle_capture_cache_telemetry(
        {},
        {
            "model": "claude-sonnet-4.6",
            "prompt_id": first_prompt_id,
            "actual_rendered_prefix": "[system_instructions]\nBe accurate.",
            "api_response": {
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 100,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 500,
                }
            },
        },
    )

    second_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": second_prompt_id,
                "actual_rendered_prefix": "[system_instructions]\nBe accurate.",
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 500,
                    }
                },
            },
        )
    )

    assert payload["telemetry"]["validation"]["cache_creation_churn"]["churn_detected"] is True
    assert payload["telemetry"]["validation"]["cache_creation_churn"]["creation_events"] == 2
    assert "cache creation churn" in payload["telemetry"]["warning"].lower()


@pytest.mark.asyncio
async def test_diagnose_cache_miss_handler_requires_known_prompt_id():
    with pytest.raises(ValueError) as exc:
        await handle_diagnose_cache_miss(
            {},
            {
                "prompt_id": "missing-prompt-id",
                "model": "claude-sonnet-4.6",
                "actual_rendered_prefix": "[system_instructions]\nBe accurate.",
                "api_response": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 0}},
            },
        )

    assert "Unknown prompt_id" in str(exc.value)


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_records_session_metrics():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": "chat-session-1",
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 400,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["session_metrics"]["session_id"] == "chat-session-1"
    assert payload["telemetry"]["session_metrics"]["cache_hits"] == 1
    assert payload["telemetry"]["session_metrics"]["templates"]["review-default"]["hits"] == 1


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_records_openai_session_cached_tokens():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": "chat-session-openai",
                "model": "gpt-5.4",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 300},
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["cached_tokens"] == 300
    assert payload["telemetry"]["session_metrics"]["total_cached_tokens"] == 300
    assert payload["telemetry"]["session_metrics"]["cache_hits"] == 1


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_prefix_siblings():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id_a = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    PromptCacheMiddleware.record_expectation(
        "review-security",
        {
            "template": {"name": "review-security"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["staging"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id_a,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 400,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["prefix_siblings"][0]["template_name"] == "review-security"


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_cache_health_degradation():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": f"baseline-health-{index}",
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 400,
                    }
                },
            },
        )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": "degraded-health",
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["cache_health"][0]["label"] == "production"
    assert payload["telemetry"]["cache_health"][0]["degraded"] is False

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": "degraded-health-2",
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
            },
        )
    )

    assert payload["telemetry"]["cache_health"][0]["degraded"] is True
    assert any(
        "hit ratio degraded" in warning.lower()
        for warning in payload["telemetry"]["cache_health"][0]["warnings"]
    )


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_prefix_integrity_on_cache_hit():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_capture_cache_telemetry(
            {},
            {
                "model": "claude-sonnet-4.6",
                "prompt_id": prompt_id,
                "actual_rendered_prefix": '[system_instructions]\nBe accurate.\n{"request_id":"req-2"}',
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 250,
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["telemetry"]["validation"]["cache_hit_detected"] is True
    assert payload["telemetry"]["validation"]["prefix_integrity"]["prefix_changed"] is True
    assert (
        payload["telemetry"]["validation"]["prefix_integrity"]["expected_prefix_hash"]
        != payload["telemetry"]["validation"]["prefix_integrity"]["actual_prefix_hash"]
    )


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_prefix_drift_trend():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for request_id in ("req-1", "req-2", "req-3"):
        payload = json.loads(
            await handle_capture_cache_telemetry(
                {},
                {
                    "model": "claude-sonnet-4.6",
                    "prompt_id": prompt_id,
                    "actual_rendered_prefix": (
                        f'[system_instructions]\nBe accurate.\n{{"request_id":"{request_id}"}}'
                    ),
                    "api_response": {
                        "usage": {
                            "input_tokens": 500,
                            "output_tokens": 100,
                            "cache_read_input_tokens": 250,
                        }
                    },
                },
            )
        )

    trend = payload["telemetry"]["validation"]["prefix_integrity"]["trend"]

    assert trend["drift_observations"] == 3
    assert trend["drift_frequency"] == 1.0
    assert trend["systematic_drift_detected"] is True


@pytest.mark.asyncio
async def test_capture_cache_telemetry_handler_surfaces_cross_label_cache_skew():
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()
    prod_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    staging_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["staging"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        await handle_capture_cache_telemetry(
            {},
            {
                "session_id": f"coherence-prod-{index}",
                "model": "claude-sonnet-4.6",
                "prompt_id": prod_prompt_id,
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 400,
                    }
                },
            },
        )

    payload = None
    for index in range(3):
        payload = json.loads(
            await handle_capture_cache_telemetry(
                {},
                {
                    "session_id": f"coherence-staging-{index}",
                    "model": "claude-sonnet-4.6",
                    "prompt_id": staging_prompt_id,
                    "api_response": {
                        "usage": {
                            "input_tokens": 500,
                            "output_tokens": 100,
                            "cache_read_input_tokens": 0,
                        }
                    },
                },
            )
        )

    assert payload["telemetry"]["cache_health"][0]["coherence"]["skew_detected"] is True
    assert payload["telemetry"]["cache_health"][0]["coherence"]["peer_labels"] == ["production"]
