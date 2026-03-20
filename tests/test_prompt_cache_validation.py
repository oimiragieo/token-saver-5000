"""Tests for prompt cache expectation tracking and validation."""

from src.prompt_cache_middleware import PromptCacheMiddleware


def setup_function():
    PromptCacheMiddleware.reset()


def test_record_expectation_marks_cache_friendly_prompt_as_expected_hit():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": '[metadata]\n{"request_id":"req-1"}',
        },
    )

    expectation = PromptCacheMiddleware.get_expectation(prompt_id)

    assert expectation is not None
    assert expectation["template_name"] == "review-default"
    assert expectation["expected_cache_hit"] is True
    assert expectation["is_cache_friendly"] is True
    assert expectation["cacheable_prefix_hash"]
    assert expectation["template_version"] == 1
    assert expectation["resolved_labels"] == ["production"]
    assert expectation["is_stale"] is False


def test_validate_provider_response_warns_on_unexpected_cache_miss():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 93, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": '[metadata]\n{"request_id":"req-1"}',
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
            }
        },
    )

    assert validation["status"] == "validated"
    assert validation["cache_hit_detected"] is False
    assert validation["expectation"]["expected_cache_hit"] is True
    assert "warning" in validation


def test_invalidate_expectations_for_redeployed_label_marks_expectation_stale():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 98, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": '[metadata]\n{"request_id":"req-1"}',
        },
    )

    invalidation = PromptCacheMiddleware.invalidate_template_expectations(
        template_name="review-default",
        label="production",
        previous_version=1,
        new_version=2,
    )

    expectation = PromptCacheMiddleware.get_expectation(prompt_id)

    assert invalidation == {
        "template_name": "review-default",
        "label": "production",
        "previous_version": 1,
        "new_version": 2,
        "stale_expectations": 1,
    }
    assert expectation is not None
    assert expectation["is_stale"] is True
    assert expectation["stale_reason"] == "redeployed_label"
    assert expectation["stale_replaced_by_version"] == 2


def test_validate_provider_response_reports_stale_expectation_details():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 96, "is_cache_friendly": True},
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

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    assert validation["status"] == "validated_against_stale_expectation"
    assert validation["stale_expectation"]["template_name"] == "review-default"
    assert validation["stale_expectation"]["label"] == "production"
    assert validation["stale_expectation"]["rendered_version"] == 1
    assert validation["stale_expectation"]["current_version"] == 2


def test_validate_provider_response_reports_stale_colliding_sibling():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 96, "is_cache_friendly": True},
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
            "audit": {"score": 96, "is_cache_friendly": True},
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

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    assert validation["sibling_coherence"]["coherence_valid"] is False
    assert validation["sibling_coherence"]["stale_siblings"] == [
        {
            "template_name": "review-security",
            "label": "staging",
            "rendered_version": 1,
            "current_version": 2,
            "reason": "redeployed_label",
        }
    ]
    assert "stale sibling" in validation["sibling_coherence"]["warning"].lower()


def test_validate_provider_response_reports_clean_sibling_coherence():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 96, "is_cache_friendly": True},
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
            "audit": {"score": 96, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    assert validation["sibling_coherence"] == {
        "coherence_valid": True,
        "stale_siblings": [],
        "warning": None,
    }


def test_validate_provider_response_reports_section_interleaving_drift():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {
                "score": 96,
                "is_cache_friendly": True,
                "present_order": ["system_instructions", "rag_context", "user_query"],
            },
            "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
            "volatile_suffix": "[user_query]\nWhat changed?",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
            }
        },
        actual_rendered_prefix=(
            "[system_instructions]\nBe accurate.\n\n"
            "[rag_context]\nStatic docs.\n\n"
            "[few_shot_examples]\nQ: hi\nA: hello\n\n"
            "[user_query]\nWhat changed?"
        ),
    )

    assert validation["section_interleaving"]["layout_changed"] is True
    assert validation["section_interleaving"]["unexpected_stable_sections"] == ["few_shot_examples"]
    assert validation["diagnostic"]["probable_cause"] == "section_interleaving"
    assert "section layout drift" in validation["warning"].lower()


def test_validate_provider_response_reports_semantic_equivalence_drift():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {
                "score": 96,
                "is_cache_friendly": True,
                "present_order": ["system_instructions", "rag_context", "user_query"],
            },
            "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
            "volatile_suffix": "[user_query]\nWhat changed?",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
            }
        },
        actual_rendered_prefix=(
            "[system_instructions]\r\nBe accurate.   \r\n\r\n\r\n"
            "[rag_context]\r\nStatic docs.\r\n"
        ),
    )

    assert validation["prefix_integrity"]["prefix_changed"] is True
    assert validation["prefix_integrity"]["semantic_equivalence"]["semantic_match"] is True
    assert validation["prefix_integrity"]["semantic_equivalence"]["drift_type"] == (
        "whitespace_only"
    )
    assert validation["diagnostic"]["probable_cause"] == "semantic_equivalence_drift"
    assert "semantically equivalent" in validation["warning"].lower()


def test_validate_provider_response_reports_partial_cache_reuse_underperformance():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {
                "score": 96,
                "is_cache_friendly": True,
                "present_order": ["system_instructions", "rag_context", "user_query"],
            },
            "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
            "volatile_suffix": "[user_query]\nWhat changed?",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 80,
            }
        },
        actual_rendered_prefix="[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
    )

    assert validation["cache_hit_detected"] is True
    assert validation["diagnostic"]["probable_cause"] == "partial_cache_reuse_underperformance"
    assert validation["diagnostic"]["partial_reuse"]["partial_reuse_detected"] is True
    assert "partial cache reuse" in validation["warning"].lower()


def test_validate_provider_response_reports_repeated_cache_creation_churn():
    first_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 96, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "[user_query]\nWhat changed?",
        },
    )
    PromptCacheMiddleware.validate_provider_response(
        prompt_id=first_prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 500,
            }
        },
        actual_rendered_prefix="[system_instructions]\nBe accurate.",
    )

    second_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 96, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "[user_query]\nWhat changed again?",
        },
    )
    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=second_prompt_id,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 500,
            }
        },
        actual_rendered_prefix="[system_instructions]\nBe accurate.",
    )

    assert validation["cache_creation_churn"]["churn_detected"] is True
    assert validation["cache_creation_churn"]["creation_events"] == 2
    assert validation["cache_creation_churn"]["creation_token_total"] == 1000
    assert "cache creation churn" in validation["warning"].lower()


def test_validate_provider_response_reports_missing_expectation():
    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id="missing-prompt-id",
        model="gpt-5.4",
        api_response={
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
        },
    )

    assert validation == {"status": "missing_expectation", "prompt_id": "missing-prompt-id"}


def test_record_session_telemetry_aggregates_cache_metrics():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    PromptCacheMiddleware.record_session_telemetry(
        session_id="session-1",
        prompt_id=prompt_id,
        telemetry={
            "cache_hit_detected": True,
            "cache_hit_ratio": 0.8,
            "estimated_uncached_input_cost_usd": 0.0015,
            "estimated_cache_savings_usd": 0.0012,
            "cache_read_input_tokens": 400,
            "total_input_tokens": 500,
        },
    )

    metrics = PromptCacheMiddleware.get_session_metrics("session-1")

    assert metrics["session_id"] == "session-1"
    assert metrics["cache_hits"] == 1
    assert metrics["cache_misses"] == 0
    assert metrics["total_cached_tokens"] == 400
    assert metrics["total_input_tokens"] == 500
    assert metrics["templates"]["review-default"]["hits"] == 1
    assert metrics["labels"]["production"]["hits"] == 1


def test_record_session_telemetry_uses_openai_cached_tokens_field():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    PromptCacheMiddleware.record_session_telemetry(
        session_id="session-openai",
        prompt_id=prompt_id,
        telemetry={
            "cache_hit_detected": True,
            "cache_hit_ratio": 0.6,
            "estimated_uncached_input_cost_usd": 0.005,
            "estimated_cache_savings_usd": 0.004,
            "cached_tokens": 300,
            "total_input_tokens": 500,
        },
    )

    metrics = PromptCacheMiddleware.get_session_metrics("session-openai")

    assert metrics["cache_hits"] == 1
    assert metrics["total_cached_tokens"] == 300
    assert metrics["total_input_tokens"] == 500


def test_record_session_telemetry_uses_gemini_cached_content_token_count():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    PromptCacheMiddleware.record_session_telemetry(
        session_id="session-gemini",
        prompt_id=prompt_id,
        telemetry={
            "cache_hit_detected": True,
            "cache_hit_ratio": 0.5,
            "estimated_uncached_input_cost_usd": 0.001,
            "estimated_cache_savings_usd": 0.0008,
            "cachedContentTokenCount": 320,
            "total_input_tokens": 640,
        },
    )

    metrics = PromptCacheMiddleware.get_session_metrics("session-gemini")

    assert metrics["cache_hits"] == 1
    assert metrics["total_cached_tokens"] == 320
    assert metrics["total_input_tokens"] == 640


def test_record_session_telemetry_adds_degradation_warning_for_repeated_misses():
    for _ in range(3):
        prompt_id = PromptCacheMiddleware.record_expectation(
            "review-default",
            {
                "template": {"name": "review-default"},
                "resolved_version": {"version": 1},
                "resolved_labels": ["production"],
                "audit": {"score": 100, "is_cache_friendly": True},
                "cacheable_prefix": "[system_instructions]\nBe accurate.",
                "volatile_suffix": "",
            },
        )
        PromptCacheMiddleware.record_session_telemetry(
            session_id="session-2",
            prompt_id=prompt_id,
            telemetry={
                "cache_hit_detected": False,
                "cache_hit_ratio": 0.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0,
                "cache_read_input_tokens": 0,
                "total_input_tokens": 500,
            },
        )

    metrics = PromptCacheMiddleware.get_session_metrics("session-2")

    assert metrics["cache_hits"] == 0
    assert metrics["cache_misses"] == 3
    assert any("review-default" in warning for warning in metrics["warnings"])


def test_record_expectation_tracks_cross_template_prefix_collisions():
    prompt_id_a = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )
    prompt_id_b = PromptCacheMiddleware.record_expectation(
        "review-security",
        {
            "template": {"name": "review-security"},
            "resolved_version": {"version": 2},
            "resolved_labels": ["staging"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    collision_map = PromptCacheMiddleware.get_prefix_collision_map()
    siblings = PromptCacheMiddleware.get_cache_siblings(prompt_id_a)

    assert len(collision_map) == 1
    assert len(next(iter(collision_map.values()))["templates"]) == 2
    assert siblings == [
        {
            "template_name": "review-security",
            "template_version": 2,
            "resolved_labels": ["staging"],
            "prompt_id": prompt_id_b,
        }
    ]


def test_record_session_telemetry_establishes_label_cache_health_baseline():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for _ in range(3):
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"session-hit-{_}",
            prompt_id=prompt_id,
            telemetry={
                "cache_hit_detected": True,
                "cache_hit_ratio": 1.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0012,
                "cache_read_input_tokens": 400,
                "total_input_tokens": 500,
            },
        )

    health = PromptCacheMiddleware.get_deployment_cache_health("review-default", "production")

    assert health == {
        "template_name": "review-default",
        "label": "production",
        "cache_hits": 3,
        "cache_misses": 0,
        "total_events": 3,
        "baseline_hit_ratio": 1.0,
        "current_hit_ratio": 1.0,
        "degraded": False,
        "degradation_amount": 0.0,
        "warnings": [],
        "coherence": {
            "skew_detected": False,
            "peer_labels": [],
            "max_hit_ratio_delta": 0.0,
            "warning": None,
        },
    }


def test_record_session_telemetry_flags_cache_health_degradation():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"baseline-{index}",
            prompt_id=prompt_id,
            telemetry={
                "cache_hit_detected": True,
                "cache_hit_ratio": 1.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0012,
                "cache_read_input_tokens": 400,
                "total_input_tokens": 500,
            },
        )

    degraded_snapshot = None
    for index in range(3):
        degraded_snapshot = PromptCacheMiddleware.record_session_telemetry(
            session_id=f"degraded-{index}",
            prompt_id=prompt_id,
            telemetry={
                "cache_hit_detected": False,
                "cache_hit_ratio": 0.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0,
                "cache_read_input_tokens": 0,
                "total_input_tokens": 500,
            },
        )

    health = PromptCacheMiddleware.get_deployment_cache_health("review-default", "production")

    assert health["baseline_hit_ratio"] == 1.0
    assert health["current_hit_ratio"] == 0.5
    assert health["degraded"] is True
    assert health["degradation_amount"] == 0.5
    assert any("hit ratio degraded" in warning.lower() for warning in health["warnings"])
    assert degraded_snapshot["cache_health"][0]["degraded"] is True


def test_record_session_telemetry_does_not_flag_small_cache_health_variance():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"healthy-hit-{index}",
            prompt_id=prompt_id,
            telemetry={
                "cache_hit_detected": True,
                "cache_hit_ratio": 1.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0012,
                "cache_read_input_tokens": 400,
                "total_input_tokens": 500,
            },
        )

    PromptCacheMiddleware.record_session_telemetry(
        session_id="healthy-miss",
        prompt_id=prompt_id,
        telemetry={
            "cache_hit_detected": False,
            "cache_hit_ratio": 0.0,
            "estimated_uncached_input_cost_usd": 0.0015,
            "estimated_cache_savings_usd": 0.0,
            "cache_read_input_tokens": 0,
            "total_input_tokens": 500,
        },
    )

    health = PromptCacheMiddleware.get_deployment_cache_health("review-default", "production")

    assert health["current_hit_ratio"] == 0.75
    assert health["degraded"] is False
    assert health["warnings"] == []


def test_validate_provider_response_reports_prefix_integrity_match():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 98, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        actual_rendered_prefix="[system_instructions]\nBe accurate.",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    assert validation["prefix_integrity"]["prefix_changed"] is False
    assert validation["prefix_integrity"]["expected_prefix_hash"] == (
        validation["expectation"]["cacheable_prefix_hash"]
    )
    assert validation["prefix_integrity"]["warning"] is None


def test_validate_provider_response_reports_prefix_integrity_drift_even_on_cache_hit():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 98, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        actual_rendered_prefix='[system_instructions]\nBe accurate.\n{"request_id":"req-2"}',
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    assert validation["cache_hit_detected"] is True
    assert validation["prefix_integrity"]["prefix_changed"] is True
    assert validation["prefix_integrity"]["first_difference_index"] is not None
    assert "diverged" in validation["prefix_integrity"]["warning"].lower()


def test_validate_provider_response_tracks_prefix_drift_frequency():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 98, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        actual_rendered_prefix="[system_instructions]\nBe accurate.",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )
    PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        actual_rendered_prefix='[system_instructions]\nBe accurate.\n{"request_id":"req-2"}',
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    validation = PromptCacheMiddleware.validate_provider_response(
        prompt_id=prompt_id,
        model="claude-sonnet-4.6",
        actual_rendered_prefix='[system_instructions]\nBe accurate.\n{"request_id":"req-3"}',
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 250,
            }
        },
    )

    trend = validation["prefix_integrity"]["trend"]

    assert trend["total_observations"] == 3
    assert trend["drift_observations"] == 2
    assert trend["drift_frequency"] == 0.6667
    assert trend["systematic_drift_detected"] is True
    assert "repeated prefix drift" in trend["warning"].lower()


def test_record_session_telemetry_detects_cross_label_cache_skew_for_shared_prefix():
    prod_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
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
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"prod-{index}",
            prompt_id=prod_prompt_id,
            telemetry={
                "cache_hit_detected": True,
                "cache_hit_ratio": 1.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0012,
                "cache_read_input_tokens": 400,
                "total_input_tokens": 500,
            },
        )
        snapshot = PromptCacheMiddleware.record_session_telemetry(
            session_id=f"staging-{index}",
            prompt_id=staging_prompt_id,
            telemetry={
                "cache_hit_detected": False,
                "cache_hit_ratio": 0.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0,
                "cache_read_input_tokens": 0,
                "total_input_tokens": 500,
            },
        )

    staging_health = PromptCacheMiddleware.get_deployment_cache_health("review-default", "staging")

    assert staging_health["coherence"]["skew_detected"] is True
    assert staging_health["coherence"]["peer_labels"] == ["production"]
    assert staging_health["coherence"]["max_hit_ratio_delta"] == 1.0
    assert any("cross-label" in warning.lower() for warning in staging_health["warnings"])
    assert snapshot["cache_health"][0]["coherence"]["skew_detected"] is True


def test_record_session_telemetry_does_not_flag_cross_label_skew_for_different_prefixes():
    prod_prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 100, "is_cache_friendly": True},
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
            "audit": {"score": 100, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe concise.",
            "volatile_suffix": "",
        },
    )

    for index in range(3):
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"prod-diff-{index}",
            prompt_id=prod_prompt_id,
            telemetry={
                "cache_hit_detected": True,
                "cache_hit_ratio": 1.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0012,
                "cache_read_input_tokens": 400,
                "total_input_tokens": 500,
            },
        )
        PromptCacheMiddleware.record_session_telemetry(
            session_id=f"staging-diff-{index}",
            prompt_id=staging_prompt_id,
            telemetry={
                "cache_hit_detected": False,
                "cache_hit_ratio": 0.0,
                "estimated_uncached_input_cost_usd": 0.0015,
                "estimated_cache_savings_usd": 0.0,
                "cache_read_input_tokens": 0,
                "total_input_tokens": 500,
            },
        )

    staging_health = PromptCacheMiddleware.get_deployment_cache_health("review-default", "staging")

    assert staging_health["coherence"]["skew_detected"] is False
    assert staging_health["coherence"]["peer_labels"] == []
