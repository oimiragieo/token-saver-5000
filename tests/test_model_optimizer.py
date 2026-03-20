from src.fidelity_advisor import FidelityLevel, UseCase
from src.model_optimizer import (
    advise_cache_threshold,
    build_prompt_cache_key,
    optimize_for_model,
    summarize_provider_cache_usage,
)


def test_optimize_for_high_cost_model_biases_toward_compaction():
    plan = optimize_for_model(
        text="Authentication flows and session rotation details. " * 80,
        model="claude-opus-4.6",
        use_case=UseCase.TOPIC_OVERVIEW,
        num_nodes=8,
        query_complexity="medium",
    )

    assert plan["provider_profile"]["provider"] == "anthropic"
    assert plan["recommended_output_format"] == "toon"
    assert plan["projected_costs"]["estimated_savings_usd"] > 0
    assert plan["recommended_fidelity"]["recommended_level"] in {
        FidelityLevel.ABSTRACT.name,
        FidelityLevel.OUTLINE.name,
        FidelityLevel.STRUCTURE.name,
    }


def test_optimize_for_large_context_model_can_keep_more_detail():
    plan = optimize_for_model(
        text="Architecture and rollout notes. " * 100,
        model="gemini-3.1-pro-preview",
        use_case=UseCase.DETAILED_ANALYSIS,
        num_nodes=6,
        query_complexity="complex",
    )

    assert plan["provider_profile"]["provider"] == "google"
    assert plan["recommended_fidelity"]["recommended_level"] in {
        FidelityLevel.DETAILED.name,
        FidelityLevel.RAW.name,
    }
    assert "cache" in plan["cache_strategy"].lower()


def test_optimize_for_model_includes_prompt_structure_guidance():
    plan = optimize_for_model(
        text="Prompt caching guidance. " * 60,
        model="gpt-5.4",
        use_case=UseCase.QUESTION_ANSWERING,
        num_nodes=5,
    )

    assert "static" in plan["prompt_structure"].lower()
    assert plan["provider_profile"]["cache_read_field"] == "cached_tokens"


def test_optimize_for_codex_family_includes_routing_stickiness_guidance():
    plan = optimize_for_model(
        text="Prompt caching guidance. " * 60,
        model="gpt-5.3-codex",
        use_case=UseCase.QUESTION_ANSWERING,
        num_nodes=5,
    )

    assert plan["provider_profile"]["provider"] == "openai"
    assert plan["routing_stickiness"]["supports_prompt_cache_key"] is True
    assert "prompt_cache_key" in plan["routing_stickiness"]["strategy"].lower()


def test_optimize_for_model_includes_cache_threshold_guidance():
    plan = optimize_for_model(
        text="Prompt caching guidance. " * 60,
        model="gpt-5.4",
        use_case=UseCase.QUESTION_ANSWERING,
        num_nodes=5,
    )

    assert plan["cache_thresholds"]["minimum_cacheable_tokens"] >= 1024
    assert "eligible" in plan["cache_thresholds"]["guidance"].lower()


def test_advise_cache_threshold_reports_gap_when_below_minimum():
    threshold = advise_cache_threshold(model="gpt-5.4", prompt_tokens=900)

    assert threshold["eligible"] is False
    assert threshold["tokens_below_minimum"] > 0
    assert threshold["minimum_cacheable_tokens"] >= 1024


def test_build_prompt_cache_key_is_stable_for_same_inputs():
    key_one = build_prompt_cache_key(
        model="gpt-5.4",
        workflow_id="review-session",
        stable_prefix="[system]\nBe accurate.\n[rag]\nStable docs",
    )
    key_two = build_prompt_cache_key(
        model="gpt-5.4",
        workflow_id="review-session",
        stable_prefix="[system]\nBe accurate.\n[rag]\nStable docs",
    )

    assert key_one == key_two
    assert key_one.startswith("gpt-5.4:review-session:")


def test_summarize_provider_cache_usage_uses_provider_profile_costs():
    telemetry = summarize_provider_cache_usage(
        model="claude-opus-4.6",
        api_response={
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_read_input_tokens": 800,
            }
        },
    )

    assert telemetry["provider"] == "anthropic"
    assert telemetry["cache_read_field"] == "cache_read_input_tokens"
    assert (
        telemetry["estimated_uncached_input_cost_usd"] > telemetry["estimated_cache_read_cost_usd"]
    )
