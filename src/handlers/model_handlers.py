"""Handlers for provider-aware model optimization surfaces."""

from __future__ import annotations

import json
from typing import Any

from ..cache_compatibility import assess_cache_compatibility
from ..cache_diagnostics import diagnose_cache_miss
from ..metrics import get_metrics
from ..fidelity_advisor import UseCase
from ..metrics import compute_cost_savings
from ..model_optimizer import optimize_for_model, summarize_provider_cache_usage
from ..observability import get_observability
from ..prompt_cache_middleware import PromptCacheMiddleware
from ..provider_telemetry_normalizer import normalize_cache_telemetry
from ..provider_profiles import get_provider_profile


def get_model_output_fields() -> list[str]:
    return [
        "status",
        "profile.model",
        "profile.provider",
        "profile.cache_read_field",
        "telemetry.saved_tokens",
        "telemetry.cost_savings_usd",
        "plan.provider_profile.provider",
        "plan.recommended_fidelity.recommended_level",
        "plan.projected_costs.estimated_savings_usd",
        "plan.routing_stickiness.supports_prompt_cache_key",
        "plan.routing_stickiness.example_prompt_cache_key",
        "plan.cache_thresholds.eligible",
        "plan.cache_thresholds.minimum_cacheable_tokens",
    ]


def get_cache_compatibility_output_fields() -> list[str]:
    return [
        "status",
        "assessment.model",
        "assessment.provider",
        "assessment.harness",
        "assessment.support_level",
        "assessment.risk_level",
        "assessment.telemetry.primary_fields",
        "assessment.telemetry.acceptable_sources",
        "assessment.telemetry.visibility_status",
        "assessment.recommendations",
    ]


def get_cache_telemetry_output_fields() -> list[str]:
    return [
        "status",
        "telemetry.model",
        "telemetry.provider",
        "telemetry.file_id",
        "telemetry.cache_read_field",
        "telemetry.total_input_tokens",
        "telemetry.total_output_tokens",
        "telemetry.cache_creation_input_tokens",
        "telemetry.cache_hit_detected",
        "telemetry.cache_hit_ratio",
        "telemetry.estimated_uncached_input_cost_usd",
        "telemetry.estimated_cache_read_cost_usd",
        "telemetry.estimated_cache_savings_usd",
        "telemetry.warning",
        "telemetry.observability.cached_input_tokens",
        "telemetry.observability.validation_status",
        "telemetry.validation.status",
        "telemetry.validation.prompt_id",
        "telemetry.validation.expectation.expected_cache_hit",
        "telemetry.validation.warning",
        "telemetry.validation.stale_expectation.current_version",
        "telemetry.validation.sibling_coherence.coherence_valid",
        "telemetry.validation.cache_creation_churn.churn_detected",
        "telemetry.validation.cache_creation_churn.creation_events",
        "telemetry.validation.cache_creation_churn.creation_token_total",
        "telemetry.validation.section_interleaving.layout_changed",
        "telemetry.validation.section_interleaving.sections_reordered",
        "telemetry.validation.diagnostic.probable_cause",
        "telemetry.validation.diagnostic.section_interleaving.layout_changed",
        "telemetry.validation.diagnostic.semantic_equivalence.semantic_match",
        "telemetry.validation.diagnostic.partial_reuse.partial_reuse_detected",
        "telemetry.validation.diagnostic.suggested_remediation",
        "telemetry.validation.prefix_integrity.prefix_changed",
        "telemetry.validation.prefix_integrity.actual_prefix_hash",
        "telemetry.validation.prefix_integrity.semantic_equivalence.drift_type",
        "telemetry.validation.prefix_integrity.first_difference_index",
        "telemetry.validation.prefix_integrity.trend.drift_frequency",
        "telemetry.validation.prefix_integrity.trend.systematic_drift_detected",
        "telemetry.session_metrics.cache_hits",
        "telemetry.session_metrics.cache_hit_ratio",
        "telemetry.cache_health[].label",
        "telemetry.cache_health[].baseline_hit_ratio",
        "telemetry.cache_health[].current_hit_ratio",
        "telemetry.cache_health[].degraded",
        "telemetry.cache_health[].coherence.skew_detected",
        "telemetry.prefix_siblings[].template_name",
    ]


def get_cache_diagnostic_output_fields() -> list[str]:
    return [
        "status",
        "diagnostic.template_name",
        "diagnostic.expected_prefix_hash",
        "diagnostic.actual_prefix_hash",
        "diagnostic.prefix_changed",
        "diagnostic.probable_cause",
        "diagnostic.suggested_remediation",
        "diagnostic.section_interleaving.layout_changed",
        "diagnostic.section_interleaving.sections_reordered",
        "diagnostic.semantic_equivalence.semantic_match",
        "diagnostic.semantic_equivalence.drift_type",
        "diagnostic.partial_reuse.partial_reuse_detected",
        "diagnostic.framework_signature.has_uuid",
        "diagnostic.framework_signature.has_timestamp",
        "diagnostic.metrics.actual_cached_tokens",
        "diagnostic.metrics.expected_cached_tokens",
        "diagnostic.metrics.estimated_missed_cache_cost_usd",
    ]


async def handle_get_provider_profile(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    model = args.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("get_provider_profile requires non-empty 'model'")
    with observe.trace("model_optimization.get_profile", model=model.strip()):
        profile = get_provider_profile(model.strip())
    return json.dumps({"status": "success", "profile": profile.to_dict()}, indent=2)


async def handle_estimate_model_cost(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    model = args.get("model")
    original_tokens = args.get("original_tokens")
    compressed_tokens = args.get("compressed_tokens")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("estimate_model_cost requires non-empty 'model'")
    if not isinstance(original_tokens, int) or original_tokens < 0:
        raise ValueError("estimate_model_cost requires integer 'original_tokens' >= 0")
    if not isinstance(compressed_tokens, int) or compressed_tokens < 0:
        raise ValueError("estimate_model_cost requires integer 'compressed_tokens' >= 0")
    with observe.trace(
        "model_optimization.estimate_cost",
        model=model.strip(),
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
    ):
        telemetry = compute_cost_savings(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            model=model.strip(),
        )
    return json.dumps({"status": "success", "telemetry": telemetry.to_dict()}, indent=2)


async def handle_optimize_for_model(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    model = args.get("model")
    text = args.get("text")
    use_case = args.get("use_case")
    num_nodes = args.get("num_nodes")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("optimize_for_model requires non-empty 'model'")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("optimize_for_model requires non-empty 'text'")
    if not isinstance(use_case, str) or use_case not in {member.value for member in UseCase}:
        raise ValueError("optimize_for_model requires valid 'use_case'")
    if not isinstance(num_nodes, int) or num_nodes <= 0:
        raise ValueError("optimize_for_model requires integer 'num_nodes' > 0")

    with observe.trace(
        "model_optimization.optimize",
        model=model.strip(),
        use_case=use_case,
        num_nodes=num_nodes,
    ):
        plan = optimize_for_model(
            text=text,
            model=model.strip(),
            use_case=UseCase(use_case),
            num_nodes=num_nodes,
            token_budget=args.get("token_budget"),
            query_complexity=args.get("query_complexity", "medium"),
        )
    return json.dumps({"status": "success", "plan": plan}, indent=2)


async def handle_assess_cache_compatibility(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    model = args.get("model")
    harness = args.get("harness")
    raw_usage_available = args.get("raw_usage_available", False)
    cli_stats_available = args.get("cli_stats_available", False)
    if not isinstance(model, str) or not model.strip():
        raise ValueError("assess_cache_compatibility requires non-empty 'model'")
    if not isinstance(harness, str) or not harness.strip():
        raise ValueError("assess_cache_compatibility requires non-empty 'harness'")
    if not isinstance(raw_usage_available, bool):
        raise ValueError("assess_cache_compatibility requires boolean 'raw_usage_available'")
    if not isinstance(cli_stats_available, bool):
        raise ValueError("assess_cache_compatibility requires boolean 'cli_stats_available'")

    with observe.trace(
        "model_optimization.assess_cache_compatibility",
        model=model.strip(),
        harness=harness.strip(),
    ):
        assessment = assess_cache_compatibility(
            model=model.strip(),
            harness=harness.strip(),
            raw_usage_available=raw_usage_available,
            cli_stats_available=cli_stats_available,
        )
    return json.dumps({"status": "success", "assessment": assessment}, indent=2)


async def handle_capture_cache_telemetry(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    model = args.get("model")
    api_response = args.get("api_response")
    file_id = args.get("file_id")
    prompt_id = args.get("prompt_id")
    session_id = args.get("session_id")
    actual_rendered_prefix = args.get("actual_rendered_prefix")
    expected_cache_hit = args.get("expected_cache_hit", False)

    if not isinstance(model, str) or not model.strip():
        raise ValueError("capture_cache_telemetry requires non-empty 'model'")
    if not isinstance(api_response, dict):
        raise ValueError("capture_cache_telemetry requires object 'api_response'")
    if file_id is not None and not isinstance(file_id, str):
        raise ValueError("capture_cache_telemetry requires string 'file_id' when provided")
    if prompt_id is not None and not isinstance(prompt_id, str):
        raise ValueError("capture_cache_telemetry requires string 'prompt_id' when provided")
    if session_id is not None and not isinstance(session_id, str):
        raise ValueError("capture_cache_telemetry requires string 'session_id' when provided")
    if actual_rendered_prefix is not None and not isinstance(actual_rendered_prefix, str):
        raise ValueError(
            "capture_cache_telemetry requires string 'actual_rendered_prefix' when provided"
        )
    if not isinstance(expected_cache_hit, bool):
        raise ValueError("capture_cache_telemetry requires boolean 'expected_cache_hit'")

    with observe.trace(
        "model_optimization.capture_cache_telemetry",
        model=model.strip(),
        expected_cache_hit=expected_cache_hit,
    ):
        telemetry = summarize_provider_cache_usage(
            model=model.strip(),
            api_response=api_response,
            file_id=file_id,
            expected_cache_hit=expected_cache_hit,
        )
        if prompt_id is not None:
            telemetry["validation"] = PromptCacheMiddleware.validate_provider_response(
                prompt_id=prompt_id,
                model=model.strip(),
                api_response=api_response,
                actual_rendered_prefix=actual_rendered_prefix,
            )
            telemetry["prefix_siblings"] = PromptCacheMiddleware.get_cache_siblings(prompt_id)
            if telemetry["validation"]["status"] == "validated_against_stale_expectation":
                stale = telemetry["validation"]["stale_expectation"]
                telemetry["warning"] = (
                    "Cache validation used a stale expectation for "
                    f"'{stale['template_name']}' label '{stale['label']}' "
                    f"(rendered version {stale['rendered_version']}, current version {stale['current_version']}). "
                    "Re-render to capture the current stable prefix."
                )
            elif "warning" in telemetry["validation"] and "warning" not in telemetry:
                telemetry["warning"] = telemetry["validation"]["warning"]
            if session_id is not None:
                telemetry["session_metrics"] = PromptCacheMiddleware.record_session_telemetry(
                    session_id=session_id,
                    prompt_id=prompt_id,
                    telemetry=telemetry,
                )
                telemetry["cache_health"] = telemetry["session_metrics"]["cache_health"]
            else:
                telemetry["cache_health"] = PromptCacheMiddleware.record_cache_health(
                    prompt_id=prompt_id,
                    telemetry=telemetry,
                )

        metrics = context.get("metrics") or get_metrics()
        metrics.set_cache_hit_ratio(telemetry["cache_hit_ratio"])
        telemetry["observability"] = normalize_cache_telemetry(telemetry)
        observe.record_cache_telemetry(telemetry["observability"])
        metrics.record_provider_cache_telemetry(telemetry["observability"])

    return json.dumps({"status": "success", "telemetry": telemetry}, indent=2)


async def handle_diagnose_cache_miss(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    prompt_id = args.get("prompt_id")
    model = args.get("model")
    actual_rendered_prefix = args.get("actual_rendered_prefix")
    api_response = args.get("api_response")

    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError("diagnose_cache_miss requires non-empty 'prompt_id'")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("diagnose_cache_miss requires non-empty 'model'")
    if not isinstance(actual_rendered_prefix, str) or not actual_rendered_prefix.strip():
        raise ValueError("diagnose_cache_miss requires non-empty 'actual_rendered_prefix'")
    if not isinstance(api_response, dict):
        raise ValueError("diagnose_cache_miss requires object 'api_response'")

    expectation = PromptCacheMiddleware.get_expectation(prompt_id.strip())
    if expectation is None:
        raise ValueError(f"Unknown prompt_id '{prompt_id}'")

    with observe.trace(
        "model_optimization.diagnose_cache_miss",
        model=model.strip(),
        prompt_id=prompt_id.strip(),
    ):
        diagnostic = diagnose_cache_miss(
            expectation=expectation,
            actual_rendered_prefix=actual_rendered_prefix,
            model=model.strip(),
            api_response=api_response,
        )

    return json.dumps(
        {"status": "success", "prompt_id": prompt_id.strip(), "diagnostic": diagnostic}, indent=2
    )
