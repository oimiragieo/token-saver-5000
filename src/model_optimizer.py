"""Model-aware optimization recommendations for cost, fidelity, and prompt shaping."""

from __future__ import annotations

import hashlib
from typing import Any

from .compression_advisor import CompressionAdvisor
from .fidelity_advisor import FidelityAdvisor, UseCase
from .metrics import compute_cost_savings
from .provider_profiles import get_provider_profile

_INPUT_TOKEN_FIELDS = (
    "input_tokens",
    "prompt_tokens",
    "promptTokenCount",
    "prompt_token_count",
    "inputTokens",
)
_OUTPUT_TOKEN_FIELDS = (
    "output_tokens",
    "completion_tokens",
    "candidatesTokenCount",
    "candidates_token_count",
    "outputTokens",
)
_CACHE_CREATION_FIELDS = ("cache_creation_input_tokens",)
_CACHE_READ_FIELD_ALIASES = {
    "cache_read_input_tokens": ("cache_read_input_tokens",),
    "cached_tokens": ("cached_tokens",),
    "cachedContentTokenCount": (
        "cachedContentTokenCount",
        "cached_content_token_count",
        "cachedTokens",
        "cache_hit_input_tokens",
    ),
}


def _find_numeric_field(payload: Any, field_name: str) -> int | None:
    if isinstance(payload, dict):
        value = payload.get(field_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        for nested in payload.values():
            nested_value = _find_numeric_field(nested, field_name)
            if nested_value is not None:
                return nested_value
    elif isinstance(payload, list):
        for item in payload:
            nested_value = _find_numeric_field(item, field_name)
            if nested_value is not None:
                return nested_value
    return None


def _require_numeric_field(
    payload: dict[str, Any], field_names: tuple[str, ...], label: str
) -> int:
    for field_name in field_names:
        value = _find_numeric_field(payload, field_name)
        if value is not None:
            return value
    supported = ", ".join(field_names)
    raise ValueError(f"Unable to determine {label}; expected one of: {supported}")


def _find_provider_cache_read_tokens(payload: dict[str, Any], field_name: str) -> int | None:
    aliases = _CACHE_READ_FIELD_ALIASES.get(field_name, (field_name,))
    for candidate in aliases:
        value = _find_numeric_field(payload, candidate)
        if value is not None:
            return value
    return None


def summarize_provider_cache_usage(
    *,
    model: str,
    api_response: dict[str, Any],
    file_id: str | None = None,
    expected_cache_hit: bool = False,
) -> dict[str, Any]:
    if not isinstance(api_response, dict):
        raise ValueError("summarize_provider_cache_usage requires dict 'api_response'")

    profile = get_provider_profile(model)
    total_input_tokens = _require_numeric_field(api_response, _INPUT_TOKEN_FIELDS, "input tokens")
    cache_read_tokens = _find_provider_cache_read_tokens(api_response, profile.cache_read_field)
    if cache_read_tokens is None:
        raise ValueError(
            f"Unable to determine provider cache read tokens; expected field "
            f"'{profile.cache_read_field}' in api_response"
        )
    total_output_tokens = _require_numeric_field(
        api_response, _OUTPUT_TOKEN_FIELDS, "output tokens"
    )

    cache_creation_tokens = _find_numeric_field(api_response, _CACHE_CREATION_FIELDS[0]) or 0
    cache_hit_ratio = (cache_read_tokens / total_input_tokens) if total_input_tokens > 0 else 0.0
    uncached_read_cost = (cache_read_tokens / 1_000_000) * profile.input_cost_per_million
    cached_read_cost = (cache_read_tokens / 1_000_000) * profile.cached_input_cost_per_million

    summary = {
        "model": profile.model,
        "provider": profile.provider,
        "file_id": file_id,
        "cache_read_field": profile.cache_read_field,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_hit_detected": cache_read_tokens > 0,
        "cache_hit_ratio": round(cache_hit_ratio, 4),
        "estimated_uncached_input_cost_usd": round(
            (total_input_tokens / 1_000_000) * profile.input_cost_per_million, 6
        ),
        "estimated_cache_read_cost_usd": round(cached_read_cost, 6),
        "estimated_cache_savings_usd": round(max(uncached_read_cost - cached_read_cost, 0.0), 6),
    }
    summary[profile.cache_read_field] = cache_read_tokens

    if expected_cache_hit and cache_read_tokens == 0:
        summary["warning"] = (
            "Expected a provider-side cache hit but observed a cache miss. "
            "Check prompt prefix stability, hidden dynamic metadata, and provider cache configuration."
        )

    return summary


def advise_cache_threshold(*, model: str, prompt_tokens: int) -> dict[str, Any]:
    """Explain whether a prompt is large enough to benefit from provider caching."""
    if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
        raise ValueError("advise_cache_threshold requires integer 'prompt_tokens' >= 0")
    profile = get_provider_profile(model)
    eligible = prompt_tokens >= profile.minimum_cacheable_tokens
    tokens_below = max(profile.minimum_cacheable_tokens - prompt_tokens, 0)
    next_increment_target = profile.minimum_cacheable_tokens
    if prompt_tokens > profile.minimum_cacheable_tokens:
        next_increment_target = (
            profile.minimum_cacheable_tokens
            + (
                (
                    prompt_tokens
                    - profile.minimum_cacheable_tokens
                    + profile.cache_token_increment
                    - 1
                )
                // profile.cache_token_increment
            )
            * profile.cache_token_increment
        )
    guidance = (
        "Prompt is eligible for cache accounting thresholds; keep the reusable prefix byte-stable."
        if eligible
        else "Prompt is not yet eligible for provider cache accounting thresholds; increase stable reusable prefix size before expecting cache eligibility."
    )
    return {
        "model": profile.model,
        "eligible": eligible,
        "minimum_cacheable_tokens": profile.minimum_cacheable_tokens,
        "cache_token_increment": profile.cache_token_increment,
        "prompt_tokens": prompt_tokens,
        "tokens_below_minimum": tokens_below,
        "next_increment_target": next_increment_target,
        "guidance": guidance,
    }


def build_prompt_cache_key(*, model: str, workflow_id: str, stable_prefix: str) -> str:
    """Build a deterministic prompt_cache_key from workflow identity and stable prefix."""
    if not isinstance(workflow_id, str) or not workflow_id.strip():
        raise ValueError("build_prompt_cache_key requires non-empty 'workflow_id'")
    if not isinstance(stable_prefix, str) or not stable_prefix.strip():
        raise ValueError("build_prompt_cache_key requires non-empty 'stable_prefix'")
    profile = get_provider_profile(model)
    digest = hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest()[:16]
    return f"{profile.model}:{workflow_id.strip()}:{digest}"


def optimize_for_model(
    *,
    text: str,
    model: str,
    use_case: UseCase,
    num_nodes: int,
    token_budget: int | None = None,
    query_complexity: str = "medium",
) -> dict[str, Any]:
    profile = get_provider_profile(model)
    compression_estimate = CompressionAdvisor().estimate_compression(text)
    effective_budget = token_budget

    if effective_budget is None:
        if profile.input_cost_per_million >= 10:
            effective_budget = max(80, num_nodes * 35)
        elif profile.context_window >= 500_000:
            effective_budget = num_nodes * 140
        else:
            effective_budget = num_nodes * 90

    fidelity = FidelityAdvisor().recommend(
        use_case=use_case,
        num_nodes=num_nodes,
        token_budget=effective_budget,
        query_complexity=query_complexity,
        model=model,
    )
    projected_costs = compute_cost_savings(
        original_tokens=compression_estimate.original_tokens,
        compressed_tokens=compression_estimate.estimated_compressed,
        model=model,
    ).to_dict()
    cache_thresholds = advise_cache_threshold(
        model=model,
        prompt_tokens=compression_estimate.original_tokens,
    )

    return {
        "provider_profile": profile.to_dict(),
        "compression_estimate": {
            "original_tokens": compression_estimate.original_tokens,
            "estimated_compressed": compression_estimate.estimated_compressed,
            "compression_ratio": round(compression_estimate.compression_ratio, 2),
            "reasoning": compression_estimate.reasoning,
        },
        "recommended_fidelity": {
            "recommended_level": fidelity.recommended_level.name,
            "confidence": round(fidelity.confidence, 2),
            "reasoning": fidelity.reasoning,
            "token_estimate": fidelity.token_estimate,
            "alternatives": fidelity.alternatives,
        },
        "recommended_output_format": profile.recommended_output_format,
        "cache_strategy": (
            f"Monitor provider cache reads via '{profile.cache_read_field}' and keep the prompt prefix stable."
        ),
        "routing_stickiness": {
            "supports_prompt_cache_key": profile.supports_prompt_cache_key,
            "strategy": profile.cache_routing_strategy,
            "example_prompt_cache_key": (
                build_prompt_cache_key(
                    model=model,
                    workflow_id="default-workflow",
                    stable_prefix="system|tools|rag-context",
                )
                if profile.supports_prompt_cache_key
                else None
            ),
        },
        "cache_thresholds": cache_thresholds,
        "prompt_structure": profile.prompt_prefix_strategy,
        "projected_costs": {
            "estimated_savings_usd": projected_costs["cost_savings_usd"],
            "estimated_original_cost_usd": projected_costs["estimated_original_cost_usd"],
            "estimated_compressed_cost_usd": projected_costs["estimated_compressed_cost_usd"],
            "saved_tokens": projected_costs["saved_tokens"],
            "savings_percent": projected_costs["savings_percent"],
        },
        "reasoning": (
            f"{profile.notes} Estimated compression is {compression_estimate.compression_ratio:.1f}x, "
            f"and the recommended fidelity is {fidelity.recommended_level.name}."
        ),
    }
