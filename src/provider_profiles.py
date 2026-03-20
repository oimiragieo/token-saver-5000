"""Provider/model profile registry for cost and cache-aware optimization."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ProviderProfile:
    model: str
    provider: str
    input_cost_per_million: float
    output_cost_per_million: float
    cached_input_cost_per_million: float
    context_window: int
    cache_read_field: str
    minimum_cacheable_tokens: int
    cache_token_increment: int
    supports_prompt_cache_key: bool
    cache_routing_strategy: str
    prompt_prefix_strategy: str
    recommended_output_format: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROFILES = {
    "claude-opus-4.6": ProviderProfile(
        model="claude-opus-4.6",
        provider="anthropic",
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        cached_input_cost_per_million=1.5,
        context_window=200_000,
        cache_read_field="cache_read_input_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=256,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Keep requests on the same workflow shape; Anthropic cache reuse depends primarily on exact stable-prefix identity.",
        prompt_prefix_strategy="Keep tool definitions, system instructions, and large docs fixed at the front; move volatile query material to the tail.",
        recommended_output_format="toon",
        notes="Premium-cost model: optimize aggressively for cache stability and compact structured outputs.",
    ),
    "claude-sonnet-4.6": ProviderProfile(
        model="claude-sonnet-4.6",
        provider="anthropic",
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.3,
        context_window=200_000,
        cache_read_field="cache_read_input_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=256,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Favor byte-identical prompt prefixes and stable tool/system ordering across turns.",
        prompt_prefix_strategy="Stabilize prefixes and pin reusable context before chat history and current query.",
        recommended_output_format="toon",
        notes="Balanced cost/quality profile suited to cache-first prompt orchestration.",
    ),
    "claude-haiku-4": ProviderProfile(
        model="claude-haiku-4",
        provider="anthropic",
        input_cost_per_million=0.8,
        output_cost_per_million=4.0,
        cached_input_cost_per_million=0.08,
        context_window=200_000,
        cache_read_field="cache_read_input_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=256,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Keep lightweight prefixes stable; routing hints are less important than exact prefix reuse.",
        prompt_prefix_strategy="Keep prompts compact and stable; prefer lower-fidelity retrieval by default.",
        recommended_output_format="toon",
        notes="Low-cost model; useful for lightweight routing and summary tasks.",
    ),
    "gpt-5.4": ProviderProfile(
        model="gpt-5.4",
        provider="openai",
        input_cost_per_million=10.0,
        output_cost_per_million=30.0,
        cached_input_cost_per_million=1.0,
        context_window=400_000,
        cache_read_field="cached_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=128,
        supports_prompt_cache_key=True,
        cache_routing_strategy="Use a stable prompt_cache_key per workflow + stable prefix to improve routing stickiness across repeated requests.",
        prompt_prefix_strategy="Protect static instructions and RAG context up front; keep dynamic metadata and user query last.",
        recommended_output_format="toon",
        notes="High-context OpenAI profile with explicit cached_tokens telemetry.",
    ),
    "gpt-4.1": ProviderProfile(
        model="gpt-4.1",
        provider="openai",
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
        cached_input_cost_per_million=0.2,
        context_window=128_000,
        cache_read_field="cached_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=128,
        supports_prompt_cache_key=True,
        cache_routing_strategy="Use stable prompt_cache_key values for repeated workflows when your integration exposes them.",
        prompt_prefix_strategy="Prefer compact structured prefixes and stable few-shot examples.",
        recommended_output_format="toon",
        notes="Mid-cost OpenAI profile suited to compact context delivery.",
    ),
    "gemini-3.1-pro-preview": ProviderProfile(
        model="gemini-3.1-pro-preview",
        provider="google",
        input_cost_per_million=1.25,
        output_cost_per_million=10.0,
        cached_input_cost_per_million=0.125,
        context_window=1_000_000,
        cache_read_field="cachedContentTokenCount",
        minimum_cacheable_tokens=1024,
        cache_token_increment=1024,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Gemini cache performance depends on keeping a large common prefix first; route stickiness is less configurable than OpenAI-style prompt_cache_key flows.",
        prompt_prefix_strategy="Large contexts are available, but stable prefix ordering still determines cache reuse.",
        recommended_output_format="json",
        notes="Gemini 3.x large-context profile: can afford higher-fidelity retrieval when correctness benefits.",
    ),
}

_ALIASES = {
    "claude-opus-4": "claude-opus-4.6",
    "claude-opus-4.5": "claude-opus-4.6",
    "claude-sonnet-4": "claude-sonnet-4.6",
    "claude-sonnet-4.5": "claude-sonnet-4.6",
    "claude-haiku-3.5": "claude-haiku-4",
    "gpt-5": "gpt-5.4",
    "gpt-5.4": "gpt-5.4",
    "gpt-5.3-codex": "gpt-5.4",
    "gpt-5.2-codex": "gpt-5.4",
    "codex": "gpt-5.4",
    "gemini-3": "gemini-3.1-pro-preview",
    "gemini-3.1-pro": "gemini-3.1-pro-preview",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
    "gemini-2.5": "gemini-3.1-pro-preview",
    "gemini-2.5-pro": "gemini-3.1-pro-preview",
}


def get_provider_profile(model: str) -> ProviderProfile:
    normalized = _ALIASES.get(model, model)
    profile = _PROFILES.get(normalized)
    if profile is None:
        raise ValueError(f"Unknown model '{model}'")
    return profile


def list_provider_profiles() -> list[dict[str, Any]]:
    return [profile.to_dict() for _, profile in sorted(_PROFILES.items())]
