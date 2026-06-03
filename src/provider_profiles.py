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
    "claude-opus-4.8": ProviderProfile(
        model="claude-opus-4.8",
        provider="anthropic",
        input_cost_per_million=5.0,
        output_cost_per_million=25.0,
        cached_input_cost_per_million=0.5,
        context_window=1_000_000,
        cache_read_field="cache_read_input_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=256,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Keep requests on the same workflow shape; Anthropic cache reuse depends primarily on exact stable-prefix identity.",
        prompt_prefix_strategy="Keep tool definitions, system instructions, and large docs fixed at the front; move volatile query material to the tail. 1M-context mode benefits from stable-prefix discipline at larger document sizes.",
        recommended_output_format="toon",
        notes="Anthropic's current flagship Opus (released 2026-05-27): $5/$25 per 1M, 1M-token context. Pricing verified vs OpenRouter 2026-06-03.",
    ),
    "claude-opus-4.7": ProviderProfile(
        model="claude-opus-4.7",
        provider="anthropic",
        input_cost_per_million=5.0,
        output_cost_per_million=25.0,
        cached_input_cost_per_million=0.5,
        context_window=1_000_000,
        cache_read_field="cache_read_input_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=256,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Keep requests on the same workflow shape; Anthropic cache reuse depends primarily on exact stable-prefix identity.",
        prompt_prefix_strategy="Keep tool definitions, system instructions, and large docs fixed at the front; move volatile query material to the tail. 1M-context mode benefits from stable-prefix discipline at larger document sizes.",
        recommended_output_format="toon",
        notes="Flagship Anthropic model. Same per-1M pricing as Opus 4.6 but with 1M-token context window — compression still pays off because output tokens dominate the bill.",
    ),
    "claude-opus-4.6": ProviderProfile(
        model="claude-opus-4.6",
        provider="anthropic",
        input_cost_per_million=5.0,
        output_cost_per_million=25.0,
        cached_input_cost_per_million=0.5,
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
        # Standard tier. Per OpenAI's 2026-04 release notes and Artificial
        # Analysis pricing index, GPT-5.4 standard input was $2.50/MTok,
        # output $15/MTok — earlier $10/$30 entry was a stale copy from a
        # GPT-5.2 profile.  GPT-5.5 (separate entry) is the one that
        # doubled standard to $5/$30 on 2026-04-23.
        input_cost_per_million=2.5,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.25,
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
    "gemini-3.1-flash": ProviderProfile(
        model="gemini-3.1-flash",
        provider="google",
        input_cost_per_million=0.30,
        output_cost_per_million=2.50,
        cached_input_cost_per_million=0.03,
        context_window=1_000_000,
        cache_read_field="cachedContentTokenCount",
        minimum_cacheable_tokens=1024,
        cache_token_increment=1024,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Same prefix-stability rules as Pro; Flash tier amplifies the value of cache reuse because the base rate is already low.",
        prompt_prefix_strategy="Keep large contexts at the head; Flash handles long inputs but still pays per token.",
        recommended_output_format="json",
        notes="Low-cost Gemini 3.1 variant; pair with aggressive compression and cache reuse to get the best $/request.",
    ),
    "gemini-auto": ProviderProfile(
        model="gemini-auto",
        provider="google",
        # Routed mix of Pro + Flash; blended effective rate
        input_cost_per_million=0.60,
        output_cost_per_million=5.0,
        cached_input_cost_per_million=0.06,
        context_window=1_000_000,
        cache_read_field="cachedContentTokenCount",
        minimum_cacheable_tokens=1024,
        cache_token_increment=1024,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Auto routing picks Pro for hard prompts and Flash for easy ones; prompt cache hits are per-family, so the same prefix can land on different tiers across turns.",
        prompt_prefix_strategy="Stable prefixes still help, but assume some routing variance — don't overoptimize for a specific tier.",
        recommended_output_format="json",
        notes="Blended-rate profile for Gemini auto routing; cost math is approximate because actual pricing depends on per-turn tier selection.",
    ),
    "gpt-5.4-mini": ProviderProfile(
        model="gpt-5.4-mini",
        provider="openai",
        input_cost_per_million=0.25,
        output_cost_per_million=2.0,
        cached_input_cost_per_million=0.025,
        context_window=400_000,
        cache_read_field="cached_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=128,
        supports_prompt_cache_key=True,
        cache_routing_strategy="Same prompt_cache_key pattern as full GPT-5.4; the mini tier shares the cache infrastructure.",
        prompt_prefix_strategy="Keep system + tool prefix stable; mini is already cheap so the value of caching comes from latency, not cost reduction.",
        recommended_output_format="toon",
        notes="Low-cost GPT-5.4 variant for high-volume routing and summary tasks.",
    ),
    "gpt-4.1-mini": ProviderProfile(
        model="gpt-4.1-mini",
        provider="openai",
        input_cost_per_million=0.40,
        output_cost_per_million=1.60,
        cached_input_cost_per_million=0.10,
        context_window=128_000,
        cache_read_field="cached_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=128,
        supports_prompt_cache_key=True,
        cache_routing_strategy="Use stable prompt_cache_key for repeatable mini workflows; cache write premium is small relative to the per-call savings.",
        prompt_prefix_strategy="Prefer compact structured prefixes; mini models benefit more from tight few-shot examples.",
        recommended_output_format="toon",
        notes="GPT-4.1 mini — low cost, smaller context; good for bulk preprocessing and light classification.",
    ),
    # Launched 2026-04-23. Same input rate as Opus 4.7 ($5/MTok) but output is
    # significantly more expensive at $30/MTok (vs $25/MTok for Opus). 1M-token
    # context window. Prompt cache read at 10% of input (same as GPT-5.4 tier).
    "gpt-5.5": ProviderProfile(
        model="gpt-5.5",
        provider="openai",
        input_cost_per_million=5.0,
        output_cost_per_million=30.0,
        cached_input_cost_per_million=0.50,
        context_window=1_000_000,
        cache_read_field="cached_tokens",
        minimum_cacheable_tokens=1024,
        cache_token_increment=128,
        supports_prompt_cache_key=True,
        cache_routing_strategy="Use stable prompt_cache_key per workflow; 1M-context window makes stable-prefix discipline especially valuable at large document sizes.",
        prompt_prefix_strategy="Pin system instructions, tool definitions, and large docs at the front; move volatile query material to the tail.",
        recommended_output_format="toon",
        notes="OpenAI GPT-5.5 flagship (2026-04-23). 1M-token context, $5/$30 per MTok input/output. Compression ROI is high given the expensive output rate.",
    ),
}

_ALIASES = {
    # Latest flagship — unversioned name resolves to the current Opus (4.8).
    "claude-opus": "claude-opus-4.8",
    "claude-opus-4": "claude-opus-4.8",
    "claude-opus-4-8": "claude-opus-4.8",
    "opus-4-8": "claude-opus-4.8",
    "opus-4.8": "claude-opus-4.8",
    # Prior versions kept as their own profiles (all same $5/$25 pricing).
    "claude-opus-4-7": "claude-opus-4.7",
    "opus-4-7": "claude-opus-4.7",
    "opus-4.7": "claude-opus-4.7",
    # Pinned earlier version kept as its own profile
    "claude-opus-4.5": "claude-opus-4.6",
    "claude-opus-4-6": "claude-opus-4.6",
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
    # Flash family
    "gemini-flash": "gemini-3.1-flash",
    "gemini-3-flash": "gemini-3.1-flash",
    "gemini-2.5-flash": "gemini-3.1-flash",
    # Auto routing
    "gemini": "gemini-auto",
    "gemini-auto-preview": "gemini-auto",
    # OpenAI mini variants
    "gpt-5-mini": "gpt-5.4-mini",
    "gpt-5.3-mini": "gpt-5.4-mini",
    "gpt-4o-mini": "gpt-4.1-mini",
    "gpt-4-mini": "gpt-4.1-mini",
    # GPT-5.5 hyphen alias (e.g. used as "gpt-5-5" in some clients)
    "gpt-5-5": "gpt-5.5",
}


def get_provider_profile(model: str) -> ProviderProfile:
    normalized = _ALIASES.get(model, model)
    profile = _PROFILES.get(normalized)
    if profile is None:
        raise ValueError(f"Unknown model '{model}'")
    return profile


def list_provider_profiles() -> list[dict[str, Any]]:
    return [profile.to_dict() for _, profile in sorted(_PROFILES.items())]
