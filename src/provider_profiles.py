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
    # Claude Haiku 4.5 — $1/$5 per MTok, $0.10 cache-read (10%). Verified vs
    # OpenRouter + tokencost.app 2026-06-03. The prior "claude-haiku-4" entry
    # carried a stale $0.80/$4.0 (a Haiku-3.5-era rate); the current small
    # Anthropic model is Haiku 4.5 at the rates below.
    "claude-haiku-4.5": ProviderProfile(
        model="claude-haiku-4.5",
        provider="anthropic",
        input_cost_per_million=1.0,
        output_cost_per_million=5.0,
        cached_input_cost_per_million=0.10,
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
        cached_input_cost_per_million=0.5,
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
    # Gemini 3.1 Pro — $2/$12 per MTok (<=200K), $0.20 cache-read. Verified vs
    # Google AI pricing + tokencost.app + aipricing.guru 2026-06-03. The prior
    # entry carried $1.25/$10/$0.125, which is actually Gemini 2.5 Pro's rate —
    # the same stale-price bug class as the Opus $15/$75 burn. (Above 200K
    # context Google steps to $4/$18; we price the common <=200K tier.)
    "gemini-3.1-pro-preview": ProviderProfile(
        model="gemini-3.1-pro-preview",
        provider="google",
        input_cost_per_million=2.0,
        output_cost_per_million=12.0,
        cached_input_cost_per_million=0.20,
        context_window=1_000_000,
        cache_read_field="cachedContentTokenCount",
        minimum_cacheable_tokens=1024,
        cache_token_increment=1024,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Gemini cache performance depends on keeping a large common prefix first; route stickiness is less configurable than OpenAI-style prompt_cache_key flows.",
        prompt_prefix_strategy="Large contexts are available, but stable prefix ordering still determines cache reuse.",
        recommended_output_format="json",
        notes="Gemini 3.1 Pro flagship: $2/$12 per MTok (<=200K), 2x in / 1.5x out above 200K. Large-context; can afford higher-fidelity retrieval when correctness benefits.",
    ),
    # Gemini 3.5 Flash — $1.50/$9 per MTok, $0.15 cache-read, flat 1M context.
    # Launched 2026-05-19 (Google I/O). Verified vs Google AI pricing +
    # OpenRouter + tokencost.app 2026-06-03. This is Google's current stable
    # premium Flash route and the headline "Gemini 3.5" model (there is NO
    # Gemini 3.5 Pro — the Pro tier remains 3.1 Pro).
    "gemini-3.5-flash": ProviderProfile(
        model="gemini-3.5-flash",
        provider="google",
        input_cost_per_million=1.50,
        output_cost_per_million=9.0,
        cached_input_cost_per_million=0.15,
        context_window=1_000_000,
        cache_read_field="cachedContentTokenCount",
        minimum_cacheable_tokens=1024,
        cache_token_increment=1024,
        supports_prompt_cache_key=False,
        cache_routing_strategy="Same prefix-stability rules as Pro; flat 1M-context pricing means stable-prefix discipline pays off across the whole window.",
        prompt_prefix_strategy="Keep large contexts at the head; 3.5 Flash handles long inputs at a flat rate but still pays per token.",
        recommended_output_format="json",
        notes="Gemini 3.5 Flash (2026-05-19): frontier-level agentic/coding/multimodal at $1.50/$9 per MTok, flat 1M context. Strong compression ROI given the $9 output rate.",
    ),
    # Gemini 2.5 Flash — $0.30/$2.50 per MTok, $0.03 cache-read. Kept as a
    # historical/low-cost entry (this was previously mislabeled "gemini-3.1-flash";
    # the $0.30/$2.50 rate is in fact Gemini 2.5 Flash's).
    "gemini-2.5-flash": ProviderProfile(
        model="gemini-2.5-flash",
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
        notes="Low-cost Gemini 2.5 Flash; pair with aggressive compression and cache reuse to get the best $/request.",
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
    # gpt-5.4-mini — $0.75/$4.50 per MTok, $0.075 cache-read. Verified vs
    # OpenAI official pricing (platform.openai.com/docs/pricing) AND OpenRouter
    # 2026-06-07; both sources agree. The prior $0.25/$2.0/$0.025 entry was a
    # stale guess (likely carried from a 4o-mini-era rate) — same stale-price
    # bug class as the Opus $15/$75 and Gemini 2.5-Pro-on-3.1-Pro burns. This
    # model is SLUG_OVERRIDES[None] in scripts/audit_model_prices.py so the
    # weekly OpenRouter auditor never flagged it.
    "gpt-5.4-mini": ProviderProfile(
        model="gpt-5.4-mini",
        provider="openai",
        input_cost_per_million=0.75,
        output_cost_per_million=4.50,
        cached_input_cost_per_million=0.075,
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
    # Haiku — unversioned + prior-name aliases resolve to the current Haiku 4.5.
    "claude-haiku": "claude-haiku-4.5",
    "claude-haiku-4": "claude-haiku-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "claude-haiku-3.5": "claude-haiku-4.5",
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
    # Flash family — current flash flagship is Gemini 3.5 Flash (2026-05-19).
    "gemini-flash": "gemini-3.5-flash",
    "gemini-3-flash": "gemini-3.5-flash",
    "gemini-3.5": "gemini-3.5-flash",
    # Back-compat: the old "gemini-3.1-flash" id was really 2.5-Flash-priced.
    "gemini-3.1-flash": "gemini-2.5-flash",
    # Auto routing. Bare "gemini" resolves to the current flagship-value model
    # (3.5 Flash) per the 2026-06-03 catalog refresh; the blended router stays
    # reachable via "gemini-auto".
    "gemini": "gemini-3.5-flash",
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
