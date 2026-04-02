"""Provider-aware cache strategy advisor.

Recommends optimal caching approach based on model and provider,
since each LLM provider handles prompt caching differently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .client_config import _detect_provider


@dataclass
class CacheStrategy:
    """Recommended caching strategy for a model/provider."""

    provider: str
    model: str
    cache_type: str  # "explicit", "automatic", "implicit", "none", "unknown"
    min_prefix_tokens: int  # minimum prefix size for caching to activate
    cache_discount_pct: int  # % discount on cached tokens (e.g., 90 = 90% off)
    ttl_description: str  # human-readable TTL info
    client_action: str  # what the user/tool needs to do
    tips: list[str]  # optimization tips
    supports_cache: bool  # whether caching is available at all


def advise_cache_strategy(model_id: str) -> CacheStrategy:
    """Get the recommended cache strategy for a model.

    Args:
        model_id: Model identifier (e.g., "claude-4-sonnet", "gpt-4.1", "gemini-2.5-flash")

    Returns:
        CacheStrategy with provider-specific recommendations.
    """
    provider = _detect_provider(model_id)
    model_lower = model_id.lower()

    if provider == "anthropic":
        return CacheStrategy(
            provider="anthropic",
            model=model_id,
            cache_type="explicit",
            min_prefix_tokens=1024,
            cache_discount_pct=90,
            ttl_description="5 minutes (1 hour for subscribers)",
            client_action=("Add cache_control: ephemeral to system prompt and last 2 messages"),
            tips=[
                "Place stable content (instructions, examples) BEFORE dynamic content",
                "Sort tool schemas alphabetically for prefix stability",
                "Cache markers on system prompt + last 2 conversation messages",
                "Cache creation costs 25% more than regular input",
                "Cache reads save 90% vs regular input",
                "Use Token Saver's cache-stable response ordering for maximum prefix reuse",
            ],
            supports_cache=True,
        )

    if provider == "openai":
        return CacheStrategy(
            provider="openai",
            model=model_id,
            cache_type="automatic",
            min_prefix_tokens=1024,
            cache_discount_pct=50,
            ttl_description="5-60 minutes (automatic eviction)",
            client_action=(
                "No client action needed. Caching is automatic for 1024+ token prefixes."
            ),
            tips=[
                "Keep system prompt and tool schemas at the START of the prompt",
                "Variable content (user input) goes at the END",
                "Cache hits happen in 128-token increments after the first 1024",
                "50% discount on cached prefix tokens (automatic)",
                "No explicit cache markers needed",
                "Token Saver responses are already cache-stable ordered",
            ],
            supports_cache=True,
        )

    if provider == "google":
        is_25_plus = any(v in model_lower for v in ["2.5", "3.0", "3.1"])
        if is_25_plus:
            return CacheStrategy(
                provider="google",
                model=model_id,
                cache_type="implicit",
                min_prefix_tokens=0,
                cache_discount_pct=90,
                ttl_description="Automatic (managed by Google)",
                client_action=(
                    "No client action needed. Implicit caching is automatic on Gemini 2.5+."
                ),
                tips=[
                    "Implicit caching provides 90% discount automatically",
                    "Explicit caching available for large static content (4096+ tokens)",
                    "Keep prompts stable between requests for best cache hit rate",
                    "Gemini compresses early (50% of window) -- use compact profiles",
                    "Token Saver's proportional truncation matches Gemini's head-preserving pattern",
                ],
                supports_cache=True,
            )
        return CacheStrategy(
            provider="google",
            model=model_id,
            cache_type="explicit",
            min_prefix_tokens=4096,
            cache_discount_pct=75,
            ttl_description="1 hour (configurable)",
            client_action=(
                "Use cachedContents.create() API for large static content (4096+ tokens)"
            ),
            tips=[
                "Explicit caching requires API call to create cache object",
                "75% discount on Gemini 2.0 models",
                "Cache large reference documents that don't change between requests",
            ],
            supports_cache=True,
        )

    # Groq -- limited or no caching
    if provider == "groq":
        return CacheStrategy(
            provider="groq",
            model=model_id,
            cache_type="automatic",
            min_prefix_tokens=0,
            cache_discount_pct=0,
            ttl_description="Limited prompt caching on select models",
            client_action=(
                "No client action. Focus on minimizing prompt size for fastest inference."
            ),
            tips=[
                "Groq optimizes for speed, not cost -- inference is already cheap",
                "Minimize prompt size to maximize Groq's speed advantage",
                "Use Token Saver's minimal/summary profiles for maximum compression",
                "Groq models have smaller context windows -- compression is more valuable",
            ],
            supports_cache=False,
        )

    # xAI / Grok -- no caching
    if provider == "xai":
        return CacheStrategy(
            provider="xai",
            model=model_id,
            cache_type="none",
            min_prefix_tokens=0,
            cache_discount_pct=0,
            ttl_description="No caching available",
            client_action="No caching support. Optimize by reducing prompt size.",
            tips=[
                "No prompt caching -- every token is billed at full price",
                "Token Saver compression provides direct cost savings",
                "Use aggressive compression profiles (minimal/summary)",
            ],
            supports_cache=False,
        )

    if provider == "local":
        return CacheStrategy(
            provider="local",
            model=model_id,
            cache_type="none",
            min_prefix_tokens=0,
            cache_discount_pct=0,
            ttl_description="N/A (local inference)",
            client_action="No caching. Smaller prompts = faster local inference.",
            tips=[
                "Local models have no token cost but benefit from smaller prompts",
                "Compression reduces inference time (less to process)",
                "Use Token Saver to fit more context into limited local model windows",
                "Most local models have 4K-32K context -- compression is essential",
            ],
            supports_cache=False,
        )

    # Unknown provider fallback
    return CacheStrategy(
        provider="unknown",
        model=model_id,
        cache_type="unknown",
        min_prefix_tokens=1024,
        cache_discount_pct=0,
        ttl_description="Unknown",
        client_action="Check provider documentation for caching support.",
        tips=[
            "Place stable content at the beginning of prompts",
            "Use Token Saver compression to reduce total token usage",
        ],
        supports_cache=False,
    )
