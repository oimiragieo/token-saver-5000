"""Cost computation for benchmark results."""

from __future__ import annotations

# Per million tokens pricing
PRICING: dict[str, dict[str, float]] = {
    # Claude models — verified vs OpenRouter + tokencost.app 2026-06-03.
    # Fable 5 (Mythos-class, 2026-06-09): $10/$50, $1 cached input — verified
    # vs anthropic.com/claude/fable + OpenRouter 2026-06-12.
    "claude-fable-5": {"input": 10.0, "output": 50.0, "cache_read": 1.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0, "cache_read": 0.30},
    # Opus dropped to $5/$25 at the 4.5 release (was $15/$75).
    "claude-opus-4-8": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    "claude-opus-4-6": {"input": 5.0, "output": 25.0, "cache_read": 0.50},
    # Haiku 4.5 is $1/$5 (was a stale $0.80/$4.0 Haiku-3.5-era rate).
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0, "cache_read": 0.10},
    # Gemini models
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0, "cache_read": 0.315},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60, "cache_read": 0.0375},
    # Gemini 3.1 Pro is $2/$12 (<=200K); the prior $1.25/$10 was 2.5-Pro's rate.
    "gemini-3.1-pro": {"input": 2.0, "output": 12.0, "cache_read": 0.20},
    "gemini-3.1-flash": {"input": 0.15, "output": 0.60, "cache_read": 0.0375},
    # Gemini 3.5 Flash (2026-05-19) — current stable premium Flash route.
    "gemini-3.5-flash": {"input": 1.50, "output": 9.0, "cache_read": 0.15},
    # GPT models
    "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075},
    # Codex CLI models
    "gpt-5.1-codex": {"input": 2.50, "output": 10.0, "cache_read": 0.625},
    "codex-mini": {"input": 1.50, "output": 6.0, "cache_read": 0.375},
    # OpenCode additional models
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.50},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_read": 0.025},
    "o1-pro": {"input": 2.0, "output": 8.0, "cache_read": 0.50},
    "groq-llama-4-scout": {"input": 0.11, "output": 0.34, "cache_read": 0.0},
    "groq-llama-4-maverick": {"input": 0.50, "output": 2.0, "cache_read": 0.0},
    "groq-deepseek-r1": {"input": 0.75, "output": 0.99, "cache_read": 0.0},
    "grok-3": {"input": 3.0, "output": 15.0, "cache_read": 0.0},
    "grok-3-mini": {"input": 0.30, "output": 0.50, "cache_read": 0.0},
    # Default fallback
    "default": {"input": 3.0, "output": 15.0, "cache_read": 0.30},
}


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
) -> float:
    """Compute cost in USD from token counts and model pricing."""
    rates = PRICING.get(model, PRICING["default"])
    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
        + cache_read_tokens * rates.get("cache_read", 0.0)
    ) / 1_000_000
    return round(cost, 6)


def get_model_rates(model: str) -> dict[str, float]:
    """Get pricing rates for a model, falling back to default."""
    return PRICING.get(model, PRICING["default"])
