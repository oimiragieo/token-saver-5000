"""Provider + harness prompt-cache compatibility guidance."""

from __future__ import annotations

from typing import Any

from .provider_profiles import get_provider_profile

_HARNESS_LABELS = {
    "anthropic_api": "Anthropic API",
    "claude_code": "Claude Code",
    "openai_api": "OpenAI API",
    "codex_cli": "Codex CLI",
    "gemini_api": "Gemini API",
    "gemini_cli": "Gemini CLI",
}

_HARNESS_SOURCES = {
    "anthropic_api": ["provider response"],
    "claude_code": ["provider response", "usage logs"],
    "openai_api": ["provider response"],
    "codex_cli": ["provider response", "verbose cache stats"],
    "gemini_api": ["provider response"],
    "gemini_cli": ["provider response", "/stats"],
}

_HARNESS_RISKS = {
    "anthropic_api": [
        "Framework wrappers can still mutate tool ordering or system prefixes before the provider call."
    ],
    "claude_code": [
        "Conversation growth and project-level preambles can perturb the stable prefix.",
        "Provider usage may be hidden unless logs or raw responses are captured.",
    ],
    "openai_api": ["System/tool reordering or hidden metadata can break exact-prefix reuse."],
    "codex_cli": [
        "Thread resume, compaction, and model switching can rewrite the stable prefix.",
        "Cache visibility may require raw usage payloads or verbose CLI stats.",
    ],
    "gemini_api": [
        "Authentication mode and prompt assembly can change what cache telemetry is visible."
    ],
    "gemini_cli": [
        "CLI prompt assembly can prepend dynamic metadata ahead of the intended stable prefix.",
        "Some auth modes may not expose the same cache counters as raw API usage.",
    ],
}


def _normalize_harness(harness: str) -> str:
    normalized = harness.strip().lower()
    if normalized not in _HARNESS_LABELS:
        raise ValueError(f"Unknown harness '{harness}'")
    return normalized


def assess_cache_compatibility(
    *,
    model: str,
    harness: str,
    raw_usage_available: bool = False,
    cli_stats_available: bool = False,
) -> dict[str, Any]:
    """Assess whether a provider+harness surface can be monitored reliably."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("assess_cache_compatibility requires non-empty 'model'")
    if not isinstance(harness, str) or not harness.strip():
        raise ValueError("assess_cache_compatibility requires non-empty 'harness'")
    if not isinstance(raw_usage_available, bool):
        raise ValueError("assess_cache_compatibility requires boolean 'raw_usage_available'")
    if not isinstance(cli_stats_available, bool):
        raise ValueError("assess_cache_compatibility requires boolean 'cli_stats_available'")

    profile = get_provider_profile(model.strip())
    normalized_harness = _normalize_harness(harness)
    acceptable_sources = list(_HARNESS_SOURCES[normalized_harness])
    visibility_available = raw_usage_available or cli_stats_available
    visibility_status = "available" if visibility_available else "missing"
    if raw_usage_available:
        visibility_status = "raw"
    elif cli_stats_available:
        visibility_status = "stats"

    support_level = "supported" if visibility_available else "conditional"
    recommendations = [
        "Keep tool definitions, system instructions, and large retrieved context fixed at the front of the prompt.",
        "Validate real cache behavior with capture_cache_telemetry instead of assuming hits from cost trends.",
    ]
    if normalized_harness in {"codex_cli", "openai_api"} and profile.supports_prompt_cache_key:
        recommendations.append(
            "Use a stable prompt_cache_key derived from workflow identity plus stable-prefix identity to improve routing stickiness."
        )
    if normalized_harness == "gemini_cli":
        recommendations.append(
            "Capture Gemini CLI /stats output when raw provider usage is unavailable so cached token counts remain observable."
        )
    if normalized_harness == "claude_code":
        recommendations.append(
            "Confirm cache_read_input_tokens and cache_creation_input_tokens are visible in logs or provider responses before trusting cache automation."
        )

    return {
        "model": profile.model,
        "provider": profile.provider,
        "harness": normalized_harness,
        "harness_label": _HARNESS_LABELS[normalized_harness],
        "support_level": support_level,
        "risk_level": "high" if normalized_harness.endswith("_cli") else "medium",
        "telemetry": {
            "primary_fields": [profile.cache_read_field],
            "acceptable_sources": acceptable_sources,
            "visibility_status": visibility_status,
            "raw_usage_available": raw_usage_available,
            "cli_stats_available": cli_stats_available,
        },
        "risks": list(_HARNESS_RISKS[normalized_harness]),
        "recommendations": recommendations,
    }
