"""Canonical cache telemetry normalization for observability surfaces."""

from __future__ import annotations

from typing import Any


def normalize_cache_telemetry(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Flatten provider cache telemetry into a stable observability payload."""
    if not isinstance(telemetry, dict):
        raise ValueError("normalize_cache_telemetry requires dict 'telemetry'")

    cache_read_field = telemetry.get("cache_read_field")
    cached_input_tokens = 0
    if isinstance(cache_read_field, str):
        value = telemetry.get(cache_read_field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            cached_input_tokens = int(value)

    validation = (
        telemetry.get("validation") if isinstance(telemetry.get("validation"), dict) else {}
    )
    prefix_integrity = (
        validation.get("prefix_integrity")
        if isinstance(validation.get("prefix_integrity"), dict)
        else {}
    )
    cache_creation_churn = (
        validation.get("cache_creation_churn")
        if isinstance(validation.get("cache_creation_churn"), dict)
        else {}
    )
    section_interleaving = (
        validation.get("section_interleaving")
        if isinstance(validation.get("section_interleaving"), dict)
        else {}
    )
    diagnostic = (
        validation.get("diagnostic") if isinstance(validation.get("diagnostic"), dict) else {}
    )
    partial_reuse = (
        diagnostic.get("partial_reuse") if isinstance(diagnostic.get("partial_reuse"), dict) else {}
    )
    stale_expectation = (
        validation.get("stale_expectation")
        if isinstance(validation.get("stale_expectation"), dict)
        else {}
    )
    session_metrics = (
        telemetry.get("session_metrics")
        if isinstance(telemetry.get("session_metrics"), dict)
        else {}
    )
    cache_health = (
        telemetry.get("cache_health") if isinstance(telemetry.get("cache_health"), list) else []
    )
    degraded_label_count = 0
    cache_health_skew_detected = False
    for item in cache_health:
        if not isinstance(item, dict):
            continue
        if item.get("degraded") is True:
            degraded_label_count += 1
        coherence = item.get("coherence")
        if isinstance(coherence, dict) and coherence.get("skew_detected") is True:
            cache_health_skew_detected = True

    return {
        "model": telemetry.get("model") or "unknown",
        "provider": telemetry.get("provider") or "unknown",
        "cache_read_field": cache_read_field or "unknown",
        "cached_input_tokens": cached_input_tokens,
        "total_input_tokens": int(telemetry.get("total_input_tokens") or 0),
        "total_output_tokens": int(telemetry.get("total_output_tokens") or 0),
        "cache_creation_input_tokens": int(telemetry.get("cache_creation_input_tokens") or 0),
        "cache_hit_detected": bool(telemetry.get("cache_hit_detected", False)),
        "cache_hit_ratio": float(telemetry.get("cache_hit_ratio") or 0.0),
        "estimated_cache_savings_usd": float(telemetry.get("estimated_cache_savings_usd") or 0.0),
        "validation_status": str(validation.get("status") or "unvalidated"),
        "prefix_changed": bool(prefix_integrity.get("prefix_changed", False)),
        "cache_creation_churn_detected": bool(cache_creation_churn.get("churn_detected", False)),
        "section_layout_changed": bool(section_interleaving.get("layout_changed", False)),
        "partial_reuse_detected": bool(partial_reuse.get("partial_reuse_detected", False)),
        "stale_expectation_detected": bool(stale_expectation),
        "session_cache_hit_ratio": float(session_metrics.get("cache_hit_ratio") or 0.0),
        "degraded_label_count": degraded_label_count,
        "cache_health_skew_detected": cache_health_skew_detected,
        "warning_present": isinstance(telemetry.get("warning"), str),
    }
