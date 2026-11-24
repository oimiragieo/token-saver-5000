"""Unit tests for src/cache_economics.py (CL 1)."""

from __future__ import annotations

import pytest

from src.cache_economics import (
    ANTHROPIC_SONNET_4_6,
    DEEPSEEK_V3,
    GEMINI_FLASH,
    OPENAI_GPT5,
    PROVIDER_CACHE_PROFILES,
    ExplainAnalyzeTrace,
    ProviderCacheProfile,
    bill_prefix_turn,
    compute_crossover_hit_rate,
    compute_tier_preserving_max_ratio,
    get_provider_cache_profile,
)


def test_presets_have_expected_parameters():
    assert ANTHROPIC_SONNET_4_6.cache_write_mult == 1.25
    assert ANTHROPIC_SONNET_4_6.cache_read_mult == 0.10
    assert ANTHROPIC_SONNET_4_6.persistent_tier_tokens == 3500
    assert set(PROVIDER_CACHE_PROFILES.keys()) == {
        "anthropic_sonnet_4_6",
        "openai_gpt5",
        "deepseek_v3",
        "gemini_flash",
    }
    assert get_provider_cache_profile("openai_gpt5") is OPENAI_GPT5
    assert get_provider_cache_profile("gpt-5.1-codex") is OPENAI_GPT5
    assert get_provider_cache_profile("o3-mini") is OPENAI_GPT5
    assert get_provider_cache_profile("deepseek_v3") is DEEPSEEK_V3
    assert get_provider_cache_profile("deepseek-chat") is DEEPSEEK_V3
    assert get_provider_cache_profile("gemini_flash") is GEMINI_FLASH
    assert get_provider_cache_profile("gemini-2.5-flash") is GEMINI_FLASH
    assert get_provider_cache_profile("claude-opus-4-6") is ANTHROPIC_SONNET_4_6
    assert get_provider_cache_profile("unknown") is ANTHROPIC_SONNET_4_6


def test_compute_crossover_hit_rate_capc_formula():
    # r <= 1.0 or alpha <= beta returns 1.0
    assert compute_crossover_hit_rate(1.0, 1.25, 0.10) == 1.0
    assert compute_crossover_hit_rate(0.5, 1.25, 0.10) == 1.0
    assert compute_crossover_hit_rate(3.0, 0.10, 0.10) == 1.0
    assert compute_crossover_hit_rate(3.0, 0.10, 0.25) == 1.0

    # r=2.0, alpha=1.25, beta=0.10 -> (1.25 - 0.5) / 1.15 = 0.75 / 1.15
    expected = (1.25 - 0.5) / (1.25 - 0.10)
    assert compute_crossover_hit_rate(2.0, 1.25, 0.10) == pytest.approx(expected)

    # Clamped to [0.0, 1.0]
    assert compute_crossover_hit_rate(1.1, 0.5, 0.1) == 0.0


def test_compute_tier_preserving_max_ratio():
    assert compute_tier_preserving_max_ratio(1000, 3500) == 1
    assert compute_tier_preserving_max_ratio(3500, 3500) == 1
    assert compute_tier_preserving_max_ratio(7000, 3500) == 2
    assert compute_tier_preserving_max_ratio(14500, 3500) == 4


def test_bill_prefix_turn_lcp_reads_and_writes():
    profile = ProviderCacheProfile(
        name="test",
        input_rate_per_1m=10.0,
        cache_write_mult=1.25,
        cache_read_mult=0.10,
        persistent_tier_tokens=4,
    )
    prev = ["alpha beta gamma", "delta epsilon", "old tail"]
    curr = ["alpha beta gamma", "delta epsilon", "new tail"]
    billed = bill_prefix_turn(prev, curr, profile)

    assert billed["cached_prefix_segments"] == 2
    assert billed["cached_tokens"] >= 4
    assert billed["fresh_tokens"] > 0
    assert billed["total_tokens"] == billed["cached_tokens"] + billed["fresh_tokens"]
    assert billed["net_billed_dollars"] == pytest.approx(
        billed["cache_read_dollars"] + billed["cache_write_dollars"], abs=1e-7
    )
    assert billed["dollars_saved_vs_uncached"] > 0.0


def test_bill_prefix_turn_enforces_persistent_tier_and_token_lcp():
    # Below persistent_tier_tokens -> cached_tokens must be 0
    small_prev = ["alpha beta gamma", "delta epsilon"]
    small_curr = ["alpha beta gamma", "delta omega"]
    below = bill_prefix_turn(small_prev, small_curr, OPENAI_GPT5)
    assert below["lcp_tokens"] > 0
    assert below["cached_tokens"] == 0
    assert below["fresh_tokens"] == below["total_tokens"]
    assert below["cache_read_dollars"] == 0.0

    # Above persistent_tier_tokens with partial token overlap in last segment
    shared_prefix = " ".join(f"tok{i}" for i in range(600))
    above_prev = [shared_prefix + " old_suffix_one old_suffix_two"]
    above_curr = [shared_prefix + " new_suffix_one new_suffix_two"]
    above = bill_prefix_turn(above_prev, above_curr, OPENAI_GPT5)
    assert above["lcp_tokens"] >= OPENAI_GPT5.persistent_tier_tokens
    assert above["cached_tokens"] == above["lcp_tokens"]
    assert above["fresh_tokens"] > 0
    assert above["cache_read_dollars"] > 0.0


def test_bill_prefix_turn_full_miss():
    prev = ["completely different"]
    curr = ["brand new prompt"]
    billed = bill_prefix_turn(prev, curr, ANTHROPIC_SONNET_4_6)
    assert billed["cached_prefix_segments"] == 0
    assert billed["cached_tokens"] == 0
    assert billed["cache_read_dollars"] == 0.0
    assert billed["cache_write_dollars"] > 0.0


def test_explain_analyze_trace_dataclass():
    trace = ExplainAnalyzeTrace(
        decision="compress",
        reason="hit rate below crossover",
        recommended_ratio=2.5,
        crossover_hit_rate=0.739,
        tier_preserving_max_ratio=4,
        estimated_net_savings_pct=60.0,
    )
    assert trace.decision == "compress"
    assert trace.tier_preserving_max_ratio == 4
