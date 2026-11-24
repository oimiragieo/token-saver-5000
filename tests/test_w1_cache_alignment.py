"""Unit tests for cache_align_sections and advise_with_cache_economics (CL 3a)."""

from __future__ import annotations

from src.cache_economics import ANTHROPIC_SONNET_4_6, ExplainAnalyzeTrace
from src.compression_advisor import CompressionAdvisor
from src.prompt_cache_stability_guard import cache_align_sections


def test_cache_align_sections_sorts_by_volatility_and_pins_anchors():
    sections = [
        {"name": "identity", "content": "You are a careful assistant."},
        {"name": "user_query", "content": "Explain the diff.", "volatility": "none"},
        {"name": "rag_context", "content": "Repo docs...", "volatility": "session"},
        {"name": "tool_definitions", "content": "read_file, grep", "volatility": "global"},
        {"name": "constraints", "content": "Never fabricate paths."},
    ]
    result = cache_align_sections(sections, max_reorder_moves=10)
    names = [s["name"] for s in result["aligned_sections"]]

    # Anchors stay at indices 0 and 4
    assert names[0] == "identity"
    assert names[4] == "constraints"
    # Non-pinned sorted: global (tool_definitions) < session (rag_context) < none (user_query)
    assert names[1] == "tool_definitions"
    assert names[2] == "rag_context"
    assert names[3] == "user_query"
    assert result["moves_used"] == 3
    assert len(result["cache_breakpoints"]) >= 1


def test_cache_align_sections_respects_max_reorder_moves():
    sections = [
        {"name": "user_query", "content": "q", "volatility": "none"},
        {"name": "rag_context", "content": "r", "volatility": "session"},
        {"name": "tool_definitions", "content": "t", "volatility": "global"},
    ]
    result = cache_align_sections(sections, max_reorder_moves=1)
    assert result["moves_used"] == 1


def test_advise_with_cache_economics_abstains_when_tier_would_break():
    advisor = CompressionAdvisor()
    # ~4000 tokens -> r_max_safe = 4000 // 3500 = 1, so compressing drops below T=3500
    doc = (
        "Quantum entanglement enables teleportation of quantum states across nodes. " * 360
    ).strip()
    trace = advisor.advise_with_cache_economics(
        doc, profile=ANTHROPIC_SONNET_4_6, expected_hit_rate=0.2, skeleton_ratio=0.2
    )
    assert isinstance(trace, ExplainAnalyzeTrace)
    assert trace.tier_preserving_max_ratio == 1
    assert trace.decision == "abstain"
    assert trace.recommended_ratio == 1.0


def test_advise_with_cache_economics_caps_ratio_to_preserve_tier():
    advisor = CompressionAdvisor()
    # ~10000 tokens -> r_max_safe = 2, while advisor estimates >2x -> caps at 2.0
    doc = (
        "Semantic graph compression preserves key structural anchors in large docs. " * 950
    ).strip()
    trace = advisor.advise_with_cache_economics(
        doc, profile=ANTHROPIC_SONNET_4_6, expected_hit_rate=0.2, skeleton_ratio=0.1
    )
    assert trace.tier_preserving_max_ratio >= 2
    assert trace.decision in ("cap", "compress")
    assert trace.recommended_ratio <= float(trace.tier_preserving_max_ratio)


def test_advise_with_cache_economics_abstains_above_crossover_hit_rate():
    advisor = CompressionAdvisor()
    doc = (
        "Semantic graph compression preserves key structural anchors in large docs. " * 950
    ).strip()
    trace = advisor.advise_with_cache_economics(
        doc, profile=ANTHROPIC_SONNET_4_6, expected_hit_rate=0.99, skeleton_ratio=0.1
    )
    assert trace.decision == "abstain"
    assert trace.recommended_ratio == 1.0


def test_advise_with_cache_economics_uses_capc_net_savings_formula():
    import pytest

    advisor = CompressionAdvisor()
    doc = (
        "Semantic graph compression preserves key structural anchors in large docs. " * 950
    ).strip()
    h = 0.2
    trace = advisor.advise_with_cache_economics(
        doc, profile=ANTHROPIC_SONNET_4_6, expected_hit_rate=h, skeleton_ratio=0.1
    )
    assert trace.decision in ("cap", "compress")
    r = trace.recommended_ratio
    alpha = ANTHROPIC_SONNET_4_6.cache_write_mult
    beta = ANTHROPIC_SONNET_4_6.cache_read_mult
    expected = round(100.0 * (1.0 - (1.0 / r) / (alpha * (1.0 - h) + beta * h)), 2)
    assert trace.estimated_net_savings_pct == pytest.approx(expected, abs=1e-2)


def test_cache_align_sections_detects_volatile_prefix_leak_and_suppresses_breakpoints():
    sections = [
        {"name": "identity", "content": "You are helpful.", "volatility": "global"},
        {"name": "user_query", "content": "Pinned volatile!", "volatility": "none", "pinned": True},
        {"name": "tool_definitions", "content": "read_file", "volatility": "global"},
    ]
    result = cache_align_sections(sections, max_reorder_moves=5)
    assert result["has_volatile_prefix_leak"] is True
    # Breakpoint at index 0 (before the leak) is valid; index 2 (after leak) must NOT be emitted.
    bp_indices = [bp["index"] for bp in result["cache_breakpoints"]]
    assert 2 not in bp_indices
