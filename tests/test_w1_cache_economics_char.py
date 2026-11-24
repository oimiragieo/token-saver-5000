"""Baseline characterization tests for Wave 1 modules.

Captures pre-Wave-1 behavior of SavingsTracker, CompressionAdvisor, and
evaluate_prompt_stability before extending them.
"""

from __future__ import annotations

from unittest.mock import patch

from src.compression_advisor import CompressionAdvisor, CompressionEstimate
from src.prompt_cache_stability_guard import evaluate_prompt_stability
from src.savings_tracker import SavingsEvent, SavingsTracker


def _make_isolated_tracker(
    session_id: str = "char-session", model: str = "claude-sonnet-4-6"
) -> SavingsTracker:
    with patch("src.savings_tracker.SavingsTracker._init_journal"):
        tracker = SavingsTracker(session_id=session_id, model=model)
    tracker._journal = None
    return tracker


def test_savings_tracker_baseline_record_and_report():
    tracker = _make_isolated_tracker()
    event = tracker.record("ingest_context", original_tokens=2000, compressed_tokens=500)
    assert isinstance(event, SavingsEvent)
    assert event.tokens_saved == 1500
    assert event.compression_ratio == 4.0
    assert event.dollars_saved > 0.0

    report = tracker.get_report()
    assert report.total_operations == 1
    assert report.total_tokens_saved == 1500
    assert report.total_dollars_saved == round(event.dollars_saved, 4)


def test_compression_advisor_baseline_estimate():
    advisor = CompressionAdvisor()
    small_est = advisor.estimate_compression("Hello world. Short text.")
    assert isinstance(small_est, CompressionEstimate)
    assert small_est.compression_ratio == 1.0
    assert small_est.confidence == "high"

    medium_text = ("Quantum computing uses superposition and entanglement. " * 40).strip()
    med_est = advisor.estimate_compression(medium_text, skeleton_ratio=0.2)
    assert med_est.original_tokens >= 200
    assert med_est.compression_ratio >= 1.2
    assert med_est.estimated_compressed < med_est.original_tokens


def test_evaluate_prompt_stability_baseline():
    sections = [
        {"name": "tool_definitions", "content": "read_file, write_file"},
        {"name": "system_instructions", "content": "Be concise and accurate."},
        {"name": "user_query", "content": "Summarize the diff."},
    ]
    result = evaluate_prompt_stability(sections)
    assert result["is_stable"] is True
    assert result["order_valid"] is True
    assert result["violations"] == []
    assert result["warning"] is None
    assert len(result["stable_prefix_hash"]) == 64

    bad_sections = [
        {"name": "user_query", "content": "Summarize the diff."},
        {"name": "system_instructions", "content": "Be concise."},
    ]
    bad_result = evaluate_prompt_stability(bad_sections)
    assert bad_result["is_stable"] is False
    assert bad_result["order_valid"] is False
