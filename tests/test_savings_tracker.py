"""
Tests for the SavingsTracker module.

Covers SavingsEvent creation, SavingsReport aggregation, ROI/breakeven
calculations, per-model pricing, journal persistence, and JSON serialisability.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.savings_tracker import SavingsEvent, SavingsTracker
from src.cli_benchmark.pricing import PRICING

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tracker(
    session_id: str = "test-session", model: str = "claude-sonnet-4-6"
) -> SavingsTracker:
    """Create a SavingsTracker with a mocked SessionJournal so tests stay isolated."""
    with patch("src.savings_tracker.SavingsTracker._init_journal"):
        tracker = SavingsTracker(session_id=session_id, model=model)
    tracker._journal = None  # ensure no real SQLite I/O
    return tracker


# ---------------------------------------------------------------------------
# SavingsTracker.record()
# ---------------------------------------------------------------------------


class TestRecord:
    def test_record_event_creates_savings_event(self):
        """record() should return a SavingsEvent instance."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", original_tokens=1000, compressed_tokens=100)
        assert isinstance(event, SavingsEvent)

    def test_record_event_calculates_dollars_saved(self):
        """dollars_saved must equal (original - compressed) * input_rate / 1e6."""
        tracker = _make_tracker(model="claude-sonnet-4-6")
        event = tracker.record("ingest_context", original_tokens=1_000_000, compressed_tokens=0)
        rate = PRICING["claude-sonnet-4-6"]["input"]  # $3.0 / MTok
        expected = round(1_000_000 * rate / 1_000_000, 6)
        assert event.dollars_saved == pytest.approx(expected, abs=1e-6)

    def test_record_event_compression_ratio(self):
        """compression_ratio = original / compressed (rounded to 1 dp)."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", original_tokens=500, compressed_tokens=50)
        assert event.compression_ratio == 10.0

    def test_record_zero_savings_when_compressed_ge_original(self):
        """If compressed >= original, tokens_saved and dollars_saved must be 0."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", original_tokens=100, compressed_tokens=100)
        assert event.tokens_saved == 0
        assert event.dollars_saved == 0.0

    def test_record_zero_savings_when_compressed_exceeds_original(self):
        """Expansion (compressed > original) must still yield 0 savings, not negative."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", original_tokens=50, compressed_tokens=200)
        assert event.tokens_saved == 0
        assert event.dollars_saved == 0.0

    def test_record_event_stored_in_events_list(self):
        """After calling record(), the event appears in _events."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)
        assert len(tracker._events) == 1


# ---------------------------------------------------------------------------
# SavingsTracker.get_report() — empty state
# ---------------------------------------------------------------------------


class TestGetReportEmpty:
    def test_get_report_empty_returns_zeroed_report(self):
        """With no events, all numeric fields must be zero/None."""
        tracker = _make_tracker()
        report = tracker.get_report()

        assert report.total_operations == 0
        assert report.total_tokens_saved == 0
        assert report.total_dollars_saved == 0.0
        assert report.avg_compression_ratio == 0.0
        assert report.avg_savings_pct == 0.0
        assert report.monthly_projected_savings == 0.0
        assert report.roi_vs_pro_plan == 0.0
        assert report.breakeven_operations == 0
        assert report.first_event_time is None
        assert report.last_event_time is None
        assert report.by_tool == {}


# ---------------------------------------------------------------------------
# SavingsTracker.get_report() — populated state
# ---------------------------------------------------------------------------


class TestGetReportSingleEvent:
    def test_get_report_single_event_correct_totals(self):
        """Single event totals must exactly match event values."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", original_tokens=1000, compressed_tokens=100)
        report = tracker.get_report()

        assert report.total_operations == 1
        assert report.total_tokens_saved == event.tokens_saved
        assert report.total_dollars_saved == pytest.approx(event.dollars_saved, abs=1e-4)

    def test_get_report_single_event_first_last_time(self):
        """first_event_time and last_event_time must be set for a single event."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)
        report = tracker.get_report()

        assert report.first_event_time is not None
        assert report.last_event_time is not None
        assert report.first_event_time == report.last_event_time


class TestGetReportMultipleEvents:
    def test_get_report_multiple_events_sums_correctly(self):
        """total_tokens_saved must equal the sum across all events."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)  # 900 saved
        tracker.record("batch_ingest_documents", 2000, 400)  # 1600 saved
        report = tracker.get_report()

        assert report.total_operations == 2
        assert report.total_tokens_saved == 2500

    def test_get_report_avg_savings_pct(self):
        """avg_savings_pct = total_tokens_saved / total_original_tokens * 100."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)  # 90% savings
        tracker.record("ingest_context", 1000, 500)  # 50% savings
        report = tracker.get_report()

        # total_saved=1400, total_original=2000 => 70%
        assert report.avg_savings_pct == pytest.approx(70.0, abs=0.1)

    def test_get_report_avg_compression_ratio(self):
        """avg_compression_ratio = total_original / total_compressed."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 200)  # ratio 5x
        tracker.record("ingest_context", 1000, 200)  # ratio 5x
        report = tracker.get_report()

        # 2000 / 400 = 5x
        assert report.avg_compression_ratio == pytest.approx(5.0, abs=0.1)

    def test_get_report_by_tool_breakdown(self):
        """by_tool must group events correctly by tool_name."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)
        tracker.record("ingest_context", 500, 50)
        tracker.record("batch_ingest_documents", 2000, 200)
        report = tracker.get_report()

        assert "ingest_context" in report.by_tool
        assert "batch_ingest_documents" in report.by_tool
        assert report.by_tool["ingest_context"]["operations"] == 2
        assert report.by_tool["batch_ingest_documents"]["operations"] == 1
        assert report.by_tool["ingest_context"]["tokens_saved"] == 1350


# ---------------------------------------------------------------------------
# ROI and Breakeven
# ---------------------------------------------------------------------------


class TestROICalculations:
    def test_get_report_roi_calculation(self):
        """roi_vs_pro_plan = monthly_projected_savings / 29.0."""
        tracker = _make_tracker()
        # inject events with a wide enough time span to trigger projection
        t0 = time.time() - 7200  # 2 hours ago
        for i in range(10):
            tracker.record("ingest_context", 10_000, 1_000)
            # backdate the event by spacing them over the span
            tracker._events[-1].timestamp = t0 + i * 720  # 12 min apart

        report = tracker.get_report()
        # ROI must be positive and equal to projected / 29
        assert report.roi_vs_pro_plan >= 0.0
        if report.monthly_projected_savings > 0:
            expected_roi = round(
                report.monthly_projected_savings / SavingsTracker.PRO_PLAN_PRICE, 1
            )
            assert report.roi_vs_pro_plan == pytest.approx(expected_roi, abs=0.1)

    def test_get_report_breakeven_operations(self):
        """breakeven_operations = PRO_PLAN_PRICE / savings_per_op (integer)."""
        tracker = _make_tracker()
        # With claude-sonnet-4-6 at $3/MTok: saving 1M tokens = $3.00
        tracker.record("ingest_context", 1_000_000, 0)
        report = tracker.get_report()

        assert report.breakeven_operations > 0
        # breakeven ≈ 29 / 3.0 = 9 (integer floor)
        assert report.breakeven_operations == int(
            SavingsTracker.PRO_PLAN_PRICE / report.total_dollars_saved
        )

    def test_pro_plan_price_constant(self):
        """PRO_PLAN_PRICE must remain $29.0."""
        assert SavingsTracker.PRO_PLAN_PRICE == 29.0


# ---------------------------------------------------------------------------
# Inline summary
# ---------------------------------------------------------------------------


class TestGetInlineSummary:
    def test_get_inline_summary_format_contains_key_info(self):
        """Inline summary must contain token count and dollar amount."""
        tracker = _make_tracker()
        event = tracker.record("ingest_context", 3400, 0)
        summary = tracker.get_inline_summary(event)

        assert "3,400" in summary
        assert "$" in summary

    def test_get_inline_summary_with_session_total(self):
        """After multiple ops, the summary must include session total."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)
        event = tracker.record("ingest_context", 2000, 200)
        summary = tracker.get_inline_summary(event)

        assert "Session total" in summary


# ---------------------------------------------------------------------------
# Model pricing
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_model_pricing_claude_opus(self):
        """claude-opus-4-6 input rate must be $15/MTok."""
        tracker = _make_tracker(model="claude-opus-4-6")
        event = tracker.record("ingest_context", 1_000_000, 0)
        # $15.00 for 1M tokens
        assert event.cost_without_compression == pytest.approx(15.0, abs=0.001)

    def test_model_pricing_gemini_flash(self):
        """gemini-2.5-flash input rate must be $0.15/MTok."""
        tracker = _make_tracker(model="gemini-2.5-flash")
        event = tracker.record("ingest_context", 1_000_000, 0)
        assert event.cost_without_compression == pytest.approx(0.15, abs=0.001)

    def test_model_pricing_codex(self):
        """codex-mini input rate must be $1.50/MTok."""
        tracker = _make_tracker(model="codex-mini")
        event = tracker.record("ingest_context", 1_000_000, 0)
        assert event.cost_without_compression == pytest.approx(1.50, abs=0.001)

    def test_unknown_model_falls_back_to_default_rate(self):
        """Unknown models must use the 'default' pricing ($3/MTok input)."""
        tracker = _make_tracker(model="unknown-model-xyz")
        event = tracker.record("ingest_context", 1_000_000, 0)
        default_rate = PRICING["default"]["input"]
        assert event.cost_without_compression == pytest.approx(default_rate, abs=0.001)


# ---------------------------------------------------------------------------
# Journal persistence
# ---------------------------------------------------------------------------


class TestJournalPersistence:
    def test_tracker_persists_to_journal_when_available(self):
        """record() must call journal.write_event() if the journal is set."""
        tracker = _make_tracker()
        mock_journal = MagicMock()
        tracker._journal = mock_journal

        tracker.record("ingest_context", 1000, 100)

        mock_journal.write_event.assert_called_once()
        call_args = mock_journal.write_event.call_args
        assert call_args[0][0] == "savings"

    def test_tracker_survives_journal_failure(self):
        """If write_event() raises, record() must still return the event."""
        tracker = _make_tracker()
        mock_journal = MagicMock()
        mock_journal.write_event.side_effect = OSError("disk full")
        tracker._journal = mock_journal

        # Must not raise
        event = tracker.record("ingest_context", 1000, 100)
        assert event.tokens_saved == 900

    def test_tracker_works_without_journal(self):
        """If _journal is None, record() and get_report() must still work."""
        tracker = _make_tracker()
        assert tracker._journal is None  # confirmed no journal

        tracker.record("ingest_context", 1000, 100)
        report = tracker.get_report()
        assert report.total_operations == 1


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------


class TestJSONSerialisable:
    def test_report_json_serializable(self):
        """All SavingsReport fields must be JSON-serialisable."""
        tracker = _make_tracker()
        tracker.record("ingest_context", 1000, 100)
        report = tracker.get_report()

        payload = {
            "session_id": report.session_id,
            "model": report.model,
            "total_operations": report.total_operations,
            "total_tokens_saved": report.total_tokens_saved,
            "total_dollars_saved": report.total_dollars_saved,
            "avg_compression_ratio": report.avg_compression_ratio,
            "avg_savings_pct": report.avg_savings_pct,
            "monthly_projected_savings": report.monthly_projected_savings,
            "roi_vs_pro_plan": report.roi_vs_pro_plan,
            "breakeven_operations": report.breakeven_operations,
            "by_tool": report.by_tool,
            "first_event_time": report.first_event_time,
            "last_event_time": report.last_event_time,
        }

        # Must not raise
        serialised = json.dumps(payload)
        assert isinstance(serialised, str)
