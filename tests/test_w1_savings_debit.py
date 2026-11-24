"""Unit tests for honest expansion debits in SavingsTracker (CL 2)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.savings_tracker import SavingsEvent, SavingsTracker


def _make_tracker(model: str = "claude-sonnet-4-6") -> SavingsTracker:
    with patch("src.savings_tracker.SavingsTracker._init_journal"):
        tracker = SavingsTracker(session_id="debit-test", model=model)
    tracker._journal = None
    return tracker


def test_savings_event_backward_compatible_defaults():
    ev = SavingsEvent(
        timestamp=1.0,
        tool_name="ingest_context",
        original_tokens=100,
        compressed_tokens=20,
        tokens_saved=80,
        model="claude-sonnet-4-6",
        cost_without_compression=0.0003,
        cost_with_compression=0.00006,
        dollars_saved=0.00024,
        compression_ratio=5.0,
    )
    assert ev.event_type == "compress"
    assert ev.cached_tokens == 0
    assert ev.uncached_tokens == 0
    assert ev.net_billed_dollars_saved == 0.0


def test_record_expansion_debit_reduces_cumulative_savings():
    tracker = _make_tracker(model="claude-sonnet-4-6")
    tracker.record(
        "ingest_context", original_tokens=10_000, compressed_tokens=2_000, file_id="doc1"
    )
    debit = tracker.record_expansion_debit(
        "modulate_region", expanded_tokens=3_000, reason="raw_read"
    )

    assert debit.event_type == "expansion_debit"
    assert debit.tokens_saved == -3_000
    assert debit.dollars_saved < 0.0
    assert debit.net_billed_dollars_saved == debit.dollars_saved

    report = tracker.get_report()
    assert report.total_operations == 2
    assert report.total_tokens_saved == 5_000
    expected_dollars = round((8_000 - 3_000) * 3.0 / 1_000_000, 4)
    assert report.total_dollars_saved == pytest.approx(expected_dollars, abs=1e-4)
    assert report.by_tool["modulate_region"]["tokens_saved"] == -3_000


def test_record_expansion_debit_clamps_negative_input():
    tracker = _make_tracker()
    debit = tracker.record_expansion_debit("get_original_output", expanded_tokens=-50)
    assert debit.tokens_saved == 0
    assert debit.dollars_saved == 0.0


def test_expansion_debit_persists_across_journal_recovery(tmp_path):
    from src.session_journal import SessionJournal

    tracker = SavingsTracker(
        session_id="persist-debit",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record(
        "ingest_context", original_tokens=10_000, compressed_tokens=2_000, file_id="doc1"
    )
    tracker.record_expansion_debit("modulate_region", expanded_tokens=3_000, reason="raw_read")
    original_report = tracker.get_report()
    tracker._journal.close()

    journal = SessionJournal("persist-debit", storage_dir=tmp_path)
    # Also simulate an explicit ingest event to verify no double-counting.
    journal.write_event(
        "ingest",
        {"file_id": "doc1", "original_tokens": 10_000, "compressed_tokens": 2_000},
    )
    summary = journal.recover()
    assert len(summary.expansion_debits) == 1
    assert summary.expansion_debits[0]["expanded_tokens"] == 3_000
    assert summary.total_tokens_saved == 5_000
    journal.close()

    restored = SavingsTracker(
        session_id="persist-debit",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    restored_report = restored.get_report()
    restored._journal.close()

    assert restored_report.total_operations == original_report.total_operations == 2
    assert restored_report.total_tokens_saved == 5_000
    assert restored_report.total_dollars_saved == pytest.approx(
        original_report.total_dollars_saved, abs=1e-4
    )


def test_savings_only_journal_recovers_net_tokens_without_ingest(tmp_path):
    from src.session_journal import SessionJournal

    tracker = SavingsTracker(
        session_id="savings-only",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record(
        "ingest_context", original_tokens=10_000, compressed_tokens=2_000, file_id="doc1"
    )
    tracker.record_expansion_debit("modulate_region", expanded_tokens=3_000, reason="raw_read")
    tracker._journal.close()

    journal = SessionJournal("savings-only", storage_dir=tmp_path)
    summary = journal.recover()
    journal.close()

    assert len(summary.ingested_files) == 0
    assert len(summary.savings_events) == 1
    assert len(summary.expansion_debits) == 1
    assert summary.total_tokens_saved == 5_000

    restored = SavingsTracker(
        session_id="savings-only",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    restored_report = restored.get_report()
    restored._journal.close()
    assert restored_report.total_operations == 2
    assert restored_report.total_tokens_saved == 5_000


def test_mixed_legacy_ingest_savings_and_debit_recovery(tmp_path):
    from src.session_journal import SessionJournal

    journal = SessionJournal("mixed-journal", storage_dir=tmp_path)
    journal.write_event(
        "ingest",
        {"file_id": "legacy_doc", "original_tokens": 6_000, "compressed_tokens": 1_000},
    )
    journal.close()

    tracker = SavingsTracker(
        session_id="mixed-journal",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record("filter_cli_output", original_tokens=4_000, compressed_tokens=1_000)
    tracker.record_expansion_debit("modulate_region", expanded_tokens=2_000, reason="raw_read")
    tracker._journal.close()

    check_journal = SessionJournal("mixed-journal", storage_dir=tmp_path)
    summary = check_journal.recover()
    check_journal.close()

    assert summary.total_tokens_saved == 6_000

    restored = SavingsTracker(
        session_id="mixed-journal",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    report = restored.get_report()
    restored._journal.close()

    assert report.total_operations == 3
    assert report.total_tokens_saved == 6_000


def test_savings_dashboard_includes_expansion_debits(tmp_path):
    from src.savings_dashboard import SavingsDashboard

    tracker = SavingsTracker(
        session_id="dash-debit",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record(
        "ingest_context", original_tokens=10_000, compressed_tokens=2_000, file_id="doc1"
    )
    tracker.record_expansion_debit("modulate_region", expanded_tokens=3_000, reason="raw_read")
    tracker._journal.close()

    report = SavingsDashboard(storage_dir=tmp_path, model="claude-sonnet-4-6").aggregate()
    assert report.total_sessions == 1
    assert report.total_events == 2
    assert report.total_tokens_saved == 5_000
    expected_dollars = round(5_000 * 3.0 / 1_000_000, 4)
    assert report.total_dollars_saved == pytest.approx(expected_dollars, abs=1e-4)


def test_distinct_file_ids_with_identical_tokens_do_not_collide(tmp_path):
    from src.session_journal import SessionJournal

    journal = SessionJournal("no-collide", storage_dir=tmp_path)
    journal.write_event(
        "ingest",
        {"file_id": "file_a", "original_tokens": 1_000, "compressed_tokens": 200},
    )
    journal.close()

    tracker = SavingsTracker(
        session_id="no-collide",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record(
        "ingest_context",
        original_tokens=1_000,
        compressed_tokens=200,
        file_id="file_b",
    )
    tracker._journal.close()

    check_journal = SessionJournal("no-collide", storage_dir=tmp_path)
    summary = check_journal.recover()
    check_journal.close()

    assert summary.total_tokens_saved == 1_600

    restored = SavingsTracker(
        session_id="no-collide",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    report = restored.get_report()
    restored._journal.close()

    assert report.total_operations == 2
    assert report.total_tokens_saved == 1_600


def test_read_skeleton_and_doc_id_alias_do_not_suppress_legacy_ingest(tmp_path):
    from src.session_journal import SessionJournal

    journal = SessionJournal("doc-alias-skeleton", storage_dir=tmp_path)
    journal.write_event(
        "ingest",
        {"doc_id": "shared_doc", "original_tokens": 1_000, "compressed_tokens": 200},
    )
    journal.close()

    tracker = SavingsTracker(
        session_id="doc-alias-skeleton",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    tracker.record(
        "read_skeleton",
        original_tokens=600,
        compressed_tokens=200,
        file_id="shared_doc",
    )
    tracker._journal.close()

    check_journal = SessionJournal("doc-alias-skeleton", storage_dir=tmp_path)
    summary = check_journal.recover()
    check_journal.close()
    assert summary.total_tokens_saved == 1_200

    restored = SavingsTracker(
        session_id="doc-alias-skeleton",
        model="claude-sonnet-4-6",
        storage_dir=tmp_path,
    )
    report = restored.get_report()
    restored._journal.close()
    assert report.total_operations == 2
    assert report.total_tokens_saved == 1_200
