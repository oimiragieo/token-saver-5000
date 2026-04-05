"""Tests for the SavingsDashboard cross-session aggregation."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from src.savings_dashboard import (
    AggregatedReport,
    SavingsDashboard,
    build_arg_parser,
    format_by_tool,
    format_cost,
    format_daily,
    format_summary,
    report_to_csv,
    report_to_dict,
)


@pytest.fixture
def tmp_storage(tmp_path: Path) -> Path:
    """Create a temp storage dir with synthetic session journals."""
    storage = tmp_path / "sessions"
    storage.mkdir()
    return storage


def _create_session_db(storage: Path, session_id: str, events: list[dict]) -> Path:
    """Write a synthetic session journal with savings events."""
    db_path = storage / f"{session_id}.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "event_type TEXT NOT NULL, timestamp REAL NOT NULL, data_json TEXT NOT NULL)"
    )
    for ev in events:
        conn.execute(
            "INSERT INTO events (event_type, timestamp, data_json) VALUES (?, ?, ?)",
            (ev.get("type", "savings"), ev.get("ts", time.time()), json.dumps(ev.get("data", {}))),
        )
    conn.commit()
    conn.close()
    return db_path


class TestSavingsDashboard:
    def test_empty_storage_returns_empty_report(self, tmp_storage: Path):
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate()
        assert report.total_events == 0
        assert report.total_tokens_saved == 0

    def test_nonexistent_storage_returns_empty_report(self, tmp_path: Path):
        dashboard = SavingsDashboard(storage_dir=tmp_path / "nonexistent")
        report = dashboard.aggregate()
        assert report.total_events == 0

    def test_single_session_aggregation(self, tmp_storage: Path):
        now = time.time()
        _create_session_db(
            tmp_storage,
            "session-1",
            [
                {
                    "type": "savings",
                    "ts": now - 100,
                    "data": {
                        "tool_name": "ingest_context",
                        "original_tokens": 1000,
                        "compressed_tokens": 100,
                        "tokens_saved": 900,
                        "dollars_saved": 0.0027,
                    },
                },
                {
                    "type": "savings",
                    "ts": now,
                    "data": {
                        "tool_name": "compress_text",
                        "original_tokens": 500,
                        "compressed_tokens": 50,
                        "tokens_saved": 450,
                        "dollars_saved": 0.00135,
                    },
                },
            ],
        )
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate()
        assert report.total_sessions == 1
        assert report.total_events == 2
        assert report.total_tokens_saved == 1350
        assert report.total_original_tokens == 1500
        assert report.total_compressed_tokens == 150

    def test_multi_session_aggregation(self, tmp_storage: Path):
        now = time.time()
        _create_session_db(
            tmp_storage,
            "session-a",
            [
                {
                    "type": "savings",
                    "ts": now,
                    "data": {
                        "original_tokens": 1000,
                        "compressed_tokens": 100,
                        "tokens_saved": 900,
                    },
                },
            ],
        )
        _create_session_db(
            tmp_storage,
            "session-b",
            [
                {
                    "type": "savings",
                    "ts": now,
                    "data": {
                        "original_tokens": 2000,
                        "compressed_tokens": 200,
                        "tokens_saved": 1800,
                    },
                },
            ],
        )
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate()
        assert report.total_sessions == 2
        assert report.total_events == 2
        assert report.total_tokens_saved == 2700

    def test_days_filter(self, tmp_storage: Path):
        now = time.time()
        old = now - 10 * 86400  # 10 days ago
        _create_session_db(
            tmp_storage,
            "session-filter",
            [
                {
                    "type": "savings",
                    "ts": old,
                    "data": {
                        "original_tokens": 1000,
                        "compressed_tokens": 100,
                        "tokens_saved": 900,
                    },
                },
                {
                    "type": "savings",
                    "ts": now,
                    "data": {"original_tokens": 500, "compressed_tokens": 50, "tokens_saved": 450},
                },
            ],
        )
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate(days=7)
        assert report.total_events == 1
        assert report.total_tokens_saved == 450

    def test_by_tool_breakdown(self, tmp_storage: Path):
        now = time.time()
        _create_session_db(
            tmp_storage,
            "session-tools",
            [
                {
                    "type": "savings",
                    "ts": now,
                    "data": {"tool_name": "ingest", "original_tokens": 1000, "tokens_saved": 900},
                },
                {
                    "type": "savings",
                    "ts": now,
                    "data": {"tool_name": "ingest", "original_tokens": 500, "tokens_saved": 400},
                },
                {
                    "type": "savings",
                    "ts": now,
                    "data": {"tool_name": "compress", "original_tokens": 200, "tokens_saved": 100},
                },
            ],
        )
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate()
        assert "ingest" in report.by_tool
        assert report.by_tool["ingest"]["operations"] == 2
        assert report.by_tool["compress"]["operations"] == 1

    def test_non_savings_events_ignored(self, tmp_storage: Path):
        now = time.time()
        _create_session_db(
            tmp_storage,
            "session-mixed",
            [
                {
                    "type": "ingest",
                    "ts": now,
                    "data": {"file_id": "test.py", "original_tokens": 1000},
                },
                {
                    "type": "savings",
                    "ts": now,
                    "data": {"original_tokens": 500, "compressed_tokens": 50, "tokens_saved": 450},
                },
            ],
        )
        dashboard = SavingsDashboard(storage_dir=tmp_storage)
        report = dashboard.aggregate()
        assert report.total_events == 1


class TestFormatters:
    def _make_report(self) -> AggregatedReport:
        now = time.time()
        return AggregatedReport(
            total_sessions=2,
            total_events=10,
            total_original_tokens=10000,
            total_compressed_tokens=1000,
            total_tokens_saved=9000,
            total_dollars_saved=0.027,
            avg_compression_ratio=10.0,
            avg_savings_pct=90.0,
            first_event_time=now - 3600,
            last_event_time=now,
            by_tool={
                "ingest": {"operations": 7, "tokens_saved": 7000, "dollars_saved": 0.021},
                "compress": {"operations": 3, "tokens_saved": 2000, "dollars_saved": 0.006},
            },
            by_day={
                "2026-04-05": {"operations": 10, "tokens_saved": 9000, "dollars_saved": 0.027},
            },
        )

    def test_format_summary(self):
        output = format_summary(self._make_report())
        assert "Token Saver" in output
        assert "9,000" in output
        assert "$0.0270" in output

    def test_format_daily(self):
        output = format_daily(self._make_report())
        assert "2026-04-05" in output

    def test_format_daily_empty(self):
        output = format_daily(AggregatedReport())
        assert "No daily data" in output

    def test_format_by_tool(self):
        output = format_by_tool(self._make_report())
        assert "ingest" in output
        assert "compress" in output

    def test_format_by_tool_empty(self):
        output = format_by_tool(AggregatedReport())
        assert "No per-tool data" in output

    def test_format_cost(self):
        output = format_cost(self._make_report())
        assert "claude-sonnet-4-6" in output
        assert "$" in output

    def test_report_to_dict(self):
        report = self._make_report()
        d = report_to_dict(report)
        assert d["total_tokens_saved"] == 9000
        assert "by_tool" in d
        assert "by_day" in d

    def test_report_to_csv(self):
        report = self._make_report()
        csv_output = report_to_csv(report)
        assert "date,operations,tokens_saved,dollars_saved" in csv_output
        assert "2026-04-05" in csv_output


class TestArgParser:
    def test_defaults(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.model == "claude-sonnet-4-6"
        assert args.daily is False

    def test_flags(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--daily", "--by-tool", "--cost", "--json", "--csv"])
        assert args.daily is True
        assert args.by_tool is True
        assert args.cost is True
        assert getattr(args, "json") is True
        assert args.csv is True

    def test_weekly_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--weekly"])
        assert args.weekly is True
