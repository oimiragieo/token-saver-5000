"""
Tests for SessionJournal (src/session_journal.py).

All tests use a temporary directory so they never touch the real data store.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from src.session_journal import JournalEvent, SessionJournal, SessionSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_journal(tmp_path: Path, session_id: str = "test-session") -> SessionJournal:
    return SessionJournal(session_id=session_id, storage_dir=tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWriteEvent:
    def test_write_event_stores_in_db(self, tmp_path: Path) -> None:
        """Event data should be persisted and retrievable via raw SQL."""
        j = _make_journal(tmp_path)
        j.write_event(
            "ingest", {"file_id": "doc1", "original_tokens": 100, "compressed_tokens": 10}
        )
        j.close()

        db_path = tmp_path / "test-session.db"
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT event_type, data_json FROM events").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "ingest"
        data = json.loads(rows[0][1])
        assert data["file_id"] == "doc1"

    def test_write_event_returns_event_with_id(self, tmp_path: Path) -> None:
        """write_event() should return a JournalEvent with a positive event_id."""
        j = _make_journal(tmp_path)
        event = j.write_event("configure", {"model_id": "gpt-4"})
        j.close()

        assert isinstance(event, JournalEvent)
        assert event.event_id > 0
        assert event.event_type == "configure"
        assert event.data == {"model_id": "gpt-4"}
        assert event.timestamp > 0

    def test_event_count_matches_number_of_writes(self, tmp_path: Path) -> None:
        """event_count() should reflect exactly how many events were written."""
        j = _make_journal(tmp_path)
        for i in range(5):
            j.write_event("tool_call", {"tool_name": f"tool_{i}"})
        assert j.event_count() == 5
        j.close()


class TestRecover:
    def test_recover_empty_session(self, tmp_path: Path) -> None:
        """Recovering an empty journal should return a zero-count summary."""
        j = _make_journal(tmp_path)
        summary = j.recover()
        j.close()

        assert isinstance(summary, SessionSummary)
        assert summary.event_count == 0
        assert summary.ingested_files == []
        assert summary.client_config is None
        assert summary.active_profile is None
        assert summary.tool_call_stats == {}
        assert summary.total_tokens_saved == 0

    def test_recover_with_ingest_events(self, tmp_path: Path) -> None:
        """Ingest events should populate ingested_files list."""
        j = _make_journal(tmp_path)
        j.write_event(
            "ingest", {"file_id": "readme.md", "original_tokens": 500, "compressed_tokens": 50}
        )
        j.write_event(
            "ingest", {"file_id": "main.py", "original_tokens": 300, "compressed_tokens": 30}
        )
        summary = j.recover()
        j.close()

        assert len(summary.ingested_files) == 2
        file_ids = [f["file_id"] for f in summary.ingested_files]
        assert "readme.md" in file_ids
        assert "main.py" in file_ids

    def test_recover_with_configure_event(self, tmp_path: Path) -> None:
        """The last configure event should populate client_config."""
        j = _make_journal(tmp_path)
        j.write_event(
            "configure",
            {"model_id": "gpt-3.5", "provider": "openai", "context_window_tokens": 4096},
        )
        summary = j.recover()
        j.close()

        assert summary.client_config is not None
        assert summary.client_config["model_id"] == "gpt-3.5"

    def test_recover_with_profile_event(self, tmp_path: Path) -> None:
        """The last profile event should populate active_profile."""
        j = _make_journal(tmp_path)
        j.write_event("profile", {"profile_name": "balanced"})
        summary = j.recover()
        j.close()

        assert summary.active_profile == "balanced"

    def test_recover_with_tool_calls(self, tmp_path: Path) -> None:
        """tool_call events should be counted per tool name."""
        j = _make_journal(tmp_path)
        j.write_event("tool_call", {"tool_name": "ingest_context"})
        j.write_event("tool_call", {"tool_name": "ingest_context"})
        j.write_event("tool_call", {"tool_name": "read_skeleton"})
        summary = j.recover()
        j.close()

        assert summary.tool_call_stats["ingest_context"] == 2
        assert summary.tool_call_stats["read_skeleton"] == 1

    def test_recover_total_tokens_saved(self, tmp_path: Path) -> None:
        """total_tokens_saved should be sum of (original - compressed)."""
        j = _make_journal(tmp_path)
        j.write_event("ingest", {"file_id": "a", "original_tokens": 1000, "compressed_tokens": 100})
        j.write_event("ingest", {"file_id": "b", "original_tokens": 500, "compressed_tokens": 50})
        summary = j.recover()
        j.close()

        # (1000 - 100) + (500 - 50) = 900 + 450 = 1350
        assert summary.total_tokens_saved == 1350

    def test_last_configure_wins(self, tmp_path: Path) -> None:
        """When multiple configure events exist, the last one should be used."""
        j = _make_journal(tmp_path)
        j.write_event(
            "configure",
            {"model_id": "old-model", "provider": "openai", "context_window_tokens": 4096},
        )
        j.write_event(
            "configure",
            {"model_id": "new-model", "provider": "anthropic", "context_window_tokens": 200000},
        )
        summary = j.recover()
        j.close()

        assert summary.client_config is not None
        assert summary.client_config["model_id"] == "new-model"

    def test_last_profile_event_wins(self, tmp_path: Path) -> None:
        """When multiple profile events exist, the last one should be used."""
        j = _make_journal(tmp_path)
        j.write_event("profile", {"profile_name": "minimal"})
        j.write_event("profile", {"profile_name": "detailed"})
        summary = j.recover()
        j.close()

        assert summary.active_profile == "detailed"

    def test_multiple_ingests_all_listed(self, tmp_path: Path) -> None:
        """All ingest events should appear in ingested_files regardless of order."""
        j = _make_journal(tmp_path)
        for i in range(10):
            j.write_event(
                "ingest", {"file_id": f"file_{i}", "original_tokens": 100, "compressed_tokens": 10}
            )
        summary = j.recover()
        j.close()

        assert len(summary.ingested_files) == 10


class TestSessionIsolation:
    def test_session_isolation(self, tmp_path: Path) -> None:
        """Two journals with different session_ids should not share events."""
        j1 = SessionJournal(session_id="session-A", storage_dir=tmp_path)
        j2 = SessionJournal(session_id="session-B", storage_dir=tmp_path)
        j1.write_event("configure", {"model_id": "claude"})
        summary_b = j2.recover()
        j1.close()
        j2.close()

        assert summary_b.event_count == 0
        assert summary_b.client_config is None


class TestPersistence:
    def test_close_and_reopen_persists_data(self, tmp_path: Path) -> None:
        """Data should survive a close/reopen cycle."""
        j = _make_journal(tmp_path)
        j.write_event("profile", {"profile_name": "minimal"})
        j.close()

        # Re-open
        j2 = _make_journal(tmp_path)
        summary = j2.recover()
        j2.close()

        assert summary.active_profile == "minimal"

    def test_auto_creates_directory(self, tmp_path: Path) -> None:
        """SessionJournal should create storage_dir if it does not exist."""
        nested_dir = tmp_path / "nested" / "deep"
        assert not nested_dir.exists()
        j = SessionJournal(session_id="new-session", storage_dir=nested_dir)
        j.close()
        assert nested_dir.exists()


class TestWALMode:
    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """The SQLite journal should be in WAL mode."""
        j = _make_journal(tmp_path)
        row = j._conn.execute("PRAGMA journal_mode").fetchone()
        j.close()
        assert row is not None
        assert row[0] == "wal"
