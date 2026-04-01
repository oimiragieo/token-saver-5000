"""
Session Journal (v0.13.0)

SQLite-backed event journal for session state recovery after conversation compaction.
Each session writes to an isolated DB file under .semantic_modulator_data/sessions/.

Supports:
- Journaling ingest, configure, profile, and tool_call events.
- Session recovery that aggregates the full event history into a compact summary.
- WAL mode for safe concurrent writes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_DEFAULT_STORAGE_DIR = Path(".semantic_modulator_data") / "sessions"


@dataclass
class JournalEvent:
    """A single persisted session event."""

    event_type: str
    timestamp: float
    data: dict[str, Any]
    event_id: int = 0


@dataclass
class SessionSummary:
    """Aggregated view of all events for a session."""

    session_id: str
    event_count: int = 0
    first_event_time: float | None = None
    last_event_time: float | None = None
    ingested_files: list[dict[str, Any]] = field(default_factory=list)
    client_config: dict[str, Any] | None = None
    active_profile: str | None = None
    tool_call_stats: dict[str, int] = field(default_factory=dict)
    total_tokens_saved: int = 0


class SessionJournal:
    """
    Append-only SQLite journal for a single session.

    Each session is isolated in its own DB file so that concurrent sessions
    never contend on the same SQLite connection.
    """

    def __init__(
        self,
        session_id: str,
        storage_dir: Path | str | None = None,
    ) -> None:
        self._session_id = session_id
        self._storage_dir = Path(storage_dir) if storage_dir is not None else _DEFAULT_STORAGE_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._storage_dir / f"{session_id}.db"
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the events table and index if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT    NOT NULL,
                timestamp   REAL    NOT NULL,
                data_json   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_event(self, event_type: str, data: dict[str, Any]) -> JournalEvent:
        """Insert an event and return it with the auto-assigned event_id."""
        ts = time.time()
        data_json = json.dumps(data)
        cur = self._conn.execute(
            "INSERT INTO events (event_type, timestamp, data_json) VALUES (?, ?, ?)",
            (event_type, ts, data_json),
        )
        self._conn.commit()
        return JournalEvent(
            event_type=event_type,
            timestamp=ts,
            data=data,
            event_id=cur.lastrowid or 0,
        )

    # ------------------------------------------------------------------
    # Read / Recovery
    # ------------------------------------------------------------------

    def recover(self) -> SessionSummary:
        """
        Aggregate all events into a SessionSummary.

        Aggregation rules:
        - ingested_files: list of {file_id, original_tokens, compressed_tokens}
          extracted from every "ingest" event.
        - client_config: data from the LAST "configure" event.
        - active_profile: profile_name from the LAST "profile" event.
        - tool_call_stats: {tool_name: count} from all "tool_call" events.
        - total_tokens_saved: sum of (original_tokens - compressed_tokens)
          across all "ingest" events.
        """
        rows = self._conn.execute(
            "SELECT id, event_type, timestamp, data_json FROM events ORDER BY id ASC"
        ).fetchall()

        if not rows:
            return SessionSummary(session_id=self._session_id)

        ingested_files: list[dict[str, Any]] = []
        client_config: dict[str, Any] | None = None
        active_profile: str | None = None
        tool_call_stats: dict[str, int] = {}
        total_tokens_saved: int = 0
        timestamps: list[float] = []

        for _row_id, event_type, timestamp, data_json in rows:
            timestamps.append(timestamp)
            try:
                data = json.loads(data_json)
            except json.JSONDecodeError:
                data = {}

            if event_type == "ingest":
                file_id = data.get("file_id", "")
                original = int(data.get("original_tokens", 0))
                compressed = int(data.get("compressed_tokens", 0))
                ingested_files.append(
                    {
                        "file_id": file_id,
                        "original_tokens": original,
                        "compressed_tokens": compressed,
                    }
                )
                total_tokens_saved += max(0, original - compressed)

            elif event_type == "configure":
                client_config = data

            elif event_type == "profile":
                active_profile = data.get("profile_name")

            elif event_type == "tool_call":
                tool_name = data.get("tool_name", "unknown")
                tool_call_stats[tool_name] = tool_call_stats.get(tool_name, 0) + 1

        return SessionSummary(
            session_id=self._session_id,
            event_count=len(rows),
            first_event_time=min(timestamps),
            last_event_time=max(timestamps),
            ingested_files=ingested_files,
            client_config=client_config,
            active_profile=active_profile,
            tool_call_stats=tool_call_stats,
            total_tokens_saved=total_tokens_saved,
        )

    def event_count(self) -> int:
        """Return the total number of events in this journal."""
        row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()
