"""Temporal fact graph for lifecycle-aware retrieval and invalidation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any


def coerce_timestamp(value: float | int | str | None = None) -> float:
    """Normalize unix/ISO timestamps into unix seconds."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return float(value)

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized).timestamp()


def format_timestamp(value: float) -> str:
    """Format unix seconds as canonical UTC ISO-8601."""
    return datetime.fromtimestamp(value, UTC).isoformat().replace("+00:00", "Z")


@dataclass
class TemporalFactVersion:
    """One observed version of a fact over time."""

    fact_id: str
    doc_id: str
    content: str
    observed_at: float
    version: int
    metadata: dict[str, Any] = field(default_factory=dict)
    invalidated_at: float | None = None
    invalidation_reason: str | None = None

    def is_active(self, as_of: float | None = None) -> bool:
        reference = time.time() if as_of is None else as_of
        return self.observed_at <= reference and (
            self.invalidated_at is None or self.invalidated_at > reference
        )

    def to_dict(self, as_of: float | None = None) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "version": self.version,
            "metadata": dict(self.metadata),
            "observed_at": format_timestamp(self.observed_at),
            "observed_at_unix": self.observed_at,
            "invalidated_at": (
                format_timestamp(self.invalidated_at) if self.invalidated_at is not None else None
            ),
            "invalidated_at_unix": self.invalidated_at,
            "invalidation_reason": self.invalidation_reason,
            "active": self.is_active(as_of),
        }


@dataclass
class TemporalEvent:
    """A timestamped lifecycle event."""

    event_id: int
    event_type: str
    timestamp: float
    doc_id: str | None = None
    fact_id: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": format_timestamp(self.timestamp),
            "timestamp_unix": self.timestamp,
            "doc_id": self.doc_id,
            "fact_id": self.fact_id,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


class TemporalGraph:
    """Tracks observed facts, invalidations, and lifecycle timeline events."""

    def __init__(self):
        self._lock = RLock()
        self._fact_history: dict[str, list[TemporalFactVersion]] = {}
        self._doc_fact_ids: dict[str, set[str]] = {}
        self._timeline: list[TemporalEvent] = []
        self._next_event_id = 1

    def record_event(
        self,
        event_type: str,
        *,
        doc_id: str | None = None,
        fact_id: str | None = None,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        timestamp: float | int | str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            event = self._append_event(
                event_type=event_type,
                doc_id=doc_id,
                fact_id=fact_id,
                summary=summary,
                metadata=metadata,
                timestamp=timestamp,
            )
            return event.to_dict()

    def record_document_state(
        self,
        doc_id: str,
        facts: list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
        timestamp: float | int | str | None = None,
    ) -> dict[str, Any]:
        """Capture a document snapshot as temporal facts."""
        observed_at = coerce_timestamp(timestamp)
        with self._lock:
            current_fact_ids: set[str] = set()
            previous_fact_ids = set(self._doc_fact_ids.get(doc_id, set()))

            for fact in facts:
                fact_id = str(fact["fact_id"])
                content = str(fact["content"])
                fact_metadata = dict(fact.get("metadata") or {})
                current_fact_ids.add(fact_id)

                history = self._fact_history.setdefault(fact_id, [])
                active_version = self._latest_version_before(history, observed_at, active_only=True)
                if active_version is not None and active_version.doc_id != doc_id:
                    raise ValueError(
                        f"Fact '{fact_id}' is already associated with document '{active_version.doc_id}'"
                    )

                if (
                    active_version is not None
                    and active_version.content == content
                    and active_version.metadata == fact_metadata
                ):
                    self._append_event(
                        "fact_observed",
                        doc_id=doc_id,
                        fact_id=fact_id,
                        summary=content[:160],
                        metadata={"version": active_version.version, **fact_metadata},
                        timestamp=observed_at,
                    )
                    continue

                if active_version is not None:
                    active_version.invalidated_at = observed_at
                    active_version.invalidation_reason = "superseded"
                    self._append_event(
                        "fact_superseded",
                        doc_id=doc_id,
                        fact_id=fact_id,
                        summary=content[:160],
                        metadata={"previous_version": active_version.version},
                        timestamp=observed_at,
                    )

                new_version = TemporalFactVersion(
                    fact_id=fact_id,
                    doc_id=doc_id,
                    content=content,
                    observed_at=observed_at,
                    version=len(history) + 1,
                    metadata=fact_metadata,
                )
                history.append(new_version)
                self._append_event(
                    "fact_observed" if new_version.version == 1 else "fact_updated",
                    doc_id=doc_id,
                    fact_id=fact_id,
                    summary=content[:160],
                    metadata={"version": new_version.version, **fact_metadata},
                    timestamp=observed_at,
                )

            for removed_fact_id in sorted(previous_fact_ids - current_fact_ids):
                self._invalidate_active_fact_locked(
                    removed_fact_id,
                    reason="removed_from_document_state",
                    timestamp=observed_at,
                    metadata={"doc_id": doc_id},
                )

            self._doc_fact_ids[doc_id] = current_fact_ids
            self._append_event(
                "document_observed",
                doc_id=doc_id,
                summary=f"Captured {len(current_fact_ids)} facts",
                metadata={"fact_count": len(current_fact_ids), **dict(metadata or {})},
                timestamp=observed_at,
            )
            return {
                "doc_id": doc_id,
                "active_fact_count": len(current_fact_ids),
                "observed_at": format_timestamp(observed_at),
            }

    def record_access(
        self,
        doc_id: str,
        *,
        access_type: str = "access",
        metadata: dict[str, Any] | None = None,
        timestamp: float | int | str | None = None,
    ) -> dict[str, Any]:
        return self.record_event(
            access_type,
            doc_id=doc_id,
            summary=f"{access_type} for {doc_id}",
            metadata=metadata,
            timestamp=timestamp,
        )

    def invalidate_fact(
        self,
        fact_id: str,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
        timestamp: float | int | str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            version = self._invalidate_active_fact_locked(
                fact_id,
                reason=reason,
                timestamp=coerce_timestamp(timestamp),
                metadata=metadata,
            )
            return version.to_dict()

    def list_fact_history(
        self,
        *,
        doc_id: str | None = None,
        fact_id: str | None = None,
        include_invalidated: bool = True,
        as_of: float | int | str | None = None,
    ) -> list[dict[str, Any]]:
        reference = None if as_of is None else coerce_timestamp(as_of)
        with self._lock:
            fact_ids = [fact_id] if fact_id is not None else sorted(self._fact_history)
            history: list[dict[str, Any]] = []
            for current_fact_id in fact_ids:
                for version in self._fact_history.get(current_fact_id, []):
                    if doc_id is not None and version.doc_id != doc_id:
                        continue
                    if reference is not None and version.observed_at > reference:
                        continue
                    if not include_invalidated and not version.is_active(reference):
                        continue
                    history.append(version.to_dict(reference))

            history.sort(
                key=lambda entry: (entry["observed_at_unix"], entry["version"]), reverse=True
            )
            return history

    def get_active_facts(
        self,
        doc_id: str,
        *,
        as_of: float | int | str | None = None,
        include_invalidated: bool = False,
    ) -> list[dict[str, Any]]:
        reference = None if as_of is None else coerce_timestamp(as_of)
        with self._lock:
            active: list[dict[str, Any]] = []
            for fact_id in sorted(self._doc_fact_ids.get(doc_id, set())):
                version = self._latest_version_before(
                    self._fact_history.get(fact_id, []), reference
                )
                if version is None:
                    continue
                if not include_invalidated and not version.is_active(reference):
                    continue
                active.append(version.to_dict(reference))

            active.sort(
                key=lambda entry: (entry["active"], entry["observed_at_unix"]), reverse=True
            )
            return active

    def is_fact_active(self, fact_id: str, *, as_of: float | int | str | None = None) -> bool:
        reference = None if as_of is None else coerce_timestamp(as_of)
        with self._lock:
            version = self._latest_version_before(self._fact_history.get(fact_id, []), reference)
            return version.is_active(reference) if version is not None else True

    def get_invalidated_fact_ids(
        self, doc_id: str, *, as_of: float | int | str | None = None
    ) -> set[str]:
        reference = None if as_of is None else coerce_timestamp(as_of)
        with self._lock:
            invalidated: set[str] = set()
            for fact_id in self._doc_fact_ids.get(doc_id, set()):
                version = self._latest_version_before(
                    self._fact_history.get(fact_id, []), reference
                )
                if version is not None and not version.is_active(reference):
                    invalidated.add(fact_id)
            return invalidated

    def search_timeline(
        self,
        *,
        query: str | None = None,
        doc_id: str | None = None,
        fact_id: str | None = None,
        event_types: list[str] | None = None,
        since: float | int | str | None = None,
        until: float | int | str | None = None,
        include_invalidated: bool = True,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query_text = (query or "").strip().lower()
        since_ts = None if since is None else coerce_timestamp(since)
        until_ts = None if until is None else coerce_timestamp(until)
        allowed_types = set(event_types or [])
        invalidation_types = {"fact_invalidated", "fact_superseded"}

        with self._lock:
            events: list[dict[str, Any]] = []
            for event in reversed(self._timeline):
                if doc_id is not None and event.doc_id != doc_id:
                    continue
                if fact_id is not None and event.fact_id != fact_id:
                    continue
                if allowed_types and event.event_type not in allowed_types:
                    continue
                if not include_invalidated and event.event_type in invalidation_types:
                    continue
                if since_ts is not None and event.timestamp < since_ts:
                    continue
                if until_ts is not None and event.timestamp > until_ts:
                    continue

                payload = event.to_dict()
                searchable = " ".join(
                    [
                        payload.get("event_type") or "",
                        payload.get("doc_id") or "",
                        payload.get("fact_id") or "",
                        payload.get("summary") or "",
                        json.dumps(payload.get("metadata") or {}, sort_keys=True),
                    ]
                ).lower()
                if query_text and query_text not in searchable:
                    continue

                events.append(payload)
                if len(events) >= limit:
                    break
            return events

    def _append_event(
        self,
        event_type: str,
        *,
        doc_id: str | None = None,
        fact_id: str | None = None,
        summary: str = "",
        metadata: dict[str, Any] | None = None,
        timestamp: float | int | str | None = None,
    ) -> TemporalEvent:
        event = TemporalEvent(
            event_id=self._next_event_id,
            event_type=event_type,
            timestamp=coerce_timestamp(timestamp),
            doc_id=doc_id,
            fact_id=fact_id,
            summary=summary,
            metadata=dict(metadata or {}),
        )
        self._next_event_id += 1
        self._timeline.append(event)
        return event

    def _invalidate_active_fact_locked(
        self,
        fact_id: str,
        *,
        reason: str,
        timestamp: float,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalFactVersion:
        history = self._fact_history.get(fact_id, [])
        active_version = self._latest_version_before(history, timestamp, active_only=True)
        if active_version is None:
            raise ValueError(f"Fact '{fact_id}' has no active version to invalidate")

        active_version.invalidated_at = timestamp
        active_version.invalidation_reason = reason
        self._append_event(
            "fact_invalidated",
            doc_id=active_version.doc_id,
            fact_id=fact_id,
            summary=reason,
            metadata=metadata,
            timestamp=timestamp,
        )
        return active_version

    @staticmethod
    def _latest_version_before(
        history: list[TemporalFactVersion],
        reference: float | None,
        *,
        active_only: bool = False,
    ) -> TemporalFactVersion | None:
        if not history:
            return None

        chosen: TemporalFactVersion | None = None
        for version in history:
            if reference is not None and version.observed_at > reference:
                continue
            chosen = version

        if chosen is None:
            return None
        if active_only and not chosen.is_active(reference):
            return None
        return chosen
