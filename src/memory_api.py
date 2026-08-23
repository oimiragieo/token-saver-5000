"""Scoped explicit memory service for add/search/list/delete/profile flows."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
import sys
from datetime import datetime, timezone

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc
from difflib import SequenceMatcher
from threading import RLock
from typing import Any, Optional

from .bounded_registry import BoundedDict
from .constants import MAX_MEMORY_ENTRIES
from .memory_classifier import classify_insight
from .personalization import build_user_profile, summarize_user_memories

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_SCOPE_FIELDS = ("workspace_id", "user_id", "agent_id", "session_id")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _scope_request(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Optional[str]]:
    return {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": session_id,
    }


def _scope_matches(
    entry: "MemoryEntry",
    *,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    requested = _scope_request(
        workspace_id=workspace_id,
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
    )

    if not any(requested.values()):
        return not any(getattr(entry, field) for field in _SCOPE_FIELDS)

    for scope_field, value in requested.items():
        if value is not None and getattr(entry, scope_field) != value:
            return False
    return True


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)}


@dataclass(frozen=True)
class MemoryEntry:
    """Immutable explicit memory record."""

    memory_id: str
    text: str
    category: str
    confidence: float
    created_at: str
    source: str = "manual"
    file_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    workspace_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "text": self.text,
            "category": self.category,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "source": self.source,
            "file_id": self.file_id,
            "metadata": deepcopy(self.metadata),
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }


class MemoryAPI:
    """Thread-safe in-memory explicit memory API with scope-aware filtering."""

    _instance: Optional["MemoryAPI"] = None

    def __init__(self, max_entries: int = MAX_MEMORY_ENTRIES):
        self._lock = RLock()
        self._entries: BoundedDict = BoundedDict(max_items=max_entries)
        self._counter = 0

    @classmethod
    def get_api(cls) -> "MemoryAPI":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def add_memory(
        self,
        *,
        text: str,
        category: str | None = None,
        source: str = "manual",
        file_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._counter += 1
            memory_id = f"mem_{self._counter}"
            classification = classify_insight(text)
            entry = MemoryEntry(
                memory_id=memory_id,
                text=text,
                category=category or classification.category,
                confidence=1.0 if category else classification.confidence,
                created_at=_utc_now(),
                source=source,
                file_id=file_id,
                metadata=deepcopy(metadata or {}),
                workspace_id=workspace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )
            self._entries[memory_id] = entry
            return entry.to_dict()

    def list_memories(
        self,
        *,
        category: str | None = None,
        limit: int | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            entries = [
                entry.to_dict()
                for entry in sorted(
                    self._entries.values(),
                    key=lambda item: (item.created_at, item.memory_id),
                    reverse=True,
                )
                if _scope_matches(
                    entry,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                )
                and (category is None or entry.category == category)
            ]
            return entries[:limit] if limit is not None else entries

    def search_memory(
        self,
        *,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        results: list[dict[str, Any]] = []
        for memory in self.list_memories(
            category=category,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        ):
            text = str(memory["text"])
            memory_tokens = _tokenize(text)
            overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            sequence = SequenceMatcher(None, query.lower(), text.lower()).ratio()
            score = round((overlap * 0.7) + (sequence * 0.3), 4)
            if score <= 0:
                continue
            results.append({**memory, "score": score})

        results.sort(key=lambda item: (item["score"], item["created_at"]), reverse=True)
        return results[:top_k]

    def delete_memory(
        self,
        memory_id: str,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            entry = self._entries.get(memory_id)
            if entry is None:
                raise ValueError(f"Unknown memory '{memory_id}'")
            if not _scope_matches(
                entry,
                workspace_id=workspace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            ):
                raise ValueError(f"Memory '{memory_id}' is not available in the requested scope")
            del self._entries[memory_id]
            return entry.to_dict()

    def summarize_user_memory(
        self,
        *,
        user_id: str,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        memories = self.list_memories(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return summarize_user_memories(memories, user_id)

    def get_user_profile(
        self,
        *,
        user_id: str,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        memories = self.list_memories(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return build_user_profile(memories, user_id)
