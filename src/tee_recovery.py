"""Tee/Recovery system for preserving original content before compression.

When compression drops information, the original is saved for on-demand recovery.
Storage uses an LRU cache with configurable retention limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_TOKEN_ESTIMATOR: Any = None


def _get_token_estimator() -> Any:
    global _TOKEN_ESTIMATOR
    if _TOKEN_ESTIMATOR is None:
        from .token_estimation import TokenEstimator

        _TOKEN_ESTIMATOR = TokenEstimator()
    return _TOKEN_ESTIMATOR


_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(
        r"\b(?:sk-ant-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{12,}|AKIA[0-9A-Z]{16})\b"
    ),
    re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{12,}=*"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|passwd|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?"
    ),
)


def redact_secrets(text: str) -> str:
    """Redact credential-shaped tokens before persisting tee entries to disk."""
    if not text:
        return text
    redacted = text
    for idx, pattern in enumerate(_SECRET_PATTERNS):
        if idx == 3:
            redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED_SECRET]", redacted)
        else:
            redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


_LEGACY_SCOPES = frozenset({"default", "__global__"})

_ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_SENSITIVE_FIELD_NAMES = (
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "private_key",
    "authorization",
    "auth",
)


def _validate_entry_id(entry_id: str) -> str:
    """Validate tee entry_id against safe filename characters."""
    if not isinstance(entry_id, str) or not _ENTRY_ID_RE.fullmatch(entry_id):
        raise ValueError(f"Invalid tee entry_id: {entry_id!r}")
    return entry_id


def _is_sensitive_key(key: Any) -> bool:
    """Return True if a dict key normalizes to a sensitive field name."""
    if not isinstance(key, str):
        return False
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    return any(term in normalized for term in _SENSITIVE_FIELD_NAMES)


def _redact_value(value: Any, redactor: Callable[[str], str]) -> tuple[Any, bool]:
    """Recursively redact secrets in strings, dicts, lists, and tuples."""
    if isinstance(value, str):
        redacted = redactor(value)
        return redacted, redacted != value
    if isinstance(value, dict):
        out_dict: Dict[Any, Any] = {}
        changed = False
        for k, v in value.items():
            rk, ck = _redact_value(k, redactor)
            if _is_sensitive_key(k):
                rv = "[REDACTED_SECRET]"
                cv = v != "[REDACTED_SECRET]"
            else:
                rv, cv = _redact_value(v, redactor)
            out_dict[rk] = rv
            changed = changed or ck or cv
        return out_dict, changed
    if isinstance(value, list):
        out_list: List[Any] = []
        changed = False
        for item in value:
            ri, ci = _redact_value(item, redactor)
            out_list.append(ri)
            changed = changed or ci
        return out_list, changed
    if isinstance(value, tuple):
        out_items: List[Any] = []
        changed = False
        for item in value:
            ri, ci = _redact_value(item, redactor)
            out_items.append(ri)
            changed = changed or ci
        return out_items, changed
    return value, False


@dataclass
class TeeEntry:
    """A single tee'd original content entry."""

    entry_id: str
    original_text: str
    compressed_text: str
    compression_pct: float
    source: str  # "cli_optimizer", "proxy", "compression"
    command_hint: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 0
    visible: bool = True
    workspace_id: str = "default"
    session_id: str = "default"
    redacted_on_disk: bool = False

    def __post_init__(self) -> None:
        if self.estimated_tokens == 0 and self.original_text:
            self.estimated_tokens = _get_token_estimator().count_tokens(self.original_text)

    @property
    def size_bytes(self) -> int:
        return len(self.original_text.encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "compression_pct": round(self.compression_pct, 1),
            "source": self.source,
            "command_hint": self.command_hint,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "original_length": len(self.original_text),
            "compressed_length": len(self.compressed_text),
            "estimated_tokens": self.estimated_tokens,
            "visible": self.visible,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "redacted_on_disk": self.redacted_on_disk,
            "metadata": self.metadata,
        }


# Tee mode controls when originals are saved
TEE_MODE_FAILURES = "failures"  # tee only on high compression (default)
TEE_MODE_ALWAYS = "always"  # tee everything
TEE_MODE_NEVER = "never"  # disable tee

# Compression threshold for "failures" mode — only tee when compression exceeds this
DEFAULT_COMPRESSION_THRESHOLD = 80.0

# Storage limits
DEFAULT_MAX_ENTRIES = 50
DEFAULT_MAX_SIZE_MB = 100.0


class TeeStore:
    """LRU-evicting store for original content before compression.

    Supports in-memory operation and optional JSON persistence to disk.
    """

    def __init__(
        self,
        mode: str = TEE_MODE_FAILURES,
        compression_threshold: float = DEFAULT_COMPRESSION_THRESHOLD,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_size_mb: float = DEFAULT_MAX_SIZE_MB,
        persist_dir: Optional[str] = None,
        redactor: Optional[Callable[[str], str]] = None,
    ):
        self.mode = mode
        self.compression_threshold = compression_threshold
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.persist_dir = persist_dir
        self._redactor = redactor or redact_secrets

        # Ordered dict behavior via list + dict for LRU
        self._entries: Dict[str, TeeEntry] = {}
        self._order: List[str] = []  # oldest first

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_id(
        text: str,
        source: str,
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> str:
        """Generate a short deterministic ID from full content, source, and scope."""
        payload = json.dumps([workspace_id, session_id, source, text], ensure_ascii=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return digest

    def should_tee(self, compression_pct: float) -> bool:
        """Check whether content should be tee'd based on mode and compression."""
        if self.mode == TEE_MODE_NEVER:
            return False
        if self.mode == TEE_MODE_ALWAYS:
            return True
        # "failures" mode: only tee when compression is aggressive
        return compression_pct >= self.compression_threshold

    def store(
        self,
        original_text: str,
        compressed_text: str,
        compression_pct: float,
        source: str,
        command_hint: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: str = "default",
        session_id: str = "default",
        visible: bool = True,
    ) -> Optional[str]:
        """Store original content if tee conditions are met.

        Returns the entry_id if stored, None if skipped.
        """
        if not self.should_tee(compression_pct):
            return None

        entry_id = self.generate_id(
            original_text,
            source,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        entry = TeeEntry(
            entry_id=entry_id,
            original_text=original_text,
            compressed_text=compressed_text,
            compression_pct=compression_pct,
            source=source,
            command_hint=command_hint,
            metadata=metadata or {},
            visible=visible,
            workspace_id=workspace_id,
            session_id=session_id,
        )

        # If already exists, update and move to end (most recent)
        if entry_id in self._entries:
            self._order.remove(entry_id)

        self._entries[entry_id] = entry
        self._order.append(entry_id)

        self._evict()

        if self.persist_dir:
            self._persist_entry(entry)

        return entry_id

    @staticmethod
    def _entry_matches_scope(
        entry: TeeEntry,
        workspace_id: Optional[str],
        session_id: Optional[str],
    ) -> bool:
        ws_ok = (
            entry.workspace_id in _LEGACY_SCOPES
            if workspace_id is None
            else entry.workspace_id == workspace_id
        )
        sess_ok = (
            entry.session_id in _LEGACY_SCOPES
            if session_id is None
            else entry.session_id == session_id
        )
        return ws_ok and sess_ok

    @staticmethod
    def _check_scope(
        entry: TeeEntry,
        workspace_id: Optional[str],
        session_id: Optional[str],
    ) -> None:
        if workspace_id is None:
            if entry.workspace_id not in _LEGACY_SCOPES:
                raise PermissionError(f"Workspace scope required for tee entry '{entry.entry_id}'")
        elif entry.workspace_id != workspace_id:
            raise PermissionError(f"Workspace scope mismatch for tee entry '{entry.entry_id}'")

        if session_id is None:
            if entry.session_id not in _LEGACY_SCOPES:
                raise PermissionError(f"Session scope required for tee entry '{entry.entry_id}'")
        elif entry.session_id != session_id:
            raise PermissionError(f"Session scope mismatch for tee entry '{entry.entry_id}'")

    def _entry_path(self, entry_id: str) -> Path:
        """Resolve and validate the on-disk JSON path for *entry_id*."""
        _validate_entry_id(entry_id)
        if not self.persist_dir:
            raise ValueError("persist_dir is not configured")
        tee_dir = Path(self.persist_dir).resolve()
        path = (tee_dir / f"{entry_id}.json").resolve()
        if not path.is_relative_to(tee_dir):
            raise ValueError(f"Tee entry path escapes persist_dir: {entry_id!r}")
        return path

    def get(
        self,
        entry_id: str,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[TeeEntry]:
        """Retrieve a tee'd entry by ID, enforcing optional scope isolation."""
        _validate_entry_id(entry_id)
        entry = self._entries.get(entry_id)
        if entry is None and self.persist_dir:
            entry = self._load_entry(entry_id)
            if entry:
                self._entries[entry_id] = entry
                self._order.append(entry_id)
        if entry is not None:
            self._check_scope(entry, workspace_id, session_id)
        return entry

    def get_original(
        self,
        entry_id: str,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Retrieve only the original text for a tee'd entry."""
        entry = self.get(entry_id, workspace_id=workspace_id, session_id=session_id)
        return entry.original_text if entry else None

    def set_visibility(
        self,
        entry_id: str,
        visible: bool,
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> TeeEntry:
        """Update visibility flag for *entry_id* with strict scope isolation."""
        entry = self.get(entry_id, workspace_id=workspace_id, session_id=session_id)
        if entry is None:
            raise KeyError(f"No tee entry found with id '{entry_id}'")
        prev_visible = entry.visible
        entry.visible = visible
        if self.persist_dir:
            try:
                self._persist_entry(entry)
            except Exception:
                entry.visible = prev_visible
                raise
        return entry

    def list_entries(
        self,
        limit: int = 20,
        source: Optional[str] = None,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List recent tee entries (newest first) within the caller's scope."""
        entries = []
        for eid in reversed(self._order):
            entry = self._entries.get(eid)
            if entry is None:
                continue
            if not self._entry_matches_scope(entry, workspace_id, session_id):
                continue
            if source and entry.source != source:
                continue
            entries.append(entry.to_dict())
            if len(entries) >= limit:
                break
        return entries

    def delete(
        self,
        entry_id: str,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> bool:
        """Delete a specific tee entry, enforcing scope isolation."""
        _validate_entry_id(entry_id)
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        self._check_scope(entry, workspace_id, session_id)
        del self._entries[entry_id]
        self._order.remove(entry_id)
        if self.persist_dir:
            self._entry_path(entry_id).unlink(missing_ok=True)
        return True

    def clear(
        self,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> int:
        """Clear entries matching the caller's scope. Returns count removed."""
        to_remove = [
            eid
            for eid in list(self._order)
            if (entry := self._entries.get(eid)) is not None
            and self._entry_matches_scope(entry, workspace_id, session_id)
        ]
        for eid in to_remove:
            self._entries.pop(eid, None)
            if eid in self._order:
                self._order.remove(eid)
            if self.persist_dir:
                self._entry_path(eid).unlink(missing_ok=True)
        return len(to_remove)

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def stats(
        self,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get store statistics scoped to the caller's workspace/session."""
        scoped_entries = [
            self._entries[eid]
            for eid in self._order
            if eid in self._entries
            and self._entry_matches_scope(self._entries[eid], workspace_id, session_id)
        ]
        return {
            "mode": self.mode,
            "entry_count": len(scoped_entries),
            "max_entries": self.max_entries,
            "total_size_bytes": sum(e.size_bytes for e in scoped_entries),
            "max_size_bytes": self.max_size_bytes,
            "compression_threshold": self.compression_threshold,
            "oldest_entry": scoped_entries[0].timestamp if scoped_entries else None,
            "newest_entry": scoped_entries[-1].timestamp if scoped_entries else None,
        }

    def get_stats(
        self,
        workspace_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Alias for :meth:`stats`."""
        return self.stats(workspace_id=workspace_id, session_id=session_id)

    # --- Internal methods ---

    def _evict(self) -> None:
        """Evict oldest entries to stay within limits."""
        # Evict by count
        while len(self._order) > self.max_entries:
            oldest_id = self._order.pop(0)
            entry = self._entries.pop(oldest_id, None)
            if entry and self.persist_dir:
                self._entry_path(oldest_id).unlink(missing_ok=True)

        # Evict by total size
        while self.total_size_bytes > self.max_size_bytes and self._order:
            oldest_id = self._order.pop(0)
            entry = self._entries.pop(oldest_id, None)
            if entry and self.persist_dir:
                self._entry_path(oldest_id).unlink(missing_ok=True)

    def _persist_entry(self, entry: TeeEntry) -> None:
        """Write entry to disk as JSON after running recursive secret redaction."""
        if not self.persist_dir:
            return
        path = self._entry_path(entry.entry_id)
        r_orig, c1 = _redact_value(entry.original_text, self._redactor)
        r_comp, c2 = _redact_value(entry.compressed_text, self._redactor)
        r_src, c3 = _redact_value(entry.source, self._redactor)
        r_hint, c4 = _redact_value(entry.command_hint, self._redactor)
        r_ws, c5 = _redact_value(entry.workspace_id, self._redactor)
        r_sess, c6 = _redact_value(entry.session_id, self._redactor)
        r_meta, c7 = _redact_value(entry.metadata, self._redactor)
        redacted_on_disk = bool(entry.redacted_on_disk or c1 or c2 or c3 or c4 or c5 or c6 or c7)
        data = {
            "entry_id": entry.entry_id,
            "original_text": r_orig,
            "compressed_text": r_comp,
            "compression_pct": entry.compression_pct,
            "source": r_src,
            "command_hint": r_hint,
            "timestamp": entry.timestamp,
            "estimated_tokens": entry.estimated_tokens,
            "visible": entry.visible,
            "workspace_id": r_ws,
            "session_id": r_sess,
            "redacted_on_disk": redacted_on_disk,
            "metadata": r_meta,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load_entry(self, entry_id: str) -> Optional[TeeEntry]:
        """Load entry from disk if it exists."""
        if not self.persist_dir:
            return None
        path = self._entry_path(entry_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TeeEntry(
                entry_id=data["entry_id"],
                original_text=data["original_text"],
                compressed_text=data["compressed_text"],
                compression_pct=data["compression_pct"],
                source=data["source"],
                command_hint=data.get("command_hint", ""),
                timestamp=data.get("timestamp", 0.0),
                metadata=data.get("metadata", {}),
                estimated_tokens=data.get("estimated_tokens", 0),
                visible=data.get("visible", True),
                workspace_id=data.get("workspace_id", "default"),
                session_id=data.get("session_id", "default"),
                redacted_on_disk=bool(data.get("redacted_on_disk", False)),
            )
        except (json.JSONDecodeError, KeyError):
            return None


def get_default_tee_dir() -> str:
    """Return the default tee storage directory."""
    return os.path.join(".semantic_modulator_data", "tee")


def create_tee_store(
    mode: Optional[str] = None,
    persist: bool = True,
) -> TeeStore:
    """Factory function to create a TeeStore with sensible defaults.

    Reads TEE_MODE from environment if not specified.
    """
    resolved_mode = mode or os.getenv("TEE_MODE", TEE_MODE_FAILURES)
    threshold = float(os.getenv("TEE_COMPRESSION_THRESHOLD", str(DEFAULT_COMPRESSION_THRESHOLD)))
    max_entries = int(os.getenv("TEE_MAX_ENTRIES", str(DEFAULT_MAX_ENTRIES)))
    max_size_mb = float(os.getenv("TEE_MAX_SIZE_MB", str(DEFAULT_MAX_SIZE_MB)))

    persist_dir = get_default_tee_dir() if persist else None

    return TeeStore(
        mode=resolved_mode,
        compression_threshold=threshold,
        max_entries=max_entries,
        max_size_mb=max_size_mb,
        persist_dir=persist_dir,
    )
