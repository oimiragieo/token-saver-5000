"""Tee/Recovery system for preserving original content before compression.

When compression drops information, the original is saved for on-demand recovery.
Storage uses an LRU cache with configurable retention limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    ):
        self.mode = mode
        self.compression_threshold = compression_threshold
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.persist_dir = persist_dir

        # Ordered dict behavior via list + dict for LRU
        self._entries: Dict[str, TeeEntry] = {}
        self._order: List[str] = []  # oldest first

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def generate_id(text: str, source: str) -> str:
        """Generate a short deterministic ID from content + source."""
        digest = hashlib.sha256(f"{source}:{text[:512]}".encode()).hexdigest()[:12]
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
    ) -> Optional[str]:
        """Store original content if tee conditions are met.

        Returns the entry_id if stored, None if skipped.
        """
        if not self.should_tee(compression_pct):
            return None

        entry_id = self.generate_id(original_text, source)
        entry = TeeEntry(
            entry_id=entry_id,
            original_text=original_text,
            compressed_text=compressed_text,
            compression_pct=compression_pct,
            source=source,
            command_hint=command_hint,
            metadata=metadata or {},
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

    def get(self, entry_id: str) -> Optional[TeeEntry]:
        """Retrieve a tee'd entry by ID."""
        entry = self._entries.get(entry_id)
        if entry is None and self.persist_dir:
            entry = self._load_entry(entry_id)
            if entry:
                self._entries[entry_id] = entry
                self._order.append(entry_id)
        return entry

    def get_original(self, entry_id: str) -> Optional[str]:
        """Retrieve only the original text for a tee'd entry."""
        entry = self.get(entry_id)
        return entry.original_text if entry else None

    def list_entries(self, limit: int = 20, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """List recent tee entries (newest first)."""
        entries = []
        for eid in reversed(self._order):
            entry = self._entries.get(eid)
            if entry is None:
                continue
            if source and entry.source != source:
                continue
            entries.append(entry.to_dict())
            if len(entries) >= limit:
                break
        return entries

    def delete(self, entry_id: str) -> bool:
        """Delete a specific tee entry."""
        if entry_id not in self._entries:
            return False
        del self._entries[entry_id]
        self._order.remove(entry_id)
        if self.persist_dir:
            path = Path(self.persist_dir) / f"{entry_id}.json"
            path.unlink(missing_ok=True)
        return True

    def clear(self) -> int:
        """Clear all entries. Returns count of entries removed."""
        count = len(self._entries)
        self._entries.clear()
        self._order.clear()
        if self.persist_dir:
            for f in Path(self.persist_dir).glob("*.json"):
                f.unlink(missing_ok=True)
        return count

    @property
    def total_size_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries.values())

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "mode": self.mode,
            "entry_count": self.entry_count,
            "max_entries": self.max_entries,
            "total_size_bytes": self.total_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "compression_threshold": self.compression_threshold,
            "oldest_entry": self._entries[self._order[0]].timestamp if self._order else None,
            "newest_entry": self._entries[self._order[-1]].timestamp if self._order else None,
        }

    # --- Internal methods ---

    def _evict(self) -> None:
        """Evict oldest entries to stay within limits."""
        # Evict by count
        while len(self._order) > self.max_entries:
            oldest_id = self._order.pop(0)
            entry = self._entries.pop(oldest_id, None)
            if entry and self.persist_dir:
                path = Path(self.persist_dir) / f"{oldest_id}.json"
                path.unlink(missing_ok=True)

        # Evict by total size
        while self.total_size_bytes > self.max_size_bytes and self._order:
            oldest_id = self._order.pop(0)
            entry = self._entries.pop(oldest_id, None)
            if entry and self.persist_dir:
                path = Path(self.persist_dir) / f"{oldest_id}.json"
                path.unlink(missing_ok=True)

    def _persist_entry(self, entry: TeeEntry) -> None:
        """Write entry to disk as JSON."""
        if not self.persist_dir:
            return
        path = Path(self.persist_dir) / f"{entry.entry_id}.json"
        data = {
            "entry_id": entry.entry_id,
            "original_text": entry.original_text,
            "compressed_text": entry.compressed_text,
            "compression_pct": entry.compression_pct,
            "source": entry.source,
            "command_hint": entry.command_hint,
            "timestamp": entry.timestamp,
            "metadata": entry.metadata,
        }
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def _load_entry(self, entry_id: str) -> Optional[TeeEntry]:
        """Load entry from disk if it exists."""
        if not self.persist_dir:
            return None
        path = Path(self.persist_dir) / f"{entry_id}.json"
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
