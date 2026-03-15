"""
Framework memory sync hooks.

Post-compression hook system that persists distilled learnings
to a memory index for cross-session knowledge retention.
"""

import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class MemoryHookManager:
    """Manages event hooks and accumulated memory entries."""

    def __init__(self):
        self._hooks: Dict[str, List[Callable]] = {}
        self._memory_index: List[Dict[str, Any]] = []

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for an event.

        Args:
            event: Event name (e.g., 'post_compress', 'post_ingest')
            callback: Function to call when event fires
        """
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(callback)

    def trigger(self, event: str, data: Dict[str, Any]) -> None:
        """Trigger all hooks for an event.

        Args:
            event: Event name
            data: Event data passed to callbacks
        """
        for callback in self._hooks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Hook error on '{event}': {e}")

    def add_memory_entry(
        self, file_id: str, insight: str, category: str
    ) -> None:
        """Add a memory entry to the index.

        Args:
            file_id: Source file identifier
            insight: Distilled insight text
            category: Classification category
        """
        self._memory_index.append({
            "file_id": file_id,
            "insight": insight,
            "category": category,
        })

    def get_memory_index(self) -> List[Dict[str, Any]]:
        """Get all accumulated memory entries."""
        return list(self._memory_index)

    def clear(self) -> None:
        """Clear all hooks and memory entries."""
        self._hooks.clear()
        self._memory_index.clear()
