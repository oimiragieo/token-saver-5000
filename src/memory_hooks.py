"""
Framework memory sync hooks.

Post-compression hook system that persists distilled learnings
to a memory index for cross-session knowledge retention.
"""

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from .memory_api import MemoryAPI

logger = logging.getLogger(__name__)


class MemoryHookManager:
    """Manages event hooks and accumulated memory entries."""

    def __init__(self, memory_api: Optional["MemoryAPI"] = None):
        self._hooks: Dict[str, List[Callable]] = {}
        self._memory_index: List[Dict[str, Any]] = []
        self._memory_api = memory_api

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
        self,
        file_id: str,
        insight: str,
        category: str,
        *,
        source: str = "hook",
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Add a memory entry to the index.

        Args:
            file_id: Source file identifier
            insight: Distilled insight text
            category: Classification category
        """
        entry = {"file_id": file_id, "insight": insight, "category": category}
        self._memory_index.append(entry)
        if self._memory_api is not None:
            self._memory_api.add_memory(
                text=insight,
                category=category,
                source=source,
                file_id=file_id,
                metadata=metadata or {},
                workspace_id=workspace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            )

    def get_memory_index(self) -> List[Dict[str, Any]]:
        """Get all accumulated memory entries."""
        return list(self._memory_index)

    def clear(self) -> None:
        """Clear all hooks and memory entries."""
        self._hooks.clear()
        self._memory_index.clear()
