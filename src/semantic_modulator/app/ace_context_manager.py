"""ACE context storage with LRU eviction semantics."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from src.constants import MAX_ACE_CONTEXTS
from src.structured_logging import get_logger

logger = get_logger("semantic-modulator")


class ACEContextManager(OrderedDict):
    """LRU-enabled ACE context storage with automatic eviction."""

    def __init__(self, max_contexts: int = MAX_ACE_CONTEXTS):
        super().__init__()
        self.max_contexts = max_contexts
        logger.info(
            "ace_context_manager_initialized",
            max_contexts=max_contexts if max_contexts > 0 else "unlimited",
        )

    def __setitem__(self, key, value):
        if key in self:
            super().move_to_end(key)

        super().__setitem__(key, value)

        if self.max_contexts > 0 and len(self) > self.max_contexts:
            oldest_key = next(iter(self))
            del self[oldest_key]
            logger.info(
                "ace_context_evicted", evicted_context=oldest_key, max_contexts=self.max_contexts
            )

    def __getitem__(self, key):
        value = super().__getitem__(key)
        super().move_to_end(key)
        return value

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_contexts": len(self),
            "max_contexts_limit": self.max_contexts if self.max_contexts > 0 else "unlimited",
            "context_ids": list(self.keys()),
            "approaching_limit": (
                len(self) / self.max_contexts > 0.9 if self.max_contexts > 0 else False
            ),
        }
