from __future__ import annotations
from collections import OrderedDict
from src.structured_logging import get_logger  # SAME import ace_context_manager.py

# uses -- NOT stdlib logging.Logger.
# get_logger()'s structured logger
# accepts arbitrary kwargs on
# .info(); a plain
# logging.Logger.info() does not
# and would raise on this call.
logger = get_logger("semantic-modulator")


class BoundedDict(OrderedDict):
    """Generic FIFO-evicting OrderedDict: past max_items, the OLDEST-INSERTED key
    is dropped first (never the least-recently-READ key -- see the
    "FIFO, not LRU" note in docs/plans/2026-08-24-a1-bounded-registries.md for
    why this is deliberately NOT read-recency LRU). Mirrors the eviction
    strategy MAX_FILE_SYNC_ENTRIES already ships in file_sync_manager.py.

    Re-assigning an EXISTING key does not move it -- it keeps its original
    insertion slot, so repeatedly updating one record cannot dodge eviction
    forever. Reads (__getitem__, .get(), iteration, len()) are the plain,
    unmodified OrderedDict/dict behavior: nothing here special-cases them.

    CAP ENFORCEMENT IS ONLY GUARANTEED THROUGH __setitem__, update(), AND
    setdefault() -- explicitly overridden below. dict.update()/setdefault()
    are C-level bulk operations on CPython's dict/OrderedDict that do NOT
    call an overridden __setitem__ by default (codex round 4 caught this --
    none of the 5 real consumers in this fix call these, all 5 use bracket
    assignment only, but overriding is cheap and correct rather than relying
    on "nobody currently does the unsafe thing").
    """

    def __init__(self, max_items: int):
        # Deliberately NO *args/**kwargs bulk-init parameter (codex round 4's
        # first option) -- construct empty, then assign, so there is exactly
        # ONE write path (__setitem__) to reason about, not two.
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items
        super().__init__()

    def __setitem__(self, key, value):
        is_new_key = key not in self
        super().__setitem__(key, value)
        if is_new_key and len(self) > self.max_items:
            oldest_key, _ = next(iter(self.items()))
            del self[oldest_key]
            logger.info(
                "bounded_registry_evicted", evicted_key=oldest_key, max_items=self.max_items
            )

    def update(self, *args, **kwargs):
        # dict.update() is a C bulk-op that bypasses __setitem__ -- route
        # through the real dict/OrderedDict.update() machinery via a plain
        # dict built from the args, then assign key-by-key so eviction still
        # fires per-key exactly like a loop of `self[k] = v` would.
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def setdefault(self, key, default=None):
        # dict.setdefault() is also a C bulk-op bypassing __setitem__.
        if key in self:
            return self[key]
        self[key] = default
        return default

    def __ior__(self, other):
        # dict/OrderedDict's `|=` in-place-or is a third C bulk-op that
        # bypasses __setitem__ independently of update() (codex round 5 --
        # the last CPython MutableMapping bulk-write path; __setitem__,
        # update(), setdefault(), and __ior__ are now the complete set of
        # ways to insert into this class, all routing through the same
        # per-key eviction logic).
        self.update(other)
        return self

    def get_stats(self) -> dict:
        return {
            "total": len(self),
            "max_items": self.max_items,
            "approaching_limit": len(self) / self.max_items > 0.9,
        }
