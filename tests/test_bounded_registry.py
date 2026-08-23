"""Tests for src.bounded_registry.BoundedDict (A1 bounded-registry fix).

See docs/plans/2026-08-24-a1-bounded-registries.md for the full design
rationale (FIFO-not-LRU, the bulk-mutation bypass rounds, etc).
"""

from __future__ import annotations

import pytest

from src.bounded_registry import BoundedDict


def test_construct_and_insert_under_cap_no_eviction():
    """Basic sanity: construct, insert under the cap, nothing evicted."""
    d = BoundedDict(max_items=3)
    d["a"] = 1
    d["b"] = 2
    assert len(d) == 2
    assert d["a"] == 1
    assert d["b"] == 2
    assert list(d.keys()) == ["a", "b"]


def test_max_items_less_than_one_raises_value_error():
    with pytest.raises(ValueError):
        BoundedDict(max_items=0)
    with pytest.raises(ValueError):
        BoundedDict(max_items=-1)


def test_eviction_past_cap_drops_oldest_inserted():
    d = BoundedDict(max_items=2)
    d["a"] = 1
    d["b"] = 2
    d["c"] = 3
    assert len(d) == 2
    assert "a" not in d
    assert set(d.keys()) == {"b", "c"}


def test_fifo_position_proof_reassign_does_not_move_key():
    """The load-bearing invariant: re-assigning an EXISTING key does NOT move
    it to the "newest" slot. A buggy 'update moves the key to newest'
    implementation would instead evict B here; only true FIFO-by-original-
    insertion-order evicts A.
    """
    d = BoundedDict(max_items=2)
    d["a"] = 1
    d["b"] = 2
    assert len(d) == 2

    # Re-assign an existing key -- must NOT change its original insertion slot.
    d["a"] = "updated"
    assert len(d) == 2
    assert d["a"] == "updated"

    # Insert a third key -- A (oldest by original insertion) must be evicted,
    # not B.
    d["c"] = 3
    assert len(d) == 2
    assert "a" not in d
    assert set(d.keys()) == {"b", "c"}
    assert d["b"] == 2
    assert d["c"] == 3


def test_bulk_update_enforces_cap():
    """dict.update() is a C bulk-op that bypasses __setitem__ by default --
    BoundedDict.update() must route through per-key eviction anyway.
    """
    d = BoundedDict(max_items=2)
    d.update({"a": 1, "b": 2, "c": 3})
    assert len(d) == 2
    assert set(d.keys()) == {"b", "c"}


def test_setdefault_enforces_cap():
    d = BoundedDict(max_items=2)
    d["a"] = 1
    d["b"] = 2
    d.setdefault("z", 1)
    assert len(d) == 2
    assert "a" not in d
    assert set(d.keys()) == {"b", "z"}

    # setdefault on an EXISTING key returns the existing value and does not
    # evict anything.
    existing = d.setdefault("b", "should-not-be-used")
    assert existing == 2
    assert len(d) == 2


def test_ior_bulk_operator_enforces_cap():
    """dict/OrderedDict's `|=` in-place-or is a third C bulk-op that bypasses
    __setitem__ independently of update().
    """
    d = BoundedDict(max_items=2)
    d["a"] = 1
    d |= {"y": 1, "z": 2}
    assert len(d) == 2
    assert set(d.keys()) == {"y", "z"}


def test_get_stats_reports_total_max_and_approaching_limit():
    d = BoundedDict(max_items=11)
    for i in range(10):
        d[f"k{i}"] = i
    stats = d.get_stats()
    assert stats["total"] == 10
    assert stats["max_items"] == 11
    assert stats["approaching_limit"] is True

    d2 = BoundedDict(max_items=10)
    d2["a"] = 1
    stats2 = d2.get_stats()
    assert stats2["total"] == 1
    assert stats2["approaching_limit"] is False


def test_read_paths_unaffected_below_cap():
    """Reads (.get(), bracket, iteration, len()) are the plain, unmodified
    OrderedDict/dict behavior -- nothing special-cased.
    """
    d = BoundedDict(max_items=5)
    d["a"] = 1
    d["b"] = 2
    assert d.get("a") == 1
    assert d.get("missing") is None
    assert d.get("missing", "default") == "default"
    assert list(d) == ["a", "b"]
    assert len(d) == 2
