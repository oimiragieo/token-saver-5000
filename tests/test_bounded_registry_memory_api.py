"""BoundedDict cap regression test for MemoryAPI (A1 bounded registries)."""

from __future__ import annotations

import pytest

from src.memory_api import MemoryAPI


def test_memory_api_evicts_oldest_entry_past_cap():
    api = MemoryAPI(max_entries=2)

    first = api.add_memory(text="first memory")
    api.add_memory(text="second memory")
    third = api.add_memory(text="third memory")

    all_memories = api.list_memories()
    ids = {entry["memory_id"] for entry in all_memories}

    assert len(all_memories) == 2
    assert first["memory_id"] not in ids
    assert third["memory_id"] in ids

    with pytest.raises(ValueError):
        api.delete_memory(first["memory_id"])
