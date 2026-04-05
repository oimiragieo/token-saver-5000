"""Tests for the tee/recovery system."""

from __future__ import annotations

import json
import os
import tempfile


from src.tee_recovery import (
    DEFAULT_COMPRESSION_THRESHOLD,
    TEE_MODE_ALWAYS,
    TEE_MODE_FAILURES,
    TEE_MODE_NEVER,
    TeeEntry,
    TeeStore,
    create_tee_store,
)

# ---------------------------------------------------------------------------
# TeeEntry basics
# ---------------------------------------------------------------------------


class TestTeeEntry:
    def test_entry_size_bytes(self):
        entry = TeeEntry(
            entry_id="abc",
            original_text="hello world",
            compressed_text="hello",
            compression_pct=55.0,
            source="test",
        )
        assert entry.size_bytes == len("hello world".encode("utf-8"))

    def test_entry_to_dict_has_required_keys(self):
        entry = TeeEntry(
            entry_id="abc",
            original_text="hello world",
            compressed_text="hello",
            compression_pct=55.0,
            source="test",
            command_hint="echo",
        )
        d = entry.to_dict()
        assert d["entry_id"] == "abc"
        assert d["compression_pct"] == 55.0
        assert d["source"] == "test"
        assert d["command_hint"] == "echo"
        assert "timestamp" in d
        assert d["original_length"] == 11
        assert d["compressed_length"] == 5

    def test_entry_to_dict_excludes_text(self):
        entry = TeeEntry(
            entry_id="abc",
            original_text="secret",
            compressed_text="s",
            compression_pct=80.0,
            source="test",
        )
        d = entry.to_dict()
        assert "original_text" not in d
        assert "compressed_text" not in d


# ---------------------------------------------------------------------------
# TeeStore - mode logic
# ---------------------------------------------------------------------------


class TestTeeStoreModes:
    def test_never_mode_rejects_all(self):
        store = TeeStore(mode=TEE_MODE_NEVER)
        assert store.should_tee(99.0) is False
        assert store.should_tee(0.0) is False

    def test_always_mode_accepts_all(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        assert store.should_tee(0.0) is True
        assert store.should_tee(99.0) is True

    def test_failures_mode_respects_threshold(self):
        store = TeeStore(mode=TEE_MODE_FAILURES, compression_threshold=80.0)
        assert store.should_tee(79.9) is False
        assert store.should_tee(80.0) is True
        assert store.should_tee(95.0) is True

    def test_default_mode_is_failures(self):
        store = TeeStore()
        assert store.mode == TEE_MODE_FAILURES

    def test_default_threshold(self):
        store = TeeStore()
        assert store.compression_threshold == DEFAULT_COMPRESSION_THRESHOLD


# ---------------------------------------------------------------------------
# TeeStore - store and retrieve
# ---------------------------------------------------------------------------


class TestTeeStoreOperations:
    def test_store_returns_id_when_tee_accepted(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("original text", "compressed", 90.0, "test")
        assert eid is not None
        assert isinstance(eid, str)
        assert len(eid) == 12

    def test_store_returns_none_when_below_threshold(self):
        store = TeeStore(mode=TEE_MODE_FAILURES, compression_threshold=80.0)
        eid = store.store("original text", "compressed", 50.0, "test")
        assert eid is None

    def test_get_returns_stored_entry(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("original text", "compressed", 90.0, "cli_optimizer")
        entry = store.get(eid)
        assert entry is not None
        assert entry.original_text == "original text"
        assert entry.compressed_text == "compressed"
        assert entry.source == "cli_optimizer"

    def test_get_original_returns_text(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("full content here", "short", 85.0, "proxy")
        text = store.get_original(eid)
        assert text == "full content here"

    def test_get_original_returns_none_for_missing(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        assert store.get_original("nonexistent") is None

    def test_get_returns_none_for_missing(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        assert store.get("nonexistent") is None

    def test_entry_count(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        assert store.entry_count == 0
        store.store("a", "b", 90.0, "test")
        assert store.entry_count == 1
        store.store("c", "d", 90.0, "test")
        assert store.entry_count == 2

    def test_total_size_bytes(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        store.store("hello", "h", 80.0, "test")
        assert store.total_size_bytes == len("hello".encode("utf-8"))

    def test_duplicate_id_updates_entry(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid1 = store.store("same text", "v1", 80.0, "test")
        eid2 = store.store("same text", "v2", 90.0, "test")
        assert eid1 == eid2
        assert store.entry_count == 1
        entry = store.get(eid1)
        assert entry.compressed_text == "v2"

    def test_store_with_metadata(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("text", "t", 85.0, "test", metadata={"tool": "pytest"})
        entry = store.get(eid)
        assert entry.metadata == {"tool": "pytest"}

    def test_store_with_command_hint(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("text", "t", 85.0, "test", command_hint="git diff")
        entry = store.get(eid)
        assert entry.command_hint == "git diff"


# ---------------------------------------------------------------------------
# TeeStore - list and delete
# ---------------------------------------------------------------------------


class TestTeeStoreListDelete:
    def test_list_entries_newest_first(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        store.store("first", "f", 90.0, "test")
        store.store("second", "s", 90.0, "test")
        store.store("third", "t", 90.0, "test")
        entries = store.list_entries()
        assert len(entries) == 3
        # newest first
        assert entries[0]["original_length"] == len("third")
        assert entries[2]["original_length"] == len("first")

    def test_list_entries_with_limit(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        for i in range(10):
            store.store(f"text_{i}", f"t{i}", 90.0, "test")
        entries = store.list_entries(limit=3)
        assert len(entries) == 3

    def test_list_entries_filter_by_source(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        store.store("cli_text", "c", 90.0, "cli_optimizer")
        store.store("proxy_text", "p", 90.0, "proxy")
        store.store("cli_text2", "c2", 90.0, "cli_optimizer")

        cli_entries = store.list_entries(source="cli_optimizer")
        assert len(cli_entries) == 2
        proxy_entries = store.list_entries(source="proxy")
        assert len(proxy_entries) == 1

    def test_delete_removes_entry(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        eid = store.store("deleteme", "d", 90.0, "test")
        assert store.delete(eid) is True
        assert store.get(eid) is None
        assert store.entry_count == 0

    def test_delete_returns_false_for_missing(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        assert store.delete("nonexistent") is False

    def test_clear_removes_all(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        store.store("a", "b", 90.0, "test")
        store.store("c", "d", 90.0, "test")
        count = store.clear()
        assert count == 2
        assert store.entry_count == 0


# ---------------------------------------------------------------------------
# TeeStore - LRU eviction
# ---------------------------------------------------------------------------


class TestTeeStoreEviction:
    def test_evicts_oldest_when_max_entries_exceeded(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS, max_entries=3)
        ids = []
        for i in range(5):
            eid = store.store(f"text_{i}_unique", f"t{i}", 90.0, "test")
            ids.append(eid)

        assert store.entry_count == 3
        # First 2 should be evicted
        assert store.get(ids[0]) is None
        assert store.get(ids[1]) is None
        # Last 3 should remain
        assert store.get(ids[2]) is not None
        assert store.get(ids[3]) is not None
        assert store.get(ids[4]) is not None

    def test_evicts_by_size_limit(self):
        # 1KB limit
        store = TeeStore(mode=TEE_MODE_ALWAYS, max_entries=100, max_size_mb=0.001)
        big_text = "x" * 600
        store.store(big_text + "_1", "t", 90.0, "test")
        store.store(big_text + "_2", "t", 90.0, "test")
        store.store(big_text + "_3", "t", 90.0, "test")
        # Only 1-2 should fit in ~1KB
        assert store.entry_count <= 2


# ---------------------------------------------------------------------------
# TeeStore - persistence
# ---------------------------------------------------------------------------


class TestTeeStorePersistence:
    def test_persist_and_load_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=tmpdir)
            eid = store.store("persistent text", "pt", 90.0, "test", command_hint="echo")

            # Verify file was created
            path = os.path.join(tmpdir, f"{eid}.json")
            assert os.path.exists(path)

            # Verify JSON content
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["original_text"] == "persistent text"
            assert data["source"] == "test"
            assert data["command_hint"] == "echo"

    def test_new_store_loads_from_disk(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=tmpdir)
            eid = store1.store("recoverable", "r", 90.0, "test")

            # Create a new store pointing at the same dir
            store2 = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=tmpdir)
            # Entry is not in memory yet but should load from disk
            entry = store2.get(eid)
            assert entry is not None
            assert entry.original_text == "recoverable"

    def test_delete_removes_persisted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=tmpdir)
            eid = store.store("delete me", "d", 90.0, "test")
            path = os.path.join(tmpdir, f"{eid}.json")
            assert os.path.exists(path)

            store.delete(eid)
            assert not os.path.exists(path)

    def test_clear_removes_persisted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TeeStore(mode=TEE_MODE_ALWAYS, persist_dir=tmpdir)
            store.store("a", "b", 90.0, "test")
            store.store("c", "d", 90.0, "test")
            json_files = list(os.listdir(tmpdir))
            assert len(json_files) == 2

            store.clear()
            json_files = list(os.listdir(tmpdir))
            assert len(json_files) == 0

    def test_eviction_removes_persisted_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = TeeStore(mode=TEE_MODE_ALWAYS, max_entries=2, persist_dir=tmpdir)
            store.store("text_1_u", "t", 90.0, "test")
            store.store("text_2_u", "t", 90.0, "test")
            store.store("text_3_u", "t", 90.0, "test")
            # Only 2 files should remain
            json_files = list(os.listdir(tmpdir))
            assert len(json_files) == 2


# ---------------------------------------------------------------------------
# TeeStore - stats
# ---------------------------------------------------------------------------


class TestTeeStoreStats:
    def test_stats_empty_store(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        s = store.stats()
        assert s["mode"] == TEE_MODE_ALWAYS
        assert s["entry_count"] == 0
        assert s["oldest_entry"] is None
        assert s["newest_entry"] is None

    def test_stats_populated_store(self):
        store = TeeStore(mode=TEE_MODE_ALWAYS)
        store.store("hello", "h", 90.0, "test")
        s = store.stats()
        assert s["entry_count"] == 1
        assert s["total_size_bytes"] == 5
        assert s["oldest_entry"] is not None
        assert s["newest_entry"] is not None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateTeeStore:
    def test_default_factory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = create_tee_store()
        assert store.mode == TEE_MODE_FAILURES
        assert store.compression_threshold == DEFAULT_COMPRESSION_THRESHOLD
        assert store.max_entries == 50

    def test_factory_with_mode_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        store = create_tee_store(mode=TEE_MODE_ALWAYS)
        assert store.mode == TEE_MODE_ALWAYS

    def test_factory_respects_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("TEE_MODE", "always")
        monkeypatch.setenv("TEE_COMPRESSION_THRESHOLD", "70")
        monkeypatch.setenv("TEE_MAX_ENTRIES", "25")
        monkeypatch.setenv("TEE_MAX_SIZE_MB", "50")
        store = create_tee_store()
        assert store.mode == "always"
        assert store.compression_threshold == 70.0
        assert store.max_entries == 25
        assert store.max_size_bytes == int(50 * 1024 * 1024)

    def test_factory_no_persist(self):
        store = create_tee_store(persist=False)
        assert store.persist_dir is None


# ---------------------------------------------------------------------------
# generate_id
# ---------------------------------------------------------------------------


class TestGenerateId:
    def test_deterministic(self):
        id1 = TeeStore.generate_id("same text", "same_source")
        id2 = TeeStore.generate_id("same text", "same_source")
        assert id1 == id2

    def test_different_text_different_id(self):
        id1 = TeeStore.generate_id("text1", "source")
        id2 = TeeStore.generate_id("text2", "source")
        assert id1 != id2

    def test_different_source_different_id(self):
        id1 = TeeStore.generate_id("text", "source1")
        id2 = TeeStore.generate_id("text", "source2")
        assert id1 != id2

    def test_id_length_is_12(self):
        eid = TeeStore.generate_id("anything", "any")
        assert len(eid) == 12
