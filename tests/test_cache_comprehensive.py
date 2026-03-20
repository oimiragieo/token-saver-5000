"""
Comprehensive Cache Tests (v1.0.0 - Phase 1)

Tests for the embedding cache to increase coverage from 86% → 95%+.

This module tests critical production scenarios for the LRU embedding cache:
- TTL expiration and eviction
- Disk persistence error handling
- Edge cases in cache operations
- Singleton pattern and memory safety

Test Categories:
- TTL Expiration Tests (6 tests)
- Persistence Error Handling (6 tests)
- Edge Cases & Boundaries (5 tests)
- Singleton & Memory Tests (3 tests)

Total: 20 comprehensive tests
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.embedding_cache import LRUEmbeddingCache, get_embedding_cache


# ===========================
# Fixtures
# ===========================


@pytest.fixture
def temp_cache_file():
    """Create temporary file for cache persistence."""
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return {
        "text1": np.random.rand(384),
        "text2": np.random.rand(384),
        "text3": np.random.rand(384),
    }


# ===========================
# TTL Expiration Tests
# ===========================


class TestTTLExpiration:
    """Test TTL (Time-To-Live) expiration functionality."""

    def test_get_with_expired_entry(self, sample_embeddings):
        """Test that expired entries are removed and return None (lines 126-131)."""
        # Create cache with 1-second TTL
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=1)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Verify it's cached
        assert cache.get("test_text") is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired and return None
        result = cache.get("test_text")
        assert result is None

        # Should count as a miss
        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_get_with_non_expired_entry(self, sample_embeddings):
        """Test that non-expired entries are returned (lines 126-131)."""
        # Create cache with 10-second TTL
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=10)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Should still be valid
        result = cache.get("test_text")
        assert result is not None
        np.testing.assert_array_equal(result, sample_embeddings["text1"])

    def test_ttl_disabled_entries_never_expire(self, sample_embeddings):
        """Test that entries never expire when TTL is None."""
        # Create cache with no TTL
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=None)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Wait (would expire with TTL)
        time.sleep(0.1)

        # Should still be valid
        result = cache.get("test_text")
        assert result is not None

    def test_ttl_expiration_boundary(self, sample_embeddings):
        """Test TTL expiration at exact boundary."""
        # Create cache with 0.5-second TTL
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=0.5)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Wait just under TTL
        time.sleep(0.4)
        assert cache.get("test_text") is not None

        # Wait past TTL
        time.sleep(0.2)
        assert cache.get("test_text") is None

    def test_ttl_with_multiple_entries(self, sample_embeddings):
        """Test TTL expiration with multiple entries at different ages."""
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=0.5)

        # Add first entry
        cache.put("text1", sample_embeddings["text1"])

        # Wait a bit
        time.sleep(0.3)

        # Add second entry (younger)
        cache.put("text2", sample_embeddings["text2"])

        # Wait for first to expire
        time.sleep(0.3)

        # First should be expired
        assert cache.get("text1") is None

        # Second should still be valid
        assert cache.get("text2") is not None

    def test_ttl_updates_on_put(self, sample_embeddings):
        """Test that TTL resets when entry is updated."""
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=0.5)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Wait almost to expiration
        time.sleep(0.4)

        # Update entry (resets TTL)
        cache.put("test_text", sample_embeddings["text2"])

        # Wait again
        time.sleep(0.4)

        # Should still be valid (TTL was reset)
        result = cache.get("test_text")
        assert result is not None


# ===========================
# Persistence Error Handling Tests
# ===========================


class TestPersistenceErrorHandling:
    """Test error handling in disk persistence operations."""

    def test_save_to_disk_with_no_persist_path(self, sample_embeddings):
        """Test that _save_to_disk returns early when persist_path is None (line 289)."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=None)

        cache.put("test_text", sample_embeddings["text1"])

        # Should not raise error
        cache._save_to_disk()

    def test_save_to_disk_with_write_error(self, temp_cache_file, sample_embeddings):
        """Test handling of write errors during persistence (lines 314-315)."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        cache.put("test_text", sample_embeddings["text1"])

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            # Should handle error gracefully (log and continue)
            cache._save_to_disk()

    def test_save_to_disk_with_msgpack_error(self, temp_cache_file, sample_embeddings):
        """Test handling of msgpack serialization errors (lines 314-315)."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        cache.put("test_text", sample_embeddings["text1"])

        # Mock msgpack.pack to raise exception
        with patch("msgpack.pack", side_effect=Exception("Msgpack error")):
            # Should handle error gracefully
            cache._save_to_disk()

    def test_load_from_disk_with_no_file(self, temp_cache_file):
        """Test that _load_from_disk returns early when file doesn't exist (line 320)."""
        # Remove file if it exists
        if Path(temp_cache_file).exists():
            Path(temp_cache_file).unlink()

        # Create cache pointing to non-existent file
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        # Should not raise error and cache should be empty
        assert len(cache._cache) == 0

    def test_load_from_disk_with_corrupted_file(self, temp_cache_file):
        """Test handling of corrupted cache files."""
        # Create corrupted file
        with open(temp_cache_file, "wb") as f:
            f.write(b"CORRUPTED DATA")

        # Should handle error gracefully during initialization
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        # Cache should be empty (failed to load)
        assert len(cache._cache) == 0

    def test_load_from_disk_with_msgpack_error(self, temp_cache_file):
        """Test handling of msgpack deserialization errors."""
        # Create file with valid msgpack but invalid structure
        import msgpack

        with open(temp_cache_file, "wb") as f:
            msgpack.pack({"invalid": "structure"}, f)

        # Mock msgpack.unpack to raise exception
        with patch("msgpack.unpack", side_effect=Exception("Unpack error")):
            # Should handle error gracefully
            cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)
            assert len(cache._cache) == 0


# ===========================
# Edge Cases & Boundaries Tests
# ===========================


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_put_updates_existing_key(self, sample_embeddings):
        """Test that put() deletes existing key before updating (line 169)."""
        cache = LRUEmbeddingCache(max_entries=10)

        # Add initial entry
        cache.put("test_text", sample_embeddings["text1"])

        # Verify it's there
        result1 = cache.get("test_text")
        np.testing.assert_array_equal(result1, sample_embeddings["text1"])

        # Update with new embedding
        cache.put("test_text", sample_embeddings["text2"])

        # Should have new embedding
        result2 = cache.get("test_text")
        np.testing.assert_array_equal(result2, sample_embeddings["text2"])

        # Should still have only 1 entry
        stats = cache.get_stats()
        assert stats["entries"] == 1

    def test_clear_method_with_logging(self, sample_embeddings, caplog):
        """Test clear() method logs properly (lines 247-249)."""
        import logging

        caplog.set_level(logging.INFO)

        cache = LRUEmbeddingCache(max_entries=10)

        # Add some entries
        for i in range(3):
            cache.put(f"text{i}", sample_embeddings["text1"])

        # Clear cache
        cache.clear()

        # Should be empty
        stats = cache.get_stats()
        assert stats["entries"] == 0

        # Should have logged
        assert "Embedding cache cleared" in caplog.text

    def test_get_stats_with_zero_requests(self):
        """Test get_stats() when there are no requests (hit_rate calculation)."""
        cache = LRUEmbeddingCache(max_entries=10)

        stats = cache.get_stats()

        # Should not crash on division by zero
        assert stats["hit_rate"] == 0.0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_with_max_entries_zero(self, sample_embeddings):
        """Test cache behavior with max_entries=0 (reveals division by zero bug)."""
        cache = LRUEmbeddingCache(max_entries=0)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Should immediately evict (cache size > 0)
        # Note: get_stats() will raise ZeroDivisionError due to bug in utilization calculation
        # This test documents the bug - max_entries=0 is an edge case that should be handled
        with pytest.raises(ZeroDivisionError):
            cache.get_stats()

        # Eviction should have occurred
        assert cache._evictions > 0

    def test_persist_on_destructor(self, temp_cache_file, sample_embeddings):
        """Test that cache persists on __del__ (destructor)."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        # Add entry
        cache.put("test_text", sample_embeddings["text1"])

        # Manually call destructor
        cache.__del__()

        # File should exist
        assert Path(temp_cache_file).exists()

        # Load in new cache
        cache2 = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        # Should have loaded entry
        assert cache2.get("test_text") is not None

    def test_destructor_swallows_shutdown_errors(self, temp_cache_file, sample_embeddings):
        """Destructor should not surface persistence errors during shutdown."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)
        cache.put("test_text", sample_embeddings["text1"])

        with patch.object(cache, "_save_to_disk", side_effect=NameError("open missing")):
            cache.__del__()


# ===========================
# Singleton & Memory Tests
# ===========================


class TestSingletonAndMemory:
    """Test singleton pattern and memory management."""

    def test_get_embedding_cache_singleton(self):
        """Test that get_embedding_cache returns singleton (lines 372-379)."""
        # Reset singleton
        import src.embedding_cache

        src.embedding_cache._cache_instance = None

        # Get cache instance
        cache1 = get_embedding_cache(max_entries=100)

        # Get again
        cache2 = get_embedding_cache(max_entries=200)  # Different params

        # Should be same instance (singleton)
        assert cache1 is cache2

        # Should use first initialization params
        assert cache1.max_entries == 100

    def test_get_embedding_cache_with_persist_path(self, temp_cache_file):
        """Test singleton creation with persist_path (lines 372-379)."""
        # Reset singleton
        import src.embedding_cache

        src.embedding_cache._cache_instance = None

        # Create with persist path
        cache = get_embedding_cache(persist_path=temp_cache_file, ttl_seconds=60)

        assert cache.persist_path == Path(temp_cache_file)
        assert cache.ttl_seconds == 60

    def test_memory_usage_calculation(self, sample_embeddings):
        """Test memory usage calculation in get_stats()."""
        cache = LRUEmbeddingCache(max_entries=10)

        # Add multiple entries
        for i in range(5):
            cache.put(f"text{i}", sample_embeddings["text1"], {"metadata": f"value{i}"})

        stats = cache.get_stats()

        # Should have non-zero memory usage
        assert stats["memory_mb"] > 0

        # Should track utilization
        assert stats["utilization"] == 0.5  # 5/10 entries


# ===========================
# Additional Coverage Tests
# ===========================


class TestAdditionalCoverage:
    """Additional tests to ensure comprehensive coverage."""

    def test_get_with_default_value(self, sample_embeddings):
        """Test get() with default value parameter."""
        cache = LRUEmbeddingCache(max_entries=10)

        default_embedding = np.zeros(384)

        # Get non-existent with default
        result = cache.get("nonexistent", default=default_embedding)

        # Should return default
        np.testing.assert_array_equal(result, default_embedding)

    def test_batch_operations_with_ttl(self, sample_embeddings):
        """Test batch operations respect TTL expiration."""
        cache = LRUEmbeddingCache(max_entries=10, ttl_seconds=0.5)

        # Add batch
        texts = ["text1", "text2", "text3"]
        embeddings = [sample_embeddings["text1"] for _ in range(3)]
        cache.put_batch(texts, embeddings)

        # Get batch immediately
        results1, miss_indices1 = cache.get_batch(texts)
        assert len(miss_indices1) == 0

        # Wait for expiration
        time.sleep(0.6)

        # Get batch after expiration
        results2, miss_indices2 = cache.get_batch(texts)
        assert len(miss_indices2) == 3  # All expired

    def test_concurrent_access_simulation(self, sample_embeddings):
        """Test that lock prevents race conditions (basic simulation)."""
        import threading

        cache = LRUEmbeddingCache(max_entries=100)

        results = []

        def add_entries(start_idx):
            for i in range(start_idx, start_idx + 10):
                cache.put(f"text{i}", sample_embeddings["text1"])
                results.append(i)

        # Create threads
        threads = []
        for i in range(3):
            t = threading.Thread(target=add_entries, args=(i * 10,))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # Should have all entries
        assert len(results) == 30

        # Cache should be consistent
        stats = cache.get_stats()
        assert stats["entries"] <= 100

    def test_persist_method_calls_save_to_disk(self, temp_cache_file, sample_embeddings):
        """Test that persist() method triggers save."""
        cache = LRUEmbeddingCache(max_entries=10, persist_path=temp_cache_file)

        cache.put("test_text", sample_embeddings["text1"])

        # Call persist
        cache.persist()

        # File should exist with data
        assert Path(temp_cache_file).exists()
        assert Path(temp_cache_file).stat().st_size > 0

    def test_cache_key_generation_consistency(self):
        """Test that _make_key generates consistent hashes."""
        cache = LRUEmbeddingCache(max_entries=10)

        # Same text should produce same key
        key1 = cache._make_key("hello world")
        key2 = cache._make_key("hello world")

        assert key1 == key2

        # Different text should produce different keys
        key3 = cache._make_key("different text")

        assert key1 != key3
