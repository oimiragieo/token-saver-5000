"""
Tests for Memory Optimization (v0.6.0)

This module tests the memory optimization features added in v0.6.0:
- ONNX embedding manager (quantized models, 60-70% memory reduction)
- TF-IDF fallback (minimal memory, adequate quality)
- LRU embedding cache (60-80% hit rate for repeated queries)
- Tier switching logic with automatic fallback

Test Categories:
- ONNX Embedding Tests (6 tests)
- TF-IDF Embedding Tests (6 tests)
- LRU Cache Tests (6 tests)
- Tier Switching Tests (6 tests)

Total: 24 tests
"""

import numpy as np
import pytest
import tempfile
import os

# Import tier-specific managers with graceful handling
try:
    from src.embeddings_onnx import ONNXEmbeddingManager

    # Check if onnxruntime AND optimum are actually available
    import onnxruntime  # noqa: F401
    import optimum.onnxruntime  # noqa: F401

    ONNX_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    ONNX_AVAILABLE = False

try:
    from src.embeddings_tfidf import TFIDFEmbeddingManager

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

try:
    from src.embedding_cache import LRUEmbeddingCache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from src.embeddings import EmbeddingManager, EmbeddingTier

    TIER_SWITCHING_AVAILABLE = True
except ImportError:
    TIER_SWITCHING_AVAILABLE = False


# ===========================
# ONNX Embedding Tests
# ===========================


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX dependencies not installed")
class TestONNXEmbeddings:
    """Test ONNX-optimized embedding functionality."""

    def test_onnx_manager_initialization(self):
        """Test ONNX embedding manager initialization."""
        manager = ONNXEmbeddingManager(model_name="sentence-transformers/all-MiniLM-L6-v2")

        assert manager.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert manager.quantized is True
        assert manager._initialized is False  # Lazy initialization

    def test_onnx_encode_single_text(self):
        """Test ONNX encoding of single text."""
        manager = ONNXEmbeddingManager()

        embedding = manager.encode("hello world")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)  # all-MiniLM-L6-v2 dimension
        assert np.linalg.norm(embedding) > 0  # Non-zero embedding

    def test_onnx_encode_multiple_texts(self):
        """Test ONNX encoding of multiple texts."""
        manager = ONNXEmbeddingManager()

        texts = ["hello world", "foo bar", "test document"]
        embeddings = manager.encode(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert not np.array_equal(embeddings[0], embeddings[1])  # Different embeddings

    def test_onnx_normalization(self):
        """Test ONNX embedding normalization."""
        manager = ONNXEmbeddingManager()

        # Normalized embeddings (default)
        embedding_normalized = manager.encode("hello world", normalize=True)
        norm = np.linalg.norm(embedding_normalized)
        assert np.isclose(norm, 1.0, atol=1e-5)  # L2 norm = 1

        # Non-normalized embeddings
        embedding_raw = manager.encode("hello world", normalize=False)
        norm_raw = np.linalg.norm(embedding_raw)
        assert not np.isclose(norm_raw, 1.0, atol=1e-5)  # L2 norm != 1

    def test_onnx_get_embedding_dim(self):
        """Test ONNX embedding dimension retrieval."""
        manager = ONNXEmbeddingManager()

        dim = manager.get_embedding_dim()
        assert dim == 384  # all-MiniLM-L6-v2

    def test_onnx_memory_usage(self):
        """Test ONNX memory usage tracking."""
        manager = ONNXEmbeddingManager()
        manager.encode("test")  # Initialize model

        memory_stats = manager.get_memory_usage()

        assert "rss_mb" in memory_stats
        assert "vms_mb" in memory_stats
        assert "percent" in memory_stats
        assert memory_stats["rss_mb"] > 0


# ===========================
# TF-IDF Embedding Tests
# ===========================


@pytest.mark.skipif(not TFIDF_AVAILABLE, reason="sklearn not installed")
class TestTFIDFEmbeddings:
    """Test TF-IDF fallback embedding functionality."""

    def test_tfidf_manager_initialization(self):
        """Test TF-IDF embedding manager initialization."""
        manager = TFIDFEmbeddingManager(max_features=5000, ngram_range=(1, 2))

        assert manager.max_features == 5000
        assert manager.ngram_range == (1, 2)
        assert manager._fitted is False

    def test_tfidf_fit_and_encode(self):
        """Test TF-IDF fit and encode workflow."""
        manager = TFIDFEmbeddingManager(max_features=100)

        # Fit on corpus
        corpus = ["hello world", "foo bar", "hello foo"]
        manager.fit(corpus)

        assert manager._fitted is True
        assert manager.get_vocabulary_size() > 0

        # Encode texts
        embeddings = manager.encode(["hello world", "foo bar"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] <= 100  # Max features

    def test_tfidf_fit_transform(self):
        """Test TF-IDF fit_transform in one step."""
        manager = TFIDFEmbeddingManager(max_features=100)

        corpus = ["hello world", "foo bar", "hello foo"]
        embeddings = manager.fit_transform(corpus)

        assert manager._fitted is True
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, manager.get_embedding_dim())

    def test_tfidf_auto_fit(self):
        """Test TF-IDF auto-fit on first encode."""
        manager = TFIDFEmbeddingManager(max_features=100)

        # Encode without explicit fit (auto-fit)
        embeddings = manager.encode(["hello world", "foo bar"])

        assert manager._fitted is True
        assert isinstance(embeddings, np.ndarray)

    def test_tfidf_get_vocabulary_size(self):
        """Test TF-IDF vocabulary size retrieval."""
        manager = TFIDFEmbeddingManager(max_features=100)

        # Before fit
        assert manager.get_vocabulary_size() == 0

        # After fit
        manager.fit(["hello world", "foo bar", "hello foo"])
        assert manager.get_vocabulary_size() > 0

    def test_tfidf_memory_usage(self):
        """Test TF-IDF memory usage tracking."""
        manager = TFIDFEmbeddingManager(max_features=100)
        manager.fit(["hello world", "foo bar"])

        memory_stats = manager.get_memory_usage()

        assert "vocabulary_mb" in memory_stats
        assert "idf_weights_mb" in memory_stats
        assert "total_mb" in memory_stats
        assert memory_stats["fitted"] is True


# ===========================
# LRU Cache Tests
# ===========================


@pytest.mark.skipif(not CACHE_AVAILABLE, reason="msgpack not installed")
class TestLRUEmbeddingCache:
    """Test LRU embedding cache functionality."""

    def test_cache_initialization(self):
        """Test LRU cache initialization."""
        cache = LRUEmbeddingCache(max_entries=100, ttl_seconds=3600)

        assert cache.max_entries == 100
        assert cache.ttl_seconds == 3600
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_put_and_get(self):
        """Test cache put and get operations."""
        cache = LRUEmbeddingCache(max_entries=100)

        # Put embedding
        embedding = np.random.rand(384)
        cache.put("hello world", embedding)

        # Get embedding
        retrieved = cache.get("hello world")
        assert retrieved is not None
        assert np.array_equal(retrieved, embedding)

        # Cache stats
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache = LRUEmbeddingCache(max_entries=100)

        # Get non-existent key
        retrieved = cache.get("nonexistent")
        assert retrieved is None

        # Cache stats
        stats = cache.get_stats()
        assert stats["misses"] == 1

    def test_cache_eviction(self):
        """Test LRU eviction when max_entries exceeded."""
        cache = LRUEmbeddingCache(max_entries=3)

        # Add 4 embeddings (should evict oldest)
        for i in range(4):
            embedding = np.random.rand(384)
            cache.put(f"text_{i}", embedding)

        # Check eviction
        stats = cache.get_stats()
        assert stats["entries"] == 3  # Max entries
        assert stats["evictions"] == 1

        # Oldest should be evicted
        assert cache.get("text_0") is None
        assert cache.get("text_3") is not None

    def test_cache_batch_operations(self):
        """Test batch get and put operations."""
        cache = LRUEmbeddingCache(max_entries=100)

        # Put batch
        texts = ["hello", "world", "foo"]
        embeddings = [np.random.rand(384) for _ in range(3)]
        cache.put_batch(texts, embeddings)

        # Get batch
        retrieved_embeddings, miss_indices = cache.get_batch(texts)

        assert len(miss_indices) == 0  # All cache hits
        assert len(retrieved_embeddings) == 3

    def test_cache_persistence(self):
        """Test cache disk persistence."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".msgpack", delete=False) as f:
            persist_path = f.name

        try:
            # Create cache with persistence
            cache = LRUEmbeddingCache(max_entries=100, persist_path=persist_path)

            # Add embeddings
            embedding = np.random.rand(384)
            cache.put("test_text", embedding)

            # Manually persist
            cache.persist()

            # Verify file exists
            assert os.path.exists(persist_path)

            # Load from disk in new cache instance
            cache2 = LRUEmbeddingCache(max_entries=100, persist_path=persist_path)
            retrieved = cache2.get("test_text")

            # Should load from disk
            assert retrieved is not None

        finally:
            if os.path.exists(persist_path):
                os.unlink(persist_path)


# ===========================
# Tier Switching Tests
# ===========================


@pytest.mark.skipif(not TIER_SWITCHING_AVAILABLE, reason="Tier switching not available")
class TestEmbeddingTierSwitching:
    """Test embedding tier switching and fallback logic."""

    @pytest.fixture(autouse=True)
    def _isolate_embedding_manager_singleton(self):
        """Reset the process-global EmbeddingManager singleton around each test.

        EmbeddingManager is a singleton whose tier LOCKS at first construction
        (src/embeddings.py EmbeddingManager.__new__): a later construction that
        asks for a different tier does NOT switch it — it just returns the
        already-locked instance. This class's tests construct EmbeddingManager
        with an explicit tier and assert `get_tier()` reflects it, which only
        holds if THIS test's construction is the one that wins the lock.

        In the full suite (order-randomized), whichever OTHER test happens to
        construct an EmbeddingManager first (anywhere in the file order) wins
        the tier lock for the whole process, so these assertions fail
        nondeterministically depending on run order — reproduced via
        `-p no:randomly` bisection: whichever of these tests runs first in a
        given ordering passes (it wins the lock), the others fail.

        conftest.py's `_reset_shared_state` only restores the *attributes* of
        whatever singleton instance already existed before the test — it
        can't undo an already-locked tier, because nothing about the locked
        instance changes during this test (the conflicting construction is a
        no-op by design). So the leaked global here is the singleton's
        *identity/lock*, not just an attribute value, and needs
        `reset_for_testing()` (full instance clear), not an attribute
        snapshot/restore.

        Fix: snapshot the existing singleton, clear it via
        `reset_for_testing()` so this test's construction wins the lock, then
        restore the prior singleton afterward so other tests in the suite are
        unaffected by whichever tier this test happened to lock in.
        """
        orig = EmbeddingManager._instance
        EmbeddingManager.reset_for_testing()
        yield
        EmbeddingManager._instance = orig

    def test_tier_initialization(self):
        """Test tier-aware EmbeddingManager initialization."""
        # Note: This tests default tier only (STANDARD) since ONNX/TFIDF
        # may not be available in CI environment
        manager = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=True)

        assert manager.get_tier() == EmbeddingTier.STANDARD

    def test_tier_switching(self):
        """Test switching between tiers."""
        manager = EmbeddingManager()

        # Switch to ONNX tier
        manager.set_tier(EmbeddingTier.ONNX)
        assert manager.get_tier() == EmbeddingTier.ONNX

        # Switch to TF-IDF tier
        manager.set_tier(EmbeddingTier.TFIDF)
        assert manager.get_tier() == EmbeddingTier.TFIDF

    def test_standard_tier_encoding(self):
        """Test encoding with STANDARD tier."""
        manager = EmbeddingManager(tier=EmbeddingTier.STANDARD)

        embeddings = manager.encode(["hello world", "foo bar"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Some embedding dimension

    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not available")
    def test_onnx_tier_encoding(self):
        """Test encoding with ONNX tier."""
        manager = EmbeddingManager(tier=EmbeddingTier.ONNX)

        embeddings = manager.encode(["hello world", "foo bar"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)

    @pytest.mark.skipif(not TFIDF_AVAILABLE, reason="TF-IDF not available")
    def test_tfidf_tier_encoding(self):
        """Test encoding with TF-IDF tier."""
        manager = EmbeddingManager(tier=EmbeddingTier.TFIDF)

        # Need to fit first for TF-IDF
        texts = ["hello world", "foo bar", "test document"]
        manager._tfidf_manager = TFIDFEmbeddingManager()
        manager._tfidf_manager.fit(texts)

        embeddings = manager.encode(["hello world", "foo bar"])

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == 2

    def test_cache_stats_with_tier(self):
        """Test cache stats include tier information."""
        manager = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=True)

        # Explicitly set tier (singleton may have different tier from previous test)
        manager.set_tier(EmbeddingTier.STANDARD)

        stats = manager.get_cache_stats()

        assert "tier" in stats
        assert stats["tier"] == "standard"
        assert "cache_enabled" in stats
        assert stats["cache_enabled"] is True
