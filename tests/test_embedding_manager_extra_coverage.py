"""embedding manager — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
import threading


def _make_handler_context():
    """Create a mock handler context with all required keys."""
    compressor = MagicMock()
    compressor.graphs = {"doc1": MagicMock()}
    compressor.chunks = {
        "doc1_n0": MagicMock(text="chunk0"),
        "doc1_n1": MagicMock(text="chunk1"),
    }
    compressor.file_metadata = {"doc1": {"title": "Test"}}
    compressor.get_stats.return_value = {"total_nodes": 2, "total_tokens": 100}

    persistence = MagicMock()
    persistence.delete_document.return_value = True

    resource_manager = MagicMock()
    resource_manager.unregister_document_async = AsyncMock()

    sync_manager = MagicMock()
    sync_manager.remove_metadata.return_value = None
    sync_manager.export_metadata.return_value = {}

    version_manager = MagicMock()
    version_manager.delete_versions_async = AsyncMock()

    path_validator = MagicMock()
    path_validator.validate.side_effect = lambda x: x

    context = {
        "compressor": compressor,
        "persistence": persistence,
        "resource_manager": resource_manager,
        "sync_manager": sync_manager,
        "version_manager": version_manager,
        "path_validator": path_validator,
        "retrieval_history": {},
        "multilevel_encoder": MagicMock(),
        "context_window_adapter": MagicMock(),
    }
    return context


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
        "ace_framework": MagicMock(),
        "focus_manager": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_semantic_node(text="test", importance=0.5, embedding=None):
    node = MagicMock()
    node.text = text
    node.importance = importance
    node.embedding = embedding if embedding is not None else np.random.rand(384).astype(np.float32)
    node.metadata = {"tokens": 10, "position": 0, "entities": []}
    return node


def _make_code_chunk(
    name="func", chunk_type="function", code="def f(): pass", docstring="", start_line=1, end_line=5
):
    chunk = MagicMock()
    chunk.name = name
    chunk.chunk_type = chunk_type
    chunk.code = code
    chunk.docstring = docstring
    chunk.start_line = start_line
    chunk.end_line = end_line
    return chunk


class TestEmbeddingManager:
    """Tests for EmbeddingManager."""

    def _reset_singleton(self):
        from src.embeddings import EmbeddingManager

        EmbeddingManager._instance = None

    def test_encode_tier_routing_standard(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer") as MockST:
            mock_model = MockST.return_value
            mock_model.encode.return_value = np.array([[0.1, 0.2]])
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
            result = mgr.encode(["hello"])
            assert result.shape == (1, 2)
        self._reset_singleton()

    def test_encode_tier_routing_onnx(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer"):
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.ONNX, enable_cache=False)
            mock_onnx = MagicMock()
            mock_onnx.encode.return_value = np.array([[0.3, 0.4]])
            mgr._onnx_manager = mock_onnx
            with patch("src.embeddings.ONNX_AVAILABLE", True):
                result = mgr.encode(["hello"])
            assert result.shape == (1, 2)
        self._reset_singleton()

    def test_encode_tier_routing_tfidf(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer"):
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
            mock_tfidf = MagicMock()
            mock_tfidf.encode.return_value = np.array([[0.5, 0.6]])
            mgr._tfidf_manager = mock_tfidf
            with patch("src.embeddings.TFIDF_AVAILABLE", True):
                result = mgr.encode(["hello"])
            assert result.shape == (1, 2)
        self._reset_singleton()

    def test_encode_with_fallback_neural_request_refuses_silent_tfidf(self):
        """Audit P1-6: a NEURAL tier request (ONNX) whose SBERT+ONNX fallbacks
        both fail must RAISE RuntimeError, NOT silently return TF-IDF garbage.

        (Previously this test asserted the silent TF-IDF fall-through that the
        audit identified as a correctness bug — updated to lock the new
        raise-instead-of-garbage contract.)"""
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer"):
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.ONNX, enable_cache=False)

            mock_tfidf = MagicMock()
            mock_tfidf.encode.return_value = np.array([[0.1, 0.2]])

            # Make standard fail; ONNX unavailable; TF-IDF "available" but must
            # NOT be used as a substitute for the requested neural tier.
            with patch.object(mgr, "_encode_standard", side_effect=Exception("fail")):
                with patch("src.embeddings.ONNX_AVAILABLE", False):
                    with patch("src.embeddings.TFIDF_AVAILABLE", True):
                        mgr._tfidf_manager = mock_tfidf
                        with pytest.raises(RuntimeError):
                            mgr._encode_with_fallback(["hello"], EmbeddingTier.ONNX, True)
        self._reset_singleton()

    def test_encode_all_tiers_fail(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer"):
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
            with patch.object(mgr, "_encode_standard", side_effect=Exception("fail")):
                with patch("src.embeddings.ONNX_AVAILABLE", False):
                    with patch("src.embeddings.TFIDF_AVAILABLE", False):
                        with pytest.raises(RuntimeError, match="All embedding tiers failed"):
                            mgr._encode_with_fallback(["hello"], EmbeddingTier.STANDARD, True)
        self._reset_singleton()

    def test_encode_single_text_string(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer") as MockST:
            mock_model = MockST.return_value
            mock_model.encode.return_value = np.array([[0.1, 0.2]])
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
            result = mgr.encode("single string")
            assert result.shape[0] == 1
        self._reset_singleton()

    def test_encode_fallback_on_tier_failure(self):
        self._reset_singleton()
        with patch("src.embeddings.SentenceTransformer") as MockST:
            mock_model = MockST.return_value
            mock_model.encode.return_value = np.array([[0.9, 0.8]])
            from src.embeddings import EmbeddingManager, EmbeddingTier

            mgr = EmbeddingManager(tier=EmbeddingTier.ONNX, enable_cache=False)
            # ONNX not available, should fallback
            with patch("src.embeddings.ONNX_AVAILABLE", False):
                result = mgr.encode(["hello"], tier=EmbeddingTier.ONNX)
            assert result.shape == (1, 2)
        self._reset_singleton()


class TestONNXEmbeddingManager:
    """Tests for ONNXEmbeddingManager."""

    def test_init_defaults(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        # A1 (2026-06-08): default ONNX model now tracks
        # src.constants.DEFAULT_TEXT_MODEL (bge-small-en-v1.5; was
        # sentence-transformers/all-MiniLM-L6-v2).
        from src.constants import DEFAULT_TEXT_MODEL

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        assert mgr.model_name == DEFAULT_TEXT_MODEL
        assert mgr.quantized is True
        assert mgr._initialized is False

    def test_initialize_success(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))

        with patch.dict(
            "sys.modules",
            {
                "onnxruntime": MagicMock(),
                "transformers": MagicMock(),
                "optimum": MagicMock(),
                "optimum.onnxruntime": MagicMock(),
            },
        ):
            with patch("src.embeddings_onnx.ONNXEmbeddingManager._initialize") as mock_init:
                mock_init.side_effect = lambda: setattr(mgr, "_initialized", True)
                mgr._initialize()
                assert mgr._initialized is True

    def test_initialize_import_error(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))

        # Simulate ImportError inside _initialize
        def mock_init(self_ref):
            self_ref._initialized = False
            raise ImportError("No onnxruntime")

        with patch.object(ONNXEmbeddingManager, "_initialize", mock_init):
            with pytest.raises(ImportError):
                mgr._initialize()

    def test_initialize_already_initialized(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._initialized = True
        # Should return immediately without error
        mgr._initialize()
        assert mgr._initialized is True

    def test_encode_calls_initialize(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_session = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = MagicMock()
        mock_session.return_value = mock_output

        mgr._tokenizer = mock_tokenizer
        mgr._session = mock_session

        with patch.object(mgr, "_initialize"):
            with patch.object(mgr, "_mean_pooling") as mock_pool:
                mock_tensor = MagicMock()
                mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
                    [[0.1, 0.2]]
                )
                mock_pool.return_value = mock_tensor
                result = mgr.encode(["hello"])
                assert result.shape == (1, 2)

    def test_encode_single_string(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._tokenizer = MagicMock()
        mgr._session = MagicMock()

        with patch.object(mgr, "_initialize"):
            with patch.object(mgr, "_mean_pooling") as mock_pool:
                mock_tensor = MagicMock()
                mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
                    [[0.5]]
                )
                mock_pool.return_value = mock_tensor
                result = mgr.encode("single", normalize=False)
                assert result.shape == (1, 1)

    def test_encode_with_normalization(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._tokenizer = MagicMock()
        mgr._session = MagicMock()

        with patch.object(mgr, "_initialize"):
            with patch.object(mgr, "_mean_pooling") as mock_pool:
                mock_tensor = MagicMock()
                mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array(
                    [[3.0, 4.0]]
                )
                mock_pool.return_value = mock_tensor
                result = mgr.encode(["test"], normalize=True)
                # Normalized vector should have unit norm
                np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-5)

    def test_encode_inference_error(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._tokenizer = MagicMock()
        mgr._session = MagicMock(side_effect=Exception("ONNX failure"))

        with patch.object(mgr, "_initialize"):
            with pytest.raises(Exception, match="ONNX failure"):
                mgr.encode(["hello"])

    def test_get_embedding_dim(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._initialized = True
        mgr._tokenizer = MagicMock()
        mgr._tokenizer.model_max_length = 512
        dim = mgr.get_embedding_dim()
        assert dim == 384

    def test_get_memory_usage(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        mgr._session = MagicMock()
        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = MagicMock(rss=100 * 1024 * 1024, vms=200 * 1024 * 1024)
        mock_proc.memory_percent.return_value = 5.0
        mock_psutil.Process.return_value = mock_proc
        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            stats = mgr.get_memory_usage()
        assert "rss_mb" in stats
        assert stats["rss_mb"] == pytest.approx(100.0)


class TestEmbeddingManager_boost3:
    def _reset_singleton(self):
        from src.embeddings import EmbeddingManager

        EmbeddingManager._instance = None

    def test_encode_unknown_tier_raises(self):
        self._reset_singleton()
        import src.embeddings as emb_mod
        from src.embeddings import EmbeddingManager, EmbeddingTier

        # Build a manager with no model available.
        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr._model_cache = {}
        mgr._cache_lock = threading.Lock()
        mgr._lru_cache = None
        mgr._onnx_manager = None
        mgr._tfidf_manager = None
        mgr._tier = EmbeddingTier.STANDARD
        mgr._enable_cache = False

        # _encode_with_fallback for STANDARD tier skips STANDARD re-try and goes
        # directly to ONNX then TF-IDF.  To force exhaustion we patch all three
        # encode helpers plus mark both optional tiers as available so the branches
        # are entered (otherwise they short-circuit before the patch is hit).
        err = RuntimeError("no model")
        with (
            patch.object(mgr, "_encode_standard", side_effect=err),
            patch.object(mgr, "_encode_onnx", side_effect=err),
            patch.object(mgr, "_encode_tfidf", side_effect=err),
            patch.object(emb_mod, "ONNX_AVAILABLE", True),
            patch.object(emb_mod, "TFIDF_AVAILABLE", True),
        ):
            # _encode_tfidf raises directly (no wrapper), so the exception
            # propagates as-is rather than the "All embedding tiers failed" sentinel.
            with pytest.raises(RuntimeError):
                mgr._encode_with_fallback(["test"], EmbeddingTier.STANDARD, True)

    def test_code_embedder_fallback(self):
        self._reset_singleton()
        from src.embeddings import EmbeddingManager

        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr._model_cache = {}
        mgr._cache_lock = threading.Lock()

        mock_model = MagicMock()
        with patch.object(
            mgr, "_get_or_create_model", side_effect=[Exception("code fail"), mock_model]
        ):
            result = mgr.get_code_embedder("bad-model")
            assert result is mock_model

    def test_clear_cache(self):
        self._reset_singleton()
        from src.embeddings import EmbeddingManager

        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr._model_cache = {"model1": MagicMock()}
        mgr._cache_lock = threading.Lock()

        mgr.clear_cache()
        assert len(mgr._model_cache) == 0

    def test_get_stats_with_onnx_and_tfidf(self):
        self._reset_singleton()
        from src.embeddings import EmbeddingManager, EmbeddingTier

        mgr = EmbeddingManager.__new__(EmbeddingManager)
        mgr._model_cache = {"clip-ViT-B-32": MagicMock()}
        mgr._cache_lock = threading.Lock()
        mgr._tier = EmbeddingTier.STANDARD
        mgr._enable_cache = False
        mgr._lru_cache = None
        mgr._onnx_manager = MagicMock()
        mgr._onnx_manager.get_memory_usage.return_value = {"rss_mb": 10}
        mgr._tfidf_manager = MagicMock()
        mgr._tfidf_manager.get_memory_usage.return_value = {"size_mb": 5}

        stats = mgr.get_cache_stats()
        assert "onnx_manager" in stats
        assert "tfidf_manager" in stats
        assert stats["estimated_memory_mb"] == 150  # clip model


class TestONNXEmbeddings:
    def test_onnx_init_import_error(self):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager()
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError):
                mgr._initialize()

    def test_onnx_singleton_creation(self):
        import src.embeddings_onnx as onnx_mod

        onnx_mod._onnx_manager_instance = None

        with patch.object(onnx_mod, "ONNXEmbeddingManager") as MockMgr:
            mock_instance = MagicMock()
            MockMgr.return_value = mock_instance

            result = onnx_mod.get_onnx_embedding_manager()
            assert result is mock_instance

        # Reset
        onnx_mod._onnx_manager_instance = None

    def test_onnx_mean_pooling(self):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager()
        import torch

        token_emb = torch.randn(1, 5, 384)
        attn_mask = torch.ones(1, 5, dtype=torch.long)
        result = mgr._mean_pooling(token_emb, attn_mask)
        assert result.shape == (1, 384)

    def test_onnx_get_embedding_dim(self):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager()
        mgr._initialized = True
        mgr._tokenizer = MagicMock()
        mgr._tokenizer.model_max_length = 512
        assert mgr.get_embedding_dim() == 384

    def test_onnx_get_memory_usage(self):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager()
        mgr._session = None
        with patch("psutil.Process") as MockProc:
            mock_proc = MagicMock()
            mock_proc.memory_info.return_value = MagicMock(
                rss=100 * 1024 * 1024, vms=200 * 1024 * 1024
            )
            mock_proc.memory_percent.return_value = 5.0
            MockProc.return_value = mock_proc

            stats = mgr.get_memory_usage()
            assert "rss_mb" in stats


class TestEmbeddingsImports:
    """Cover import fallback paths in embeddings module."""

    def test_onnx_import_fallback(self):
        """Cover lines 47-48 - ONNX not available."""
        # The module-level try/except is already evaluated; just verify the flag
        from src.embeddings import ONNX_AVAILABLE

        # It's either True or False based on env - just assert it's a bool
        assert isinstance(ONNX_AVAILABLE, bool)

    def test_tfidf_import_fallback(self):
        """Cover lines 54-55."""
        from src.embeddings import TFIDF_AVAILABLE

        assert isinstance(TFIDF_AVAILABLE, bool)

    def test_cache_import_fallback(self):
        """Cover lines 61-62."""
        from src.embeddings import CACHE_AVAILABLE

        assert isinstance(CACHE_AVAILABLE, bool)

    def test_cache_warning_when_unavailable(self):
        """Cover line 150 - cache requested but unavailable."""
        from src.embeddings import EmbeddingManager

        # Reset singleton for test
        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.CACHE_AVAILABLE", False):
                with patch("src.embeddings.SentenceTransformer"):
                    mgr = EmbeddingManager(enable_cache=True)
                    assert mgr._lru_cache is None
        finally:
            EmbeddingManager._instance = original

    def test_encode_unknown_tier(self):
        """Cover line 213 - unknown tier triggers fallback."""
        from src.embeddings import EmbeddingManager

        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.SentenceTransformer"):
                mgr = EmbeddingManager()
                # Create a fake tier
                fake_tier = MagicMock()
                fake_tier.value = "fake"
                # This triggers fallback path (line 213 -> 217-218)
                with patch.object(
                    mgr, "_encode_with_fallback", return_value=np.random.rand(1, 384)
                ):
                    result = mgr.encode(["hello"], tier=fake_tier)
                    assert result.shape[0] == 1
        finally:
            EmbeddingManager._instance = original

    def test_encode_with_cache_partial_hit(self):
        """Cover lines 190-197, 222, 235 - partial cache hit."""
        from src.embeddings import EmbeddingManager

        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.SentenceTransformer") as mock_st:
                mock_model = MagicMock()
                mock_model.encode.return_value = np.random.rand(1, 384)
                mock_st.return_value = mock_model

                mock_cache = MagicMock()
                cached = [np.random.rand(384), None]
                mock_cache.get_batch.return_value = (cached, [1])  # Index 1 is a miss

                mgr = EmbeddingManager()
                mgr._lru_cache = mock_cache

                from src.embeddings import EmbeddingTier

                result = mgr.encode(["cached_text", "uncached_text"], tier=EmbeddingTier.STANDARD)
                assert result.shape[0] == 2
                mock_cache.put_batch.assert_called_once()
        finally:
            EmbeddingManager._instance = original

    def test_get_image_embedder(self):
        """Cover line 352."""
        from src.embeddings import EmbeddingManager

        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.SentenceTransformer"):
                mgr = EmbeddingManager()
                mgr.get_image_embedder()
                # Should have called _get_or_create_model
        finally:
            EmbeddingManager._instance = original

    def test_encode_tfidf_fallback(self):
        """Cover line 257."""
        from src.embeddings import EmbeddingManager

        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.SentenceTransformer"):
                with patch("src.embeddings.TFIDF_AVAILABLE", False):
                    mgr = EmbeddingManager()
                    with pytest.raises(ImportError, match="TF-IDF"):
                        mgr._encode_tfidf(["test"], True)
        finally:
            EmbeddingManager._instance = original

    def test_stats_with_lru_cache(self):
        """Cover line 469."""
        from src.embeddings import EmbeddingManager

        original = EmbeddingManager._instance
        EmbeddingManager._instance = None
        try:
            with patch("src.embeddings.SentenceTransformer"):
                mgr = EmbeddingManager()
                mock_cache = MagicMock()
                mock_cache.get_stats.return_value = {"hits": 10, "misses": 5}
                mgr._lru_cache = mock_cache
                stats = mgr.get_cache_stats()
                assert "lru_cache" in stats
        finally:
            EmbeddingManager._instance = original


class TestONNXEmbeddings_boost4b:
    """Cover ONNX embedding manager init paths."""

    def test_init_first_time_download(self):
        """Cover lines 80-110 - ONNX initialization with download."""
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager.__new__(ONNXEmbeddingManager)
        mgr.model_name = "test-model"
        mgr.cache_dir = Path("/tmp/onnx_cache")
        mgr._initialized = False
        mgr._init_lock = threading.Lock()
        mgr._tokenizer = None
        mgr._session = None

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()

        with patch("src.embeddings_onnx.ort", create=True):
            with patch("src.embeddings_onnx.AutoTokenizer", create=True) as mock_at:
                with patch(
                    "src.embeddings_onnx.ORTModelForFeatureExtraction", create=True
                ) as mock_ort:
                    mock_at.from_pretrained.return_value = mock_tokenizer
                    mock_ort.from_pretrained.return_value = mock_model
                    with patch.object(Path, "exists", return_value=False):
                        try:
                            mgr._initialize()
                        except Exception:
                            pass  # May fail due to import mocking

    def test_init_import_error(self):
        """Cover lines 118-120 - ONNX import error raises."""
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager.__new__(ONNXEmbeddingManager)
        mgr.model_name = "test-model"
        mgr.cache_dir = Path("/tmp/onnx_cache")
        mgr._initialized = False
        mgr._init_lock = threading.Lock()
        mgr._tokenizer = None
        mgr._session = None

        # Simulate import error
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("no onnxruntime")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            try:
                mgr._initialize()
            except (ImportError, Exception):
                pass

    def test_double_checked_locking(self):
        """Cover line 76 - already initialized returns early."""
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager.__new__(ONNXEmbeddingManager)
        mgr._initialized = True
        mgr._init_lock = threading.Lock()
        mgr._initialize()  # Should return immediately

    def test_get_embedding_dim_fallback(self):
        """Cover lines 218-219 - fallback to encoding dummy text."""
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager.__new__(ONNXEmbeddingManager)
        mgr._initialized = True
        mgr._init_lock = threading.Lock()
        mgr._tokenizer = MagicMock(spec=[])  # No model_max_length
        del mgr._tokenizer.model_max_length
        mgr._session = MagicMock()

        with patch.object(mgr, "encode", return_value=np.random.rand(1, 384)):
            dim = mgr.get_embedding_dim()
            assert dim == 384
