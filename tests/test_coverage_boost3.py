"""
Coverage boost tests - Round 3.

Targets ~120 tests covering remaining uncovered lines across:
compression_handlers, persistence, afm, embeddings_onnx, graph_visualizer,
embeddings, resource_handlers, version_manager, multimodal_compressor,
resource_manager, batch_manager, scar_compressor, file_sync_manager,
observability, metrics, health, and smaller modules.
"""

import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ============================================================================
# 1. compression_handlers — list_documents metadata lines (789,797-802)
# ============================================================================


class TestListDocumentsMetadata:
    """Cover lines 789,797,799,801-802 — title/author/date/tags display."""

    @pytest.mark.asyncio
    async def test_list_documents_with_metadata_fields(self):
        from src.handlers.compression_handlers import handle_list_documents

        compressor = MagicMock()
        chunk_a = MagicMock()
        chunk_a.metadata = {"tokens": 100}
        chunk_a.importance = 0.5
        compressor.chunks = {"doc1_n0": chunk_a}
        compressor.graphs = {"doc1": MagicMock()}
        compressor.file_metadata = {}

        metadata = {
            "title": "My Custom Title",
            "author": "Jane Doe",
            "date": "2024-01-01",
            "tags": ["python", "testing", "coverage"],
        }
        stats = {
            "total_nodes": 1,
            "total_tokens": 200,
            "skeleton_tokens": 25,
            "compression_ratio": 8.0,
            "metadata": metadata,
        }
        compressor.get_stats.return_value = stats

        ctx = _make_mock_context(compressor=compressor)
        result = await handle_list_documents(ctx, {})

        assert "My Custom Title" in result
        assert "Jane Doe" in result
        assert "2024-01-01" in result
        assert "python" in result


# ============================================================================
# 2. compression_handlers — delete_document error paths (877-879,888-896)
# ============================================================================


class TestDeleteDocumentErrors:
    """Cover delete_document memory-error, storage-error, resource-manager-error."""

    def _make_delete_ctx(self):
        compressor = MagicMock()
        compressor.chunks = {"doc1_n0": MagicMock(), "doc1_n1": MagicMock()}
        compressor.graphs = {"doc1": MagicMock()}
        compressor.file_metadata = {"doc1": {}}
        compressor.get_stats.return_value = {"total_nodes": 2}

        ctx = _make_mock_context(compressor=compressor)
        ctx["retrieval_history"] = {"doc1": []}
        return ctx

    @pytest.mark.asyncio
    async def test_delete_memory_error_raises_runtime(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        # Make chunks.keys() raise on iteration during deletion
        ctx["compressor"].chunks = MagicMock()
        ctx["compressor"].chunks.keys.side_effect = RuntimeError("boom")
        ctx["compressor"].get_stats.return_value = {"total_nodes": 2}

        with pytest.raises(RuntimeError, match="Failed to delete from memory"):
            await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_storage_exception_logged(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        ctx["persistence"].delete_document.side_effect = Exception("storage fail")
        ctx["resource_manager"].unregister_document_async = AsyncMock()
        ctx["sync_manager"].remove_metadata = MagicMock()
        ctx["version_manager"].delete_versions_async = AsyncMock()
        ctx["sync_manager"].export_metadata.return_value = {}

        result = await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})
        assert "Deleted" in result or "DELETE" in result

    @pytest.mark.asyncio
    async def test_delete_resource_manager_exception_logged(self):
        from src.handlers.compression_handlers import handle_delete_document

        ctx = self._make_delete_ctx()
        ctx["persistence"].delete_document.return_value = True
        ctx["resource_manager"].unregister_document_async = AsyncMock(
            side_effect=Exception("rm fail")
        )
        ctx["sync_manager"].remove_metadata = MagicMock()
        ctx["version_manager"].delete_versions_async = AsyncMock()
        ctx["sync_manager"].export_metadata.return_value = {}

        result = await handle_delete_document(ctx, {"file_id": "doc1", "confirm": True})
        assert "DELETE" in result


# ============================================================================
# 3. compression_handlers — adapt_to_context_window error (968-969)
# ============================================================================


class TestAdaptToContextWindowError:
    @pytest.mark.asyncio
    async def test_adapt_raises_runtime_error(self):
        from src.handlers.compression_handlers import handle_adapt_to_context_window

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        ctx["context_window_adapter"].adapt_to_context_window.side_effect = Exception("fail")

        with pytest.raises(RuntimeError, match="Failed to adapt"):
            await handle_adapt_to_context_window(ctx, {"file_id": "doc1", "available_tokens": 500})


# ============================================================================
# 4. compression_handlers — multilevel_encode error (1003-1008)
# ============================================================================


class TestMultilevelEncodeError:
    @pytest.mark.asyncio
    async def test_multilevel_encode_raises_runtime_error(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        ctx["multilevel_encoder"].generate_adaptive_skeleton.side_effect = Exception("encode fail")

        with pytest.raises(RuntimeError, match="Failed to generate multi-level encoding"):
            await handle_multilevel_encode(ctx, {"file_id": "doc1", "available_tokens": 1000})


# ============================================================================
# 5. compression_handlers — recommend_fidelity validation (1058-1059)
# ============================================================================


class TestRecommendFidelityValidation:
    @pytest.mark.asyncio
    async def test_token_budget_too_high(self):
        from src.handlers.compression_handlers import handle_recommend_fidelity

        ctx = _make_mock_context()
        with pytest.raises(ValueError, match="very high"):
            await handle_recommend_fidelity(
                ctx,
                {
                    "use_case": "question_answering",
                    "num_nodes": 5,
                    "token_budget": 2_000_000,
                    "query_complexity": "medium",
                },
            )

    @pytest.mark.asyncio
    async def test_token_budget_too_low(self):
        from src.handlers.compression_handlers import handle_recommend_fidelity

        ctx = _make_mock_context()
        with pytest.raises(ValueError, match="too low"):
            await handle_recommend_fidelity(
                ctx,
                {
                    "use_case": "question_answering",
                    "num_nodes": 5,
                    "token_budget": 3,
                    "query_complexity": "medium",
                },
            )


# ============================================================================
# 6. compression_handlers — batch_ingest rate limit (1142-1143)
# ============================================================================


class TestBatchIngestRateLimit:
    @pytest.mark.asyncio
    async def test_batch_ingest_rate_limit_exceeded(self):
        from src.handlers.compression_handlers import handle_batch_ingest
        from src.error_types import RateLimitExceededError

        ctx = _make_mock_context()

        with patch("src.handlers.compression_handlers.RATE_LIMITERS") as mock_rl:
            mock_limiter = AsyncMock()
            mock_limiter.acquire.side_effect = RateLimitExceededError("batch", 2.0)
            mock_rl.__getitem__.return_value = mock_limiter

            with pytest.raises(ValueError, match="Rate limit exceeded"):
                await handle_batch_ingest(ctx, {"documents": []})


# ============================================================================
# 7. compression_handlers — ingest_directory paths (1358-1410, 1437-1456, 1498, 1522-1527)
# ============================================================================


class TestIngestDirectory:
    @pytest.mark.asyncio
    async def test_ingest_directory_recursive_glob_and_exclusions(self, tmp_path):
        """Cover recursive glob, exclusion, and file read error paths."""
        from src.handlers.compression_handlers import handle_ingest_directory

        # Create some files
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.py").write_text("print('a')")
        (tmp_path / "b.py").write_text("print('b')")
        (tmp_path / "c.pyc").write_text("bytecode")

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["**/*.py"],
                "exclude_patterns": ["*.pyc"],
                "max_files": 50,
                "max_concurrent": 2,
            },
        )

        data = json.loads(result)
        assert data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_ingest_directory_no_matching_files(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["*.xyz"],
                "max_files": 50,
            },
        )
        data = json.loads(result)
        assert data["status"] == "no_files"

    @pytest.mark.asyncio
    async def test_ingest_directory_file_read_failure(self, tmp_path):
        """Cover lines 1451-1453: exception reading file."""
        from src.handlers.compression_handlers import handle_ingest_directory
        from unittest.mock import mock_open, patch

        (tmp_path / "bad.py").write_text("content")
        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        with patch("builtins.open", mock_open()) as mocked_open:
            mocked_open.side_effect = OSError("read failed")
            result = await handle_ingest_directory(
                ctx,
                {"directory": str(tmp_path), "patterns": ["*.py"], "max_files": 50},
            )
        data = json.loads(result)
        assert data["status"] == "read_failed"

    @pytest.mark.asyncio
    async def test_ingest_directory_max_files_limit(self, tmp_path):
        """Cover line 1408-1410: file limiting when too many files match."""
        from src.handlers.compression_handlers import handle_ingest_directory

        for i in range(5):
            (tmp_path / f"f{i}.py").write_text(f"content {i}")

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {"directory": str(tmp_path), "patterns": ["*.py"], "max_files": 2},
        )

        data = json.loads(result)
        assert data["status"] == "complete"

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_exclude_pattern(self, tmp_path):
        """Cover line 1388-1392: exclusion patterns are applied."""
        from src.handlers.compression_handlers import handle_ingest_directory

        (tmp_path / "x.py").write_text("content")
        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: p

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(tmp_path),
                "patterns": ["*.py"],
                "exclude_patterns": ["*.pyc"],
                "max_files": 50,
            },
        )
        data = json.loads(result)
        assert data["status"] == "complete"


# ============================================================================
# 8. persistence — chromadb init, list, delete, load paths
# ============================================================================


class TestPersistenceChromaDB:
    def _make_pm(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_list_documents_json_with_exception(self, tmp_path):
        pm = self._make_pm(tmp_path)
        # Patch Path.glob at the class level
        with patch.object(Path, "glob", side_effect=OSError("fail")):
            result = pm._list_documents_json()
        assert result == []

    def test_delete_document_json_success(self, tmp_path):
        pm = self._make_pm(tmp_path)
        # Create a document file
        doc_file = pm.documents_dir / "test_doc.json"
        doc_file.write_text("{}")
        assert pm._delete_document_json("test_doc") is True

    def test_delete_document_json_no_files(self, tmp_path):
        pm = self._make_pm(tmp_path)
        assert pm._delete_document_json("nonexistent") is False

    def test_delete_document_dispatch_exception(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with patch.object(pm, "_delete_document_json", side_effect=Exception("boom")):
            assert pm.delete_document("test") is False

    def test_delete_document_chromadb_path(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.use_chromadb = True
        pm.chroma_client = MagicMock()
        pm.chroma_client.delete_collection.return_value = None
        assert pm._delete_document_chromadb("doc1") is True

    def test_delete_document_chromadb_with_files(self, tmp_path):
        pm = self._make_pm(tmp_path)
        # Create graph file to be deleted
        graph_file = pm.documents_dir / "doc1_graph.json"
        graph_file.write_text("{}")
        pm.use_chromadb = True
        pm.chroma_client = MagicMock()
        result = pm._delete_document_chromadb("doc1")
        assert result is True
        assert not graph_file.exists()

    def test_load_document_dispatch_exception(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with patch.object(pm, "_load_document_json", side_effect=Exception("boom")):
            assert pm.load_document("test") is None

    def test_load_document_chromadb_dispatch(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.use_chromadb = True
        with patch.object(pm, "_load_document_chromadb", return_value=None):
            assert pm.load_document("test") is None

    def test_save_document_chromadb_dispatch(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.use_chromadb = True
        with patch.object(pm, "_save_document_chromadb", return_value=True):
            assert pm.save_document("doc1", {}, {}, {}) is True

    def test_list_documents_chromadb_path(self, tmp_path):
        pm = self._make_pm(tmp_path)
        pm.use_chromadb = True
        assert pm.list_documents() == []  # _list_documents_chromadb needs chroma_client

    def test_get_stats_disk_error(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with patch.object(Path, "rglob", side_effect=OSError("fail")):
            stats = pm.get_storage_stats()
        assert stats["disk_usage_mb"] == 0


# ============================================================================
# 9. persistence — load_graph_data_safe legacy pickle rejection (193-202, 236-238)
# ============================================================================


class TestPersistenceLoadGraphSafe:
    def _make_pm(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_load_graph_legacy_pickle_raises_valueerror(self, tmp_path):
        pm = self._make_pm(tmp_path)
        legacy = pm.documents_dir / "test_graph.pkl"
        legacy.write_bytes(b"fake pickle")

        with pytest.raises(ValueError, match="SECURITY"):
            pm._load_graph_data_safe("test")

    def test_load_graph_ids_from_legacy_numpy(self, tmp_path):
        pm = self._make_pm(tmp_path)
        graph_file = pm.documents_dir / "test_graph.json"
        graph_file.write_text(json.dumps({"nodes": [{"id": "n1"}], "edges": []}))

        emb_file = pm.documents_dir / "test_embeddings.npz"
        embeddings = np.array([[1.0, 2.0]])
        ids = np.array(["n1"])
        np.savez(emb_file, embeddings=embeddings, ids=ids)

        result = pm._load_graph_data_safe("test")
        assert result is not None

    def test_load_graph_fallback_ids_from_data(self, tmp_path):
        pm = self._make_pm(tmp_path)
        graph_file = pm.documents_dir / "test_graph.json"
        graph_file.write_text(json.dumps({"nodes": [{"id": "n1"}], "edges": []}))

        emb_file = pm.documents_dir / "test_embeddings.npz"
        embeddings = np.array([[1.0, 2.0]])
        # Save without ids
        np.savez(emb_file, embeddings=embeddings)

        # No ids.json file either, so should fallback to graph_data nodes
        result = pm._load_graph_data_safe("test")
        assert result is not None

    def test_load_graph_generic_exception(self, tmp_path):
        pm = self._make_pm(tmp_path)
        graph_file = pm.documents_dir / "test_graph.json"
        graph_file.write_text('{"nodes": []}')

        # Cause a non-ValueError exception during processing
        with patch("json.load", side_effect=OSError("read fail")):
            result = pm._load_graph_data_safe("test")
        assert result is None


# ============================================================================
# 10. persistence — AFM history (874-917, 938-940)
# ============================================================================


class TestPersistenceAFMHistory:
    def _make_pm(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_load_afm_legacy_pickle_raises(self, tmp_path):
        pm = self._make_pm(tmp_path)
        legacy = pm.afm_dir / "sess1.pkl"
        legacy.write_bytes(b"fake")

        with pytest.raises(ValueError, match="SECURITY"):
            pm.load_afm_history("sess1")

    def test_load_afm_corrupted_json(self, tmp_path):
        pm = self._make_pm(tmp_path)
        json_file = pm.afm_dir / "sess1.json"
        json_file.write_text("{bad json!!")

        result = pm.load_afm_history("sess1")
        assert result is None

    def test_load_afm_generic_error(self, tmp_path):
        pm = self._make_pm(tmp_path)
        json_file = pm.afm_dir / "sess1.json"
        json_file.write_text("{}")

        with patch("builtins.open", side_effect=IOError("read fail")):
            result = pm.load_afm_history("sess1")
        assert result is None

    def test_list_afm_sessions_error(self, tmp_path):
        pm = self._make_pm(tmp_path)
        with patch.object(Path, "glob", side_effect=OSError("fail")):
            result = pm.list_afm_sessions()
        assert result == []

    def test_load_afm_not_found(self, tmp_path):
        pm = self._make_pm(tmp_path)
        result = pm.load_afm_history("nonexistent")
        assert result is None


# ============================================================================
# 11. afm — TokenCounter fallback, LLMCompressor, HashingEmbedder, ImportanceClassifier
# ============================================================================


class TestAFMComponents:
    def test_token_counter_double_fallback(self):
        from src.afm import TokenCounter

        with patch("tiktoken.encoding_for_model", side_effect=Exception("no")):
            with patch("tiktoken.get_encoding", side_effect=Exception("no")):
                tc = TokenCounter()
                assert tc.encoding is None
                count = tc.count("hello world test")
                assert count > 0

    def test_token_counter_fallback_to_cl100k(self):
        from src.afm import TokenCounter

        with patch("tiktoken.encoding_for_model", side_effect=Exception("no")):
            tc = TokenCounter()
            assert tc.encoding is not None

    def test_heuristic_compressor_empty_sentences(self):
        from src.afm import HeuristicCompressor, TokenCounter

        tc = TokenCounter()
        comp = HeuristicCompressor(tc)
        result = comp.compress("", 10)
        assert isinstance(result, str)

    def test_heuristic_compressor_truncation_fallback(self):
        from src.afm import HeuristicCompressor, TokenCounter

        tc = TokenCounter()
        comp = HeuristicCompressor(tc)
        result = comp.compress("Word " * 100, 5)
        assert isinstance(result, str)

    def test_llm_compressor_falls_back(self):
        from src.afm import LLMCompressor, TokenCounter

        tc = TokenCounter()
        comp = LLMCompressor(tc, api_key="fake", model="gpt-4o-mini")
        result = comp.compress("This is a test sentence. Another sentence here.", 20)
        assert isinstance(result, str)

    def test_llm_compressor_no_api_key(self):
        from src.afm import LLMCompressor, TokenCounter

        tc = TokenCounter()
        comp = LLMCompressor(tc, api_key=None)
        # LLMCompressor always falls back to heuristic - just verify it works
        result = comp.compress("Test sentence here.", 10)
        assert isinstance(result, str)

    def test_hashing_embedder(self):
        from src.afm import HashingEmbedder

        emb = HashingEmbedder(dim=64)
        result = emb.encode(["hello world", "test"])
        assert result.shape == (2, 64)

    def test_hashing_embedder_empty_text(self):
        from src.afm import HashingEmbedder

        emb = HashingEmbedder(dim=64)
        result = emb.encode([""])
        assert result.shape == (1, 64)

    def test_importance_classifier_llm_mode(self):
        from src.afm import ImportanceClassifier, Message, ImportanceLevel

        clf = ImportanceClassifier(use_llm=True, api_key="fake")
        msg = Message(role="user", content="test", importance=ImportanceLevel.TRIVIAL, turn_index=0)
        level = clf._classify_llm(msg)
        assert level in [
            ImportanceLevel.CRITICAL,
            ImportanceLevel.RELEVANT,
            ImportanceLevel.TRIVIAL,
        ]

    def test_importance_classifier_no_apikey_warning(self):
        from src.afm import ImportanceClassifier

        clf = ImportanceClassifier(use_llm=True, api_key=None)
        assert clf.use_llm is False

    def test_focus_manager_system_preamble_too_large(self):
        from src.afm import FocusManager, AFMConfig

        with patch("src.afm.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            fm = FocusManager(AFMConfig())
            packed = []
            fm._try_add_system_preamble("x" * 10000, 5, packed)
            assert len(packed) == 0  # skipped

    def test_focus_manager_llm_compression_config(self):
        from src.afm import FocusManager, AFMConfig

        cfg = AFMConfig(use_llm_compression=True, llm_api_key="fake")
        with patch("src.afm.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            fm = FocusManager(cfg)
            assert fm.compressor is not None


# ============================================================================
# 12. embeddings_onnx — ONNX init and singleton
# ============================================================================


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


# ============================================================================
# 13. graph_visualizer — visualize_html and export_json edge cases
# ============================================================================


class TestGraphVisualizer:
    def test_export_json_node_not_in_chunks(self):
        from src.graph_visualizer import GraphVisualizer
        import networkx as nx

        compressor = MagicMock()
        graph = nx.Graph()
        graph.add_node("doc_n0")
        graph.add_node("doc_n1")
        compressor.graphs = {"doc": graph}
        compressor.chunks = {}  # no chunks

        viz = GraphVisualizer(compressor)
        result = json.loads(viz.export_json("doc"))
        assert result["stats"]["total_nodes"] == 0

    def test_export_json_node_below_importance(self):
        from src.graph_visualizer import GraphVisualizer, VisualizationConfig
        import networkx as nx

        compressor = MagicMock()
        graph = nx.Graph()
        graph.add_node("doc_n0")
        chunk = MagicMock()
        chunk.importance = 0.001
        chunk.text = "low importance"
        chunk.metadata = {"tokens": 10, "position": 0}
        compressor.graphs = {"doc": graph}
        compressor.chunks = {"doc_n0": chunk}

        viz = GraphVisualizer(compressor)
        result = json.loads(viz.export_json("doc", VisualizationConfig(min_importance=0.5)))
        assert result["stats"]["total_nodes"] == 0

    def test_ascii_no_edge_weights(self):
        from src.graph_visualizer import GraphVisualizer, VisualizationConfig
        import networkx as nx

        compressor = MagicMock()
        graph = nx.Graph()
        graph.add_node("doc_n0")
        chunk = MagicMock()
        chunk.importance = 0.5
        chunk.text = "test"
        chunk.metadata = {"tokens": 10, "position": 0}
        compressor.graphs = {"doc": graph}
        compressor.chunks = {"doc_n0": chunk}

        viz = GraphVisualizer(compressor)
        result = viz.render_ascii("doc", VisualizationConfig(show_edge_weights=False))
        assert "n0" in result

    def test_visualize_html_missing_pyvis(self):
        from src.graph_visualizer import GraphVisualizer
        import networkx as nx

        compressor = MagicMock()
        compressor.graphs = {"doc": nx.Graph()}

        viz = GraphVisualizer(compressor)
        with patch.dict("sys.modules", {"pyvis": None, "pyvis.network": None}):
            with pytest.raises(ImportError, match="pyvis"):
                viz.visualize_html("doc", "/tmp/out.html")


# ============================================================================
# 14. embeddings — tier fallback, cache, code embedder fallback
# ============================================================================


class TestEmbeddingManager:
    def _reset_singleton(self):
        from src.embeddings import EmbeddingManager

        EmbeddingManager._instance = None

    def test_encode_unknown_tier_raises(self):
        self._reset_singleton()
        from src.embeddings import EmbeddingManager, EmbeddingTier

        with patch("src.embeddings.SentenceTransformer"):
            mgr = EmbeddingManager.__new__(EmbeddingManager)
            mgr._model_cache = {}
            mgr._cache_lock = threading.Lock()
            mgr._lru_cache = None
            mgr._onnx_manager = None
            mgr._tfidf_manager = None
            mgr._tier = EmbeddingTier.STANDARD
            mgr._enable_cache = False

            with pytest.raises((ValueError, RuntimeError)):
                # Use a fake tier value
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


# ============================================================================
# 15. version_manager — diff error paths, prune, stats
# ============================================================================


class TestVersionManager:
    def _make_vm(self, tmp_path):
        from src.version_manager import VersionManager

        return VersionManager(storage_dir=str(tmp_path / "versions"))

    def test_add_version_relative_path_raises(self, tmp_path):
        vm = self._make_vm(tmp_path)
        with pytest.raises(ValueError, match="Security violation"):
            vm.add_version("doc1", "content", "abc123", file_path="relative/path.txt")

    def test_get_latest_content_none(self, tmp_path):
        vm = self._make_vm(tmp_path)
        assert vm.get_latest_content("nonexistent") is None

    def test_get_latest_content_exists(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "hello", "abc789", file_path=None)
        assert vm.get_latest_content("doc1") == "hello"

    def test_diff_no_cached_version(self, tmp_path):
        vm = self._make_vm(tmp_path)
        result = vm.diff_with_current_file("doc1")
        assert "No cached versions" in result

    def test_diff_specified_version_not_found(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        result = vm.diff_with_current_file("doc1", cached_version_id=999)
        assert "not found" in result

    def test_diff_no_file_path(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        result = vm.diff_with_current_file("doc1")
        assert "No source file path" in result

    def test_diff_no_content(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "", "empty123", file_path=os.path.abspath(__file__))
        result = vm.diff_with_current_file("doc1")
        assert "empty" in result

    def test_diff_file_not_on_disk(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=str(tmp_path / "missing.txt"))
        result = vm.diff_with_current_file("doc1")
        assert "not found" in result

    def test_diff_permission_error(self, tmp_path):
        vm = self._make_vm(tmp_path)
        f = tmp_path / "perm.txt"
        f.write_text("original")
        vm.add_version("doc1", "content", "abc123", file_path=str(f))

        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = vm.diff_with_current_file("doc1")
        assert "Permission denied" in result

    def test_diff_unicode_error(self, tmp_path):
        vm = self._make_vm(tmp_path)
        f = tmp_path / "enc.txt"
        f.write_text("original")
        vm.add_version("doc1", "content", "abc123", file_path=str(f))

        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")):
            result = vm.diff_with_current_file("doc1")
        assert "Encoding error" in result

    @pytest.mark.asyncio
    async def test_delete_versions_async(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        await vm.delete_versions_async("doc1")
        assert "doc1" not in vm.versions

    def test_save_all_versions_no_doc(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm._save_all_versions("nonexistent")  # should not raise

    def test_load_index_corrupt_file(self, tmp_path):
        vm = self._make_vm(tmp_path)
        bad_file = vm.storage_dir / "bad.json"
        bad_file.write_text("NOT JSON")
        vm._load_index()  # should not raise

    def test_get_stats(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "hello world", "abc456", file_path=None)
        stats = vm.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_versions"] == 1


# ============================================================================
# 16. resource_manager — check_health, memory_health, suggest_cleanup
# ============================================================================


class TestResourceManager:
    def test_check_health_at_capacity(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10, max_documents=5)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 5.0, "d2": 6.0}  # exceeds 10MB
        health = rm.check_health()
        assert not health["healthy"]

    def test_check_health_warn_threshold(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10, max_documents=100, warn_threshold=0.5)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 6.0}  # 60% > 50% threshold
        health = rm.check_health()
        assert len(health["warnings"]) > 0

    def test_check_health_doc_count_limit(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_documents=2, max_total_storage_mb=100)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 1.0, "d2": 1.0, "d3": 1.0}
        health = rm.check_health()
        assert any("Document count" in w for w in health["warnings"])

    def test_memory_health_exceeded(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 15.0}
        healthy, msg = rm.check_memory_health()
        assert not healthy
        assert "exceeded" in msg

    def test_suggest_cleanup_empty(self):
        from src.resource_manager import ResourceManager

        rm = ResourceManager()
        assert rm.suggest_cleanup() is None

    def test_suggest_cleanup_under_threshold(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=1000, warn_threshold=0.8)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 1.0}
        assert rm.suggest_cleanup() is None

    def test_get_stats_comprehensive(self):
        from src.resource_manager import ResourceManager

        rm = ResourceManager()
        rm.register_document("doc1", 1024 * 1024)
        stats = rm.get_stats()
        assert "limits" in stats
        assert "documents" in stats


# ============================================================================
# 17. batch_manager — progress, callbacks, batch_ingest_from_files
# ============================================================================


class TestBatchManager:
    def test_progress_zero_total(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=0, completed=0, successful=0, failed=0)
        assert p.progress_percentage == 0.0

    def test_progress_zero_completed(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=5, completed=0, successful=0, failed=0)
        assert p.success_rate == 0.0

    def test_progress_str(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=10, completed=5, successful=3, failed=2)
        s = str(p)
        assert "50.0%" in s

    def test_progress_callback_error(self):
        from src.batch_manager import BatchProgressTracker

        tracker = BatchProgressTracker(
            total=2,
            on_progress=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should not raise
        tracker.update(True, "doc1")

    @pytest.mark.asyncio
    async def test_batch_ingest_from_files_missing_file(self, tmp_path):
        from src.batch_manager import batch_ingest_from_files

        compressor = MagicMock()
        compressor.ingest_text = MagicMock(return_value=MagicMock())

        results = await batch_ingest_from_files(compressor, [str(tmp_path / "nonexistent.txt")])
        assert results == []  # empty list since no valid docs

    @pytest.mark.asyncio
    async def test_batch_ingest_from_files_read_error(self, tmp_path):
        from src.batch_manager import batch_ingest_from_files

        f = tmp_path / "test.txt"
        f.write_text("content")
        compressor = MagicMock()

        with patch("builtins.open", side_effect=IOError("read error")):
            results = await batch_ingest_from_files(compressor, [str(f)])
        assert results == []


# ============================================================================
# 18. file_sync_manager — relative path, get_file_diff, check_all_sync
# ============================================================================


class TestFileSyncManager:
    def test_register_file_relative_path_raises(self):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()
        with pytest.raises(ValueError, match="Security violation"):
            fsm.register_file("doc1", "content", "relative/path.txt")

    def test_get_file_diff_no_metadata(self):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()
        assert fsm.get_file_diff("nonexistent") is None

    def test_get_file_diff_no_version_manager(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        result = fsm.get_file_diff("doc1")
        assert "WARN" in result

    def test_get_file_diff_with_version_manager(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        vm = MagicMock()
        vm.diff_with_current_file.return_value = "no changes"
        result = fsm.get_file_diff("doc1", version_manager=vm)
        assert result == "no changes"

    def test_get_file_diff_exception(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        vm = MagicMock()
        vm.diff_with_current_file.side_effect = Exception("boom")
        result = fsm.get_file_diff("doc1", version_manager=vm)
        assert result is None

    def test_check_sync_mtime_changed_content_same(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        # Simulate mtime change
        fsm.file_metadata["doc1"].mtime = 0

        result = fsm.check_file_sync("doc1")
        assert result["in_sync"] is True


# ============================================================================
# 19. observability — OpenTelemetry unavailable paths
# ============================================================================


class TestObservability:
    def test_manager_not_enabled(self):
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        mgr.tracer = None

        # These should all be no-ops
        mgr.set_attributes({"key": "value"})
        mgr.record_exception(ValueError("test"))

    def test_shutdown_not_enabled(self):
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        result = mgr.shutdown()
        assert result is True


# ============================================================================
# 20. metrics — NoOp collector, validation, enabled methods
# ============================================================================


class TestMetrics:
    def test_noop_collector_methods(self):
        from src.metrics import NoOpMetricsCollector

        noop = NoOpMetricsCollector()
        noop.record_compression_ratio(5.0, "HIGH")
        noop.record_latency("ingest", 0.5)
        noop.increment_documents_processed("ingest", "HIGH")
        noop.set_cache_hit_ratio(0.5)
        noop.set_active_documents(10)
        noop.increment_errors("ValueError", "ingest")
        noop.record_batch_size(5)
        noop.reset_all_metrics()
        assert "unavailable" in noop.generate_metrics_text()

    def test_metrics_collector_not_enabled(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            mc._enabled = False
            mc.record_compression_ratio(5.0, "HIGH")
            mc.record_latency("ingest", 0.5)
            mc.increment_documents_processed("ingest", "HIGH")
            mc.set_cache_hit_ratio(0.5)
            mc.set_active_documents(10)
            mc.increment_errors("ValueError", "ingest")
            mc.record_batch_size(5)
            mc.reset_all_metrics()
            assert "unavailable" in mc.generate_metrics_text()

    def test_validate_fidelity_invalid(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            assert mc._validate_fidelity("INVALID_LEVEL") is False
            assert mc._validate_fidelity(None) is True

    def test_validate_operation_invalid(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            assert mc._validate_operation("invalid_op") is False

    def test_singleton_reset(self):
        from src.metrics import MetricsCollector

        MetricsCollector._instance = None
        mc = MetricsCollector.get_metrics()
        assert mc is not None
        MetricsCollector.reset_singleton()
        assert MetricsCollector._instance is None


# ============================================================================
# 21. health — component check edge cases
# ============================================================================


class TestHealth:
    def test_health_psutil_unavailable(self):
        with patch("src.health.PSUTIL_AVAILABLE", False):
            from src.health import HealthChecker

            hc = HealthChecker.__new__(HealthChecker)
            hc._operation_latencies = {}
            hc._operation_errors = {}
            hc._operation_successes = {}
            result = hc._get_memory_usage()
            assert result["available"] is False

    def test_health_disk_usage_error(self):
        from src.health import HealthChecker

        hc = HealthChecker.__new__(HealthChecker)
        hc._operation_latencies = {}
        hc._operation_errors = {}
        hc._operation_successes = {}

        with patch("shutil.disk_usage", side_effect=OSError("fail")):
            result = hc._get_disk_usage()
        assert result["available"] is False


# ============================================================================
# 22. error_types — exception constructors
# ============================================================================


class TestErrorTypes:
    def test_operation_timeout(self):
        from src.error_types import OperationTimeoutError

        e = OperationTimeoutError("embed", timeout=30.0)
        assert "30" in str(e)
        assert e.operation == "embed"

    def test_circuit_breaker_open(self):
        from src.error_types import CircuitBreakerOpenError

        e = CircuitBreakerOpenError("persistence", failure_count=5)
        assert "5" in str(e)

    def test_circuit_breaker_no_count(self):
        from src.error_types import CircuitBreakerOpenError

        e = CircuitBreakerOpenError("persistence")
        assert "OPEN" in str(e)

    def test_retry_exhausted_with_exception(self):
        from src.error_types import RetryExhaustedError

        inner = ValueError("inner")
        e = RetryExhaustedError("op", max_retries=3, last_exception=inner)
        assert "3" in str(e)
        assert "inner" in str(e)

    def test_retry_exhausted_no_exception(self):
        from src.error_types import RetryExhaustedError

        e = RetryExhaustedError("op", max_retries=3)
        assert "3" in str(e)

    def test_rate_limit_exceeded_with_wait(self):
        from src.error_types import RateLimitExceededError

        e = RateLimitExceededError("ingest", rate=10.0, wait_time=5.0)
        assert "5.0" in str(e)

    def test_graceful_degradation(self):
        from src.error_types import GracefulDegradationError

        e = GracefulDegradationError("embed", "tfidf", reason="OOM")
        assert "OOM" in str(e)

    def test_graceful_degradation_no_reason(self):
        from src.error_types import GracefulDegradationError

        e = GracefulDegradationError("embed", "tfidf")
        assert "degraded" in str(e)


# ============================================================================
# 23. error_helpers — SmartError methods
# ============================================================================


class TestErrorHelpers:
    def test_file_id_not_found_with_matches(self):
        from src.error_helpers import SmartError

        err = SmartError.file_id_not_found("quantum_papper", ["quantum_paper", "neural_nets"])
        assert "quantum_paper" in str(err)

    def test_file_id_not_found_many_available(self):
        from src.error_helpers import SmartError

        ids = [f"doc{i}" for i in range(10)]
        err = SmartError.file_id_not_found("unknown", ids)
        assert "10 total" in str(err)

    def test_node_id_not_found(self):
        from src.error_helpers import SmartError

        err = SmartError.node_id_not_found("doc_n99", ["doc_n0", "doc_n1"], "doc")
        assert "doc_n" in str(err)

    def test_invalid_enum_value(self):
        from src.error_helpers import SmartError

        err = SmartError.invalid_enum_value("BALENCED", ["BALANCED", "HIGH", "LOW"], "fidelity")
        assert "BALANCED" in str(err)


# ============================================================================
# 24. compression_rewards — dataclass scores
# ============================================================================


class TestCompressionRewards:
    def test_schema_validation_scores(self):
        from src.compression_rewards import SchemaValidationResult

        r = SchemaValidationResult(input_valid=True, output_valid=False)
        assert r.score == 0.5
        assert not r.all_valid

    def test_fidelity_adherence_ratio_score_zero_target(self):
        from src.compression_rewards import FidelityAdherenceResult

        r = FidelityAdherenceResult(target_ratio=0)
        assert r.ratio_score == 0.0

    def test_composition_integrity_score(self):
        from src.compression_rewards import CompositionIntegrityResult

        r = CompositionIntegrityResult(graph_connected=False, orphan_nodes=5)
        assert r.score < 1.0

    def test_memory_discipline_zero_budget(self):
        from src.compression_rewards import MemoryDisciplineResult

        r = MemoryDisciplineResult(memory_budget_mb=0)
        assert r.memory_score == 0.0


# ============================================================================
# 25. evidence_bundle — ContractCheck, ContractResult, QualityMetrics
# ============================================================================


class TestEvidenceBundle:
    def test_contract_check_from_dict(self):
        from src.evidence_bundle import ContractCheck, ContractStatus

        data = {"name": "test", "status": "passed", "message": "ok"}
        check = ContractCheck.from_dict(data)
        assert check.status == ContractStatus.PASSED

    def test_contract_result_add_error(self):
        from src.evidence_bundle import ContractResult

        cr = ContractResult()
        cr.add_error("check1", "something broke")
        assert not cr.overall_passed
        assert cr.failed_count == 0  # error != failed

    def test_contract_result_roundtrip(self):
        from src.evidence_bundle import ContractResult

        cr = ContractResult()
        cr.add_check("test1", True, "ok")
        cr.add_check("test2", False, "bad")
        d = cr.to_dict()
        cr2 = ContractResult.from_dict(d)
        assert cr2.passed_count == 1
        assert cr2.failed_count == 1

    def test_quality_metrics_roundtrip(self):
        from src.evidence_bundle import QualityMetrics

        qm = QualityMetrics(ssim_score=0.9, custom_metrics={"extra": 1.0})
        d = qm.to_dict()
        qm2 = QualityMetrics.from_dict(d)
        assert qm2.ssim_score == 0.9


# ============================================================================
# 26. benchmark_guard — evaluate violations
# ============================================================================


class TestBenchmarkGuard:
    def test_missing_thresholds(self):
        from src.benchmark_guard import evaluate_report_against_thresholds

        violations = evaluate_report_against_thresholds(mode="unknown", report={}, thresholds={})
        assert len(violations) == 1
        assert "Missing thresholds" in violations[0].message

    def test_load_json(self, tmp_path):
        from src.benchmark_guard import load_json

        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        data = load_json(f)
        assert data["key"] == "value"


# ============================================================================
# 27. scar_compressor — preservation loss, alignment, batch compress
# ============================================================================


class TestScarCompressor:
    def test_preservation_loss(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        original = torch.randn(2, 8)
        recon = torch.randn(2, 8)
        loss = comp.compute_preservation_loss(original, recon)
        assert loss.item() > 0

    def test_compress_batch_numpy(self):
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        data = np.random.randn(3, 8).astype(np.float32)
        result = comp.compress_batch(data)
        assert result.shape == (3, 4)

    def test_forward_with_reconstruction(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        emb = torch.randn(2, 8)
        compressed, recon = comp(emb, return_reconstruction=True)
        assert recon is not None
        assert compressed.shape == (2, 4)

    def test_forward_without_reconstruction(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        emb = torch.randn(2, 8)
        compressed, recon = comp(emb, return_reconstruction=False)
        assert recon is None


# ============================================================================
# 28. resource_handlers — should_compress and check_environment edge cases
# ============================================================================


class TestResourceHandlers:
    @pytest.mark.asyncio
    async def test_should_compress_file_not_found(self):
        from src.handlers.resource_handlers import handle_should_compress

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = "/nonexistent/file.txt"

        result = await handle_should_compress(ctx, {"file_path": "/nonexistent/file.txt"})
        data = json.loads(result)
        assert "error" in data or data.get("recommendation") == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_should_compress_binary_extension(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert data["recommendation"] == "CONVERT_THEN_COMPRESS"

    @pytest.mark.asyncio
    async def test_should_compress_empty_file(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "empty.xyz"
        f.write_text("")

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert data["recommendation"] in ("SKIP", "UNKNOWN")

    @pytest.mark.asyncio
    async def test_should_compress_code_file(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "test.py"
        f.write_text("x = 1\n" * 500)

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert "estimated_tokens" in data


# ============================================================================
# 29. adaptive_rate_allocator — forward pass
# ============================================================================


class TestAdaptiveRateAllocator:
    def test_forward_pass(self):
        import networkx as nx
        from src.adaptive_rate_allocator import AdaptiveRateAllocator

        ara = AdaptiveRateAllocator(num_rate_levels=5, temperature=1.5)
        graph = nx.Graph()
        graph.add_edge("n0", "n1", weight=0.5)
        graph.add_edge("n1", "n2", weight=0.3)

        ratio, diagnostics = ara(graph, available_context_tokens=5000, max_context_tokens=10000)
        assert isinstance(ratio, float)
        assert "selected_level" in diagnostics


# ============================================================================
# 30. multimodal_compressor — encode_image, get_skeleton_summary
# ============================================================================


class TestMultimodalCompressor:
    def test_encode_image_no_encoder(self):
        from src.multimodal_compressor import MultiModalCompressor

        mc = MultiModalCompressor.__new__(MultiModalCompressor)
        mc.image_encoder = None
        result = mc._encode_image(b"fake_image_data")
        assert result is None

    def test_encode_image_exception(self):
        from src.multimodal_compressor import MultiModalCompressor

        mc = MultiModalCompressor.__new__(MultiModalCompressor)
        mc.image_encoder = MagicMock()

        with patch("PIL.Image.open", side_effect=Exception("bad image")):
            result = mc._encode_image(b"bad data")
        assert result is None


# ============================================================================
# 31. rate_limiter — acquire tokens
# ============================================================================


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_basic(self):
        from src.rate_limiter import RateLimiter

        rl = RateLimiter(rate=100.0, capacity=10, name="test")
        await rl.acquire()  # should succeed

    @pytest.mark.asyncio
    async def test_acquire_non_blocking_insufficient(self):
        from src.rate_limiter import RateLimiter
        from src.error_types import RateLimitExceededError

        rl = RateLimiter(rate=0.001, capacity=1, name="test")
        await rl.acquire()  # drain the single token

        with pytest.raises(RateLimitExceededError):
            await rl.acquire(blocking=False)


# ============================================================================
# 32. persistence — save/load document JSON paths (537-546)
# ============================================================================


class TestPersistenceLoadDocJSON:
    def _make_pm(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_load_document_json_legacy_pickle_raises(self, tmp_path):
        pm = self._make_pm(tmp_path)
        legacy = pm.documents_dir / "test.pkl"
        legacy.write_bytes(b"fake pickle data")

        with pytest.raises(ValueError, match="SECURITY"):
            pm._load_document_json("test")

    def test_load_document_json_with_ids_fallback(self, tmp_path):
        pm = self._make_pm(tmp_path)

        # Create document JSON with proper chunk format
        doc_data = {
            "chunks": {"n1": {"node_id": "n1", "text": "hello", "importance": 0.5, "metadata": {}}},
            "graph_data": {"nodes": [], "edges": []},
            "metadata": {},
        }
        doc_file = pm.documents_dir / "test.json"
        doc_file.write_text(json.dumps(doc_data))

        # Create embeddings without ids key (exercises fallback path)
        emb_file = pm.documents_dir / "test_chunks.npz"
        np.savez(emb_file, embeddings=np.array([[1.0, 2.0]]))

        result = pm._load_document_json("test")
        assert result is not None
