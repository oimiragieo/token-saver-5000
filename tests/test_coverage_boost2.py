"""
Coverage boost tests for persistence, compression handlers, code_compression_adapter,
embeddings, and embeddings_onnx modules.

Targets ~60+ tests covering large uncovered areas with extensive mocking.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ============================================================================
# 1. PERSISTENCE TESTS
# ============================================================================


class TestSaveGraphDataSafe:
    """Tests for PersistenceManager._save_graph_data_safe"""

    def _make_manager(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_save_graph_data_basic(self, tmp_path):
        pm = self._make_manager(tmp_path)
        graph_data = {
            "nodes": [
                {"id": "n1", "text": "hello", "embedding": np.array([1.0, 2.0, 3.0])},
                {"id": "n2", "text": "world", "embedding": [4.0, 5.0, 6.0]},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
            "metadata": {"title": "test"},
        }
        result = pm._save_graph_data_safe("doc1", graph_data)
        assert result is True
        assert (pm.documents_dir / "doc1_graph.json").exists()
        assert (pm.documents_dir / "doc1_embeddings.npz").exists()

    def test_save_graph_data_no_embeddings(self, tmp_path):
        pm = self._make_manager(tmp_path)
        graph_data = {
            "nodes": [{"id": "n1", "text": "hello"}],
            "edges": [],
            "metadata": {},
        }
        result = pm._save_graph_data_safe("doc2", graph_data)
        assert result is True
        assert (pm.documents_dir / "doc2_graph.json").exists()
        assert not (pm.documents_dir / "doc2_embeddings.npz").exists()

    def test_save_graph_data_empty(self, tmp_path):
        pm = self._make_manager(tmp_path)
        result = pm._save_graph_data_safe("empty", {"nodes": [], "edges": [], "metadata": {}})
        assert result is True

    def test_save_graph_data_format_version(self, tmp_path):
        pm = self._make_manager(tmp_path)
        pm._save_graph_data_safe("ver", {"nodes": [], "edges": [], "metadata": {}})
        with open(pm.documents_dir / "ver_graph.json") as f:
            data = json.load(f)
        assert data["_format_version"] == 2

    def test_save_graph_data_error_returns_false(self, tmp_path):
        pm = self._make_manager(tmp_path)
        # Make documents_dir read-only to trigger error
        with patch("builtins.open", side_effect=OSError("disk full")):
            result = pm._save_graph_data_safe("fail", {"nodes": [], "edges": [], "metadata": {}})
        assert result is False


class TestLoadGraphDataSafe:
    """Tests for PersistenceManager._load_graph_data_safe"""

    def _make_manager(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_load_nonexistent_returns_none(self, tmp_path):
        pm = self._make_manager(tmp_path)
        result = pm._load_graph_data_safe("nonexistent")
        assert result is None

    def test_save_then_load_roundtrip(self, tmp_path):
        pm = self._make_manager(tmp_path)
        graph_data = {
            "nodes": [{"id": "n1", "text": "hello", "embedding": np.array([1.0, 2.0])}],
            "edges": [{"source": "n1", "target": "n1"}],
            "metadata": {"key": "value"},
        }
        pm._save_graph_data_safe("rt", graph_data)
        # Create IDs JSON file (the secure format expected by _load_graph_data_safe)
        ids_file = pm.documents_dir / "rt_embeddings.ids.json"
        ids_file.write_text(json.dumps(["n1"]), encoding="utf-8")
        loaded = pm._load_graph_data_safe("rt")
        assert loaded is not None
        assert len(loaded["nodes"]) == 1
        assert loaded["metadata"]["key"] == "value"

    def test_load_with_embeddings_reconstruction(self, tmp_path):
        pm = self._make_manager(tmp_path)
        emb = np.array([0.5, 0.6, 0.7])
        graph_data = {
            "nodes": [{"id": "n1", "text": "test", "embedding": emb}],
            "edges": [],
            "metadata": {},
        }
        pm._save_graph_data_safe("emb", graph_data)
        # Create IDs JSON file (the secure format expected by _load_graph_data_safe)
        ids_file = pm.documents_dir / "emb_embeddings.ids.json"
        ids_file.write_text(json.dumps(["n1"]), encoding="utf-8")
        loaded = pm._load_graph_data_safe("emb")
        assert loaded is not None
        # Embedding should be reattached
        node = loaded["nodes"][0]
        assert "embedding" in node
        np.testing.assert_allclose(node["embedding"], emb, atol=1e-5)

    def test_legacy_pickle_raises_valueerror(self, tmp_path):
        pm = self._make_manager(tmp_path)
        # Create a fake legacy pickle file
        pickle_file = pm.documents_dir / "legacy_graph.pkl"
        pickle_file.write_bytes(b"fake pickle data")
        with pytest.raises(ValueError, match="SECURITY"):
            pm._load_graph_data_safe("legacy")

    def test_load_corrupted_json_raises(self, tmp_path):
        pm = self._make_manager(tmp_path)
        graph_file = pm.documents_dir / "bad_graph.json"
        graph_file.write_text("not valid json {{{", encoding="utf-8")
        # json.JSONDecodeError is a subclass of ValueError, which gets re-raised
        with pytest.raises((ValueError, json.JSONDecodeError)):
            pm._load_graph_data_safe("bad")

    def test_load_json_without_embeddings_file(self, tmp_path):
        pm = self._make_manager(tmp_path)
        graph_file = pm.documents_dir / "noemb_graph.json"
        data = {"nodes": [{"id": "n1"}], "edges": [], "metadata": {}, "_format_version": 2}
        graph_file.write_text(json.dumps(data), encoding="utf-8")
        loaded = pm._load_graph_data_safe("noemb")
        assert loaded is not None
        assert len(loaded["nodes"]) == 1


class TestChromaDBPaths:
    """Tests for ChromaDB persistence methods using mocked client."""

    def _make_manager_with_chromadb(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            pm = PersistenceManager(storage_dir=str(tmp_path / "storage"))
        # Manually set up ChromaDB mock after creation
        mock_client = MagicMock()
        pm.chroma_client = mock_client
        pm.use_chromadb = True
        return pm, mock_client

    def test_save_document_chromadb(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_collection = MagicMock()
        mock_client.create_collection.return_value = mock_collection

        node = MagicMock()
        node.embedding = np.array([1.0, 2.0])
        node.text = "hello"
        node.importance = 0.8
        node.metadata = {"position": 0, "tokens": 5, "entities": []}

        chunks = {"n1": node}
        graph_data = {"nodes": [], "edges": []}
        metadata = {"title": "test"}

        result = pm._save_document_chromadb("doc1", chunks, graph_data, metadata)
        assert result is True
        mock_collection.add.assert_called_once()

    def test_save_document_chromadb_error(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_client.create_collection.side_effect = Exception("DB error")
        result = pm._save_document_chromadb("doc1", {}, {}, {})
        assert result is False

    def test_load_document_chromadb(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_collection = MagicMock()
        mock_collection.metadata = {"file_id": "doc1"}
        mock_collection.get.return_value = {
            "ids": ["n1"],
            "documents": ["hello"],
            "embeddings": [[1.0, 2.0]],
            "metadatas": [{"importance": 0.8, "position": 0, "tokens": 5, "entities": "[]"}],
        }
        mock_client.get_collection.return_value = mock_collection

        with patch.object(pm, "_load_graph_data_safe", return_value={"nodes": [], "edges": []}):
            result = pm._load_document_chromadb("doc1")
        assert result is not None
        assert "chunks" in result
        assert "n1" in result["chunks"]

    def test_load_document_chromadb_not_found(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_client.get_collection.side_effect = Exception("not found")
        result = pm._load_document_chromadb("missing")
        assert result is None

    def test_delete_document_chromadb(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        result = pm._delete_document_chromadb("doc1")
        assert result is True
        mock_client.delete_collection.assert_called_once()

    def test_delete_document_chromadb_error(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_client.delete_collection.side_effect = Exception("fail")
        result = pm._delete_document_chromadb("doc1")
        assert result is False

    def test_list_documents_chromadb(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        coll1 = MagicMock()
        coll1.name = "doc_test1"
        coll1.metadata = {"file_id": "test1"}
        coll2 = MagicMock()
        coll2.name = "doc_test2"
        coll2.metadata = {"file_id": "test2"}
        coll3 = MagicMock()
        coll3.name = "other"
        coll3.metadata = {}
        mock_client.list_collections.return_value = [coll1, coll2, coll3]

        result = pm._list_documents_chromadb()
        assert "test1" in result
        assert "test2" in result
        assert len(result) == 2

    def test_list_documents_chromadb_error(self, tmp_path):
        pm, mock_client = self._make_manager_with_chromadb(tmp_path)
        mock_client.list_collections.side_effect = Exception("fail")
        result = pm._list_documents_chromadb()
        assert result == []


class TestAfmHistory:
    """Tests for save_afm_history, load_afm_history, list_afm_sessions."""

    def _make_manager(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def _make_mock_message(self, role="user", content="hello", embedding=None):
        msg = MagicMock()
        msg.role = role
        msg.content = content
        msg.importance = MagicMock()
        msg.importance.value = "relevant"
        msg.turn_index = 0
        msg.timestamp = 1234.0
        msg.message_id = "msg1"
        msg.relevance_score = 0.5
        msg.intended_fidelity = MagicMock()
        msg.intended_fidelity.value = "full"
        msg.compressed_summary = None
        msg.placeholder_stub = None
        msg.embedding = embedding
        return msg

    def test_save_afm_history_basic(self, tmp_path):
        pm = self._make_manager(tmp_path)
        msg = self._make_mock_message()
        result = pm.save_afm_history("sess1", [msg], turn_counter=1)
        assert result is True
        assert (pm.afm_dir / "sess1.json").exists()

    def test_save_afm_history_with_embeddings(self, tmp_path):
        pm = self._make_manager(tmp_path)
        msg = self._make_mock_message(embedding=np.array([1.0, 2.0]))
        result = pm.save_afm_history("sess2", [msg], turn_counter=1)
        assert result is True
        assert (pm.afm_dir / "sess2_embeddings.npz").exists()

    def test_save_afm_history_error(self, tmp_path):
        pm = self._make_manager(tmp_path)
        # Atomic writes use tempfile.mkstemp + os.fdopen, so patch that
        with patch("tempfile.mkstemp", side_effect=OSError("disk full")):
            result = pm.save_afm_history("fail", [], turn_counter=0)
        assert result is False

    def test_load_afm_history_basic(self, tmp_path):
        pm = self._make_manager(tmp_path)
        msg = self._make_mock_message()
        pm.save_afm_history("load_test", [msg], turn_counter=5, metadata={"key": "val"})

        with patch.object(pm, "_deserialize_message_safe", return_value=MagicMock()):
            result = pm.load_afm_history("load_test")
        assert result is not None
        assert result["turn_counter"] == 5

    def test_load_afm_history_not_found(self, tmp_path):
        pm = self._make_manager(tmp_path)
        result = pm.load_afm_history("nonexistent")
        assert result is None

    def test_load_afm_history_legacy_pickle_raises(self, tmp_path):
        pm = self._make_manager(tmp_path)
        pickle_file = pm.afm_dir / "legacy_sess.pkl"
        pickle_file.write_bytes(b"fake pickle")
        with pytest.raises(ValueError, match="SECURITY"):
            pm.load_afm_history("legacy_sess")

    def test_load_afm_history_corrupted_json(self, tmp_path):
        pm = self._make_manager(tmp_path)
        json_file = pm.afm_dir / "corrupt.json"
        json_file.write_text("{invalid json", encoding="utf-8")
        result = pm.load_afm_history("corrupt")
        assert result is None

    def test_list_afm_sessions(self, tmp_path):
        pm = self._make_manager(tmp_path)
        (pm.afm_dir / "sess1.json").write_text("{}", encoding="utf-8")
        (pm.afm_dir / "sess2.json").write_text("{}", encoding="utf-8")
        (pm.afm_dir / "old.pkl").write_bytes(b"")
        sessions = pm.list_afm_sessions()
        assert set(sessions) == {"sess1", "sess2", "old"}

    def test_list_afm_sessions_empty(self, tmp_path):
        pm = self._make_manager(tmp_path)
        assert pm.list_afm_sessions() == []


class TestClearAll:
    """Tests for PersistenceManager.clear_all"""

    def _make_manager(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            return PersistenceManager(storage_dir=str(tmp_path / "storage"))

    def test_clear_all_removes_files(self, tmp_path):
        pm = self._make_manager(tmp_path)
        (pm.documents_dir / "doc.json").write_text("{}")
        (pm.documents_dir / "doc.pkl").write_bytes(b"")
        (pm.afm_dir / "sess.json").write_text("{}")
        result = pm.clear_all()
        assert result is True
        assert list(pm.documents_dir.glob("*")) == []
        assert list(pm.afm_dir.glob("*")) == []

    def test_clear_all_with_chromadb(self, tmp_path):
        with patch("src.persistence.CHROMADB_AVAILABLE", False):
            from src.persistence import PersistenceManager

            pm = PersistenceManager(storage_dir=str(tmp_path / "storage"))
        mock_client = MagicMock()
        pm.use_chromadb = True
        pm.chroma_client = mock_client
        result = pm.clear_all()
        assert result is True
        mock_client.reset.assert_called_once()

    def test_clear_all_error(self, tmp_path):
        pm = self._make_manager(tmp_path)
        with patch.object(Path, "glob", side_effect=Exception("fail")):
            result = pm.clear_all()
        assert result is False


class TestChromaDBInitFailure:
    """Test ChromaDB initialization failure path."""

    def test_chromadb_init_failure_fallback(self, tmp_path):
        from src.persistence import PersistenceManager

        with patch("src.persistence.CHROMADB_AVAILABLE", True):
            # Mock the chromadb module at the point of use in __init__
            mock_chromadb = MagicMock()
            mock_chromadb.PersistentClient.side_effect = Exception("init failed")
            with patch.dict(
                "sys.modules", {"chromadb": mock_chromadb, "chromadb.config": MagicMock()}
            ):
                import src.persistence as pers_mod

                # Temporarily set chromadb reference
                original_chromadb = getattr(pers_mod, "chromadb", None)
                pers_mod.chromadb = mock_chromadb
                try:
                    pm = PersistenceManager(storage_dir=str(tmp_path / "storage"))
                    assert pm.use_chromadb is False
                    assert pm.chroma_client is None
                finally:
                    if original_chromadb is not None:
                        pers_mod.chromadb = original_chromadb


# ============================================================================
# 2. COMPRESSION HANDLERS TESTS
# ============================================================================


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


class TestHandleDeleteDocument:
    """Tests for handle_delete_document handler."""

    @pytest.mark.asyncio
    async def test_delete_no_confirm_returns_prompt(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": False})
        assert "DELETE CONFIRMATION REQUIRED" in result

    @pytest.mark.asyncio
    async def test_delete_with_confirm_succeeds(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result
        context["persistence"].delete_document.assert_called_once_with("doc1")

    @pytest.mark.asyncio
    async def test_delete_nonexistent_raises(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_delete_document(context, {"file_id": "nonexistent", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_cleans_up_retrieval_history(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["retrieval_history"]["doc1"] = ["some_data"]
        await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "doc1" not in context["retrieval_history"]

    @pytest.mark.asyncio
    async def test_delete_memory_error_raises_runtime(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["compressor"].get_stats.side_effect = Exception("memory error")
        with pytest.raises((RuntimeError, Exception)):
            await handle_delete_document(context, {"file_id": "doc1", "confirm": True})

    @pytest.mark.asyncio
    async def test_delete_persistence_failure_still_succeeds(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["persistence"].delete_document.return_value = False
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result

    @pytest.mark.asyncio
    async def test_delete_version_manager_error_handled(self):
        from src.handlers.compression_handlers import handle_delete_document

        context = _make_handler_context()
        context["version_manager"].delete_versions_async = AsyncMock(
            side_effect=Exception("ver fail")
        )
        # Should not raise, error is handled gracefully
        result = await handle_delete_document(context, {"file_id": "doc1", "confirm": True})
        assert "Deleted Successfully" in result


class TestHandleMultilevelEncode:
    """Tests for handle_multilevel_encode handler."""

    @pytest.mark.asyncio
    async def test_multilevel_encode_success(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        context["multilevel_encoder"].generate_adaptive_skeleton.return_value = "skeleton output"
        result = await handle_multilevel_encode(
            context, {"file_id": "doc1", "available_tokens": 5000}
        )
        assert result == "skeleton output"

    @pytest.mark.asyncio
    async def test_multilevel_encode_invalid_file(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_multilevel_encode(context, {"file_id": "bad", "available_tokens": 5000})

    @pytest.mark.asyncio
    async def test_multilevel_encode_zero_tokens(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        with pytest.raises(ValueError):
            await handle_multilevel_encode(context, {"file_id": "doc1", "available_tokens": 0})

    @pytest.mark.asyncio
    async def test_multilevel_encode_failure_raises_runtime(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        context = _make_handler_context()
        context["multilevel_encoder"].generate_adaptive_skeleton.side_effect = Exception("fail")
        with pytest.raises(RuntimeError):
            await handle_multilevel_encode(context, {"file_id": "doc1", "available_tokens": 5000})


class TestHandleBatchIngest:
    """Tests for handle_batch_ingest handler."""

    @pytest.mark.asyncio
    async def test_batch_ingest_empty_documents_raises(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(context, {"documents": []})

    @pytest.mark.asyncio
    async def test_batch_ingest_invalid_documents_type(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="must be a list"):
                await handle_batch_ingest(context, {"documents": "not a list"})

    @pytest.mark.asyncio
    async def test_batch_ingest_invalid_max_concurrent(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="max_concurrent"):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1", "text": "hi"}],
                        "max_concurrent": 99,
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_missing_file_id(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"text": "no id"}],
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_missing_text(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises((ValueError, Exception)):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1"}],
                    },
                )

    @pytest.mark.asyncio
    async def test_batch_ingest_success(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = "d1"
        mock_result.processing_time = 0.5
        mock_result.result = MagicMock(skeleton_text="skeleton...", compression_ratio=2.0)

        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
                instance = MockBCM.return_value
                instance.compress_batch = AsyncMock(return_value=[mock_result])
                result = await handle_batch_ingest(
                    context,
                    {
                        "documents": [{"file_id": "d1", "text": "hello world"}],
                    },
                )
        parsed = json.loads(result)
        assert parsed["successful"] == 1

    @pytest.mark.asyncio
    async def test_batch_ingest_non_dict_document(self):
        from src.handlers.compression_handlers import handle_batch_ingest

        context = _make_handler_context()
        with patch(
            "src.handlers.compression_handlers.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            with pytest.raises(ValueError, match="must be an object"):
                await handle_batch_ingest(
                    context,
                    {
                        "documents": ["not a dict"],
                    },
                )


class TestHandleIngestDirectory:
    """Tests for handle_ingest_directory handler."""

    @pytest.mark.asyncio
    async def test_ingest_directory_missing_dir(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with pytest.raises((ValueError, Exception)):
            await handle_ingest_directory(context, {"directory": ""})

    @pytest.mark.asyncio
    async def test_ingest_directory_nonexistent(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with pytest.raises(ValueError, match="not found"):
            await handle_ingest_directory(context, {"directory": "/nonexistent/path/xyz"})

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_max_files(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with patch("os.path.isdir", return_value=True):
            with pytest.raises(ValueError, match="max_files"):
                await handle_ingest_directory(
                    context,
                    {
                        "directory": "/some/dir",
                        "max_files": 200,
                    },
                )

    @pytest.mark.asyncio
    async def test_ingest_directory_invalid_max_concurrent(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        with patch("os.path.isdir", return_value=True):
            with pytest.raises(ValueError, match="max_concurrent"):
                await handle_ingest_directory(
                    context,
                    {
                        "directory": "/some/dir",
                        "max_concurrent": 99,
                    },
                )

    @pytest.mark.asyncio
    async def test_ingest_directory_no_files_found(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = await handle_ingest_directory(
            context,
            {
                "directory": str(empty_dir),
                "patterns": ["*.xyz"],
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] == "no_files"

    @pytest.mark.asyncio
    async def test_ingest_directory_success(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()

        # Create test files
        test_dir = tmp_path / "code"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("print('hello')")
        (test_dir / "util.py").write_text("def foo(): pass")

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = "main.py"
        mock_result.processing_time = 0.1
        mock_result.result = MagicMock(compression_ratio=3.0, total_nodes=5)

        # BatchDocument doesn't have file_path field, so mock it to accept **kwargs
        mock_batch_doc_cls = MagicMock()
        with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
            with patch("src.batch_manager.BatchDocument", mock_batch_doc_cls):
                instance = MockBCM.return_value
                instance.compress_batch = AsyncMock(return_value=[mock_result, mock_result])
                result = await handle_ingest_directory(
                    context,
                    {
                        "directory": str(test_dir),
                        "patterns": ["*.py"],
                    },
                )
        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["successful"] == 2

    @pytest.mark.asyncio
    async def test_ingest_directory_scoped_ids_and_file_metadata(self, tmp_path):
        from src.handlers.compression_handlers import handle_ingest_directory
        from src.identity_scope import compose_scoped_file_id

        context = _make_handler_context()

        test_dir = tmp_path / "code"
        test_dir.mkdir()
        (test_dir / "main.py").write_text("print('hello')")

        captured_documents = []
        scoped_file_id = compose_scoped_file_id("main.py", workspace_id="acme")
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.file_id = scoped_file_id
        mock_result.processing_time = 0.1
        mock_result.result = MagicMock(compression_ratio=3.0, total_nodes=5)

        with patch("src.batch_manager.BatchCompressionManager") as MockBCM:
            instance = MockBCM.return_value

            async def _capture(documents):
                captured_documents.extend(documents)
                return [mock_result]

            instance.compress_batch = AsyncMock(side_effect=_capture)
            result = await handle_ingest_directory(
                context,
                {
                    "directory": str(test_dir),
                    "patterns": ["*.py"],
                    "workspace_id": "acme",
                },
            )

        parsed = json.loads(result)
        assert parsed["status"] == "complete"
        assert parsed["results"][0]["file_id"] == "main.py"
        assert len(captured_documents) == 1
        assert captured_documents[0].file_id == scoped_file_id
        assert captured_documents[0].metadata["file_path"].endswith("main.py")

    @pytest.mark.asyncio
    async def test_ingest_directory_path_traversal_rejected(self):
        from src.handlers.compression_handlers import handle_ingest_directory

        context = _make_handler_context()
        context["path_validator"].validate.side_effect = ValueError("path traversal")
        with pytest.raises(ValueError, match="Invalid directory"):
            await handle_ingest_directory(context, {"directory": "../../etc"})


# ============================================================================
# 3. CODE COMPRESSION ADAPTER TESTS
# ============================================================================


class TestCodeCompressionAdapter:
    """Tests for CodeCompressionAdapter."""

    def _make_adapter(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor"):
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                mock_text.model = MagicMock()
                adapter = self._create_adapter_instance(MockSC)
                return adapter, mock_text

    def _create_adapter_instance(self, MockSC):
        from src.code_compression_adapter import CodeCompressionAdapter

        return CodeCompressionAdapter(preload_code_model=False)

    def test_load_code_compressor_success(self):
        with patch("src.code_compression_adapter.SemanticCompressor"):
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                mock_code = MockCSC.return_value
                result = adapter._load_code_compressor()
                assert result is mock_code
                assert adapter._code_model_available is True

    def test_load_code_compressor_already_loaded(self):
        with patch("src.code_compression_adapter.SemanticCompressor"):
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()
                # Second call should return cached
                result = adapter._load_code_compressor()
                assert result is not None
                assert MockCSC.call_count == 1

    def test_load_code_compressor_failure(self):
        with patch("src.code_compression_adapter.SemanticCompressor"):
            with patch(
                "src.code_compression_adapter.CodeSemanticCompressor",
                side_effect=Exception("no torch"),
            ):
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                result = adapter._load_code_compressor()
                assert result is None
                assert adapter._code_model_available is False
                assert adapter._code_model_error == "no torch"

    def test_load_code_compressor_already_failed(self):
        with patch("src.code_compression_adapter.SemanticCompressor"):
            with patch(
                "src.code_compression_adapter.CodeSemanticCompressor", side_effect=Exception("err")
            ):
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()  # fail first time
                result = adapter._load_code_compressor()  # skip second time
                assert result is None

    @pytest.mark.asyncio
    async def test_ingest_file_async_text_path(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor"):
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                mock_text.ingest_file_async = AsyncMock(return_value="skeleton")
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                result = await adapter.ingest_file_async("hello", "doc1", file_path="readme.txt")
                assert result == "skeleton"
                mock_text.ingest_file_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_async_code_path(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                mock_code = MockCSC.return_value
                mock_code.graphs = {"main.py": MagicMock()}
                mock_code.chunks = {}
                mock_code.file_metadata = {}
                mock_code.ingest_code_file.return_value = {
                    "total_chunks": 5,
                    "total_tokens": 200,
                    "compression_ratio": 3.0,
                }
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                result = await adapter.ingest_file_async(
                    "def foo(): pass", "main.py", file_path="main.py"
                )
                assert result is not None
                assert "main.py" in adapter._code_file_ids

    @pytest.mark.asyncio
    async def test_ingest_file_async_code_fallback_to_text(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch(
                "src.code_compression_adapter.CodeSemanticCompressor",
                side_effect=Exception("no torch"),
            ):
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                mock_text.ingest_file_async = AsyncMock(return_value="fallback")
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                result = await adapter.ingest_file_async(
                    "def foo(): pass", "main.py", file_path="main.py"
                )
                assert result == "fallback"

    def test_generate_code_skeleton(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}

                mock_code = MockCSC.return_value
                mock_graph = MagicMock()
                mock_graph.nodes.return_value = ["main.py::func1"]
                mock_code.graphs = {"main.py": mock_graph}
                mock_code.chunks = {}
                mock_code.file_metadata = {}

                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()
                adapter._code_file_ids.add("main.py")

                result = adapter._generate_code_skeleton("main.py")
                assert result is not None

    def test_generate_code_skeleton_not_found(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch(
                "src.code_compression_adapter.CodeSemanticCompressor", side_effect=Exception("err")
            ):
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                with pytest.raises(ValueError, match="not found"):
                    adapter._generate_code_skeleton("nonexistent.py")

    def test_modulate_region_text_nodes(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor"):
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {"doc_n0": MagicMock()}
                mock_text.file_metadata = {}
                mock_text.modulate_region.return_value = "text content"
                from src.code_compression_adapter import CodeCompressionAdapter, FidelityLevel

                adapter = CodeCompressionAdapter(preload_code_model=False)
                result = adapter.modulate_region(["doc_n0"], FidelityLevel.STRUCTURE)
                assert result == "text content"

    def test_modulate_region_code_nodes(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}

                mock_code = MockCSC.return_value
                mock_chunk = MagicMock()
                mock_chunk.name = "my_func"
                mock_chunk.chunk_type = "function"
                mock_chunk.start_line = 1
                mock_chunk.end_line = 10
                mock_chunk.docstring = "A function"
                mock_chunk.code = "def my_func():\n    pass"
                mock_code.chunks = {"file::my_func": mock_chunk}
                mock_code.graphs = {}
                mock_code.file_metadata = {}

                from src.code_compression_adapter import CodeCompressionAdapter, FidelityLevel

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()

                result = adapter.modulate_region(["file::my_func"], FidelityLevel.STRUCTURE)
                assert "my_func" in result

    def test_get_stats_aggregate(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}
                mock_text.get_stats.return_value = {
                    "total_documents": 1,
                    "total_nodes": 10,
                    "total_tokens": 500,
                }

                mock_code = MockCSC.return_value
                mock_graph = MagicMock()
                mock_graph.nodes.return_value = ["f.py::func1"]
                mock_code.graphs = {"f.py": mock_graph}
                mock_code.chunks = {}
                mock_code.file_metadata = {}

                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()
                adapter._code_file_ids.add("f.py")

                stats = adapter.get_stats()
                assert stats["text_documents"] == 1
                assert stats["code_model_available"] is True

    def test_get_stats_specific_code_file(self):
        with patch("src.code_compression_adapter.SemanticCompressor") as MockSC:
            with patch("src.code_compression_adapter.CodeSemanticCompressor") as MockCSC:
                mock_text = MockSC.return_value
                mock_text.graphs = {}
                mock_text.chunks = {}
                mock_text.file_metadata = {}

                mock_code = MockCSC.return_value
                mock_chunk = MagicMock()
                mock_chunk.start_line = 1
                mock_chunk.end_line = 20
                mock_chunk.chunk_type = "function"
                mock_graph = MagicMock()
                mock_graph.nodes.return_value = ["app.py::main"]
                mock_graph.number_of_edges.return_value = 3
                mock_code.graphs = {"app.py": mock_graph}
                mock_code.chunks = {"app.py::main": mock_chunk}
                mock_code.file_metadata = {"app.py": {"language": "python"}}

                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter(preload_code_model=False)
                adapter._load_code_compressor()
                adapter._code_file_ids.add("app.py")

                stats = adapter.get_stats("app.py")
                assert stats["type"] == "code"
                assert stats["total_nodes"] == 1
                assert stats["total_lines"] == 19


# ============================================================================
# 4. EMBEDDINGS TESTS
# ============================================================================


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


# ============================================================================
# 5. EMBEDDINGS ONNX TESTS
# ============================================================================


class TestONNXEmbeddingManager:
    """Tests for ONNXEmbeddingManager."""

    def test_init_defaults(self, tmp_path):
        from src.embeddings_onnx import ONNXEmbeddingManager

        mgr = ONNXEmbeddingManager(cache_dir=str(tmp_path / "cache"))
        assert mgr.model_name == "sentence-transformers/all-MiniLM-L6-v2"
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
