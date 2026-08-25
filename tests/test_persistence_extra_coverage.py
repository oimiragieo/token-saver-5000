"""persistence — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
import time


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


class TestPersistence:
    """Cover persistence edge cases."""

    def test_chromadb_not_available(self):
        """Cover lines 34-36 - ChromaDB import fallback."""
        from src.persistence import CHROMADB_AVAILABLE

        assert isinstance(CHROMADB_AVAILABLE, bool)

    def test_chromadb_init_failure(self, tmp_path):
        """Cover line 74 - ChromaDB init failure falls back."""
        from src.persistence import PersistenceManager

        mock_chroma = MagicMock()
        mock_chroma.PersistentClient.side_effect = Exception("ChromaDB error")
        mock_settings = MagicMock()
        with patch("src.persistence.CHROMADB_AVAILABLE", True):
            with patch("src.persistence.chromadb", mock_chroma, create=True):
                with patch("src.persistence.Settings", mock_settings, create=True):
                    mgr = PersistenceManager(storage_dir=str(tmp_path))
                    assert mgr.use_chromadb is False

    def test_serialize_non_ndarray_embedding(self):
        """Cover line 266 - embedding that's not ndarray."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager.__new__(PersistenceManager)
        node = MagicMock()
        node.text = "hello"
        node.importance = 0.5
        node.metadata = {}
        node.embedding = [0.1, 0.2, 0.3]  # List, not ndarray
        chunks = {"n0": node}
        result = mgr._serialize_chunks_safe(chunks)
        assert result["n0"]["embedding"] == [0.1, 0.2, 0.3]

    def test_load_document_json_with_data(self, tmp_path):
        """Cover JSON load path - valid data."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager(storage_dir=str(tmp_path))
        # Test that loading non-existent doc returns None
        result = mgr._load_document_json("nonexistent")
        assert result is None

    def test_load_document_json_legacy_ids(self, tmp_path):
        """Cover lines 539-543 - legacy numpy IDs trigger warning."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager(storage_dir=str(tmp_path))

        json_file = mgr.documents_dir / "doc1.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        json_file.write_text(
            json.dumps({"chunks": {"n0": {"text": "hello", "importance": 0.5, "metadata": {}}}})
        )

        emb_file = mgr.documents_dir / "doc1_chunks.npz"
        np.savez(emb_file, embeddings=np.random.rand(1, 384), ids=np.array(["n0"]))

        # This triggers the legacy IDs warning path (line 539-543)
        # It may fail on deserialization but the target lines are executed
        mgr._load_document_json("doc1")
        # The legacy IDs path is hit even if full deserialization fails

    def test_delete_document_error(self, tmp_path):
        """Cover line 654 - delete error."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager.__new__(PersistenceManager)
        mgr.use_chromadb = False
        with patch.object(mgr, "_delete_document_json", side_effect=Exception("fail")):
            result = mgr.delete_document("doc1")
            assert result is False

    def test_deserialize_message_safe(self):
        """Cover lines 749, 758."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager.__new__(PersistenceManager)
        msg_data = {
            "role": "user",
            "content": "hello",
            "turn": 1,
            "turn_index": 0,
            "importance": "critical",
            "fidelity": "full",
            "embedding": [0.1, 0.2, 0.3],
            "timestamp": time.time(),
            "token_count": 5,
            "placeholder_stub": None,
        }
        result = mgr._deserialize_message_safe(msg_data)
        assert result is not None

    def test_load_afm_state_with_embeddings(self, tmp_path):
        """Cover lines 874-881 - AFM state with embeddings."""
        from src.persistence import PersistenceManager

        mgr = PersistenceManager(storage_dir=str(tmp_path))

        json_file = mgr.afm_dir / "default.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        msg_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "turn": 1,
                    "turn_index": 0,
                    "importance": "trivial",
                    "fidelity": "full",
                    "embedding": None,
                    "timestamp": time.time(),
                    "token_count": 1,
                    "placeholder_stub": None,
                }
            ],
            "current_turn": 1,
        }
        json_file.write_text(json.dumps(msg_data))

        emb_file = mgr.afm_dir / "default_embeddings.npz"
        np.savez(emb_file, embeddings=np.random.rand(1, 384), indices=np.array([0]))

        result = mgr.load_afm_history("default")
        assert result is not None
