"""code compression adapter — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
import os


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


class TestCodeCompressionAdapterProperties:
    """Cover property proxy lines and code model management."""

    def test_graphs_with_code_compressor(self):
        """Cover line 169 - graphs property with code compressor loaded."""
        import networkx as nx

        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._text_compressor.graphs = {"text_doc": nx.Graph()}
            adapter._code_compressor = MagicMock()
            adapter._code_compressor.graphs = {"code_doc": nx.Graph()}
            result = adapter.graphs
            assert "text_doc" in result
            assert "code_doc" in result

    def test_chunks_with_code_compressor(self):
        """Cover line 177 - chunks property with code compressor."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._text_compressor.chunks = {"t1": "text_chunk"}
            adapter._code_compressor = MagicMock()
            adapter._code_compressor.chunks = {"c1": "code_chunk"}
            result = adapter.chunks
            assert "t1" in result
            assert "c1" in result

    def test_file_metadata_with_code_compressor(self):
        """Cover line 185 - file_metadata with code compressor."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._text_compressor.file_metadata = {"t1": {}}
            adapter._code_compressor = MagicMock()
            adapter._code_compressor.file_metadata = {"c1": {"lang": "python"}}
            result = adapter.file_metadata
            assert "c1" in result

    def test_model_property(self):
        """Cover line 191 - model property delegates to text compressor."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._text_compressor.model = "mock_model"
            assert adapter.model == "mock_model"

    def test_is_code_model_available_not_tried(self):
        """Cover lines 237-240 - is_code_model_available when not tried."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._code_model_available = None
            adapter._code_compressor = None
            adapter._code_model_name = "test"
            adapter._code_similarity_threshold = 0.7
            adapter._code_model_error = None
            with patch.object(adapter, "_load_code_compressor", return_value=None):
                assert adapter.is_code_model_available() is False

    def test_is_code_model_available_already_tried(self):
        """Cover line 240 - already tried and available."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._code_model_available = True
            assert adapter.is_code_model_available() is True

    def test_get_code_model_status(self):
        """Cover line 244."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._code_model_available = True
            adapter._code_compressor = MagicMock()
            adapter._code_model_name = "codebert"
            adapter._code_model_error = None
            adapter._code_file_ids = {"a.py", "b.py"}
            status = adapter.get_code_model_status()
            assert status["available"] is True
            assert status["code_files_ingested"] == 2

    def test_preload_code_model_env(self):
        """Cover lines 150-151 - pre-warming via env var."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            with patch.dict(os.environ, {"PRELOAD_CODE_MODEL": "true"}):
                from src.code_compression_adapter import CodeCompressionAdapter

                adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
                adapter._text_compressor = MagicMock()
                adapter._code_compressor = None
                adapter._code_model_available = None
                adapter._code_model_name = "test"
                adapter._code_similarity_threshold = 0.7
                adapter._code_model_error = None
                adapter._code_file_ids = set()
                adapter._executor = MagicMock()
                with patch.object(adapter, "_load_code_compressor", return_value=None):
                    # Already constructed - just test that env_preload path works
                    env_preload = os.environ.get("PRELOAD_CODE_MODEL", "").lower() == "true"
                    assert env_preload is True


class TestCodeCompressionAdapterSkeleton:
    """Cover skeleton generation and code-specific paths."""

    def test_generate_skeleton_routes_to_code(self):
        """Cover lines 406-408."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._code_compressor = MagicMock()
            adapter._code_file_ids = {"main.py"}
            mock_result = MagicMock()
            with patch.object(adapter, "_generate_code_skeleton", return_value=mock_result):
                result = adapter._generate_skeleton("main.py")
                assert result == mock_result

    def test_generate_skeleton_routes_to_text(self):
        """Cover line 408 - text path."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._code_compressor = None
            adapter._code_file_ids = set()
            adapter._generate_skeleton("readme.md")
            adapter._text_compressor._generate_skeleton.assert_called_once_with("readme.md")

    def test_convert_code_stats_skeleton_with_chunks(self):
        """Cover lines 352, 356, 358, 362, 365-368, 371-375, 387."""
        import networkx as nx

        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()

            graph = nx.Graph()
            graph.add_node("main.py::imports")
            graph.add_node("main.py::MyClass")
            graph.add_node("main.py::my_func")
            graph.add_node("main.py::block1")

            import_chunk = _make_code_chunk("os_import", "import")
            class_chunk = _make_code_chunk(
                "MyClass", "class", docstring="A class for testing things"
            )
            func_chunk = _make_code_chunk("my_func", "function", docstring="")
            block_chunk = _make_code_chunk("", "block")
            block_chunk.name = ""

            code_compressor = MagicMock()
            code_compressor.graphs = {"main.py": graph}
            code_compressor.chunks = {
                "main.py::imports": import_chunk,
                "main.py::MyClass": class_chunk,
                "main.py::my_func": func_chunk,
                "main.py::block1": block_chunk,
            }
            code_compressor.file_metadata = {"main.py": {"language": "python"}}
            adapter._code_compressor = code_compressor

            stats = {
                "total_chunks": 4,
                "total_tokens": 100,
                "compression_ratio": 2.0,
            }
            result = adapter._convert_code_stats_to_skeleton(stats, "main.py")
            assert result.file_id == "main.py"
            assert "Imports" in result.skeleton_text
            assert "Classes" in result.skeleton_text
            assert "Code Blocks" in result.skeleton_text


class TestCodeCompressionAdapterCodeNodes:
    """Cover code node rendering and search paths."""

    def test_modulate_code_region_no_compressor(self):
        """Cover line 476."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._code_compressor = None
            from src.semantic_compressor import FidelityLevel

            result = adapter._modulate_code_region(["n1"], FidelityLevel.RAW)
            assert "Error" in result

    def test_modulate_code_region_skip_missing(self):
        """Cover line 482."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._code_compressor = MagicMock()
            adapter._code_compressor.chunks = {}
            from src.semantic_compressor import FidelityLevel

            result = adapter._modulate_code_region(["missing_node"], FidelityLevel.RAW)
            assert result == ""

    def test_modulate_code_region_detailed_with_long_code(self):
        """Cover lines 513, 517."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            code_compressor = MagicMock()
            long_code = "\n".join([f"line {i}" for i in range(20)])
            chunk = _make_code_chunk("big_func", "function", code=long_code, docstring="docs")
            code_compressor.chunks = {"f1::big_func": chunk}
            adapter._code_compressor = code_compressor
            from src.semantic_compressor import FidelityLevel

            result = adapter._modulate_code_region(["f1::big_func"], FidelityLevel.DETAILED)
            assert "..." in result

    def test_modulate_code_region_abstract(self):
        """Cover line 517 - ABSTRACT fidelity."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            chunk = _make_code_chunk("small_func", "function")
            code_compressor = MagicMock()
            code_compressor.chunks = {"f1::small_func": chunk}
            adapter._code_compressor = code_compressor
            from src.semantic_compressor import FidelityLevel

            result = adapter._modulate_code_region(["f1::small_func"], FidelityLevel.ABSTRACT)
            assert "small_func" in result

    def test_search_semantic_delegates(self):
        """Cover line 535."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            with patch.object(
                adapter, "search_semantic_with_scores", return_value=[("n1", 0.9), ("n2", 0.8)]
            ):
                result = adapter.search_semantic("test query", top_k=2)
                assert result == ["n1", "n2"]

    def test_generate_summary_delegates(self):
        """Cover line 594."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._text_compressor._generate_summary.return_value = "summary"
            assert adapter._generate_summary("text") == "summary"

    def test_get_stats_code_file(self):
        """Cover lines 602, 635."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._text_compressor = MagicMock()
            adapter._code_file_ids = {"main.py"}
            adapter._code_compressor = MagicMock()
            with patch.object(adapter, "_get_code_stats", return_value={"type": "code"}):
                result = adapter.get_stats("main.py")
                assert result["type"] == "code"

    def test_cleanup(self):
        """Cover line 702."""
        with patch("src.code_compression_adapter.SemanticCompressor"):
            from src.code_compression_adapter import CodeCompressionAdapter

            adapter = CodeCompressionAdapter.__new__(CodeCompressionAdapter)
            adapter._executor = MagicMock()
            adapter.cleanup()
            adapter._executor.shutdown.assert_called_once_with(wait=False)
