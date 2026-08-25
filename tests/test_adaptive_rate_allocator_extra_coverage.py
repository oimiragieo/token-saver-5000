"""adaptive rate allocator — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest


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


class TestAdaptiveRateAllocator_boost4b:
    """Cover adaptive rate allocator and multi-level encoder."""

    def test_context_window_adapter_no_graph(self):
        """Cover line 230 - file not found."""
        from src.adaptive_rate_allocator import ContextWindowAdapter

        compressor = MagicMock()
        compressor.graphs = {}
        adapter = ContextWindowAdapter(compressor)
        with pytest.raises(ValueError, match="not found"):
            adapter.adapt_to_context_window("missing", 1000, 5000)

    def test_multilevel_encoder_include_auxiliary(self):
        """Cover lines 332-338 - include auxiliary and detail nodes."""
        from src.adaptive_rate_allocator import MultiLevelSemanticEncoder

        compressor = MagicMock()
        import networkx as nx

        graph = nx.Graph()
        nodes = {}
        for i in range(20):
            nid = f"doc_n{i}"
            graph.add_node(nid)
            nodes[nid] = _make_semantic_node(f"Node {i}", importance=0.1 + i * 0.04)
        compressor.graphs = {"doc": graph}
        compressor.chunks = nodes

        encoder = MultiLevelSemanticEncoder(compressor)
        levels = encoder.encode_multilevel("doc", available_tokens=50000)
        assert "main" in levels
        assert "auxiliary" in levels
        assert "detail" in levels

    def test_generate_adaptive_skeleton_all_levels(self):
        """Cover lines 350-367 - all levels included."""
        from src.adaptive_rate_allocator import MultiLevelSemanticEncoder

        compressor = MagicMock()
        import networkx as nx

        graph = nx.Graph()
        nodes = {}
        for i in range(20):
            nid = f"doc_n{i}"
            graph.add_node(nid)
            nodes[nid] = _make_semantic_node(f"Node {i} content here", importance=0.1 + i * 0.04)
        compressor.graphs = {"doc": graph}
        compressor.chunks = nodes
        compressor._generate_summary.return_value = "summary text"

        encoder = MultiLevelSemanticEncoder(compressor)
        result = encoder.generate_adaptive_skeleton("doc", available_tokens=100000)
        assert "MULTI-LEVEL" in result
        assert "[MAIN]" in result
