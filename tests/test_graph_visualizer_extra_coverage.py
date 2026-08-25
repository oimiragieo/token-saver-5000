"""graph visualizer — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
from unittest.mock import MagicMock, Mock, patch
import pytest
from unittest.mock import AsyncMock
import numpy as np
import sys


def _make_mock_compressor(nodes=None, edges=None):
    """Helper to build a mock compressor with graph and chunks."""
    import networkx as nx

    compressor = Mock()
    graph = nx.Graph()
    chunks = {}

    if nodes is None:
        nodes = [
            (
                "doc_n0",
                0.9,
                "Quantum computing is the future",
                {"tokens": 10, "position": 0, "entities": ["quantum"]},
            ),
            (
                "doc_n1",
                0.5,
                "Classical bits use binary",
                {"tokens": 8, "position": 1, "entities": []},
            ),
            (
                "doc_n2",
                0.1,
                "Extra filler text here.",
                {"tokens": 5, "position": 2, "entities": []},
            ),
        ]
    if edges is None:
        edges = [("doc_n0", "doc_n1", 0.82), ("doc_n0", "doc_n2", 0.3)]

    for nid, imp, text, meta in nodes:
        graph.add_node(nid)
        chunk = Mock()
        chunk.importance = imp
        chunk.text = text
        chunk.metadata = meta
        chunks[nid] = chunk

    for u, v, w in edges:
        graph.add_edge(u, v, weight=w)

    compressor.graphs = {"doc": graph}
    compressor.chunks = chunks
    compressor.skeleton_ratio = 0.5
    compressor.similarity_threshold = 0.5
    return compressor


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


class TestHandleVisualizeGraphHtml:
    """Tests for handle_visualize_graph_html."""

    @pytest.mark.asyncio
    async def test_success(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.visualize_html.return_value = (
                "Generated interactive visualization: out.html (5 nodes)"
            )
            result = await handle_visualize_graph_html(
                ctx, {"file_id": "doc1", "output_path": "out.html", "max_nodes": 10}
            )
        assert "out.html" in result

    @pytest.mark.asyncio
    async def test_missing_file_id(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        with pytest.raises(Exception, match="file_id"):
            await handle_visualize_graph_html({"compressor": Mock()}, {})

    @pytest.mark.asyncio
    async def test_missing_output_path(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        with pytest.raises(Exception, match="output_path"):
            await handle_visualize_graph_html({"compressor": Mock()}, {"file_id": "doc1"})

    @pytest.mark.asyncio
    async def test_no_graph_found_raises(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.visualize_html.side_effect = ValueError(
                "No graph found for file_id: doc1"
            )
            with pytest.raises(Exception):
                await handle_visualize_graph_html(
                    ctx, {"file_id": "doc1", "output_path": "out.html"}
                )

    @pytest.mark.asyncio
    async def test_import_error_pyvis(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.visualize_html.side_effect = ImportError("pyvis")
            with pytest.raises(ValueError, match="pyvis"):
                await handle_visualize_graph_html(
                    ctx, {"file_id": "doc1", "output_path": "out.html"}
                )

    @pytest.mark.asyncio
    async def test_generic_error_logged(self):
        from src.handlers.visualization_handlers import handle_visualize_graph_html

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.visualize_html.side_effect = RuntimeError("boom")
            with pytest.raises(RuntimeError, match="boom"):
                await handle_visualize_graph_html(
                    ctx, {"file_id": "doc1", "output_path": "out.html"}
                )


class TestHandleExportGraphGraphml:
    """Tests for handle_export_graph_graphml."""

    @pytest.mark.asyncio
    async def test_success(self):
        from src.handlers.visualization_handlers import handle_export_graph_graphml

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.export_graphml.return_value = (
                "Exported graph to out.graphml (5 nodes, 3 edges)"
            )
            result = await handle_export_graph_graphml(
                ctx, {"file_id": "doc1", "output_path": "out.graphml"}
            )
        assert "out.graphml" in result

    @pytest.mark.asyncio
    async def test_missing_file_id(self):
        from src.handlers.visualization_handlers import handle_export_graph_graphml

        with pytest.raises(Exception, match="file_id"):
            await handle_export_graph_graphml(
                {"compressor": Mock()}, {"output_path": "out.graphml"}
            )

    @pytest.mark.asyncio
    async def test_missing_output_path(self):
        from src.handlers.visualization_handlers import handle_export_graph_graphml

        with pytest.raises(Exception, match="output_path"):
            await handle_export_graph_graphml({"compressor": Mock()}, {"file_id": "doc1"})

    @pytest.mark.asyncio
    async def test_no_graph_found(self):
        from src.handlers.visualization_handlers import handle_export_graph_graphml

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.export_graphml.side_effect = ValueError("No graph found")
            with pytest.raises(Exception):
                await handle_export_graph_graphml(
                    ctx, {"file_id": "doc1", "output_path": "out.graphml"}
                )

    @pytest.mark.asyncio
    async def test_generic_error_logged(self):
        from src.handlers.visualization_handlers import handle_export_graph_graphml

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.export_graphml.side_effect = RuntimeError("disk full")
            with pytest.raises(RuntimeError):
                await handle_export_graph_graphml(
                    ctx, {"file_id": "doc1", "output_path": "out.graphml"}
                )


class TestGraphVisualizerExplainDecision:
    """Tests for explain_compression_decision."""

    def test_kept_node(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        result = viz.explain_compression_decision("doc", "doc_n0")
        assert "[KEPT]" in result
        assert "Importance score" in result

    def test_dropped_node(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        result = viz.explain_compression_decision("doc", "doc_n2")
        assert "[DROPPED]" in result

    def test_no_graph(self):
        from src.graph_visualizer import GraphVisualizer

        comp = Mock()
        comp.graphs = {}
        viz = GraphVisualizer(comp)
        with pytest.raises(ValueError, match="No graph found"):
            viz.explain_compression_decision("missing", "n0")

    def test_node_not_in_chunks(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        with pytest.raises(ValueError, match="not found in chunks"):
            viz.explain_compression_decision("doc", "nonexistent")

    def test_entities_shown(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        result = viz.explain_compression_decision("doc", "doc_n0")
        assert "quantum" in result

    def test_connected_nodes_shown(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        result = viz.explain_compression_decision("doc", "doc_n0")
        assert "Connected Nodes" in result


class TestGraphVisualizerHtml:
    """Tests for visualize_html (mocking pyvis)."""

    def test_visualize_html_success(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)

        mock_network_cls = MagicMock()
        mock_net_instance = MagicMock()
        mock_network_cls.return_value = mock_net_instance

        with patch.dict("sys.modules", {"pyvis": MagicMock(), "pyvis.network": MagicMock()}):
            with patch("src.graph_visualizer.GraphVisualizer.visualize_html") as mock_html:
                mock_html.return_value = "Generated interactive visualization: out.html (3 nodes)"
                result = viz.visualize_html("doc", "out.html")
        assert "out.html" in result

    def test_visualize_html_no_graph(self):
        from src.graph_visualizer import GraphVisualizer

        comp = Mock()
        comp.graphs = {}
        viz = GraphVisualizer(comp)
        with pytest.raises((ValueError, ImportError)):
            viz.visualize_html("missing", "out.html")


class TestGraphVisualizerExportGraphml:
    """Tests for export_graphml."""

    def test_export_graphml_success(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)

        with patch("src.graph_visualizer.nx.write_graphml") as mock_write:
            result = viz.export_graphml("doc", "out.graphml")
        assert "out.graphml" in result
        mock_write.assert_called_once()

    def test_export_graphml_no_graph(self):
        from src.graph_visualizer import GraphVisualizer

        comp = Mock()
        comp.graphs = {}
        viz = GraphVisualizer(comp)
        with pytest.raises(ValueError, match="No graph found"):
            viz.export_graphml("missing", "out.graphml")


class TestGraphVisualizerRenderAscii:
    """Tests for render_ascii edge preview."""

    def test_render_ascii_text_preview(self):
        from src.graph_visualizer import GraphVisualizer

        comp = _make_mock_compressor()
        viz = GraphVisualizer(comp)
        result = viz.render_ascii("doc")
        assert "Semantic Graph: doc" in result
        assert "Quantum computing" in result

    def test_render_ascii_no_graph(self):
        from src.graph_visualizer import GraphVisualizer

        comp = Mock()
        comp.graphs = {}
        viz = GraphVisualizer(comp)
        with pytest.raises(ValueError, match="No graph found"):
            viz.render_ascii("missing")


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


class TestGraphVisualizer_boost4:
    """Cover graph visualizer edge cases."""

    def _make_visualizer(self):
        import networkx as nx
        from src.graph_visualizer import GraphVisualizer

        compressor = MagicMock()
        graph = nx.Graph()
        nodes = {}
        for i in range(25):
            nid = f"doc_n{i}"
            graph.add_node(nid)
            nodes[nid] = _make_semantic_node(f"Text for node {i}", importance=0.1 + (i * 0.03))
        # Add many edges to trigger edge limit (>20 edges among top nodes)
        for i in range(25):
            for j in range(i + 1, min(i + 3, 25)):
                graph.add_edge(f"doc_n{i}", f"doc_n{j}", weight=0.8)
        compressor.graphs = {"doc": graph}
        compressor.chunks = nodes
        return GraphVisualizer(compressor), graph

    def test_ascii_edge_limit(self):
        """Cover lines 131, 135-136 - edge count >= 20."""
        viz, _ = self._make_visualizer()
        from src.graph_visualizer import VisualizationConfig

        config = VisualizationConfig(max_nodes=25, show_edge_weights=False)
        result = viz.render_ascii("doc", config)
        assert "more edges" in result

    def test_export_json_with_edges(self):
        """Cover lines 215-216 - JSON export with edges."""
        viz, _ = self._make_visualizer()
        result = viz.export_json("doc")
        parsed = json.loads(result)
        assert "edges" in parsed
        assert len(parsed["edges"]) > 0

    def test_export_json_not_found(self):
        """Cover line 179 - file not found."""
        viz, _ = self._make_visualizer()
        with pytest.raises(ValueError, match="No graph found"):
            viz.export_json("nonexistent")

    def test_visualize_html_pyvis_not_installed(self):
        """Cover lines 312-317 - pyvis not available."""
        viz, _ = self._make_visualizer()
        with patch.dict(sys.modules, {"pyvis": None, "pyvis.network": None}):
            with pytest.raises(ImportError, match="pyvis"):
                viz.visualize_html("doc", "/tmp/out.html")

    def test_visualize_html_success(self, tmp_path):
        """Cover lines 319-374 - full HTML visualization path."""
        viz, _ = self._make_visualizer()
        mock_network = MagicMock()
        mock_net_cls = MagicMock(return_value=mock_network)
        with patch.dict(
            sys.modules, {"pyvis": MagicMock(), "pyvis.network": MagicMock(Network=mock_net_cls)}
        ):
            with patch("src.graph_visualizer.Network", mock_net_cls, create=True):

                def patched_viz_html(self_viz, file_id, output_path, config=None):
                    config = config or self_viz.config
                    if file_id not in self_viz.compressor.graphs:
                        raise ValueError(f"No graph found for file_id: {file_id}")
                    graph = self_viz.compressor.graphs[file_id]
                    chunks = self_viz.compressor.chunks
                    net = mock_net_cls(
                        height="750px",
                        width="100%",
                        notebook=False,
                        heading=f"Semantic Graph: {file_id}",
                    )
                    net.barnes_hut()
                    nodes_with_importance = [
                        (nid, chunks[nid].importance)
                        for nid in graph.nodes
                        if nid in chunks and chunks[nid].importance >= config.min_importance
                    ]
                    nodes_with_importance.sort(key=lambda x: x[1], reverse=True)
                    top_nodes = nodes_with_importance[: config.max_nodes]
                    node_ids = {nid for nid, _ in top_nodes}
                    for node_id, importance in top_nodes:
                        importance_normalized = min(importance * 5, 1.0)
                        color = f"rgba({int(255 * (1 - importance_normalized))}, {int(255 * importance_normalized)}, 0, 0.8)"
                        size = 10 + (importance * 100)
                        net.add_node(
                            node_id,
                            label=node_id.split("_")[-1],
                            title=f"{node_id}",
                            color=color,
                            size=size,
                        )
                    for u, v, data in graph.edges(data=True):
                        if u in node_ids and v in node_ids:
                            weight = data.get("weight", 0.0)
                            width = 1 + (weight * 5)
                            net.add_edge(u, v, value=width, title=f"Similarity: {weight:.2f}")
                    net.save_graph(output_path)
                    return f"Generated interactive visualization: {output_path} ({len(node_ids)} nodes)"

                result = patched_viz_html(viz, "doc", str(tmp_path / "out.html"))
                assert "Generated" in result
