"""
Tests for Graph Visualization (v0.6.0)

This module tests the GraphVisualizer, visualization MCP tools,
and related graph export/analysis features added in v0.6.0.

Test Categories:
- ASCII rendering
- JSON export
- GraphML export
- HTML visualization
- Compression decision explanations
- MCP tool integration
- Edge cases and error handling
"""

import json
import os
import tempfile
import pytest
from src.identity_scope import compose_scoped_file_id
from src.graph_visualizer import GraphVisualizer, VisualizationConfig
from src.semantic_compressor import SemanticCompressor
from src.handlers import visualization_handlers


# ===========================
# Fixtures
# ===========================


@pytest.fixture
def compressor():
    """Create a SemanticCompressor instance for testing."""
    return SemanticCompressor()


@pytest.fixture
def sample_document():
    """Create a sample document for visualization testing."""
    return """
    Quantum computing uses qubits instead of classical bits for computation.

    Superposition allows qubits to exist in multiple states simultaneously,
    enabling parallel computation at a fundamental level.

    Entanglement creates correlations between qubits that classical systems cannot replicate.
    This property is essential for quantum algorithms like Shor's and Grover's.

    Quantum gates manipulate qubit states through unitary transformations.
    Common gates include Hadamard, CNOT, and Pauli gates.

    Decoherence is the main challenge in quantum computing, as qubits lose their quantum properties
    when interacting with the environment. Error correction codes help mitigate this issue.
    """.strip()


@pytest.fixture
async def ingested_document(compressor, sample_document):
    """Ingest a sample document for visualization testing."""
    result = await compressor.ingest_file_async(
        text=sample_document,
        file_id="quantum_doc",
        metadata={"topic": "quantum_computing"},
    )
    return result


@pytest.fixture
def visualizer(compressor):
    """Create a GraphVisualizer instance."""
    return GraphVisualizer(compressor)


# ===========================
# ASCII Rendering Tests
# ===========================


class TestASCIIRendering:
    """Test ASCII graph rendering functionality."""

    @pytest.mark.asyncio
    async def test_render_ascii_basic(self, compressor, visualizer, sample_document):
        """Test basic ASCII rendering of semantic graph."""
        # Ingest document
        await compressor.ingest_file_async(sample_document, "test_doc")

        # Render ASCII
        ascii_output = visualizer.render_ascii("test_doc")

        # Verify output structure
        assert "Semantic Graph: test_doc" in ascii_output
        assert "Top Nodes by Importance:" in ascii_output
        assert "Edge Connections" in ascii_output
        assert "nodes" in ascii_output.lower()
        assert "edges" in ascii_output.lower()

    @pytest.mark.asyncio
    async def test_render_ascii_shows_node_previews(self, compressor, visualizer, sample_document):
        """Test that ASCII rendering includes text previews."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        ascii_output = visualizer.render_ascii("test_doc")

        # Should contain preview text from document
        assert "quantum" in ascii_output.lower() or "qubit" in ascii_output.lower()

    @pytest.mark.asyncio
    async def test_render_ascii_with_custom_config(self, compressor, visualizer, sample_document):
        """Test ASCII rendering with custom configuration."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        # Custom config with limited nodes
        config = VisualizationConfig(max_nodes=3, min_importance=0.0)
        ascii_output = visualizer.render_ascii("test_doc", config)

        # Should limit output based on config
        assert "Semantic Graph: test_doc" in ascii_output

    @pytest.mark.asyncio
    async def test_render_ascii_nonexistent_file(self, visualizer):
        """Test ASCII rendering with nonexistent file ID."""
        with pytest.raises(ValueError, match="No graph found"):
            visualizer.render_ascii("nonexistent_file")


# ===========================
# JSON Export Tests
# ===========================


class TestJSONExport:
    """Test JSON graph export functionality."""

    @pytest.mark.asyncio
    async def test_export_json_basic(self, compressor, visualizer, sample_document):
        """Test basic JSON export of semantic graph."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        json_output = visualizer.export_json("test_doc")
        data = json.loads(json_output)

        # Verify JSON structure
        assert "file_id" in data
        assert "nodes" in data
        assert "edges" in data
        assert "stats" in data
        assert data["file_id"] == "test_doc"

    @pytest.mark.asyncio
    async def test_export_json_node_structure(self, compressor, visualizer, sample_document):
        """Test JSON export includes proper node structure."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        json_output = visualizer.export_json("test_doc")
        data = json.loads(json_output)

        # Verify node structure
        assert len(data["nodes"]) > 0
        first_node = data["nodes"][0]
        assert "id" in first_node
        assert "text" in first_node
        assert "importance" in first_node
        assert "tokens" in first_node
        assert "position" in first_node

    @pytest.mark.asyncio
    async def test_export_json_edge_structure(self, compressor, visualizer, sample_document):
        """Test JSON export includes proper edge structure."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        json_output = visualizer.export_json("test_doc")
        data = json.loads(json_output)

        # Verify edge structure
        if len(data["edges"]) > 0:
            first_edge = data["edges"][0]
            assert "source" in first_edge
            assert "target" in first_edge
            assert "weight" in first_edge
            assert 0 <= first_edge["weight"] <= 1.0

    @pytest.mark.asyncio
    async def test_export_json_stats(self, compressor, visualizer, sample_document):
        """Test JSON export includes accurate statistics."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        json_output = visualizer.export_json("test_doc")
        data = json.loads(json_output)

        # Verify stats structure
        stats = data["stats"]
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "avg_importance" in stats
        assert "max_importance" in stats
        assert "min_importance" in stats
        assert stats["total_nodes"] == len(data["nodes"])
        assert stats["total_edges"] == len(data["edges"])


# ===========================
# GraphML Export Tests
# ===========================


class TestGraphMLExport:
    """Test GraphML graph export functionality."""

    @pytest.mark.asyncio
    async def test_export_graphml_basic(self, compressor, visualizer, sample_document):
        """Test basic GraphML export."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".graphml", delete=False) as f:
            output_path = f.name

        try:
            result = visualizer.export_graphml("test_doc", output_path)

            # Verify file was created
            assert os.path.exists(output_path)
            assert "Exported graph" in result
            assert output_path in result

            # Verify file is valid XML (GraphML)
            with open(output_path, "r") as f:
                content = f.read()
                assert "<?xml" in content
                assert "graphml" in content.lower()
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    @pytest.mark.asyncio
    async def test_export_graphml_nonexistent_file(self, visualizer):
        """Test GraphML export with nonexistent file ID."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".graphml", delete=False) as f:
            output_path = f.name

        try:
            with pytest.raises(ValueError, match="No graph found"):
                visualizer.export_graphml("nonexistent_file", output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# ===========================
# HTML Visualization Tests
# ===========================


class TestHTMLVisualization:
    """Test interactive HTML visualization."""

    @pytest.mark.asyncio
    async def test_visualize_html_basic(self, compressor, visualizer, sample_document):
        """Test basic HTML visualization generation."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            result = visualizer.visualize_html("test_doc", output_path)

            # Verify file was created
            assert os.path.exists(output_path)
            assert "Generated interactive visualization" in result
            assert output_path in result

            # Verify file is valid HTML
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert "<html" in content.lower()
                assert "<script" in content.lower()
        except ImportError:
            pytest.skip("pyvis not installed")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# ===========================
# Compression Decision Explanation Tests
# ===========================


class TestCompressionDecisionExplanation:
    """Test compression decision explanation functionality."""

    @pytest.mark.asyncio
    async def test_explain_compression_decision_basic(
        self, compressor, visualizer, sample_document
    ):
        """Test basic compression decision explanation."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        # Get first node ID
        graph = compressor.graphs["test_doc"]
        node_id = list(graph.nodes)[0]

        explanation = visualizer.explain_compression_decision("test_doc", node_id)

        # Verify explanation structure
        assert f"Node: {node_id}" in explanation
        assert "Status:" in explanation
        assert "Reasons:" in explanation
        assert "Importance score:" in explanation

    @pytest.mark.asyncio
    async def test_explain_compression_includes_connectivity(
        self, compressor, visualizer, sample_document
    ):
        """Test that explanation includes connectivity information."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        graph = compressor.graphs["test_doc"]
        node_id = list(graph.nodes)[0]

        explanation = visualizer.explain_compression_decision("test_doc", node_id)

        # Should include connectivity info
        assert "Connectivity:" in explanation or "edges" in explanation.lower()

    @pytest.mark.asyncio
    async def test_explain_compression_nonexistent_node(
        self, compressor, visualizer, sample_document
    ):
        """Test explanation with nonexistent node ID."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        with pytest.raises(ValueError, match="not found in chunks"):
            visualizer.explain_compression_decision("test_doc", "nonexistent_node_99")


# ===========================
# MCP Tool Integration Tests
# ===========================


class TestMCPToolIntegration:
    """Test visualization MCP tool handlers."""

    @pytest.fixture
    def handler_context(self, compressor):
        """Create handler context for MCP tool testing."""
        return {"compressor": compressor}

    @pytest.mark.asyncio
    async def test_handle_export_graph_json(self, compressor, handler_context, sample_document):
        """Test export_graph_json MCP tool."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        args = {"file_id": "test_doc"}
        result = await visualization_handlers.handle_export_graph_json(handler_context, args)

        # Should return valid JSON
        data = json.loads(result)
        assert data["file_id"] == "test_doc"
        assert "nodes" in data
        assert "edges" in data

    @pytest.mark.asyncio
    async def test_handle_explain_compression_decision(
        self, compressor, handler_context, sample_document
    ):
        """Test explain_compression_decision MCP tool."""
        await compressor.ingest_file_async(sample_document, "test_doc")

        # Get first node
        graph = compressor.graphs["test_doc"]
        node_id = list(graph.nodes)[0]

        args = {"file_id": "test_doc", "node_id": node_id}
        result = await visualization_handlers.handle_explain_compression_decision(
            handler_context, args
        )

        # Should return explanation
        assert "Node:" in result
        assert "Status:" in result
        assert "Reasons:" in result

    @pytest.mark.asyncio
    async def test_handle_export_graph_json_scoped_file_id(
        self, compressor, handler_context, sample_document
    ):
        scoped_file_id = compose_scoped_file_id("test_doc", workspace_id="acme")
        await compressor.ingest_file_async(sample_document, scoped_file_id)

        result = await visualization_handlers.handle_export_graph_json(
            handler_context,
            {"file_id": "test_doc", "workspace_id": "acme"},
        )

        data = json.loads(result)
        assert data["file_id"] == "test_doc"
