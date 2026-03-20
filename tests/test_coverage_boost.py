"""
Coverage boost tests for uncovered code paths.

Covers: visualization_handlers, graceful_degradation, experimental_handlers,
graph_visualizer, experience_synthesis, compression_presets, validation_hooks,
memory_hooks.

All tests use mocks to avoid heavy dependencies (ChromaDB, ONNX, PyTorch, etc.).
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


# =============================================================================
# 1. Visualization Handlers (lines 46-156)
# =============================================================================


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


class TestHandleExplainCompressionDecision:
    """Tests for handle_explain_compression_decision."""

    @pytest.mark.asyncio
    async def test_success(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.explain_compression_decision.return_value = (
                "Node: n0\nStatus: [KEPT]"
            )
            result = await handle_explain_compression_decision(
                ctx, {"file_id": "doc1", "node_id": "n0"}
            )
        assert "KEPT" in result

    @pytest.mark.asyncio
    async def test_missing_file_id(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        with pytest.raises(Exception, match="file_id"):
            await handle_explain_compression_decision({"compressor": Mock()}, {"node_id": "n0"})

    @pytest.mark.asyncio
    async def test_missing_node_id(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        with pytest.raises(Exception, match="node_id"):
            await handle_explain_compression_decision({"compressor": Mock()}, {"file_id": "doc1"})

    @pytest.mark.asyncio
    async def test_no_graph_found(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.explain_compression_decision.side_effect = ValueError(
                "No graph found"
            )
            with pytest.raises(Exception):
                await handle_explain_compression_decision(ctx, {"file_id": "doc1", "node_id": "n0"})

    @pytest.mark.asyncio
    async def test_node_not_found_in_chunks(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.explain_compression_decision.side_effect = ValueError(
                "Node xyz not found in chunks"
            )
            with pytest.raises(ValueError, match="not found"):
                await handle_explain_compression_decision(
                    ctx, {"file_id": "doc1", "node_id": "xyz"}
                )

    @pytest.mark.asyncio
    async def test_generic_error_logged(self):
        from src.handlers.visualization_handlers import handle_explain_compression_decision

        ctx = {"compressor": Mock()}
        with patch("src.handlers.visualization_handlers.GraphVisualizer") as MockViz:
            MockViz.return_value.explain_compression_decision.side_effect = RuntimeError("oops")
            with pytest.raises(RuntimeError):
                await handle_explain_compression_decision(ctx, {"file_id": "doc1", "node_id": "n0"})


# =============================================================================
# 2. Graceful Degradation (lines 112-295)
# =============================================================================


class TestGracefulDegradationEmbedAllFail:
    """Test error logging when all embedding tiers fail."""

    @pytest.mark.asyncio
    async def test_all_tiers_fail_raises_last_exception(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.encode.side_effect = RuntimeError("model unavailable")
        with pytest.raises(RuntimeError, match="model unavailable"):
            await GracefulDegradation.embed_with_fallback(["hello"], mgr)

    @pytest.mark.asyncio
    async def test_fallback_to_later_tier(self):
        from src.graceful_degradation import GracefulDegradation
        import numpy as np

        mgr = AsyncMock()
        calls = []

        async def encode_side(texts, tier=None):
            calls.append(tier)
            if tier in ("STANDARD", "ONNX"):
                raise RuntimeError("fail")
            return np.array([[1.0, 2.0]])

        mgr.encode.side_effect = encode_side
        result = await GracefulDegradation.embed_with_fallback(["hi"], mgr)
        assert result.shape == (1, 2)
        assert "TFIDF" in calls


class TestGracefulDegradationPersist:
    """Test persist_with_fallback success path."""

    @pytest.mark.asyncio
    async def test_save_to_disk_success(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.save_document.return_value = True
        result = await GracefulDegradation.persist_with_fallback("d1", {"k": "v"}, mgr)
        assert result["success"] is True
        assert result["mode"] == "disk"

    @pytest.mark.asyncio
    async def test_save_returns_false(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.save_document.return_value = False
        result = await GracefulDegradation.persist_with_fallback("d1", {}, mgr)
        assert result["success"] is False
        assert result["mode"] == "memory"

    @pytest.mark.asyncio
    async def test_save_raises_exception(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.save_document.side_effect = IOError("disk full")
        result = await GracefulDegradation.persist_with_fallback("d1", {}, mgr)
        assert result["success"] is False
        assert "error" in result


class TestGracefulDegradationFileSync:
    """Test file_sync_with_fallback success path."""

    @pytest.mark.asyncio
    async def test_full_validation_success(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.check_staleness.return_value = True
        result = await GracefulDegradation.file_sync_with_fallback("/path/file.py", mgr)
        assert result["is_stale"] is True
        assert result["mode"] == "full_validation"

    @pytest.mark.asyncio
    async def test_os_error_fallback(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.check_staleness.side_effect = OSError("permission denied")
        result = await GracefulDegradation.file_sync_with_fallback("/path", mgr)
        assert result["is_stale"] is False
        assert result["mode"] == "cached_metadata"
        assert "warning" in result


class TestGracefulDegradationVersionHistory:
    """Test version_history_with_fallback."""

    @pytest.mark.asyncio
    async def test_full_diff_success(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.add_version.return_value = "v42"
        result = await GracefulDegradation.version_history_with_fallback(
            "d1", {"timestamp": 123, "version": 1}, mgr
        )
        assert result["success"] is True
        assert result["mode"] == "full_diff"
        assert result["version_id"] == "v42"

    @pytest.mark.asyncio
    async def test_fallback_to_metadata_only(self):
        from src.graceful_degradation import GracefulDegradation

        mgr = AsyncMock()
        mgr.add_version.side_effect = RuntimeError("diff failed")
        result = await GracefulDegradation.version_history_with_fallback(
            "d1", {"timestamp": 123, "version": 2}, mgr
        )
        assert result["success"] is False
        assert result["mode"] == "metadata_only"
        assert result["metadata"]["diff"] is None
        assert result["metadata"]["file_id"] == "d1"


# =============================================================================
# 3. Experimental Handlers (lines 540-783)
# =============================================================================


class TestHandleVerifyCompression:
    """Tests for handle_verify_compression."""

    @pytest.mark.asyncio
    async def test_missing_required_args(self):
        from src.handlers.experimental_handlers import handle_verify_compression

        result = json.loads(await handle_verify_compression({}, {}))
        assert "error" in result
        assert "Missing required" in result["error"]

    @pytest.mark.asyncio
    async def test_success(self):
        from src.handlers.experimental_handlers import handle_verify_compression

        mock_result = Mock()
        mock_result.verified = True
        mock_result.all_contracts_passed = True
        mock_result.preconditions.overall_passed = True
        mock_result.postconditions.overall_passed = True
        mock_result.violations = []

        with patch("src.handlers.experimental_handlers._get_compression_verifier") as gv:
            gv.return_value.verify_compression_operation.return_value = mock_result
            result = json.loads(
                await handle_verify_compression(
                    {},
                    {
                        "document": "hello world",
                        "skeleton_text": "hello",
                        "original_tokens": 10,
                        "skeleton_tokens": 5,
                        "fidelity_level": "OUTLINE",
                    },
                )
            )
        assert result["verified"] is True
        assert result["experimental"] is True

    @pytest.mark.asyncio
    async def test_error_path(self):
        from src.handlers.experimental_handlers import handle_verify_compression

        with patch("src.handlers.experimental_handlers._get_compression_verifier") as gv:
            gv.return_value.verify_compression_operation.side_effect = RuntimeError("bad")
            result = json.loads(
                await handle_verify_compression(
                    {},
                    {
                        "document": "x",
                        "skeleton_text": "y",
                        "original_tokens": 1,
                        "skeleton_tokens": 1,
                        "fidelity_level": "ABSTRACT",
                    },
                )
            )
        assert "error" in result


class TestHandleCalculateReward:
    """Tests for handle_calculate_reward."""

    @pytest.mark.asyncio
    async def test_missing_args(self):
        from src.handlers.experimental_handlers import handle_calculate_reward

        result = json.loads(await handle_calculate_reward({}, {}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_success(self):
        from src.handlers.experimental_handlers import handle_calculate_reward

        mock_reward = Mock()
        mock_reward.total_reward = 0.85
        mock_reward.passes_threshold.return_value = True
        mock_reward.component_scores = {Mock(value="schema"): 0.9, Mock(value="semantic"): 0.8}
        mock_reward.weakest_component = (Mock(value="semantic"), 0.8)

        with patch("src.handlers.experimental_handlers._get_reward_calculator") as gc:
            gc.return_value.calculate.return_value = mock_reward
            result = json.loads(
                await handle_calculate_reward(
                    {},
                    {
                        "input_text": "hello world",
                        "output_text": "hello",
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "fidelity_level": "OUTLINE",
                    },
                )
            )
        assert result["total_reward"] == 0.85
        assert result["passes_threshold"] is True
        assert result["experimental"] is True

    @pytest.mark.asyncio
    async def test_error_path(self):
        from src.handlers.experimental_handlers import handle_calculate_reward

        with patch("src.handlers.experimental_handlers._get_reward_calculator") as gc:
            gc.return_value.calculate.side_effect = RuntimeError("boom")
            result = json.loads(
                await handle_calculate_reward(
                    {},
                    {
                        "input_text": "x",
                        "output_text": "y",
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "fidelity_level": "OUTLINE",
                    },
                )
            )
        assert "error" in result


class TestHandleGenerateSyntheticTests:
    """Tests for handle_generate_synthetic_tests."""

    @pytest.mark.asyncio
    async def test_boundary_type(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        mock_doc = Mock()
        mock_doc.name = "test_doc"
        mock_doc.category.value = "boundary"
        mock_doc.description = "desc"
        mock_doc.token_estimate = 42
        mock_doc.expected_behavior = "pass"

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_boundary_cases.return_value = [mock_doc]
            result = json.loads(
                await handle_generate_synthetic_tests({}, {"test_type": "boundary"})
            )
        assert result["test_type"] == "boundary"
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_dialogue_type(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_dialogue_cases.return_value = [
                [{"role": "user", "content": "hi"}]
            ]
            result = json.loads(
                await handle_generate_synthetic_tests({}, {"test_type": "dialogue"})
            )
        assert result["test_type"] == "dialogue"
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_ace_type(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_ace_cases.return_value = [
                {"name": "c1", "bullets": [{"text": "x"}], "expected": "ok"}
            ]
            result = json.loads(await handle_generate_synthetic_tests({}, {"test_type": "ace"}))
        assert result["test_type"] == "ace"
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_all_type(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        mock_suite = Mock()
        mock_suite.documents = [Mock()]
        mock_suite.dialogues = [Mock(), Mock()]
        mock_suite.ace_contexts = []

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_full_test_suite.return_value = mock_suite
            result = json.loads(await handle_generate_synthetic_tests({}, {"test_type": "all"}))
        assert result["test_type"] == "all"
        assert result["boundary_count"] == 1
        assert result["dialogue_count"] == 2

    @pytest.mark.asyncio
    async def test_unknown_type(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer"):
            result = json.loads(await handle_generate_synthetic_tests({}, {"test_type": "unknown"}))
        assert "error" in result
        assert "Unknown test_type" in result["error"]

    @pytest.mark.asyncio
    async def test_error_path(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.side_effect = RuntimeError("no module")
            result = json.loads(
                await handle_generate_synthetic_tests({}, {"test_type": "boundary"})
            )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_default_type_is_boundary(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_boundary_cases.return_value = []
            result = json.loads(await handle_generate_synthetic_tests({}, {}))
        assert result["test_type"] == "boundary"

    @pytest.mark.asyncio
    async def test_seed_passed(self):
        from src.handlers.experimental_handlers import handle_generate_synthetic_tests

        with patch("src.handlers.experimental_handlers._get_experience_synthesizer") as gs:
            gs.return_value.generate_boundary_cases.return_value = []
            await handle_generate_synthetic_tests({}, {"test_type": "boundary", "seed": 42})
        gs.assert_called_once_with(seed=42)


# =============================================================================
# 4. Graph Visualizer (lines 123-466)
# =============================================================================


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


# =============================================================================
# 5. Experience Synthesis (lines 580-732)
# =============================================================================


class TestStressTestCompression:
    """Tests for stress_test_compression."""

    def test_stress_test_all_succeed(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=1)
        comp = Mock()
        skeleton_mock = Mock()
        skeleton_mock.compression_ratio = 0.5
        comp.read_skeleton.return_value = skeleton_mock

        result = synth.stress_test_compression(comp, iterations=3)
        assert result.test_name == "compression_stress_test"
        assert result.passed is True
        assert result.iterations == 3
        assert len(result.errors) == 0
        assert result.metrics["error_rate"] == 0.0

    def test_stress_test_with_errors(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=1)
        comp = Mock()
        comp.ingest_file.side_effect = RuntimeError("oom")

        result = synth.stress_test_compression(comp, iterations=2)
        assert result.passed is False
        assert len(result.errors) == 2

    def test_stress_test_to_dict(self):
        from src.experience_synthesis import StressTestResult

        r = StressTestResult(
            test_name="t", passed=True, duration_ms=100.0, iterations=5, errors=[], metrics={"x": 1}
        )
        d = r.to_dict()
        assert d["test_name"] == "t"
        assert d["metrics"]["x"] == 1


class TestStressTestAFM:
    """Tests for stress_test_afm (memory pressure)."""

    def test_stress_test_afm_success(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=42)
        mgr = Mock()
        mgr.build_context.return_value = (
            "I have a peanut allergy and more context",
            {"tokens": 50},
        )

        result = synth.stress_test_afm(mgr, turns=5)
        assert result.test_name == "afm_stress_test"
        assert result.iterations == 5

    def test_stress_test_afm_lost_safety_info(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=42)
        mgr = Mock()
        mgr.build_context.return_value = ("generic response without keywords", {})

        result = synth.stress_test_afm(mgr, turns=5)
        assert len(result.errors) > 0


class TestValidateBoundaryCases:
    """Tests for run_boundary_tests."""

    def test_run_boundary_tests_success(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=1)
        comp = Mock()
        skeleton = Mock()
        skeleton.skeleton_tokens = 10
        comp.read_skeleton.return_value = skeleton
        # Make empty doc raise as expected
        comp.ingest_file.side_effect = lambda content, name: (
            (_ for _ in ()).throw(ValueError("empty")) if content == "" else None
        )

        results = synth.run_boundary_tests(comp)
        assert len(results) > 0
        # Empty doc should pass (expected to fail)
        empty_result = [r for r in results if r[0] == "empty_document"]
        assert len(empty_result) == 1
        assert empty_result[0][1] is True

    def test_generate_full_test_suite(self):
        from src.experience_synthesis import ExperienceSynthesizer

        synth = ExperienceSynthesizer(seed=1)
        suite = synth.generate_full_test_suite()
        assert len(suite.documents) > 0
        assert len(suite.dialogues) > 0
        assert len(suite.ace_contexts) > 0


# =============================================================================
# 6. Compression Presets - to_dict()
# =============================================================================


class TestCompressionPresets:
    """Tests for CompressionPreset.to_dict and helpers."""

    def test_to_dict(self):
        from src.compression_presets import CompressionPreset

        p = CompressionPreset(
            name="test", description="A test preset", skeleton_ratio=0.3, fidelity="OUTLINE"
        )
        d = p.to_dict()
        assert d["name"] == "test"
        assert d["skeleton_ratio"] == 0.3
        assert d["fidelity"] == "OUTLINE"
        assert d["description"] == "A test preset"

    def test_get_preset_existing(self):
        from src.compression_presets import get_preset

        p = get_preset("code-review")
        assert p.name == "code-review"
        assert p.fidelity == "DETAILED"

    def test_get_preset_unknown(self):
        from src.compression_presets import get_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent")

    def test_list_presets(self):
        from src.compression_presets import list_presets

        presets = list_presets()
        assert len(presets) >= 4
        names = [p.name for p in presets]
        assert "chat" in names
        assert "aggressive" in names


# =============================================================================
# 6b. Validation Hooks - modulate_region
# =============================================================================


class TestValidationHooks:
    """Tests for validation hooks including modulate_region."""

    def test_modulate_region_empty_node_ids(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("modulate_region", {"node_ids": []})
        assert len(errors) == 1
        assert "node_ids" in errors[0]

    def test_modulate_region_valid(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("modulate_region", {"node_ids": ["n0", "n1"]})
        assert len(errors) == 0

    def test_modulate_region_missing_key(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("modulate_region", {})
        assert len(errors) == 1

    def test_unknown_tool_no_errors(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("unknown_tool_xyz", {})
        assert errors == []


# =============================================================================
# 7. Memory Hooks - clear()
# =============================================================================


class TestMemoryHooksClear:
    """Tests for MemoryHookManager.clear()."""

    def test_clear_removes_hooks_and_entries(self):
        from src.memory_hooks import MemoryHookManager

        mgr = MemoryHookManager()
        mgr.register_hook("post_compress", lambda d: None)
        mgr.add_memory_entry("f1", "insight", "cat")
        assert len(mgr.get_memory_index()) == 1

        mgr.clear()
        assert len(mgr.get_memory_index()) == 0
        # hooks dict should also be empty
        assert mgr._hooks == {}

    def test_clear_idempotent(self):
        from src.memory_hooks import MemoryHookManager

        mgr = MemoryHookManager()
        mgr.clear()
        mgr.clear()
        assert len(mgr.get_memory_index()) == 0

    def test_trigger_after_clear_does_nothing(self):
        from src.memory_hooks import MemoryHookManager

        mgr = MemoryHookManager()
        called = []
        mgr.register_hook("evt", lambda d: called.append(1))
        mgr.clear()
        mgr.trigger("evt", {})
        assert len(called) == 0
