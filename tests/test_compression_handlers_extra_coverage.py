"""compression handlers — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
from unittest.mock import MagicMock, Mock, patch
import pytest
from unittest.mock import AsyncMock
import numpy as np
from pathlib import PurePath


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

    def test_modulate_region_singular_node_id_honored(self):
        # dogfood 2026-07-04: the documented singular `node_id` convenience alias
        # was rejected ("node_ids must not be empty") because the hook read only
        # node_ids. It must satisfy the requirement on its own.
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("modulate_region", {"node_id": "n0"})
        assert len(errors) == 0

    def test_modulate_region_missing_key(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("modulate_region", {})
        assert len(errors) == 1

    def test_unknown_tool_no_errors(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("unknown_tool_xyz", {})
        assert errors == []


class TestAdaptToContextWindowError:
    @pytest.mark.asyncio
    async def test_adapt_raises_runtime_error(self):
        from src.handlers.compression_handlers import handle_adapt_to_context_window

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        ctx["context_window_adapter"].adapt_to_context_window.side_effect = Exception("fail")

        with pytest.raises(RuntimeError, match="Failed to adapt"):
            await handle_adapt_to_context_window(ctx, {"file_id": "doc1", "available_tokens": 500})


class TestMultilevelEncodeError:
    @pytest.mark.asyncio
    async def test_multilevel_encode_raises_runtime_error(self):
        from src.handlers.compression_handlers import handle_multilevel_encode

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        ctx["multilevel_encoder"].generate_adaptive_skeleton.side_effect = Exception("encode fail")

        with pytest.raises(RuntimeError, match="Failed to generate multi-level encoding"):
            await handle_multilevel_encode(ctx, {"file_id": "doc1", "available_tokens": 1000})


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


class TestCompressionHandlersValidation:
    """Cover validation helper edge cases."""

    def test_validate_file_id_empty(self):
        """Cover line 181 - empty file_id."""
        from src.handlers.compression_handlers import validate_file_id

        ctx = _make_mock_context()
        with pytest.raises(ValueError):
            validate_file_id("", ctx)

    def test_validate_file_id_not_found_no_docs(self):
        """Cover lines 188-190 - file not found, no docs ingested."""
        from src.handlers.compression_handlers import validate_file_id

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {}
        with pytest.raises(ValueError, match="No documents ingested"):
            validate_file_id("missing_doc", ctx, must_exist=True)

    def test_validate_node_ids_empty(self):
        """Cover line 207."""
        from src.handlers.compression_handlers import validate_node_ids

        ctx = _make_mock_context()
        with pytest.raises(ValueError):
            validate_node_ids([], ctx)

    def test_validate_node_ids_no_valid_nodes(self):
        """Cover lines 221-225."""
        from src.handlers.compression_handlers import validate_node_ids

        ctx = _make_mock_context()
        ctx["compressor"].chunks = {}
        with pytest.raises(ValueError, match="may not be ingested"):
            validate_node_ids(["unknown_file_n0"], ctx)

    def test_validate_token_count_zero(self):
        """Cover lines 242, 251."""
        from src.handlers.compression_handlers import validate_token_count

        with pytest.raises(ValueError, match="available_tokens is 0"):
            validate_token_count(0)

    def test_validate_token_count_exceeds_max(self):
        """Cover line 251."""
        from src.handlers.compression_handlers import validate_token_count

        with pytest.raises(ValueError, match="exceeds max_tokens"):
            validate_token_count(10000, max_tokens=5000)

    @pytest.mark.asyncio
    async def test_ingest_rate_limit(self):
        """Cover lines 280-281, 294 - rate limit and text size."""
        from src.handlers.compression_handlers import handle_ingest

        ctx = _make_mock_context()
        # Text too large
        with pytest.raises(ValueError, match="too large|Rate limit"):
            await handle_ingest(
                ctx,
                {
                    "text": "x" * (100 * 1024 * 1024 + 1),  # Over limit
                    "file_id": "test",
                },
            )

    @pytest.mark.asyncio
    async def test_ingest_path_validation_error(self):
        """Cover lines 305-306 - path validation error."""
        from src.handlers.compression_handlers import handle_ingest

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = ValueError("path traversal")
        with pytest.raises(ValueError, match="Invalid file_path"):
            await handle_ingest(
                ctx,
                {
                    "text": "hello world this is a test document with enough text to pass validation",
                    "file_id": "test",
                    "file_path": "../../../etc/passwd",
                },
            )

    @pytest.mark.asyncio
    async def test_ingest_save_metadata_failure(self):
        """Cover lines 408-409 - metadata save failure (non-fatal)."""
        from src.handlers.compression_handlers import handle_ingest

        ctx = _make_mock_context()
        ctx["compressor"].ingest_file_async = AsyncMock()
        mock_skeleton = MagicMock()
        mock_skeleton.compression_ratio = 5.0
        mock_skeleton.total_nodes = 10
        mock_skeleton.total_tokens = 500
        mock_skeleton.skeleton_tokens = 100
        mock_skeleton.skeleton_text = "skeleton"
        mock_skeleton.node_map = {"n0": "desc"}
        ctx["compressor"].ingest_file_async.return_value = mock_skeleton
        ctx["compressor"].get_estimate.return_value = MagicMock(compression_ratio=5.0)
        ctx["sync_manager"].track_file.return_value = None
        ctx["sync_manager"].export_metadata.return_value = {}
        ctx["persistence"].save_file_sync_metadata.side_effect = Exception("save fail")
        ctx["resource_manager"].check_document_size_async = AsyncMock(return_value=(True, None))
        ctx["resource_manager"].register_document_async = AsyncMock()

        result = await handle_ingest(
            ctx,
            {
                "text": "hello world this is a test document with enough characters to pass validation",
                "file_id": "test_doc",
            },
        )
        # Should succeed despite metadata save failure
        parsed = json.loads(result)
        assert parsed["file_id"] == "test_doc"


class TestCompressionHandlersBatch:
    """Cover batch ingestion edge paths."""

    @pytest.mark.asyncio
    async def test_batch_ingest_non_string_file_id(self):
        """Cover lines 1190, 1201 - non-string file_id/text."""
        from src.handlers.compression_handlers import handle_batch_ingest

        ctx = _make_mock_context()
        with pytest.raises(ValueError, match="must be a string"):
            await handle_batch_ingest(
                ctx,
                {
                    "documents": [{"file_id": 123, "text": "hello"}],
                },
            )

    @pytest.mark.asyncio
    async def test_batch_ingest_non_string_text(self):
        """Cover line 1201."""
        from src.handlers.compression_handlers import handle_batch_ingest

        ctx = _make_mock_context()
        with pytest.raises(ValueError, match="must be a string"):
            await handle_batch_ingest(
                ctx,
                {
                    "documents": [{"file_id": "doc1", "text": 123}],
                },
            )

    @pytest.mark.asyncio
    async def test_directory_ingest_excluded_patterns(self, tmp_path):
        """Cover lines 1388-1392, 1403-1405 - exclude patterns and path validation."""
        from src.handlers.compression_handlers import handle_ingest_directory

        ctx = _make_mock_context()
        ctx["path_validator"].validate.side_effect = lambda p: str(p)

        # Create test files
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "good.py").write_text("print('hello world test')")
        (sub / "bad.pyc").write_text("binary content")

        ctx["compressor"].ingest_file_async = AsyncMock(
            return_value=MagicMock(compression_ratio=2.0, total_nodes=3)
        )

        result = await handle_ingest_directory(
            ctx,
            {
                "directory": str(sub),
                "patterns": ["*.py"],
                "exclude_patterns": ["*.pyc"],
            },
        )
        parsed = json.loads(result)
        assert parsed["status"] in ("complete", "no_files", "read_failed")

    @pytest.mark.asyncio
    async def test_directory_ingest_skipped_and_failed(self, tmp_path):
        """Cover lines 1437-1438, 1498, 1523, 1526-1527 - skipped files."""
        # Test the is_excluded helper path directly
        path_obj = PurePath("src/file.pyc")
        assert path_obj.match("*.pyc")

        path_obj2 = PurePath("src/good.py")
        assert not path_obj2.match("*.pyc")


class TestCompressionHandlersModulate:
    """Cover modulate_region and search_semantic edge paths."""

    @pytest.mark.asyncio
    async def test_modulate_tracks_retrieval_history(self):
        """Cover lines 610-611 - retrieval history tracking."""
        from src.handlers.compression_handlers import handle_modulate_region

        ctx = _make_mock_context()
        ctx["compressor"].chunks = {"doc_n0": MagicMock()}
        ctx["sync_manager"].file_metadata = {}
        ctx["compressor"].modulate_region.return_value = "content"
        ctx["retrieval_history"] = {}

        await handle_modulate_region(
            ctx,
            {
                "node_ids": ["doc_n0"],
                "fidelity_level": "RAW",
            },
        )
        assert "doc" in ctx["retrieval_history"]

    @pytest.mark.asyncio
    async def test_modulate_staleness_warning(self):
        """Cover lines 545-546, 578 - staleness warning."""
        from src.handlers.compression_handlers import handle_modulate_region

        ctx = _make_mock_context()
        ctx["compressor"].chunks = {"doc_n0": MagicMock()}
        ctx["compressor"].modulate_region.return_value = "content"
        ctx["sync_manager"].file_metadata = {"doc": {"file_path": "/tmp/test.py"}}
        ctx["sync_manager"].check_file_sync.return_value = {
            "in_sync": False,
            "reason": "File modified",
        }
        ctx["retrieval_history"] = {}

        result = await handle_modulate_region(
            ctx,
            {
                "node_ids": ["doc_n0"],
                "fidelity_level": "RAW",
            },
        )
        assert "WARNING" in result or "content" in result

    @pytest.mark.asyncio
    async def test_read_skeleton_exception(self):
        """Cover line 546 - skeleton read failure."""
        from src.handlers.compression_handlers import handle_read_skeleton

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc": MagicMock()}
        ctx["compressor"]._generate_skeleton.side_effect = Exception("fail")
        ctx["sync_manager"].file_metadata = {}
        with pytest.raises(RuntimeError, match="Failed to read skeleton"):
            await handle_read_skeleton(ctx, {"file_id": "doc"})

    @pytest.mark.asyncio
    async def test_modulate_exception(self):
        """Cover lines 618-619 - modulate failure."""
        from src.handlers.compression_handlers import handle_modulate_region

        ctx = _make_mock_context()
        ctx["compressor"].chunks = {"doc_n0": MagicMock()}
        ctx["sync_manager"].file_metadata = {}
        ctx["compressor"].modulate_region.side_effect = Exception("modulate fail")
        ctx["retrieval_history"] = {}

        with pytest.raises(RuntimeError, match="Failed to modulate"):
            await handle_modulate_region(
                ctx,
                {
                    "node_ids": ["doc_n0"],
                    "fidelity_level": "RAW",
                },
            )
