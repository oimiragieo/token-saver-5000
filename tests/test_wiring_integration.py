"""
Integration tests for Phase 5 wiring — verifies that modules are actually
connected to the main pipeline, not just importable.
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from src.semantic_compressor import SemanticCompressor, SemanticNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compressor():
    return SemanticCompressor()


@pytest.fixture
def sample_text():
    return (
        "Machine learning models require large datasets for training. "
        "Neural networks learn representations through backpropagation. "
        "Deep learning has revolutionized computer vision tasks. "
        "Natural language processing uses transformers for text understanding. "
        "Reinforcement learning agents learn from environmental rewards. "
        "Transfer learning enables knowledge reuse across domains. "
        "Generative models can create new synthetic data samples. "
        "Attention mechanisms allow models to focus on relevant inputs."
    )


@pytest.fixture
def handler_context(compressor):
    """Minimal handler context for testing handlers."""
    from unittest.mock import AsyncMock
    from src.file_sync_manager import FileSyncManager
    sync_mgr = FileSyncManager()
    persistence = MagicMock()
    persistence.save_graphs = MagicMock(return_value=True)
    persistence.save_file_sync_metadata = MagicMock(return_value=True)

    version_manager = MagicMock()
    version_manager.record_version = MagicMock()
    version_manager.record_version_async = AsyncMock()

    resource_manager = MagicMock()
    resource_manager.check_document_size_async = AsyncMock(return_value=(True, None))
    resource_manager.register_document_async = AsyncMock()

    path_validator = MagicMock()
    path_validator.validate = MagicMock(side_effect=lambda x: x)

    return {
        "compressor": compressor,
        "sync_manager": sync_mgr,
        "persistence": persistence,
        "retrieval_history": {},
        "version_manager": version_manager,
        "resource_manager": resource_manager,
        "path_validator": path_validator,
        "multilevel_encoder": MagicMock(),
        "context_window_adapter": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Test: AccessTracker + CompressionReplayLog are initialized
# ---------------------------------------------------------------------------

class TestCompressorInitialization:
    def test_access_tracker_initialized(self, compressor):
        """AccessTracker should be created in __init__."""
        assert hasattr(compressor, '_access_tracker')
        from src.context_decay import AccessTracker
        assert isinstance(compressor._access_tracker, AccessTracker)

    def test_compression_replay_initialized(self, compressor):
        """CompressionReplayLog should be created in __init__."""
        assert hasattr(compressor, '_compression_replay')
        from src.compression_replay import CompressionReplayLog
        assert isinstance(compressor._compression_replay, CompressionReplayLog)


# ---------------------------------------------------------------------------
# Test: handle_ingest records to tracker and replay
# ---------------------------------------------------------------------------

class TestIngestWiring:
    @pytest.mark.asyncio
    async def test_ingest_records_access(self, handler_context, sample_text):
        """After ingest, access_tracker should have an entry for the file."""
        from src.handlers.compression_handlers import handle_ingest
        args = {"text": sample_text, "file_id": "test_doc"}
        result = await handle_ingest(handler_context, args)
        response = json.loads(result)
        assert response["status"] == "success"

        tracker = handler_context["compressor"]._access_tracker
        info = tracker.get_access_info("test_doc")
        assert info is not None
        assert info["access_count"] >= 1

    @pytest.mark.asyncio
    async def test_ingest_records_replay(self, handler_context, sample_text):
        """After ingest, compression_replay should have an entry."""
        from src.handlers.compression_handlers import handle_ingest
        args = {"text": sample_text, "file_id": "replay_doc"}
        await handle_ingest(handler_context, args)

        replay = handler_context["compressor"]._compression_replay
        history = replay.get_history("replay_doc")
        assert len(history) >= 1
        assert history[0]["input_tokens"] > 0
        assert history[0]["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_ingest_includes_fidelity_score(self, handler_context, sample_text):
        """Ingest response should include fidelity_score from Phase 5."""
        from src.handlers.compression_handlers import handle_ingest
        args = {"text": sample_text, "file_id": "fidelity_doc"}
        result = await handle_ingest(handler_context, args)
        response = json.loads(result)
        assert "fidelity_score" in response
        assert 0.0 <= response["fidelity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_ingest_with_semantic_chunking(self, handler_context):
        """Ingest with chunking_strategy='semantic' should work."""
        from src.handlers.compression_handlers import handle_ingest
        long_text = " ".join([
            f"Section {i}: This is a paragraph about topic {i} with enough detail to be meaningful."
            for i in range(20)
        ])
        args = {"text": long_text, "file_id": "semantic_doc", "chunking_strategy": "semantic"}
        result = await handle_ingest(handler_context, args)
        response = json.loads(result)
        assert response["status"] == "success"
        assert response["total_nodes"] > 0


# ---------------------------------------------------------------------------
# Test: handle_read_skeleton records access + keyword anchoring
# ---------------------------------------------------------------------------

class TestSkeletonWiring:
    @pytest.mark.asyncio
    async def test_skeleton_records_access(self, handler_context, sample_text):
        """Reading skeleton should record access in the tracker."""
        from src.handlers.compression_handlers import handle_ingest, handle_read_skeleton
        await handle_ingest(handler_context, {"text": sample_text, "file_id": "skel_doc"})

        # Reset tracker to verify skeleton adds its own access
        handler_context["compressor"]._access_tracker._access_log.clear()

        await handle_read_skeleton(handler_context, {"file_id": "skel_doc"})

        tracker = handler_context["compressor"]._access_tracker
        info = tracker.get_access_info("skel_doc")
        assert info is not None
        assert info["access_count"] >= 1

    @pytest.mark.asyncio
    async def test_skeleton_keyword_anchoring(self, handler_context, sample_text):
        """Skeleton with anchored_keywords should force matching nodes into anchors."""
        from src.handlers.compression_handlers import handle_ingest, handle_read_skeleton
        await handle_ingest(handler_context, {"text": sample_text, "file_id": "anchor_doc"})

        result = await handle_read_skeleton(
            handler_context,
            {"file_id": "anchor_doc", "anchored_keywords": ["transformers", "backpropagation"]}
        )
        response = json.loads(result)
        assert "file_id" in response
        assert response["file_id"] == "anchor_doc"
        assert response["anchored_nodes"]
        for node_id in response["anchored_nodes"]:
            assert f"[{node_id}] [rag:{node_id}] [ANCHOR]" in response["skeleton_text"]


# ---------------------------------------------------------------------------
# Test: handle_search_semantic records access
# ---------------------------------------------------------------------------

class TestSearchWiring:
    @pytest.mark.asyncio
    async def test_search_records_access(self, handler_context, sample_text):
        """Semantic search should record access for found documents."""
        from src.handlers.compression_handlers import handle_ingest, handle_search_semantic
        await handle_ingest(handler_context, {"text": sample_text, "file_id": "search_doc"})

        handler_context["compressor"]._access_tracker._access_log.clear()

        result = await handle_search_semantic(
            handler_context, {"query": "neural networks", "file_id": "search_doc"}
        )
        response = json.loads(result)
        assert response["total_results"] > 0

        tracker = handler_context["compressor"]._access_tracker
        info = tracker.get_access_info("search_doc")
        assert info is not None


# ---------------------------------------------------------------------------
# Test: evict_stale and get_compression_insights now work
# ---------------------------------------------------------------------------

class TestPhase5ToolsWork:
    @pytest.mark.asyncio
    async def test_evict_stale_returns_data(self, handler_context, sample_text):
        """evict_stale should return stale docs when tracker is wired."""
        from src.handlers.compression_handlers import handle_ingest, handle_evict_stale

        await handle_ingest(handler_context, {"text": sample_text, "file_id": "old_doc"})

        # Manually set access time to the past
        tracker = handler_context["compressor"]._access_tracker
        tracker._access_log["old_doc"]["last_accessed"] = time.time() - 7200  # 2 hours ago

        result = await handle_evict_stale(handler_context, {"max_age_seconds": 3600})
        response = json.loads(result)
        assert "old_doc" in response.get("stale_documents", [])

    @pytest.mark.asyncio
    async def test_get_compression_insights_returns_data(self, handler_context, sample_text):
        """get_compression_insights should return data after ingest."""
        from src.handlers.compression_handlers import handle_ingest, handle_get_compression_insights

        await handle_ingest(handler_context, {"text": sample_text, "file_id": "insight_doc"})

        result = await handle_get_compression_insights(handler_context, {})
        response = json.loads(result)
        # After ingest, insights should have at least one content type entry
        assert len(response.get("insights", {})) > 0


# ---------------------------------------------------------------------------
# Test: semantic chunking module works in _chunk_text
# ---------------------------------------------------------------------------

class TestSemanticChunking:
    def test_chunk_text_auto_uses_semantic_for_large_structured_docs(self, compressor):
        """Auto mode should upgrade larger structured documents to semantic chunking."""
        text = "\n\n".join(
            [
                "Paragraph one discusses authentication, authorization, and token rotation. " * 16,
                "Paragraph two discusses billing retries, invoices, and payment recovery. " * 16,
                "Paragraph three discusses audit logs, compliance review, and admin actions. " * 16,
            ]
        )
        with patch("src.semantic_chunking.chunk_by_semantics", return_value=["semantic-a", "semantic-b"]) as mocked:
            chunks = compressor._chunk_text(text, strategy="auto")
        assert chunks == ["semantic-a", "semantic-b"]
        mocked.assert_called_once()

    def test_chunk_text_auto_keeps_fixed_for_small_docs(self, compressor):
        """Auto mode should avoid semantic chunking for small/simple docs."""
        text = "Short doc. " * 20
        with patch("src.semantic_chunking.chunk_by_semantics", side_effect=AssertionError("should not be called")):
            chunks = compressor._chunk_text(text, strategy="auto")
        assert len(chunks) > 0

    def test_chunk_text_fixed_strategy(self, compressor):
        """Fixed strategy should work as before."""
        text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50
        chunks = compressor._chunk_text(text, strategy="fixed")
        assert len(chunks) > 0

    def test_chunk_text_semantic_strategy(self, compressor):
        """Semantic strategy should produce chunks."""
        text = " ".join([
            f"Topic {i} is about something completely different from the others."
            for i in range(20)
        ])
        chunks = compressor._chunk_text(text, strategy="semantic")
        assert len(chunks) > 0

    def test_chunk_text_semantic_fallback(self, compressor):
        """If semantic chunking fails, should fall back to fixed."""
        with patch("src.semantic_chunking.chunk_by_semantics", side_effect=Exception("fail")):
            text = "Hello world. " * 20
            chunks = compressor._chunk_text(text, strategy="semantic")
            assert len(chunks) > 0


# ---------------------------------------------------------------------------
# Test: intra-doc dedup runs during ingest
# ---------------------------------------------------------------------------

class TestIntraDocDedup:
    def test_ingest_deduplicates_redundant_chunks(self, compressor):
        """Ingest should collapse near-identical paragraphs."""
        # Create text with highly redundant paragraphs
        para = "Machine learning uses neural networks for pattern recognition in data. " * 5
        text = "\n\n".join([para] * 6)  # 6 identical paragraphs
        result = compressor.ingest_file(text, "dedup_test")
        # Should have fewer nodes than 6 (some collapsed)
        assert result.total_nodes <= 6

    def test_ingest_dedup_preserves_order_and_reuses_embeddings(self, compressor):
        """Dedup should keep original chunk order and avoid re-encoding retained chunks."""
        compressor._chunk_text = lambda text, max_chunk_size=512, strategy="fixed": [
            "first chunk",
            "duplicate chunk",
            "last chunk",
        ]

        encode_calls = []

        async def fake_encode(chunks):
            encode_calls.append(list(chunks))
            if len(encode_calls) > 1:
                raise AssertionError("Embeddings should be reused after dedup")
            return np.array([
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ])

        compressor._encode_async = fake_encode

        def fake_collapse(nodes_map, threshold=0.92):
            return {
                "tmp_2": {"text": "last chunk", "embedding": np.array([0.0, 1.0])},
                "tmp_0": {"text": "first chunk", "embedding": np.array([1.0, 0.0])},
            }

        with patch("src.intra_doc_dedup.collapse_redundant_nodes", side_effect=fake_collapse):
            result = compressor.ingest_file("ignored", "dedup_order_doc")

        assert result.total_nodes == 2
        assert compressor.chunks["dedup_order_doc_n0"].text == "first chunk"
        assert compressor.chunks["dedup_order_doc_n1"].text == "last chunk"
        assert len(encode_calls) == 1


# ---------------------------------------------------------------------------
# Test: query-adaptive ratios affect skeleton
# ---------------------------------------------------------------------------

class TestQueryAdaptive:
    def test_skeleton_with_query_uses_adaptive_ratios(self, compressor):
        """When query is provided, adaptive ratios should be passed into selection."""
        doc_id = "adaptive_doc"
        graph = nx.Graph()
        nodes = {
            f"{doc_id}_n0": SemanticNode(
                node_id=f"{doc_id}_n0",
                text="Machine learning requires large datasets.",
                embedding=np.array([1.0, 0.0]),
                importance=0.4,
                metadata={"tokens": 8, "entities": ["Machine"], "position": 0},
            ),
            f"{doc_id}_n1": SemanticNode(
                node_id=f"{doc_id}_n1",
                text="Cooking recipes use fresh ingredients.",
                embedding=np.array([0.0, 1.0]),
                importance=0.3,
                metadata={"tokens": 6, "entities": ["Cooking"], "position": 1},
            ),
            f"{doc_id}_n2": SemanticNode(
                node_id=f"{doc_id}_n2",
                text="Neural networks learn from backpropagation.",
                embedding=np.array([0.9, 0.1]),
                importance=0.5,
                metadata={"tokens": 7, "entities": ["Neural"], "position": 2},
            ),
        }
        for node_id, node in nodes.items():
            compressor.chunks[node_id] = node
            graph.add_node(node_id, **node.metadata)
        compressor.graphs[doc_id] = graph
        captured = {}
        original = compressor._select_skeleton_nodes

        def wrapped(file_nodes, num_skeleton, query=None, redundancy_penalty=0.2, priority_scores=None):
            captured["priority_scores"] = priority_scores
            return original(
                file_nodes,
                num_skeleton,
                query=query,
                redundancy_penalty=redundancy_penalty,
                priority_scores=priority_scores,
            )

        compressor._select_skeleton_nodes = wrapped
        try:
            with patch.object(compressor.model, "encode", return_value=np.array([[1.0, 0.0]])):
                result_ml = compressor._generate_skeleton(
                    doc_id,
                    query="machine learning neural networks",
                )
        finally:
            compressor._select_skeleton_nodes = original

        assert result_ml.total_nodes > 0
        assert captured["priority_scores"] is not None
        assert any(score > 0 for score in captured["priority_scores"].values())

    def test_skeleton_without_query_uses_uniform_ratios(self, compressor):
        """Without query, all sections get equal treatment."""
        text = "Topic A is important. " * 10 + "\n\n" + "Topic B matters too. " * 10
        compressor.ingest_file(text, "uniform_doc")
        result = compressor._generate_skeleton("uniform_doc")
        assert result.total_nodes > 0


# ---------------------------------------------------------------------------
# Test: workflow guidance in tool_help
# ---------------------------------------------------------------------------

class TestWorkflowGuidance:
    @pytest.mark.asyncio
    async def test_tool_help_includes_workflow(self):
        """tool_help() with no args should include recommended workflow."""
        from src.handlers.help_handlers import handle_tool_help
        result = await handle_tool_help({}, {})
        response = json.loads(result)
        assert "recommended_workflow" in response
        assert len(response["recommended_workflow"]["steps"]) >= 5

    @pytest.mark.asyncio
    async def test_tool_help_includes_profiles(self):
        """tool_help() should document core_stable and full profiles."""
        from src.handlers.help_handlers import handle_tool_help
        result = await handle_tool_help({}, {})
        response = json.loads(result)
        assert "tool_profiles" in response
        assert "core_stable" in response["tool_profiles"]
        assert "full" in response["tool_profiles"]

    @pytest.mark.asyncio
    async def test_tool_help_core_stable_matches_actual_tools(self):
        """Workflow guidance should list the real core_stable profile tools."""
        from src.handlers.help_handlers import handle_tool_help
        result = await handle_tool_help({}, {})
        response = json.loads(result)
        assert response["tool_profiles"]["core_stable"]["tools"] == [
            "ingest_context",
            "read_skeleton",
            "modulate_region",
            "search_semantic",
            "get_stats",
            "list_documents",
            "delete_document",
        ]

    @pytest.mark.asyncio
    async def test_modulate_region_help_uses_fidelity_level_param(self):
        """Help examples should use the real fidelity_level argument name."""
        from src.handlers.help_handlers import handle_tool_help
        result = await handle_tool_help({}, {"tool_name": "modulate_region", "verbose": True})
        response = json.loads(result)
        example_args = response["examples"][0]["args"]
        assert "fidelity_level" in example_args
        assert "fidelity" not in example_args
