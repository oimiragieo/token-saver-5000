"""Tests for bug fixes discovered during deep audit.

Covers:
- compute_adaptive_ratio() input validation
- compute_cost_savings() negative savings (expansion)
- find_duplicates() null embedding guard
- diff_reingest() null embedding guard
- handle_ingest div-by-zero protection
- handle_ingest cost_savings exception protection
- Atomic persistence writes
- New MCP tool handlers (diff_reingest, find_duplicates, get_presets)
"""

import json
import os
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass

from src.semantic_compressor import compute_adaptive_ratio, SemanticCompressor
from src.metrics import compute_cost_savings


# =========================================================================
# compute_adaptive_ratio validation
# =========================================================================

class TestComputeAdaptiveRatioValidation:
    def test_negative_tokens_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_adaptive_ratio(-1)

    def test_zero_tokens_returns_high_ratio(self):
        assert compute_adaptive_ratio(0) == 0.8

    def test_small_tokens(self):
        assert compute_adaptive_ratio(100) == 0.8

    def test_medium_tokens(self):
        assert compute_adaptive_ratio(10000) == 0.5

    def test_large_tokens(self):
        assert compute_adaptive_ratio(50000) == 0.2

    def test_huge_tokens(self):
        assert compute_adaptive_ratio(200000) == 0.1


# =========================================================================
# compute_cost_savings negative savings (expansion case)
# =========================================================================

class TestComputeCostSavingsNegative:
    def test_expansion_shows_negative_savings(self):
        result = compute_cost_savings(100, 150)
        assert result.saved_tokens == -50
        assert result.savings_percent < 0
        assert result.cost_savings_usd < 0

    def test_zero_input_tokens(self):
        result = compute_cost_savings(0, 0)
        assert result.saved_tokens == 0
        assert result.savings_percent == 0.0

    def test_normal_savings(self):
        result = compute_cost_savings(1000, 200)
        assert result.saved_tokens == 800
        assert result.savings_percent == 80.0
        assert result.cost_savings_usd > 0


# =========================================================================
# find_duplicates null embedding guard
# =========================================================================

class TestFindDuplicatesNullGuard:
    def test_skips_none_embeddings(self):
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        node_a = MagicMock()
        node_a.embedding = None
        node_b = MagicMock()
        node_b.embedding = np.array([1.0, 0.0])
        compressor.chunks = {"fileA_0": node_a, "fileB_0": node_b}
        # Should not crash
        result = compressor.find_duplicates(threshold=0.9)
        assert result == []

    def test_works_with_valid_embeddings(self):
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        node_a = MagicMock()
        node_a.embedding = np.array([1.0, 0.0, 0.0])
        node_b = MagicMock()
        node_b.embedding = np.array([1.0, 0.0, 0.0])
        compressor.chunks = {"fileA_0": node_a, "fileB_0": node_b}
        result = compressor.find_duplicates(threshold=0.9)
        assert len(result) == 1

    def test_skips_dimension_mismatch(self):
        """Embeddings with different dimensions (e.g. MiniLM 384 vs CodeBERT 768) should be skipped."""
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        node_a = MagicMock()
        node_a.embedding = np.array([1.0, 0.0, 0.0])  # 3-dim
        node_b = MagicMock()
        node_b.embedding = np.array([1.0, 0.0, 0.0, 0.0, 0.0])  # 5-dim
        compressor.chunks = {"fileA_0": node_a, "fileB_0": node_b}
        # Should not crash, should skip mismatched pairs
        result = compressor.find_duplicates(threshold=0.5)
        assert result == []


# =========================================================================
# diff_reingest null embedding guard
# =========================================================================

class TestDiffReingestNullGuard:
    def test_preserves_only_non_null_embeddings(self):
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        compressor.graphs = {"test": MagicMock()}

        node_with_emb = MagicMock()
        node_with_emb.text = "kept text"
        node_with_emb.embedding = np.array([1.0, 2.0])

        node_without_emb = MagicMock()
        node_without_emb.text = "another text"
        node_without_emb.embedding = None

        compressor.chunks = {
            "test_0": node_with_emb,
            "test_1": node_without_emb,
        }
        compressor._chunk_text = MagicMock(return_value=["kept text", "another text", "new text"])

        stats = compressor._compute_diff_stats("test", "kept text\nanother text\nnew text")
        # Only the node with non-None embedding should be preserved
        assert "kept text" in stats["preserved"]
        assert "another text" not in stats["preserved"]


# =========================================================================
# handle_ingest div-by-zero and cost_savings protection
# =========================================================================

class TestHandleIngestProtection:
    @pytest.mark.asyncio
    async def test_zero_total_tokens_no_crash(self):
        from src.handlers.compression_handlers import handle_ingest

        mock_skeleton = MagicMock()
        mock_skeleton.total_tokens = 0
        mock_skeleton.skeleton_tokens = 0
        mock_skeleton.total_nodes = 0
        mock_skeleton.compression_ratio = 0

        mock_estimate = MagicMock()
        mock_estimate.compression_ratio = 0

        compressor = AsyncMock()
        compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
        compressor.generate_skeleton = MagicMock(return_value=mock_skeleton)
        compressor.estimate_compression = MagicMock(return_value=mock_estimate)
        compressor.chunks = {}
        compressor.graphs = {}

        context = {
            "compressor": compressor,
            "persistence": MagicMock(),
            "sync_manager": MagicMock(),
            "version_manager": AsyncMock(),
            "validate_file_id": MagicMock(),
            "resource_manager": AsyncMock(),
            "retrieval_history": {},
        }
        context["resource_manager"].check_document_size_async = AsyncMock(return_value=(True, None))
        context["resource_manager"].register_document_async = AsyncMock()
        context["persistence"].save_document = MagicMock(return_value=True)

        args = {"text": "This is a sufficiently long test document for semantic analysis purposes.", "file_id": "test"}
        result = await handle_ingest(context, args)
        parsed = json.loads(result)
        assert parsed["token_savings_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_cost_savings_exception_handled(self):
        from src.handlers.compression_handlers import handle_ingest

        mock_skeleton = MagicMock()
        mock_skeleton.total_tokens = 100
        mock_skeleton.skeleton_tokens = 20
        mock_skeleton.total_nodes = 5
        mock_skeleton.compression_ratio = 5.0

        mock_estimate = MagicMock()
        mock_estimate.compression_ratio = 5.0

        compressor = AsyncMock()
        compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
        compressor.generate_skeleton = MagicMock(return_value=mock_skeleton)
        compressor.estimate_compression = MagicMock(return_value=mock_estimate)
        compressor.chunks = {}
        compressor.graphs = {}

        context = {
            "compressor": compressor,
            "persistence": MagicMock(),
            "sync_manager": MagicMock(),
            "version_manager": AsyncMock(),
            "validate_file_id": MagicMock(),
            "resource_manager": AsyncMock(),
            "retrieval_history": {},
        }
        context["resource_manager"].check_document_size_async = AsyncMock(return_value=(True, None))
        context["resource_manager"].register_document_async = AsyncMock()
        context["persistence"].save_document = MagicMock(return_value=True)

        args = {"text": "This is a sufficiently long test document for semantic analysis purposes.", "file_id": "test"}
        with patch("src.handlers.compression_handlers.compute_cost_savings", side_effect=Exception("boom")):
            result = await handle_ingest(context, args)
        parsed = json.loads(result)
        assert parsed["cost_savings"] is None
        assert parsed["status"] == "success"


# =========================================================================
# Atomic persistence writes
# =========================================================================

class TestAtomicPersistence:
    def test_atomic_write_json(self, tmp_path):
        from src.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_path / "store"))
        target = tmp_path / "store" / "test.json"
        pm._atomic_write_json(target, {"key": "value"})
        with open(target) as f:
            data = json.load(f)
        assert data["key"] == "value"
        # No temp files left
        assert not list(tmp_path.glob("*.tmp"))

    def test_atomic_write_json_cleanup_on_error(self, tmp_path):
        from src.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_path / "store"))
        # Use a non-existent directory to cause write failure
        target = tmp_path / "nonexistent_dir" / "deep" / "test.json"
        with pytest.raises(Exception):
            pm._atomic_write_json(target, {"key": "value"})
        assert not target.exists()

    def test_atomic_write_npz(self, tmp_path):
        from src.persistence import PersistenceManager
        pm = PersistenceManager(str(tmp_path / "store"))
        target = tmp_path / "store" / "test.npz"
        pm._atomic_write_npz(target, embeddings=np.array([1.0, 2.0]))
        loaded = np.load(target)
        np.testing.assert_array_equal(loaded["embeddings"], [1.0, 2.0])


# =========================================================================
# New MCP tool handlers
# =========================================================================

class TestNewMCPHandlers:
    @pytest.mark.asyncio
    async def test_handle_diff_reingest_missing_args(self):
        from src.handlers.compression_handlers import handle_diff_reingest
        result = await handle_diff_reingest({}, {"file_id": "test"})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_not_found(self):
        from src.handlers.compression_handlers import handle_diff_reingest
        compressor = AsyncMock()
        compressor.diff_reingest_async = AsyncMock(side_effect=ValueError("not found"))
        result = await handle_diff_reingest(
            {"compressor": compressor, "persistence": MagicMock(), "version_manager": AsyncMock()},
            {"file_id": "missing", "text": "hello"}
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"]

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_success(self):
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.semantic_compressor import DiffReingestionResult
        mock_result = DiffReingestionResult(
            file_id="test", chunks_unchanged=3, chunks_updated=1,
            chunks_added=1, chunks_removed=0
        )
        compressor = AsyncMock()
        compressor.diff_reingest_async = AsyncMock(return_value=mock_result)
        compressor.graphs = {"test": MagicMock()}
        compressor.chunks = {"test_n0": MagicMock(), "test_n1": MagicMock()}
        compressor.file_metadata = {"test": {"source": "test"}}
        context = {
            "compressor": compressor,
            "persistence": MagicMock(save_document=MagicMock(return_value=True)),
            "version_manager": AsyncMock(add_version_async=AsyncMock()),
        }
        result = await handle_diff_reingest(
            context,
            {"file_id": "test", "text": "new content"}
        )
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["chunks_unchanged"] == 3

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_persists_to_disk(self):
        """Verify diff_reingest calls persistence.save_document."""
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.semantic_compressor import DiffReingestionResult
        mock_result = DiffReingestionResult(
            file_id="doc1", chunks_unchanged=2, chunks_updated=1,
            chunks_added=0, chunks_removed=0
        )
        compressor = AsyncMock()
        compressor.diff_reingest_async = AsyncMock(return_value=mock_result)
        compressor.graphs = {"doc1": MagicMock()}
        compressor.chunks = {"doc1_n0": MagicMock()}
        compressor.file_metadata = {"doc1": {}}
        mock_persistence = MagicMock(save_document=MagicMock(return_value=True))
        mock_version_mgr = AsyncMock(add_version_async=AsyncMock())
        context = {
            "compressor": compressor,
            "persistence": mock_persistence,
            "version_manager": mock_version_mgr,
        }
        result = await handle_diff_reingest(context, {"file_id": "doc1", "text": "updated"})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Verify persistence was called
        mock_persistence.save_document.assert_called_once()
        call_kwargs = mock_persistence.save_document.call_args
        assert call_kwargs[1]["file_id"] == "doc1" or call_kwargs.kwargs["file_id"] == "doc1"

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_saves_version(self):
        """Verify diff_reingest calls version_manager.add_version_async."""
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.semantic_compressor import DiffReingestionResult
        mock_result = DiffReingestionResult(
            file_id="doc1", chunks_unchanged=2, chunks_updated=1,
            chunks_added=0, chunks_removed=0
        )
        compressor = AsyncMock()
        compressor.diff_reingest_async = AsyncMock(return_value=mock_result)
        compressor.graphs = {"doc1": MagicMock()}
        compressor.chunks = {"doc1_n0": MagicMock()}
        compressor.file_metadata = {"doc1": {}}
        mock_version_mgr = AsyncMock(add_version_async=AsyncMock())
        context = {
            "compressor": compressor,
            "persistence": MagicMock(save_document=MagicMock(return_value=True)),
            "version_manager": mock_version_mgr,
        }
        result = await handle_diff_reingest(context, {"file_id": "doc1", "text": "updated text"})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Verify version manager was called
        mock_version_mgr.add_version_async.assert_called_once()
        call_kwargs = mock_version_mgr.add_version_async.call_args
        assert call_kwargs.kwargs["doc_id"] == "doc1"
        assert call_kwargs.kwargs["checksum"] is not None

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_success(self):
        from src.handlers.compression_handlers import handle_find_duplicates
        compressor = MagicMock()
        compressor.find_duplicates = MagicMock(return_value=[])
        result = await handle_find_duplicates(
            {"compressor": compressor}, {"threshold": 0.95}
        )
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["duplicate_count"] == 0

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_error(self):
        from src.handlers.compression_handlers import handle_find_duplicates
        compressor = MagicMock()
        compressor.find_duplicates = MagicMock(side_effect=RuntimeError("fail"))
        result = await handle_find_duplicates(
            {"compressor": compressor}, {}
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_handle_get_presets(self):
        from src.handlers.compression_handlers import handle_get_presets
        result = await handle_get_presets({}, {})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert len(parsed["presets"]) == 5
        names = {p["name"] for p in parsed["presets"]}
        assert "code-review" in names
        assert "aggressive" in names


# =========================================================================
# MCP tool registration
# =========================================================================

class TestToolRegistration:
    def test_new_tools_in_schema(self):
        from src.handlers.mcp_core import setup_mcp_tools
        tools = setup_mcp_tools()
        tool_names = {t.name for t in tools}
        assert "diff_reingest" in tool_names
        assert "find_duplicates" in tool_names
        assert "get_compression_presets" in tool_names
        assert "check_context_budget" in tool_names

    @pytest.mark.asyncio
    async def test_router_dispatches_new_tools(self):
        from src.handlers.mcp_core import route_tool_call
        context = {"compressor": MagicMock()}
        context["compressor"].find_duplicates = MagicMock(return_value=[])
        result = await route_tool_call("find_duplicates", {}, context)
        parsed = json.loads(result)
        assert parsed["status"] == "success"

    @pytest.mark.asyncio
    async def test_router_dispatches_presets(self):
        from src.handlers.mcp_core import route_tool_call
        result = await route_tool_call("get_compression_presets", {}, {})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert len(parsed["presets"]) >= 5
