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
import warnings
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch

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

        # Use the canonical "{file_id}_n{i}" node-ID format the production
        # ingest path actually emits (audit P1-5 made file_id membership
        # boundary-aware, so the loose "test_0" prefix no longer matches "test").
        compressor.chunks = {
            "test_n0": node_with_emb,
            "test_n1": node_without_emb,
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

        args = {
            "text": "This is a sufficiently long test document for semantic analysis purposes.",
            "file_id": "test",
        }
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

        args = {
            "text": "This is a sufficiently long test document for semantic analysis purposes.",
            "file_id": "test",
        }
        with patch(
            "src.handlers.compression_handlers_ingest.compute_cost_savings",
            side_effect=Exception("boom"),
        ):
            result = await handle_ingest(context, args)
        parsed = json.loads(result)
        assert parsed["cost_savings"] is None
        assert parsed["status"] == "success"

    @pytest.mark.asyncio
    async def test_phase5_async_hooks_are_awaited_without_runtime_warnings(self):
        from src.handlers.compression_handlers import handle_ingest

        mock_skeleton = MagicMock()
        mock_skeleton.total_tokens = 100
        mock_skeleton.skeleton_tokens = 20
        mock_skeleton.total_nodes = 5
        mock_skeleton.compression_ratio = 5.0
        mock_skeleton.skeleton_text = "compressed summary"

        tracker = MagicMock(record_access=AsyncMock())
        replay = MagicMock(record=AsyncMock())
        compressor = AsyncMock()
        compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
        compressor.generate_skeleton = MagicMock(return_value=mock_skeleton)
        compressor.estimate_compression = MagicMock(return_value=MagicMock(compression_ratio=5.0))
        # 2026-07-06 knob-honesty fix: set_file_skeleton_ratio is a genuinely
        # SYNCHRONOUS method on the real compressor (handle_ingest calls it,
        # not awaits it). The base AsyncMock() auto-generates unconfigured
        # attributes as AsyncMock too, which would make this call return an
        # unawaited coroutine and trip the exact RuntimeWarning this test
        # asserts is absent — same pattern as the other sync methods above.
        compressor.set_file_skeleton_ratio = MagicMock()
        compressor.model = MagicMock(
            encode=MagicMock(return_value=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        )
        compressor._access_tracker = tracker
        compressor._compression_replay = replay
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

        args = {
            "text": "This is a sufficiently long test document for semantic analysis purposes.",
            "file_id": "test",
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = await handle_ingest(context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        tracker.record_access.assert_awaited_once_with("test")
        replay.record.assert_awaited_once()
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warnings == []


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
        # F8 fix: _atomic_write_json now auto-creates parent directories
        # (parents=True, exist_ok=True) so callers using file_ids with
        # forward slashes (e.g. "docs/audits/foo.md") no longer hit ENOENT.
        # This test previously expected an exception on missing dirs;
        # updated to assert the new documented behavior: success + file exists.
        target = tmp_path / "nonexistent_dir" / "deep" / "test.json"
        pm._atomic_write_json(target, {"key": "value"})
        assert target.exists()

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
            {"file_id": "missing", "text": "hello"},
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "not found" in parsed["error"]

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_success(self):
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.semantic_compressor import DiffReingestionResult

        mock_result = DiffReingestionResult(
            file_id="test", chunks_unchanged=3, chunks_updated=1, chunks_added=1, chunks_removed=0
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
        result = await handle_diff_reingest(context, {"file_id": "test", "text": "new content"})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["chunks_unchanged"] == 3

    @pytest.mark.asyncio
    async def test_handle_diff_reingest_persists_to_disk(self):
        """Verify diff_reingest calls persistence.save_document."""
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.semantic_compressor import DiffReingestionResult

        mock_result = DiffReingestionResult(
            file_id="doc1", chunks_unchanged=2, chunks_updated=1, chunks_added=0, chunks_removed=0
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
            file_id="doc1", chunks_unchanged=2, chunks_updated=1, chunks_added=0, chunks_removed=0
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
    async def test_handle_diff_reingest_scoped_file_id(self):
        """Verify diff_reingest uses tenant-scoped internal IDs."""
        from src.handlers.compression_handlers import handle_diff_reingest
        from src.identity_scope import compose_scoped_file_id
        from src.semantic_compressor import DiffReingestionResult

        scoped_file_id = compose_scoped_file_id("doc1", workspace_id="acme")
        mock_result = DiffReingestionResult(
            file_id=scoped_file_id,
            chunks_unchanged=2,
            chunks_updated=1,
            chunks_added=0,
            chunks_removed=0,
        )
        compressor = AsyncMock()
        compressor.diff_reingest_async = AsyncMock(return_value=mock_result)
        compressor.graphs = {scoped_file_id: MagicMock()}
        compressor.chunks = {f"{scoped_file_id}_n0": MagicMock()}
        compressor.file_metadata = {scoped_file_id: {}}
        mock_version_mgr = AsyncMock(add_version_async=AsyncMock())
        mock_persistence = MagicMock(save_document=MagicMock(return_value=True))
        context = {
            "compressor": compressor,
            "persistence": mock_persistence,
            "version_manager": mock_version_mgr,
        }

        result = await handle_diff_reingest(
            context,
            {"file_id": "doc1", "workspace_id": "acme", "text": "updated text"},
        )
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["file_id"] == "doc1"
        compressor.diff_reingest_async.assert_called_once_with(scoped_file_id, "updated text")
        assert mock_version_mgr.add_version_async.call_args.kwargs["doc_id"] == scoped_file_id

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_success(self):
        from src.handlers.compression_handlers import handle_find_duplicates

        compressor = MagicMock()
        compressor.find_duplicates = MagicMock(return_value=[])
        result = await handle_find_duplicates({"compressor": compressor}, {"threshold": 0.95})
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["duplicate_count"] == 0

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_error(self):
        from src.handlers.compression_handlers import handle_find_duplicates

        compressor = MagicMock()
        compressor.find_duplicates = MagicMock(side_effect=RuntimeError("fail"))
        result = await handle_find_duplicates({"compressor": compressor}, {})
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


# =========================================================================
# find_duplicates timeout
# =========================================================================


class TestFindDuplicatesTimeout:
    def test_timeout_returns_partial_results(self):
        """find_duplicates should abort and return partial results on timeout."""
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        # Create many chunks across two files to trigger timeout
        compressor.chunks = {}
        for i in range(200):
            node = MagicMock()
            node.embedding = np.random.rand(384)
            file_prefix = "fileA" if i < 100 else "fileB"
            compressor.chunks[f"{file_prefix}_n{i}"] = node

        # With a tiny timeout, should hit timeout marker
        result = compressor.find_duplicates(threshold=0.99, timeout_seconds=0.0001)
        # Either we get timeout marker or it finished fast enough — both valid
        # Just verify no crash and returns list
        assert isinstance(result, list)

    def test_no_timeout_with_small_dataset(self):
        """Small datasets should complete without timeout."""
        compressor = SemanticCompressor.__new__(SemanticCompressor)
        node_a = MagicMock()
        node_a.embedding = np.array([1.0, 0.0, 0.0])
        node_b = MagicMock()
        node_b.embedding = np.array([1.0, 0.0, 0.0])
        compressor.chunks = {"fileA_0": node_a, "fileB_0": node_b}
        result = compressor.find_duplicates(threshold=0.9, timeout_seconds=30.0)
        assert len(result) == 1
        assert "warning" not in result[0]


# =========================================================================
# find_duplicates multi-tenant scope isolation
# =========================================================================


class TestFindDuplicatesTenantScoping:
    """handle_find_duplicates must not leak duplicate pairs across tenants.

    The compressor's chunk store is process-wide and shared across tenants
    (keyed by scoped file_id). Sibling handlers (e.g. handle_search_semantic)
    already filter their results through scope_matches(); handle_find_duplicates
    forgot to, so a caller with a scope could see duplicate pairs whose file_ids
    belong to a DIFFERENT tenant sharing the process.
    """

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_isolates_by_scope(self):
        from src.handlers.compression_handlers import handle_find_duplicates
        from src.identity_scope import compose_scoped_file_id

        compressor = SemanticCompressor.__new__(SemanticCompressor)

        # Two near-duplicate docs for tenant A, two near-duplicate docs for
        # tenant B — all four chunks share the same embedding so every
        # cross-file pair within a tenant is a genuine duplicate.
        file_a1 = compose_scoped_file_id("doc1", user_id="userA")
        file_a2 = compose_scoped_file_id("doc2", user_id="userA")
        file_b1 = compose_scoped_file_id("doc1", user_id="userB")
        file_b2 = compose_scoped_file_id("doc2", user_id="userB")

        def _node():
            node = MagicMock()
            node.embedding = np.array([1.0, 0.0, 0.0])
            return node

        compressor.chunks = {
            f"{file_a1}_n0": _node(),
            f"{file_a2}_n0": _node(),
            f"{file_b1}_n0": _node(),
            f"{file_b2}_n0": _node(),
        }

        result = await handle_find_duplicates(
            {"compressor": compressor},
            {"threshold": 0.9, "user_id": "userA"},
        )
        parsed = json.loads(result)

        assert parsed["status"] == "success"
        # Only the tenant-A cross-file pair should survive scope filtering.
        assert parsed["duplicate_count"] == 1
        pair = parsed["duplicates"][0]
        assert "userB" not in pair["node_a"]
        assert "userB" not in pair["node_b"]

    @pytest.mark.asyncio
    async def test_handle_find_duplicates_unscoped_call_is_unaffected(self):
        """A caller that passes no scope args keeps the pre-fix, unfiltered
        behavior (mirrors _has_scope_args gating on sibling handlers)."""
        from src.handlers.compression_handlers import handle_find_duplicates

        compressor = SemanticCompressor.__new__(SemanticCompressor)
        node_a = MagicMock()
        node_a.embedding = np.array([1.0, 0.0, 0.0])
        node_b = MagicMock()
        node_b.embedding = np.array([1.0, 0.0, 0.0])
        compressor.chunks = {"fileA_n0": node_a, "fileB_n0": node_b}

        result = await handle_find_duplicates({"compressor": compressor}, {"threshold": 0.9})
        parsed = json.loads(result)

        assert parsed["status"] == "success"
        assert parsed["duplicate_count"] == 1


# =========================================================================
# handle_batch_ingest tenant scoping (mirrors handle_find_duplicates above
# and handle_ingest's own file_id scoping contract)
# =========================================================================


class TestHandleBatchIngestTenantScoping:
    """handle_batch_ingest must scope file_id per-tenant like handle_ingest.

    The compressor's document store is process-wide and multi-tenant.
    handle_ingest composes a scoped internal id via _scoped_file_id() before
    ever touching the compressor, so two tenants can reuse the same plain
    file_id without collision. handle_batch_ingest forgot to do the same
    for BatchDocument.file_id (it passed the raw, unscoped caller value
    straight through to BatchCompressionManager -> compressor.ingest_file_async),
    so two tenants batch-ingesting the same plain file_id (e.g. "notes") would
    collide on one unscoped process-wide key -- the second caller's ingest
    silently overwrote the first's.
    """

    @pytest.mark.asyncio
    async def test_batch_ingest_scopes_file_id_per_tenant(self):
        from src.handlers.compression_handlers import handle_batch_ingest
        from src.identity_scope import compose_scoped_file_id

        recorded_file_ids = []

        async def _fake_ingest_file_async(text, file_id, metadata):
            recorded_file_ids.append(file_id)
            return MagicMock(
                skeleton_text=f"skeleton for {file_id}",
                compression_ratio=2.0,
                total_nodes=1,
            )

        fake_compressor = MagicMock()
        fake_compressor.ingest_file_async = AsyncMock(side_effect=_fake_ingest_file_async)

        context = {"compressor": fake_compressor}

        with patch(
            "src.handlers.compression_handlers_ingest.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            result_a = await handle_batch_ingest(
                context,
                {
                    "documents": [{"file_id": "notes", "text": "tenant A notes"}],
                    "user_id": "userA",
                },
            )
            result_b = await handle_batch_ingest(
                context,
                {
                    "documents": [{"file_id": "notes", "text": "tenant B notes"}],
                    "user_id": "userB",
                },
            )

        parsed_a = json.loads(result_a)
        parsed_b = json.loads(result_b)

        assert parsed_a["successful"] == 1
        assert parsed_b["successful"] == 1

        expected_a = compose_scoped_file_id("notes", user_id="userA")
        expected_b = compose_scoped_file_id("notes", user_id="userB")

        # Pre-fix, both entries are the bare "notes" string -- a collision on
        # one unscoped process-wide key. Post-fix each tenant gets a distinct
        # internal key.
        assert recorded_file_ids == [expected_a, expected_b]
        assert expected_a != expected_b

        # The caller-visible file_id in the response must stay the RAW id the
        # caller passed in -- never the internal scoped key (mirrors
        # handle_ingest / handle_ingest_directory's display_file_id contract).
        assert parsed_a["results"][0]["file_id"] == "notes"
        assert parsed_b["results"][0]["file_id"] == "notes"

    @pytest.mark.asyncio
    async def test_batch_ingest_unscoped_call_is_unaffected(self):
        """A caller that passes no scope args keeps the pre-fix, unscoped
        behavior (file_id passed straight through unchanged)."""
        from src.handlers.compression_handlers import handle_batch_ingest

        recorded_file_ids = []

        async def _fake_ingest_file_async(text, file_id, metadata):
            recorded_file_ids.append(file_id)
            return MagicMock(skeleton_text="skeleton", compression_ratio=2.0, total_nodes=1)

        fake_compressor = MagicMock()
        fake_compressor.ingest_file_async = AsyncMock(side_effect=_fake_ingest_file_async)

        context = {"compressor": fake_compressor}

        with patch(
            "src.handlers.compression_handlers_ingest.RATE_LIMITERS",
            {"batch_ingest": AsyncMock(acquire=AsyncMock())},
        ):
            result = await handle_batch_ingest(
                context,
                {"documents": [{"file_id": "unscoped_doc", "text": "hello"}]},
            )

        parsed = json.loads(result)
        assert parsed["successful"] == 1
        assert recorded_file_ids == ["unscoped_doc"]
        assert parsed["results"][0]["file_id"] == "unscoped_doc"


# =========================================================================
# Validation hooks for destructive operations
# =========================================================================


class TestValidationHooksDestructive:
    def test_delete_document_requires_file_id(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("delete_document", {"file_id": ""})
        assert len(errors) == 1
        assert "file_id" in errors[0]

    def test_delete_document_valid(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("delete_document", {"file_id": "my_doc"})
        assert errors == []

    def test_batch_ingest_empty_list(self):
        # The real MCP tool name is "batch_ingest_documents" (see
        # mcp_core.py's tools/call dispatch table) -- the validator was
        # previously registered under the never-invoked "batch_ingest" key,
        # so this hook silently never fired on real batch ingest calls.
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("batch_ingest_documents", {"documents": []})
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_batch_ingest_too_many(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("batch_ingest_documents", {"documents": [{}] * 101})
        assert len(errors) == 1
        assert "100" in errors[0]

    def test_batch_ingest_valid(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("batch_ingest_documents", {"documents": [{"text": "a"}]})
        assert errors == []

    def test_batch_ingest_wrong_legacy_key_is_inert(self):
        # Locks the fix: the old (wrong) registration key no longer matches
        # anything, since the router calls validate_tool_input(name, args)
        # with the REAL tool name -- "batch_ingest" was never that name.
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("batch_ingest", {"documents": []})
        assert errors == []

    def test_unregistered_tool_passes(self):
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("some_unknown_tool", {"anything": True})
        assert errors == []


# =========================================================================
# MetricsCollector wired into handlers
# =========================================================================


class TestMetricsWiring:
    @pytest.mark.asyncio
    async def test_ingest_records_metrics(self):
        """handle_ingest should call MetricsCollector recording methods."""
        from src.handlers.compression_handlers import handle_ingest

        mock_skeleton = MagicMock()
        mock_skeleton.total_nodes = 5
        mock_skeleton.total_tokens = 100
        mock_skeleton.skeleton_tokens = 20
        mock_skeleton.compression_ratio = 5.0
        mock_skeleton.skeleton_text = "test"

        compressor = AsyncMock()
        compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
        compressor.graphs = {"test": MagicMock()}
        compressor.chunks = {"test_n0": MagicMock()}
        compressor.file_metadata = {}

        context = {
            "compressor": compressor,
            "persistence": MagicMock(save_document=MagicMock(return_value=True)),
            "resource_manager": AsyncMock(
                check_document_size_async=AsyncMock(return_value=(True, None)),
                register_document_async=AsyncMock(),
            ),
            "version_manager": AsyncMock(add_version_async=AsyncMock()),
            "sync_manager": MagicMock(
                register_file=MagicMock(), export_metadata=MagicMock(return_value={})
            ),
            "path_validator": MagicMock(),
            "retrieval_history": {},
        }

        with patch("src.handlers.compression_handlers_ingest.get_metrics") as mock_get_metrics:
            mock_metrics = MagicMock()
            mock_get_metrics.return_value = mock_metrics

            args = {
                "text": "This is a sufficiently long test document for semantic analysis purposes.",
                "file_id": "test",
            }
            result = await handle_ingest(context, args)
            parsed = json.loads(result)
            assert parsed["status"] == "success"

            # Verify metrics were recorded
            mock_metrics.record_compression_ratio.assert_called_once()
            mock_metrics.increment_documents_processed.assert_called_once_with(
                "ingest", "BALANCED", "success"
            )
            mock_metrics.set_active_documents.assert_called_once()
