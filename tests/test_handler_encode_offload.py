"""
Receipts for roadmap A3: the read_skeleton + search_semantic MCP handlers run
their synchronous CPU-bound work (query encode + skeleton/ranking) OFF the event
loop via ``asyncio.to_thread``, not inline on the loop thread.

No output change — these tests only assert the sync work executed on a worker
thread (different thread id than the running event loop) so a long compression
cannot stall the loop / other concurrent MCP sessions.

Run with:
    cd token-saver-5000
    PYTHONPATH=. pytest tests/test_handler_encode_offload.py -v
"""

import threading
from unittest.mock import MagicMock

import pytest

from src.handlers.compression_handlers import (
    handle_read_skeleton,
    handle_search_semantic,
)


def _loop_thread_id() -> int:
    return threading.get_ident()


class _RecordingCompressor:
    """Records the thread id each sync entrypoint runs on."""

    def __init__(self):
        self.chunks = {}
        self.thread_ids = {}

    # --- search_semantic path ------------------------------------------------
    def search_semantic_with_scores(self, query, file_id, top_k):
        self.thread_ids["search_semantic_with_scores"] = threading.get_ident()
        return [("doc_n0", 0.9)]

    def retrieve_evidence(self, query, file_id, top_k, min_similarity):
        self.thread_ids["retrieve_evidence"] = threading.get_ident()
        ev = MagicMock()
        ev.scores = [("doc_n0", 0.9)]
        ev.sufficient = True
        ev.best_score = 0.9
        ev.threshold = min_similarity
        ev.used_expanded_search = False
        ev.message = "ok"
        return ev

    def _generate_summary(self, text, max_length=100):
        return text[:max_length]


def _search_context():
    comp = _RecordingCompressor()
    node = MagicMock()
    node.text = "hello world content"
    node.importance = 0.5
    node.metadata = {"tokens": 3}
    comp.chunks = {"doc_n0": node}
    return {"compressor": comp}, comp


class TestSearchSemanticOffload:
    @pytest.mark.asyncio
    async def test_search_semantic_runs_sync_work_on_worker_thread(self):
        context, comp = _search_context()
        loop_tid = _loop_thread_id()

        await handle_search_semantic(context, {"query": "find content", "top_k": 1})

        ran_on = comp.thread_ids.get("search_semantic_with_scores")
        assert ran_on is not None, "search_semantic_with_scores was never called"
        assert ran_on != loop_tid, (
            "search_semantic_with_scores ran on the EVENT-LOOP thread — A3 offload "
            "missing (it must run via asyncio.to_thread on a worker thread)."
        )

    @pytest.mark.asyncio
    async def test_search_semantic_evidence_aware_runs_on_worker_thread(self):
        context, comp = _search_context()
        loop_tid = _loop_thread_id()

        await handle_search_semantic(
            context, {"query": "find content", "top_k": 1, "evidence_aware": True}
        )

        ran_on = comp.thread_ids.get("retrieve_evidence")
        assert ran_on is not None, "retrieve_evidence was never called"
        assert (
            ran_on != loop_tid
        ), "retrieve_evidence ran on the EVENT-LOOP thread — A3 offload missing."


class TestReadSkeletonOffload:
    @pytest.mark.asyncio
    async def test_read_skeleton_runs_pipeline_on_worker_thread(self, monkeypatch):
        loop_tid = _loop_thread_id()
        captured = {}

        # Replace the (synchronous) pipeline with a recorder so we can confirm
        # WHICH thread it executes on — without needing a fully-wired compressor.
        def fake_pipeline(**kwargs):
            captured["thread_id"] = threading.get_ident()
            skel = MagicMock()
            skel.skeleton_text = "=== SKELETON ===\nnode summary"
            skel.total_nodes = 1
            skel.total_tokens = 10
            skel.skeleton_tokens = 4
            skel.compression_ratio = 2.5
            skel.node_map = {}
            return {
                "final_skeleton": skel,
                "final_stage": "baseline",
                "selection_mode_resolved": "baseline",
                "stage_count": 1,
                "stages": ["baseline"],
                "evidence": None,
            }

        monkeypatch.setattr(
            "src.handlers.compression_handlers_ingest.run_read_skeleton_pipeline",
            fake_pipeline,
        )
        monkeypatch.setattr(
            "src.handlers.compression_handlers_ingest.validate_file_id",
            lambda *a, **k: None,
        )

        compressor = MagicMock()
        compressor.chunks = {}
        compressor._baseline_skeleton_cache = {}
        sync_manager = MagicMock()
        sync_manager.file_metadata = {}
        context = {"compressor": compressor, "sync_manager": sync_manager}

        await handle_read_skeleton(context, {"file_id": "doc", "selection_mode": "baseline"})

        assert "thread_id" in captured, "pipeline was never invoked"
        assert captured["thread_id"] != loop_tid, (
            "run_read_skeleton_pipeline ran on the EVENT-LOOP thread — A3 offload "
            "missing (it must run via asyncio.to_thread on a worker thread)."
        )
