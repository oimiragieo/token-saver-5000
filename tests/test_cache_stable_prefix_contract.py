"""N5 contract test: cache_stable_prefix must never silently return
query-conditioned text.

Bug (verified against src/handlers/compression_handlers.py:1171-1178 before the
fix): in a non-baseline (query_guided / evidence_aware) selection_mode, on a
`_baseline_skeleton_cache` MISS, `.get(scoped_file_id, skeleton_response.skeleton_text)`
silently fell back to the query-conditioned skeleton text. A customer caching
that value under a "stable prefix" contract caches something that varies per
query -- exactly the prefix-cache defeat arXiv 2607.15516 measures.

`_baseline_skeleton_cache` is in-process (semantic_compressor.py:1451) so a
miss is routine (restart / eviction / cross-worker ingest), not exotic.
"""

import json
from unittest.mock import patch

import pytest

import src.handlers.compression_handlers as ch


class _FakeSkeletonResponse:
    """Minimal stand-in for SkeletonResponse -- JSON-serializable fields only."""

    def __init__(self, skeleton_text: str):
        self.skeleton_text = skeleton_text
        self.total_nodes = 3
        self.total_tokens = 300
        self.skeleton_tokens = 30
        self.compression_ratio = 10.0
        self.node_map = {}


class _FakeCompressor:
    """A compressor whose `_generate_skeleton` output actually varies by
    query, unlike the `unittest.mock.Mock` used elsewhere in this test suite
    (which returns the same canned object regardless of args and therefore
    cannot discriminate this bug class -- the whole point of this file).

    `_baseline_skeleton_cache` starts EMPTY to model the routine miss case
    (restart / eviction / cross-worker ingest) rather than the freshly-ingested
    case where ingest already populated it.
    """

    def __init__(self):
        self._baseline_skeleton_cache: dict = {}

    def _generate_skeleton(self, file_id, query=None, **kwargs):
        if query is None:
            # The query-free / baseline call.
            return _FakeSkeletonResponse(f"BASELINE::{file_id}")
        # Query-conditioned call -- different text per query, by construction.
        return _FakeSkeletonResponse(f"QUERY::{file_id}::{query}")


def _context():
    compressor = _FakeCompressor()
    sync_manager = type("SM", (), {"file_metadata": {}})()
    return {"compressor": compressor, "sync_manager": sync_manager}, compressor


async def _read(context, query):
    args = {
        "file_id": "doc1",
        "selection_mode": "query_guided",
        "query": query,
    }
    with patch("src.handlers.compression_handlers.validate_file_id"):
        result = await ch.handle_read_skeleton(context, args)
    return json.loads(result)


class TestCacheStablePrefixContract:
    @pytest.mark.asyncio
    async def test_red_two_queries_must_not_diverge_on_cache_miss(self):
        """THE RED ARM. With an empty `_baseline_skeleton_cache` (a routine
        miss) and two DIFFERENT queries in query_guided mode, both responses'
        `cache_stable_prefix` must be equal and query-independent. Pre-fix,
        each returned its own query-conditioned skeleton text -- this
        assertion is exactly the one that failed before the fix.
        """
        context, _ = _context()

        data_a = await _read(context, "query about topic A")
        data_b = await _read(context, "an entirely different query B")

        # Non-vacuity: two empty strings comparing equal would prove nothing.
        assert data_a["cache_stable_prefix"], "cache_stable_prefix must be non-empty"
        assert data_b["cache_stable_prefix"], "cache_stable_prefix must be non-empty"

        assert data_a["cache_stable_prefix"] == data_b["cache_stable_prefix"], (
            "cache_stable_prefix diverged across two different queries on a "
            "baseline-cache MISS -- it silently returned query-conditioned "
            "text, defeating the prefix-cache contract."
        )
        # And it must actually be the query-FREE baseline, not merely "the
        # same value by coincidence" -- pin it to the baseline marker.
        assert data_a["cache_stable_prefix"] == "BASELINE::doc1"

    @pytest.mark.asyncio
    async def test_cache_hit_still_returns_cached_baseline(self):
        """Control: when `_baseline_skeleton_cache` already has an entry (the
        common case -- populated at ingest), the fix must not bypass it and
        recompute instead. The cached value wins verbatim.
        """
        context, compressor = _context()
        compressor._baseline_skeleton_cache["doc1"] = "CACHED-AT-INGEST::doc1"

        data = await _read(context, "some query")

        assert data["cache_stable_prefix"] == "CACHED-AT-INGEST::doc1"

    @pytest.mark.asyncio
    async def test_baseline_mode_path_is_unchanged(self):
        """Control: baseline selection_mode never went through the
        `_baseline_skeleton_cache` substitution branch at all -- must still
        be the case after the fix (no regression to the common case).
        """
        context, _ = _context()
        args = {"file_id": "doc1", "selection_mode": "baseline"}

        with patch("src.handlers.compression_handlers.validate_file_id"):
            result = await ch.handle_read_skeleton(context, args)
        data = json.loads(result)

        assert data["cache_stable_prefix"], "cache_stable_prefix must be non-empty"
        assert data["cache_stable_prefix"] == data["skeleton_text"] == "BASELINE::doc1"
