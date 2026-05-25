"""
F11 Path C — BM25 + Reciprocal Rank Fusion hybrid retrieval tests.

Plan: docs/superpowers/plans/2026-05-25-f11-path-c-bm25-rrf-plan.md

Council patches verified:
  P1 — BM25 receives file_id-filtered candidate set (NOT self.chunks globally).
       test_bm25_uses_file_id_filtered_candidates_not_full_chunks
  P2 — score_type: "rrf"|"cosine" in handler response.
       test_response_includes_score_type_field_rrf / _cosine
  P3 — IDF-pollution regression test (two-document setup, doc B must not
       pollute doc A BM25 scores).
       test_bm25_uses_file_id_filtered_candidates_not_full_chunks (covers P3)

EmbeddingManager singleton canary (Phase 7c-4 carry-forward):
  If RRF scores are byte-identical to cosine scores across 3+ distinct
  query/chunk pairs → BM25 is silently not running. Mathematical impossibility
  if BM25 contributed. Verified by test_rrf_scores_differ_from_dense_only.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs — avoid importing the full server stack in CI
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Returns deterministic embeddings so tests are reproducible without models."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Hash-based deterministic embedding (dim=8, L2-normalized)."""
        vecs = []
        for text in texts:
            seed = sum(ord(c) for c in text) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(8).astype(np.float32)
            norm = np.linalg.norm(v)
            vecs.append(v / max(norm, 1e-9))
        return np.array(vecs, dtype=np.float32)


@pytest.fixture()
def fake_embedder():
    return _FakeEmbedder()


# ---------------------------------------------------------------------------
# Inline SemanticNode + SemanticCompressor stubs
# (avoids needing live ONNX models in CI)
# ---------------------------------------------------------------------------


def _make_compressor_with_chunks(
    chunks: Dict[str, Dict[str, Any]],
    embedder: _FakeEmbedder,
) -> Any:
    """Build a minimal SemanticCompressor-like object backed by fake_embedder.

    Each chunk dict must have: text (str), importance (float).
    The node_id is the dict key.
    """
    from src.semantic_compressor import SemanticCompressor, SemanticNode

    compressor = object.__new__(SemanticCompressor)
    compressor.model = embedder

    # Build chunks dict with SemanticNode instances
    compressor.chunks = {}
    for node_id, data in chunks.items():
        node = object.__new__(SemanticNode)
        node.text = data["text"]
        node.importance = data.get("importance", 0.5)
        node.metadata = {"tokens": len(data["text"].split())}
        # Compute embedding via fake embedder
        node.embedding = embedder.encode([data["text"]])[0]
        compressor.chunks[node_id] = node

    # Minimal stubs for unused attributes
    compressor.graphs = {}

    return compressor


# ---------------------------------------------------------------------------
# bm25_utils unit tests
# ---------------------------------------------------------------------------


class TestBM25Utils:
    """Unit tests for the shared BM25 utility module."""

    def test_bm25_tokenize_lowercases(self):
        from src.bm25_utils import bm25_tokenize

        tokens = bm25_tokenize("Hello World_foo BAR123")
        assert "hello" in tokens
        assert "world_foo" in tokens
        assert "bar123" in tokens

    def test_bm25_scores_empty_query_returns_zeros(self):
        from src.bm25_utils import bm25_scores

        result = bm25_scores("", ["doc one", "doc two"])
        assert result == [0.0, 0.0]

    def test_bm25_scores_empty_texts_returns_empty(self):
        from src.bm25_utils import bm25_scores

        result = bm25_scores("query", [])
        assert result == []

    def test_bm25_scores_exact_match_is_nonzero(self):
        from src.bm25_utils import bm25_scores

        texts = ["needle in a haystack", "completely unrelated content"]
        scores = bm25_scores("needle", texts)
        assert scores[0] > 0.0
        assert scores[1] == 0.0

    def test_bm25_idf_zero_division_safe(self):
        """Term present in ALL documents should return near-zero IDF, not raise."""
        from src.bm25_utils import bm25_idf

        tokenized = [["the", "cat"], ["the", "dog"], ["the", "fish"]]
        idf_val = bm25_idf("the", tokenized)
        # log(1 + (3 - 3 + 0.5) / (3 + 0.5)) = log(1 + 0.5/3.5) ≈ 0.133
        assert not math.isnan(idf_val)
        assert not math.isinf(idf_val)
        assert idf_val >= 0.0

    def test_bm25_prefix_stemming_matches_variant(self):
        """Query 'authentication' should match 'authenticate' via prefix."""
        from src.bm25_utils import bm25_scores

        texts = ["authenticate users via token", "completely different content"]
        scores = bm25_scores("authentication", texts)
        # Prefix match: "authen" matches "authenticate"
        assert scores[0] > 0.0, "Expected prefix-stem match"

    def test_bm25_short_term_requires_exact_match(self):
        """Terms < 6 chars need exact match (no prefix stemming)."""
        from src.bm25_utils import bm25_scores

        texts = ["authentication required", "auth token valid"]
        # "auth" (4 chars) requires exact match, should NOT match "authentication"
        scores = bm25_scores("auth", texts)
        # "authentication" does NOT start with "auth" match under exact mode
        # (exact match: "authentication" != "auth")
        # "auth" in "auth token valid" IS an exact match
        assert scores[1] > 0.0, "'auth' should match 'auth' exactly in doc[1]"
        assert (
            scores[0] == 0.0
        ), "'auth' should NOT match 'authentication' (exact-only for <6 chars)"


# ---------------------------------------------------------------------------
# SemanticCompressor._bm25_scores_for_nodes tests
# ---------------------------------------------------------------------------


class TestBM25ScoresForNodes:
    """Tests for the file_id-filtered BM25 helper on SemanticCompressor."""

    def test_returns_empty_for_empty_candidate_nodes(self, fake_embedder):
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        result = compressor._bm25_scores_for_nodes("query", [])
        assert result == []

    def test_excludes_zero_score_nodes(self, fake_embedder):
        chunks = {
            "doc_a::1": {"text": "python async coroutines", "importance": 0.5},
            "doc_a::2": {"text": "totally unrelated gibberish zzz", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)
        candidates = list(compressor.chunks.items())
        result = compressor._bm25_scores_for_nodes("python coroutines", candidates)
        # Zero-score nodes must be excluded
        for nid, score in result:
            assert score > 0.0, f"Node {nid} has score {score} (expected >0)"

    def test_sorted_descending(self, fake_embedder):
        chunks = {
            "doc_a::1": {"text": "bm25 rrf fusion ranking retrieval", "importance": 0.5},
            "doc_a::2": {"text": "bm25 algorithm information retrieval", "importance": 0.5},
            "doc_a::3": {"text": "completely different topic", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)
        candidates = list(compressor.chunks.items())
        result = compressor._bm25_scores_for_nodes("bm25 retrieval", candidates)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True), "BM25 results must be sorted descending"


# ---------------------------------------------------------------------------
# Council Patch P1 + P3 — IDF pollution regression test
# ---------------------------------------------------------------------------


class TestIdfPollutionRegression:
    """
    Council patch P1: BM25 MUST score only file_id-filtered nodes.
    Council patch P3: Two-document corpus, query against doc A only;
    assert doc B IDF does NOT pollute doc A scores.
    """

    def test_bm25_uses_file_id_filtered_candidates_not_full_chunks(self, fake_embedder):
        """
        Setup: Two documents with very different vocabulary.
        - doc_a: contains "ValueError datetime string conversion" (technical error terms)
        - doc_b: contains "recipe flour sugar butter bake cookie cake" (food terms)

        Query: "ValueError datetime" scoped to doc_a.

        IDF test: If BM25 scored self.chunks (both docs), the IDF for "ValueError"
        would be computed over a 2-doc corpus where only 1 doc contains it:
            idf_2doc("ValueError") = log(1 + (2 - 1 + 0.5) / (1 + 0.5)) = log(2) ≈ 0.693

        If BM25 scored only doc_a's nodes (correct, filtered), the IDF would be
        computed over just doc_a's nodes, producing a different value.

        More importantly: doc_b terms ("flour", "sugar") must have ZERO influence
        on doc_a scores regardless of corpus size.

        We verify: the BM25 scores for doc_a nodes when filtered to doc_a's candidates
        differ from scores computed over the full self.chunks corpus.
        """
        chunks = {
            "doc_a::1": {
                "text": "ValueError datetime string conversion asyncpg failed",
                "importance": 0.5,
            },
            "doc_a::2": {
                "text": "datetime object required got string instead",
                "importance": 0.5,
            },
            "doc_b::1": {
                "text": "recipe flour sugar butter bake cookie cake",
                "importance": 0.5,
            },
            "doc_b::2": {
                "text": "chocolate brownie frosting vanilla extract",
                "importance": 0.5,
            },
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)
        query = "ValueError datetime"

        # Filtered: only doc_a candidates (what Path C does)
        doc_a_candidates = [
            (nid, node) for nid, node in compressor.chunks.items() if nid.startswith("doc_a")
        ]
        filtered_scores = compressor._bm25_scores_for_nodes(query, doc_a_candidates)
        filtered_node_ids = {nid for nid, _ in filtered_scores}

        # Verify: doc_b nodes are NOT in filtered scores (P1 guard)
        doc_b_ids = {nid for nid in compressor.chunks if nid.startswith("doc_b")}
        assert filtered_node_ids.isdisjoint(doc_b_ids), (
            f"P1 violation: doc_b node(s) {filtered_node_ids & doc_b_ids} "
            "appeared in file_id-filtered BM25 results"
        )

        # Verify: at least one doc_a node scored > 0 on the query
        doc_a_ids = {nid for nid, _ in filtered_scores}
        assert (
            len(doc_a_ids) > 0
        ), "Expected at least one doc_a node to score on 'ValueError datetime'"

        # IDF pollution check: compute BM25 over FULL corpus (the bad path)
        # and compare to filtered path.  Scores MUST differ because corpus size
        # changes IDF (2 doc_a docs vs 4 total docs).  This is the P3 regression
        # test: if they are equal, _bm25_scores_for_nodes is ignoring the filtered
        # candidates and using the full corpus (IDF pollution).
        all_candidates = list(compressor.chunks.items())
        full_scores = compressor._bm25_scores_for_nodes(query, all_candidates)
        full_score_map = {nid: score for nid, score in full_scores}
        filtered_score_map = {nid: score for nid, score in filtered_scores}

        differs_found = False
        for nid in doc_a_ids:
            full_s = full_score_map.get(nid, 0.0)
            filtered_s = filtered_score_map.get(nid, 0.0)
            assert filtered_s > 0.0, f"Filtered BM25 should score {nid} > 0"
            if abs(full_s - filtered_s) > 1e-9:
                differs_found = True
        assert differs_found, (
            "P3 regression: full-corpus and filtered BM25 scores are identical — "
            "IDF corpus isolation is not working"
        )

    @pytest.mark.asyncio
    async def test_search_semantic_with_scores_does_not_pollute_across_files(self, fake_embedder):
        """
        Council P3 — handler dispatch path:

        End-to-end test through handle_search_semantic verifying that
        file_id scoping prevents IDF pollution from a second document.

        Setup:
          - doc_a: technical Python/asyncpg error terms
          - doc_b: food/recipe terms (semantically orthogonal)

        Under Path C (BM25+RRF), querying "ValueError asyncpg" scoped to
        doc_a must return only doc_a nodes.  If _bm25_scores_for_nodes
        receives the full corpus instead of the filtered candidates, doc_b
        IDF values bleed into doc_a scores — this test catches that.
        """
        import src.semantic_compressor as _sc
        import src.handlers.compression_handlers as _ch
        from src.handlers.compression_handlers import handle_search_semantic

        orig_sc = _sc.F11_RANKER_PATH
        orig_ch = _ch.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        _ch.F11_RANKER_PATH = "c"

        chunks = {
            "doc_a::1": {
                "text": "ValueError asyncpg datetime string conversion failed",
                "importance": 0.5,
            },
            "doc_a::2": {
                "text": "asyncpg expects datetime object not string isoformat",
                "importance": 0.5,
            },
            "doc_b::1": {
                "text": "recipe flour sugar butter bake cookie cake oven",
                "importance": 0.5,
            },
            "doc_b::2": {
                "text": "chocolate brownie frosting vanilla extract confection",
                "importance": 0.5,
            },
        }

        compressor = _make_compressor_with_chunks(chunks, fake_embedder)
        compressor._generate_summary = lambda text, max_length=100: text[:max_length]
        compressor._access_tracker = None
        context = {"compressor": compressor}

        args = {"query": "ValueError asyncpg", "file_id": "doc_a", "top_k": 5}

        try:
            raw = await handle_search_semantic(context, args)
            response = json.loads(raw)

            result_ids = [r["node_id"] for r in response.get("results", [])]

            # Primary assertion: no doc_b node should appear in doc_a-scoped search
            doc_b_leaks = [nid for nid in result_ids if nid.startswith("doc_b")]
            assert doc_b_leaks == [], (
                f"IDF pollution detected: doc_b nodes {doc_b_leaks} "
                "appeared in a doc_a-scoped search_semantic call"
            )

            # Sanity: at least one doc_a result must be returned
            doc_a_results = [nid for nid in result_ids if nid.startswith("doc_a")]
            assert (
                len(doc_a_results) > 0
            ), "Expected at least one doc_a result for 'ValueError asyncpg' query"
        finally:
            _sc.F11_RANKER_PATH = orig_sc
            _ch.F11_RANKER_PATH = orig_ch

    def test_full_corpus_bm25_differs_from_filtered(self, fake_embedder):
        """
        Explicitly verify that scoring the full corpus produces different IDF
        than scoring only a subset — proving our filtering actually matters.
        """
        from src.bm25_utils import bm25_scores as raw_bm25

        # doc_a text
        doc_a_texts = [
            "ValueError datetime string conversion asyncpg",
            "datetime object required got string",
        ]
        # doc_b text (different vocabulary)
        doc_b_texts = [
            "recipe flour sugar butter bake cookie",
            "chocolate brownie frosting vanilla extract",
        ]

        query = "ValueError datetime"

        # BM25 over doc_a only
        scores_doc_a_only = raw_bm25(query, doc_a_texts)
        # BM25 over full corpus
        scores_full = raw_bm25(query, doc_a_texts + doc_b_texts)

        # doc_a scores under full corpus (first 2 elements)
        scores_full_doc_a = scores_full[:2]

        # They MUST differ because IDF is computed over different N
        # (N=2 vs N=4 changes the denominator in log((N - df + 0.5) / (df + 0.5)))
        any_differ = any(abs(a - b) > 1e-9 for a, b in zip(scores_doc_a_only, scores_full_doc_a))
        assert any_differ, (
            "BM25 IDF scores must differ when corpus size changes "
            "(N=2 vs N=4). If identical, IDF is not corpus-aware."
        )


# ---------------------------------------------------------------------------
# _rrf_fuse unit tests
# ---------------------------------------------------------------------------


class TestRRFFuse:
    """Unit tests for the Reciprocal Rank Fusion helper."""

    def test_rrf_degrades_to_dense_when_bm25_empty(self, fake_embedder):
        """Null hypothesis: empty BM25 → RRF output matches dense-only."""
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        bm25 = []  # empty
        fused = compressor._rrf_fuse(dense, bm25, k=60, top_k=3)
        fused_ids = [nid for nid, _ in fused]
        dense_ids = [nid for nid, _ in dense]
        assert fused_ids == dense_ids, "Empty BM25 must produce dense-only ranking"

    def test_rrf_k_parameter_affects_scores(self, fake_embedder):
        """k=1 produces higher scores than k=60 for the same inputs."""
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        dense = [("a", 0.9), ("b", 0.8)]
        bm25 = [("a", 1.0), ("b", 0.5)]
        fused_k1 = compressor._rrf_fuse(dense, bm25, k=1, top_k=2)
        fused_k60 = compressor._rrf_fuse(dense, bm25, k=60, top_k=2)
        # k=1: score_a = 1/(1+1) + 1/(1+1) = 1.0; k=60: score_a = 1/61 + 1/61 ≈ 0.033
        assert fused_k1[0][1] > fused_k60[0][1], "k=1 should produce higher RRF scores than k=60"

    def test_rrf_formula_correctness(self, fake_embedder):
        """Verify exact RRF formula: RRF(d) = Σ 1/(k + rank)."""
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        # node "a": dense rank 1, bm25 rank 2
        # node "b": dense rank 2, bm25 rank 1
        dense = [("a", 0.9), ("b", 0.5)]
        bm25 = [("b", 1.5), ("a", 0.3)]
        k = 60
        fused = compressor._rrf_fuse(dense, bm25, k=k, top_k=2)
        score_a = 1 / (k + 1) + 1 / (k + 2)  # dense rank-1 + bm25 rank-2
        score_b = 1 / (k + 2) + 1 / (k + 1)  # dense rank-2 + bm25 rank-1
        result_map = {nid: s for nid, s in fused}
        assert (
            abs(result_map["a"] - score_a) < 1e-9
        ), f"RRF score for 'a': {result_map['a']} ≠ {score_a}"
        assert (
            abs(result_map["b"] - score_b) < 1e-9
        ), f"RRF score for 'b': {result_map['b']} ≠ {score_b}"
        # Both have same score when contributions are symmetric
        assert abs(score_a - score_b) < 1e-9

    def test_rrf_top_k_truncates(self, fake_embedder):
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]
        bm25 = [("a", 1.0), ("c", 0.8), ("b", 0.5)]
        fused = compressor._rrf_fuse(dense, bm25, k=60, top_k=2)
        assert len(fused) == 2

    def test_rrf_zero_division_safe(self, fake_embedder):
        """k must never be zero (would divide by zero at rank=0)."""
        compressor = _make_compressor_with_chunks({}, fake_embedder)
        # k=0 with start=1 → 1/(0+1) = 1.0, safe
        # But we test k=1 as the minimum practical value
        dense = [("a", 0.9)]
        bm25 = [("a", 1.0)]
        result = compressor._rrf_fuse(dense, bm25, k=1, top_k=1)
        assert len(result) == 1
        assert not math.isnan(result[0][1])
        assert not math.isinf(result[0][1])


# ---------------------------------------------------------------------------
# search_semantic_with_scores integration tests
# ---------------------------------------------------------------------------


class TestSearchSemanticWithScoresPathDispatch:
    """Tests for the ranker dispatch in search_semantic_with_scores."""

    def test_feature_flag_off_uses_dense_only(self, fake_embedder, monkeypatch):
        """F11_RANKER_PATH=a → dense-only, scores are cosine-like."""
        monkeypatch.setenv("F11_RANKER_PATH", "a")
        # Reload constant so monkeypatch takes effect
        import importlib
        import src.constants as _c

        importlib.reload(_c)

        chunks = {
            "doc_a::1": {"text": "python fastapi async endpoint route", "importance": 0.5},
            "doc_a::2": {"text": "database migration alembic postgresql", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)

        # Patch the module-level constant in semantic_compressor
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "a"
        try:
            results = compressor.search_semantic_with_scores("python async", top_k=2)
            assert len(results) == 2
            # Under Path A, scores are cosine similarity (floats, any range)
            for nid, score in results:
                assert isinstance(score, float)
        finally:
            _sc.F11_RANKER_PATH = orig

    def test_feature_flag_c_enables_hybrid(self, fake_embedder):
        """F11_RANKER_PATH=c → BM25+RRF hybrid path is taken."""
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        try:
            chunks = {
                "doc_a::1": {
                    "text": "reciprocal rank fusion bm25 hybrid retrieval",
                    "importance": 0.5,
                },
                "doc_a::2": {"text": "cosine similarity dense vector embedding", "importance": 0.5},
                "doc_a::3": {"text": "information retrieval ranking evaluation", "importance": 0.5},
            }
            compressor = _make_compressor_with_chunks(chunks, fake_embedder)
            results = compressor.search_semantic_with_scores("bm25 ranking", top_k=3)
            assert len(results) > 0
            # RRF scores are positive
            for nid, score in results:
                assert score > 0.0, f"RRF score for {nid} must be positive"
        finally:
            _sc.F11_RANKER_PATH = orig

    def test_rrf_scores_differ_from_dense_only(self, fake_embedder):
        """
        CANARY (EmbeddingManager singleton tier-lock smoke test, Phase 7c-4):

        If RRF scores are byte-identical to cosine scores across 3+ distinct
        query/chunk pairs, BM25 is silently not running. Mathematical impossibility
        if BM25 contributed to the fusion.

        Setup: inject a chunk whose text exactly matches the query term
        ("_rrf_fuse") but whose EMBEDDING is semantically distant (hash-based
        fake embedder guarantees distinctness by text content).

        Under dense-only: the exact-match chunk may rank poorly (low cosine).
        Under Path C: BM25 lifts the exact-match chunk via rank-1 BM25 hit.

        Assert: at least one of the 3 query pairs produces different top-1
        ranking between Path A and Path C.
        """
        import src.semantic_compressor as _sc

        # Build a corpus where BM25 and dense will disagree on at least one query
        chunks = {
            # This chunk has the exact query term but fake embedder may give low cosine
            "doc_a::exact": {"text": "_rrf_fuse scoring term exact match bm25", "importance": 0.5},
            # These have higher-similarity embeddings to other queries
            "doc_a::2": {"text": "embedding vector space cosine similarity", "importance": 0.5},
            "doc_a::3": {
                "text": "semantic search information retrieval ranking",
                "importance": 0.5,
            },
            "doc_a::4": {"text": "neural network language model attention", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)

        test_queries = [
            "_rrf_fuse",  # exact lexical match → BM25 lifts "exact"
            "bm25 match",  # partial match → BM25 contributes
            "exact term scoring",  # multi-term match
        ]

        orig_path = _sc.F11_RANKER_PATH
        try:
            # Collect Path A scores
            _sc.F11_RANKER_PATH = "a"
            dense_results = {}
            for q in test_queries:
                dense_results[q] = compressor.search_semantic_with_scores(q, top_k=4)

            # Collect Path C scores
            _sc.F11_RANKER_PATH = "c"
            rrf_results = {}
            for q in test_queries:
                rrf_results[q] = compressor.search_semantic_with_scores(q, top_k=4)

            # Check: at least ONE query should produce different top-1 node
            # (if all identical, BM25 is silently no-op'd — smoking gun)
            any_differ = False
            for q in test_queries:
                dense_top1 = dense_results[q][0][0] if dense_results[q] else None
                rrf_top1 = rrf_results[q][0][0] if rrf_results[q] else None
                if dense_top1 != rrf_top1:
                    any_differ = True
                    break
                # Also check scores differ (not just node order)
                dense_scores = [round(s, 10) for _, s in dense_results[q]]
                rrf_scores = [round(s, 10) for _, s in rrf_results[q]]
                if dense_scores != rrf_scores:
                    any_differ = True
                    break

            assert any_differ, (
                "CANARY FAILED: RRF scores are byte-identical to dense-only scores "
                "across all 3 query/chunk pairs. BM25 is silently not contributing to "
                "the hybrid fusion. Check F11_RANKER_PATH dispatch and _bm25_scores_for_nodes."
            )
        finally:
            _sc.F11_RANKER_PATH = orig_path

    def test_single_document_corpus_does_not_raise(self, fake_embedder):
        """Edge case: corpus with one node — BM25 avgdl = doc_len, no division issues."""
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        try:
            chunks = {"doc_a::1": {"text": "single document corpus", "importance": 0.5}}
            compressor = _make_compressor_with_chunks(chunks, fake_embedder)
            results = compressor.search_semantic_with_scores("single document", top_k=1)
            assert len(results) == 1
        finally:
            _sc.F11_RANKER_PATH = orig

    def test_empty_corpus_returns_empty(self, fake_embedder):
        """Empty corpus returns empty list without raising."""
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        try:
            compressor = _make_compressor_with_chunks({}, fake_embedder)
            results = compressor.search_semantic_with_scores("query", top_k=5)
            assert results == []
        finally:
            _sc.F11_RANKER_PATH = orig

    def test_file_id_filter_applied_under_path_c(self, fake_embedder):
        """file_id filter is respected under Path C (BM25 sees only filtered nodes)."""
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        try:
            chunks = {
                "doc_a::1": {"text": "python authentication fastapi endpoint", "importance": 0.5},
                "doc_b::1": {"text": "authentication JWT token validation", "importance": 0.5},
            }
            compressor = _make_compressor_with_chunks(chunks, fake_embedder)
            results = compressor.search_semantic_with_scores(
                "authentication", file_id="doc_a", top_k=5
            )
            result_ids = [nid for nid, _ in results]
            for nid in result_ids:
                assert nid.startswith("doc_a"), f"file_id filter leak: {nid} is not in doc_a"
        finally:
            _sc.F11_RANKER_PATH = orig

    def test_path_a_regression_lock_unaffected_under_path_c(self, fake_embedder):
        """
        Path A (chunking fix) is orthogonal to Path C (ranker fix).
        Run a structured-doc query under Path C and verify results are returned
        without error. (Exact threshold is Path A's concern; this test guards
        against Path C breaking the handler call chain.)
        """
        import src.semantic_compressor as _sc

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"
        try:
            # Minimal structured-doc fixture
            chunks = {
                "handoff::1": {
                    "text": "## Deployment\nFly.io deploy with flyctl",
                    "importance": 0.7,
                },
                "handoff::2": {
                    "text": "## Database\nAlembic migration upgrade head",
                    "importance": 0.6,
                },
                "handoff::3": {"text": "## Auth\nClerk JWT verification JWKS", "importance": 0.5},
            }
            compressor = _make_compressor_with_chunks(chunks, fake_embedder)
            results = compressor.search_semantic_with_scores("Fly deployment", top_k=3)
            assert len(results) > 0, "Path C must return results for structured-doc fixture"
        finally:
            _sc.F11_RANKER_PATH = orig


# ---------------------------------------------------------------------------
# Council patch P2 — score_type in handler response
# (tested via direct handler invocation with mock context)
# ---------------------------------------------------------------------------


class TestScoreTypeInHandlerResponse:
    """Council patch P2: score_type: 'rrf'|'cosine' in handle_search_semantic."""

    def _build_mock_context(self, fake_embedder, chunks_data):
        """Build a minimal HandlerContext-compatible dict."""
        compressor = _make_compressor_with_chunks(chunks_data, fake_embedder)

        # Mock the compressor's _generate_summary method
        compressor._generate_summary = lambda text, max_length=100: text[:max_length]

        # Mock access tracker (optional)
        compressor._access_tracker = None

        # Build context dict matching HandlerContext shape
        context = {"compressor": compressor}
        return context

    @pytest.mark.asyncio
    async def test_response_includes_score_type_cosine_path_a(self, fake_embedder):
        """Path A response must include score_type: 'cosine'."""
        import src.semantic_compressor as _sc
        from src.handlers.compression_handlers import handle_search_semantic

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "a"

        # Patch F11_RANKER_PATH in the handlers module too
        import src.handlers.compression_handlers as _ch

        orig_ch = _ch.F11_RANKER_PATH
        _ch.F11_RANKER_PATH = "a"

        try:
            chunks = {
                "doc_a::1": {"text": "python fastapi authentication endpoint", "importance": 0.5},
                "doc_a::2": {"text": "database migration alembic model", "importance": 0.5},
            }
            context = self._build_mock_context(fake_embedder, chunks)
            args = {"query": "authentication", "file_id": "doc_a", "top_k": 2}

            raw = await handle_search_semantic(context, args)
            response = json.loads(raw)

            assert "score_type" in response, "Top-level score_type missing"
            assert (
                response["score_type"] == "cosine"
            ), f"Expected 'cosine', got {response['score_type']!r}"
            for result in response["results"]:
                assert (
                    "score_type" in result
                ), f"Per-result score_type missing for {result['node_id']}"
                assert result["score_type"] == "cosine"
        finally:
            _sc.F11_RANKER_PATH = orig
            _ch.F11_RANKER_PATH = orig_ch

    @pytest.mark.asyncio
    async def test_response_includes_score_type_rrf_path_c(self, fake_embedder):
        """Path C response must include score_type: 'rrf'."""
        import src.semantic_compressor as _sc
        from src.handlers.compression_handlers import handle_search_semantic

        orig = _sc.F11_RANKER_PATH
        _sc.F11_RANKER_PATH = "c"

        import src.handlers.compression_handlers as _ch

        orig_ch = _ch.F11_RANKER_PATH
        _ch.F11_RANKER_PATH = "c"

        try:
            chunks = {
                "doc_a::1": {"text": "bm25 rrf hybrid retrieval ranking search", "importance": 0.5},
                "doc_a::2": {"text": "cosine similarity vector embedding space", "importance": 0.5},
            }
            context = self._build_mock_context(fake_embedder, chunks)
            args = {"query": "bm25 ranking", "file_id": "doc_a", "top_k": 2}

            raw = await handle_search_semantic(context, args)
            response = json.loads(raw)

            assert "score_type" in response, "Top-level score_type missing from Path C response"
            assert (
                response["score_type"] == "rrf"
            ), f"Expected 'rrf', got {response['score_type']!r}"
            for result in response["results"]:
                assert (
                    result.get("score_type") == "rrf"
                ), f"Per-result score_type should be 'rrf' for {result['node_id']}"
        finally:
            _sc.F11_RANKER_PATH = orig
            _ch.F11_RANKER_PATH = orig_ch
