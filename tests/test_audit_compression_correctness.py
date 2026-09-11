"""
Enterprise audit — compression-engine correctness regression locks.

Eight findings remediated under STRICT TDD (failing test first):

P1-3  retrieve_evidence() compared RRF scores (≈0.033 max under Path C, k=60,
      2 rankers) against a 0.35 cosine threshold → sufficient permanently False.
      Lock: test_rrf_sufficient_reachable.
P1-4  MIG re-ranking mutated shared SemanticNode.importance IN PLACE → PageRank
      corruption across queries on the long-lived singleton compressor.
      Lock: test_mig_does_not_mutate_node_importance / _fresh_importance_per_query.
P1-5  startswith(file_id) prefix collision: file_id 'foo' matched 'foobar_n0'.
      Lock: test_file_id_prefix_isolation (skeleton + retrieval + stats + diff).
P1-6  TFIDF silent fallback returned garbage vectors when SBERT+ONNX both failed.
      Lock: test_encode_with_fallback_raises_when_sbert_and_onnx_fail.
P1-7  EmbeddingManager singleton locked tier to first caller (process poisoning).
      Lock: test_reset_for_testing_clears_singleton / _conflicting_tier_warns.
P2-3  score_type mislabeled 'cosine' when evidence_aware under Path C.
      Lock: test_score_type_rrf_under_path_c_even_when_evidence_aware.
P2-4  get_stats() triggered full skeleton generation (+ MIG mutation) as a side
      effect.  Lock: test_get_stats_does_not_invoke_skeleton_generation.
P2-5  BM25 6-char prefix match false positives ('python' matched 'pythonic').
      Lock: test_bm25_python_does_not_match_pythonic.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
import pytest

from tests.f11_ranker_path_helpers import restore_f11_ranker_path, set_f11_ranker_path

# ---------------------------------------------------------------------------
# Deterministic fake embedder (mirrors test_f11_path_c._FakeEmbedder)
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Hash-based deterministic embeddings — reproducible without models."""

    def encode(self, texts: list[str]) -> np.ndarray:
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


def _make_compressor_with_chunks(
    chunks: Dict[str, Dict[str, Any]],
    embedder: _FakeEmbedder,
) -> Any:
    """Minimal SemanticCompressor backed by the fake embedder (no live models)."""
    from src.semantic_compressor import SemanticCompressor, SemanticNode

    compressor = object.__new__(SemanticCompressor)
    compressor.model = embedder
    compressor.chunks = {}
    for node_id, data in chunks.items():
        node = object.__new__(SemanticNode)
        node.text = data["text"]
        node.importance = data.get("importance", 0.5)
        node.metadata = {"tokens": len(data["text"].split()), "entities": []}
        node.embedding = embedder.encode([data["text"]])[0]
        compressor.chunks[node_id] = node
    compressor.graphs = {}
    compressor._access_tracker = None
    return compressor


# ===========================================================================
# P1-3 — RRF sufficiency threshold reachable under Path C
# ===========================================================================


class TestRrfSufficiencyReachable:
    """Under Path C, retrieve_evidence().sufficient must be reachable.

    Max single-ranker RRF contribution is 1/(k+1) = 1/61 ≈ 0.0164; with two
    rankers both ranking a node #1 the ceiling is ≈ 0.0328 — far below the 0.35
    cosine threshold, so the old code returned sufficient=False forever.
    """

    def test_rrf_sufficient_reachable(self, fake_embedder):
        chunks = {
            # Strong lexical + dense match for the query → top RRF rank.
            "doc_a_n0": {"text": "python fastapi async endpoint routing", "importance": 0.5},
            "doc_a_n1": {"text": "database migration alembic postgresql", "importance": 0.5},
            "doc_a_n2": {"text": "redis cache invalidation strategy ttl", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)

        orig = set_f11_ranker_path("c")
        try:
            evidence = compressor.retrieve_evidence(
                query="python fastapi async endpoint routing",
                file_id="doc_a",
                top_k=3,
            )
            # The cited bug: RRF top score (~0.03) < 0.35 cosine → always False.
            assert evidence.sufficient is True, (
                "Under Path C a high-relevance query must be able to report "
                f"sufficient=True; got best_score={evidence.best_score!r} "
                f"threshold={evidence.threshold!r}"
            )
        finally:
            restore_f11_ranker_path(orig)

    def test_rrf_irrelevant_query_reports_insufficient(self, fake_embedder):
        """LOAD-BEARING (audit re-fix): under Path C, a genuinely-irrelevant /
        low-cosine query MUST report sufficient=False.

        The original P1-3 fix gated sufficiency on the RRF *fusion* score, which
        encodes rank POSITION not relevance MAGNITUDE: the rank-1 node of ANY
        non-empty doc has RRF >= 1/(60+1) ≈ 0.01639, so 0.01639 >= 0.015 was
        unconditionally True — `sufficient` was True for EVERY query, including
        gibberish. The corrected gate must threshold on the DENSE COSINE
        similarity of the top-ranked node against the cosine bar (min_similarity).
        """
        chunks = {
            "doc_a_n0": {"text": "python fastapi async endpoint routing", "importance": 0.5},
            "doc_a_n1": {"text": "database migration alembic postgresql", "importance": 0.5},
            "doc_a_n2": {"text": "redis cache invalidation strategy ttl", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)

        orig = set_f11_ranker_path("c")
        try:
            evidence = compressor.retrieve_evidence(
                query="zzzqqq gibberish nonsense xylophone quasar wibble",
                file_id="doc_a",
                top_k=3,
                # A near-impossible cosine bar: no hash-based fake-embedding pair
                # reaches it, so cosine-gated sufficiency MUST be False.
                min_similarity=0.999,
            )
            assert evidence.sufficient is False, (
                "Under Path C, an irrelevant low-cosine query must report "
                f"sufficient=False; got best_score={evidence.best_score!r} "
                f"threshold={evidence.threshold!r}. The sufficiency gate is "
                "thresholding on the RRF fusion score (rank position), not the "
                "dense cosine magnitude."
            )
        finally:
            restore_f11_ranker_path(orig)

    def test_cosine_threshold_unchanged_under_path_a(self, fake_embedder):
        """Path A must keep the cosine min_similarity semantics (no regression)."""
        chunks = {
            "doc_a_n0": {"text": "alpha beta gamma", "importance": 0.5},
            "doc_a_n1": {"text": "delta epsilon zeta", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)

        orig = set_f11_ranker_path("a")
        try:
            evidence = compressor.retrieve_evidence(
                query="completely unrelated xylophone quasar",
                file_id="doc_a",
                top_k=2,
                min_similarity=0.99,  # near-impossible cosine bar
            )
            # Cosine path: a near-impossible threshold still gates correctly.
            assert evidence.sufficient is False
            assert evidence.threshold == 0.99
        finally:
            restore_f11_ranker_path(orig)


# ===========================================================================
# P1-4 — MIG re-ranking must not mutate shared node.importance
# ===========================================================================


class TestMigDoesNotMutateImportance:
    def test_fresh_importance_per_query(self):
        """PageRank importance must be identical before and after a MIG query."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        doc = (
            "# Authentication\n\n"
            "The login flow validates the bearer token against the JWKS endpoint. "
            "Sessions are issued as signed JWTs with a short TTL.\n\n"
            "# Billing\n\n"
            "Polar is the merchant of record. Checkout and webhooks are Svix-signed. "
            "Overage metering runs nightly via a cron sweep.\n\n"
            "# Storage\n\n"
            "Postgres with pgvector holds the semantic cache entries and HNSW index."
        )
        file_id = "p1_4_doc"
        compressor.ingest_file(doc, file_id)

        before = {nid: n.importance for nid, n in compressor.chunks.items()}

        # Run a MIG-strategy query — this used to overwrite node.importance.
        compressor.read_skeleton(file_id, query="authentication token", selection_strategy="mig")

        after = {nid: n.importance for nid, n in compressor.chunks.items()}

        assert after == before, (
            "MIG re-ranking corrupted shared node.importance in place. "
            "Compute a query-local importance_override instead of writing node.importance."
        )

    def test_second_query_not_poisoned_by_first(self):
        """A second MIG query must compute from PageRank, not first query's MIG scores."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        doc = (
            "# Authentication\n\n"
            "Login validates the bearer token against the JWKS endpoint.\n\n"
            "# Billing\n\n"
            "Polar checkout and Svix-signed webhooks drive plan changes.\n\n"
            "# Storage\n\n"
            "Postgres with pgvector holds the semantic cache entries."
        )
        file_id = "p1_4_doc_b"
        compressor.ingest_file(doc, file_id)

        baseline = {nid: n.importance for nid, n in compressor.chunks.items()}

        compressor.read_skeleton(file_id, query="authentication token", selection_strategy="mig")
        compressor.read_skeleton(file_id, query="billing webhooks", selection_strategy="mig")

        after = {nid: n.importance for nid, n in compressor.chunks.items()}
        assert after == baseline, "Sequential MIG queries poisoned node.importance."


# ===========================================================================
# P1-5 — file_id prefix-collision isolation
# ===========================================================================


class TestFileIdPrefixIsolation:
    def test_file_id_prefix_isolation(self):
        """Doc 'foo' must never surface 'foobar' nodes in skeleton/search/stats."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        foo_text = (
            "Foo service handles authentication tokens and session lifetimes. "
            "It exposes a login endpoint and a refresh endpoint for clients."
        )
        foobar_text = (
            "Foobar analytics pipeline aggregates billing events and emits "
            "Prometheus counters for the dashboard. Completely unrelated to foo."
        )
        compressor.ingest_file(foo_text, "foo")
        compressor.ingest_file(foobar_text, "foobar")

        # 1. Skeleton for 'foo' must contain zero 'foobar' nodes.
        skeleton = compressor.read_skeleton("foo")
        assert "foobar_n" not in skeleton, "foo skeleton leaked foobar_* nodes"

        # 2. Semantic search scoped to 'foo' must not return any 'foobar' node.
        results = compressor.search_semantic_with_scores(
            "billing analytics dashboard", file_id="foo", top_k=10
        )
        leaked = [nid for nid, _ in results if nid.startswith("foobar")]
        assert leaked == [], f"foo-scoped search leaked foobar nodes: {leaked}"

        # 3. get_stats('foo') node count must not include foobar nodes.
        stats = compressor.get_stats("foo")
        foobar_node_count = sum(1 for nid in compressor.chunks if nid.startswith("foobar"))
        assert foobar_node_count > 0  # sanity: foobar really has nodes
        assert stats["total_nodes"] == sum(
            1 for nid in compressor.chunks if nid.startswith("foo_n")
        ), "get_stats('foo') counted foobar_* nodes due to prefix collision"

    def test_diff_reingest_does_not_touch_sibling_prefix(self):
        """diff_reingest of 'foo' must not preserve/restore 'foobar' chunks."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        compressor.ingest_file("foo alpha content one two three four five", "foo")
        compressor.ingest_file("foobar beta content six seven eight nine ten", "foobar")

        foobar_nodes_before = {nid for nid in compressor.chunks if nid.startswith("foobar_n")}
        # _compute_diff_stats must scope strictly to 'foo'.
        diff_stats = compressor._compute_diff_stats("foo", "foo alpha content one two three")
        # preserved keys are chunk TEXTS — none should belong to foobar.
        foobar_texts = {compressor.chunks[nid].text for nid in foobar_nodes_before}
        assert not (
            set(diff_stats["preserved"].keys()) & foobar_texts
        ), "diff stats for 'foo' preserved 'foobar' chunk text (prefix collision)"

    @pytest.mark.asyncio
    async def test_prune_by_relevance_handler_excludes_prefix_sibling(self, monkeypatch):
        """handle_prune_by_relevance must scope nodes by boundary-safe file_id match,
        not bare startswith (audit P1-5 straggler). Model-free: fake chunks + a patched
        embedder so this never touches the SBERT/ONNX model cache."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        from src.handlers import compression_handlers as ch

        compressor = Mock()
        compressor.graphs = {"report": object(), "report_archive": object()}
        compressor.chunks = {
            "report_n0": SimpleNamespace(embedding=np.array([1.0, 0.0], dtype=np.float32)),
            "report_n1": SimpleNamespace(embedding=np.array([0.0, 1.0], dtype=np.float32)),
            "report_archive_n0": SimpleNamespace(embedding=np.array([1.0, 1.0], dtype=np.float32)),
        }
        fake_mgr = Mock()
        fake_mgr.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        monkeypatch.setattr("src.embeddings.EmbeddingManager", lambda *a, **k: fake_mgr)

        result = json.loads(
            await ch.handle_prune_by_relevance(
                {"compressor": compressor},
                {"doc_id": "report", "query": "auth", "keep_ratio": 0.5},
            )
        )
        assert (
            result["total_nodes"] == 2
        ), "prune_by_relevance counted report_archive_* nodes via bare startswith"
        leaked = [nid for nid in result["kept_node_ids"] if nid.startswith("report_archive")]
        assert leaked == [], f"prune leaked prefix-sibling nodes: {leaked}"

    @pytest.mark.asyncio
    async def test_multi_level_skeleton_handler_excludes_prefix_sibling(self):
        """handle_multi_level_skeleton must scope nodes by boundary-safe file_id match.
        Model-free: fake chunks, no embedder involved."""
        from types import SimpleNamespace
        from unittest.mock import Mock

        from src.handlers import compression_handlers as ch

        compressor = Mock()
        compressor.graphs = {"report": object(), "report_archive": object()}
        compressor.chunks = {
            "report_n0": SimpleNamespace(text="auth tokens login refresh", importance=0.9),
            "report_archive_n0": SimpleNamespace(
                text="ZZZSENTINEL billing cold storage aggregation", importance=0.9
            ),
        }

        result = json.loads(
            await ch.handle_multi_level_skeleton({"compressor": compressor}, {"doc_id": "report"})
        )
        # The sibling's unique sentinel text must never appear in 'report's skeleton.
        assert "ZZZSENTINEL" not in json.dumps(
            result["levels"]
        ), "multi_level_skeleton leaked report_archive_* nodes via bare startswith"


# ===========================================================================
# P1-6 — TFIDF silent fallback must raise when SBERT+ONNX fail
# ===========================================================================


class TestTfidfSilentFallbackRaises:
    def test_encode_with_fallback_raises_when_sbert_and_onnx_fail(self, monkeypatch):
        """When a non-TFIDF tier is requested and SBERT+ONNX both fail, the
        fallback must RAISE RuntimeError, not silently return TFIDF garbage."""
        from src import embeddings as emb_mod
        from src.embeddings import EmbeddingManager, EmbeddingTier

        EmbeddingManager.reset_for_testing()
        mgr = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)

        # Force SBERT (STANDARD) and ONNX to both fail.
        boom = RuntimeError("model load failed")

        def _fail_standard(texts, normalize):
            raise boom

        def _fail_onnx(texts, normalize):
            raise RuntimeError("onnx unavailable")

        monkeypatch.setattr(mgr, "_encode_standard", _fail_standard)
        monkeypatch.setattr(mgr, "_encode_onnx", _fail_onnx)
        monkeypatch.setattr(emb_mod, "ONNX_AVAILABLE", True)
        monkeypatch.setattr(emb_mod, "TFIDF_AVAILABLE", True)

        with pytest.raises(RuntimeError):
            # Requested tier ONNX (non-TFIDF). STANDARD + ONNX fail → must raise,
            # NOT fall through to TF-IDF garbage vectors.
            mgr._encode_with_fallback(["some text to embed"], EmbeddingTier.ONNX, normalize=True)
        EmbeddingManager.reset_for_testing()

    def test_explicit_tfidf_tier_still_allowed(self, monkeypatch):
        """A caller who explicitly requests TFIDF tier still gets TF-IDF vectors."""
        from src.embeddings import EmbeddingManager, EmbeddingTier

        EmbeddingManager.reset_for_testing()
        mgr = EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
        try:
            vecs = mgr.encode(["hello world", "foo bar baz"], tier=EmbeddingTier.TFIDF)
            assert vecs.shape[0] == 2
        finally:
            EmbeddingManager.reset_for_testing()


# ===========================================================================
# P1-7 — EmbeddingManager singleton tier poisoning
# ===========================================================================


class TestEmbeddingManagerSingletonTier:
    def test_reset_for_testing_clears_singleton(self):
        from src.embeddings import EmbeddingManager, EmbeddingTier

        EmbeddingManager.reset_for_testing()
        first = EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
        assert EmbeddingManager._instance is first
        EmbeddingManager.reset_for_testing()
        assert EmbeddingManager._instance is None
        # A new construction after reset is a brand-new instance.
        second = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
        assert second is not first
        EmbeddingManager.reset_for_testing()

    def test_conflicting_tier_second_construction_warns(self, caplog):
        """Constructing with a CONFLICTING tier than the locked one must surface
        a CRITICAL warning (the Phase 7c-4 silent-poisoning war story)."""
        import logging

        from src.embeddings import EmbeddingManager, EmbeddingTier

        EmbeddingManager.reset_for_testing()
        EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
        with caplog.at_level(logging.CRITICAL, logger="src.embeddings"):
            EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
        critical = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        assert critical, (
            "Second construction with a conflicting tier must log CRITICAL "
            "(process-wide tier poisoning is silent otherwise)."
        )
        EmbeddingManager.reset_for_testing()

    def test_same_tier_second_construction_silent(self, caplog):
        """Re-constructing with the SAME tier is fine — no CRITICAL noise."""
        import logging

        from src.embeddings import EmbeddingManager, EmbeddingTier

        EmbeddingManager.reset_for_testing()
        EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
        with caplog.at_level(logging.CRITICAL, logger="src.embeddings"):
            EmbeddingManager(tier=EmbeddingTier.TFIDF, enable_cache=False)
        assert not [r for r in caplog.records if r.levelno >= logging.CRITICAL]
        EmbeddingManager.reset_for_testing()


# ===========================================================================
# P2-3 — score_type label decoupled from evidence_aware under Path C
# ===========================================================================


class TestScoreTypeUnderPathC:
    @pytest.mark.asyncio
    async def test_score_type_rrf_under_path_c_even_when_evidence_aware(self, fake_embedder):
        from src.handlers.compression_handlers import handle_search_semantic

        chunks = {
            "doc_a_n0": {"text": "bm25 rrf hybrid retrieval ranking search", "importance": 0.5},
            "doc_a_n1": {"text": "cosine similarity vector embedding space", "importance": 0.5},
        }
        compressor = _make_compressor_with_chunks(chunks, fake_embedder)
        compressor._generate_summary = lambda text, max_length=100: text[:max_length]
        context = {"compressor": compressor}

        orig = set_f11_ranker_path("c")
        try:
            args = {
                "query": "bm25 ranking",
                "file_id": "doc_a",
                "top_k": 2,
                "evidence_aware": True,  # the cited mislabel trigger
            }
            response = json.loads(await handle_search_semantic(context, args))
            assert response["score_type"] == "rrf", (
                "Under Path C the scores ARE RRF even when evidence_aware=True; "
                f"score_type must be 'rrf', got {response['score_type']!r}"
            )
            for result in response["results"]:
                assert result["score_type"] == "rrf"
        finally:
            restore_f11_ranker_path(orig)


# ===========================================================================
# P2-4 — get_stats() must not trigger skeleton generation
# ===========================================================================


class TestGetStatsNoSkeletonSideEffect:
    def test_get_stats_does_not_invoke_skeleton_generation(self):
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        compressor.ingest_file(
            "# Title\n\nAlpha section content.\n\n# Two\n\nBeta section content here.",
            "p2_4_doc",
        )

        calls = {"n": 0}
        original = compressor._generate_skeleton

        def _spy(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        compressor._generate_skeleton = _spy
        stats = compressor.get_stats("p2_4_doc")

        assert calls["n"] == 0, (
            "get_stats() must serve from ingested/cached data — it invoked "
            f"_generate_skeleton {calls['n']} time(s)."
        )
        # Contract preservation: stats still carry the documented fields.
        for key in (
            "file_id",
            "total_nodes",
            "total_edges",
            "total_tokens",
            "skeleton_tokens",
            "compression_ratio",
        ):
            assert key in stats, f"get_stats lost the '{key}' field"

    def test_get_stats_does_not_mutate_importance(self):
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        compressor.ingest_file(
            "# Title\n\nAlpha content paragraph.\n\n# Two\n\nBeta content paragraph.",
            "p2_4_doc_b",
        )
        before = {nid: n.importance for nid, n in compressor.chunks.items()}
        compressor.get_stats("p2_4_doc_b")
        after = {nid: n.importance for nid, n in compressor.chunks.items()}
        assert after == before, "get_stats() mutated node.importance as a side effect"

    def test_get_stats_cold_baseline_cache_reports_real_ratio(self):
        """LOAD-BEARING (audit P2-4 edge case): when the baseline caches are cold
        but the graph + nodes exist (e.g. document restored from persistence
        without re-ingest), get_stats() must report a REAL compression ratio
        derived from the graph nodes — NOT a misleading flat 1.0 — and must STILL
        NOT invoke _generate_skeleton (the side-effect-free P2-4 property).
        """
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        # Multi-section doc with LARGE bodies → multiple graph nodes so the
        # baseline skeleton is a strict subset (a tiny doc collapses to one
        # chunk → no compression headroom → ratio legitimately ~1.0).
        sections = []
        for name in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"):
            body = " ".join(f"{name.lower()}word{i}" for i in range(120))
            sections.append(f"# {name}\n\n{body}")
        compressor.ingest_file("\n\n".join(sections), "p2_4_cold_doc")

        # Simulate a persistence restore that rebuilt the graph + chunks but did
        # NOT repopulate the baseline skeleton caches.
        compressor._baseline_skeleton_stats.pop("p2_4_cold_doc", None)
        compressor._baseline_skeleton_cache.pop("p2_4_cold_doc", None)

        calls = {"n": 0}
        original = compressor._generate_skeleton

        def _spy(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        compressor._generate_skeleton = _spy
        stats = compressor.get_stats("p2_4_cold_doc")

        assert calls["n"] == 0, (
            "get_stats() must remain side-effect-free even on a cold baseline "
            f"cache — it invoked _generate_skeleton {calls['n']} time(s)."
        )
        assert stats["total_tokens"] > 0
        assert stats["skeleton_tokens"] > 0
        assert stats["compression_ratio"] > 1.0, (
            "Cold-baseline get_stats() must compute a real >1.0 compression ratio "
            "from the graph nodes, not the misleading flat 1.0 fallback; got "
            f"compression_ratio={stats['compression_ratio']!r}, "
            f"skeleton_tokens={stats['skeleton_tokens']!r}, "
            f"total_tokens={stats['total_tokens']!r}"
        )


# ===========================================================================
# P2-5 — BM25 prefix match must not produce 'python'→'pythonic' false positives
# ===========================================================================


class TestBm25PrefixMatch:
    def test_bm25_python_does_not_match_pythonic(self):
        from src.bm25_utils import bm25_term_freq_with_stemming

        # 'python' must NOT count 'pythonic' as a hit under the tightened rule.
        freq = bm25_term_freq_with_stemming("python", ["pythonic", "guidelines", "apply"])
        assert freq == 0, "'python' falsely matched 'pythonic' (prefix collision)"

    def test_bm25_exact_match_still_counts(self):
        from src.bm25_utils import bm25_term_freq_with_stemming

        freq = bm25_term_freq_with_stemming("python", ["python", "is", "python"])
        assert freq == 2, "exact-token match regressed"

    def test_bm25_genuine_long_stem_still_matches(self):
        from src.bm25_utils import bm25_term_freq_with_stemming

        # A genuine long morphological variant must still stem-match.
        freq = bm25_term_freq_with_stemming("authentication", ["authenticating", "the", "request"])
        assert freq >= 1, "legitimate long-term stemming regressed"
