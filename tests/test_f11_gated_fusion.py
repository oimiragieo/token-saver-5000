"""F11 Path G — gated fusion (design memo idea #1) — TDD unit tests.

EXPERIMENTAL ranker path behind a NEW ``F11_RANKER_PATH=g`` value. Path A
("a", default) and Path C ("c", HOLD) are UNTOUCHED — see
``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 2, idea #1
for the design this file locks.

Two layers of coverage, both model-free (no ONNX/SBERT load required):

1. The gate PREDICATE in isolation (``query_has_lexical_shape`` /
   ``bm25_top1_is_discriminative`` in ``src/bm25_utils.py``, and the
   combining dispatch ``_gate_should_fuse_g`` in ``src/semantic_compressor.py``)
   — table-driven, exact IDF math checked by hand where it matters.
2. The CORE CORRECTNESS CONTRACT at the ``SemanticCompressor`` level: Path G
   must equal Path A's output when the gate stays closed, and Path C's
   output when the gate opens — using the same ``_FakeEmbedder`` +
   ``object.__new__(SemanticCompressor)`` model-free pattern as
   ``tests/test_f11_path_c.py``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict

import numpy as np
import pytest

from src.bm25_utils import (
    _query_has_quoted_phrase,
    bm25_idf,
    bm25_scores,
    bm25_tokenize,
    bm25_top1_is_discriminative,
    query_has_lexical_shape,
)

# ===========================================================================
# Layer 1a: query_has_lexical_shape (gate predicate #1) — pure string predicate
# ===========================================================================


class TestQueryHasLexicalShape:
    def test_bare_digit_token_gates_open(self) -> None:
        assert query_has_lexical_shape("8000") is True

    def test_digit_embedded_in_word_gates_open(self) -> None:
        assert query_has_lexical_shape("port8080") is True

    def test_underscore_identifier_gates_open(self) -> None:
        assert query_has_lexical_shape("API_KEY_HMAC_SECRET") is True

    def test_camel_case_identifier_gates_open(self) -> None:
        assert query_has_lexical_shape("getUserById") is True

    def test_double_quoted_phrase_gates_open(self) -> None:
        assert query_has_lexical_shape('"release_command runs alembic"') is True

    def test_single_quoted_text_does_not_open_via_quoted_phrase_path(self) -> None:
        # Double-only policy (blocker-1 final fix): a single-quoted span is too
        # ambiguous with contractions to count as a quoted phrase. This query
        # has no digit/identifier/camelCase token either, so the WHOLE
        # predicate stays closed.
        assert query_has_lexical_shape("find the 'default value' parameter") is False

    def test_pure_natural_language_paraphrase_gates_closed(self) -> None:
        query = "how does it stop someone calling too often"
        assert query_has_lexical_shape(query) is False

    def test_plain_keyword_nl_without_identifiers_gates_closed(self) -> None:
        query = "what is the default rank fusion constant value"
        assert query_has_lexical_shape(query) is False

    def test_empty_query_gates_closed(self) -> None:
        assert query_has_lexical_shape("") is False


class TestBlocker1QuotedPhraseSignalIsDoubleQuoteOnly:
    """BLOCKER-1 FINAL LOCK (2026-07-08 codex re-gate, 3rd pass): the
    quoted-phrase signal is DOUBLE-QUOTE ONLY. Single quotes / apostrophes are
    inherently ambiguous with contractions & possessives -- codex found both a
    false-CLOSE (``'don't retry'`` -- balanced single-quoted phrase broken by
    the internal apostrophe) and a false-OPEN (``'don't retry`` -- open quote,
    no close) under any single-quote heuristic. We ELIMINATE the class: only a
    balanced pair of double quotes (straight or smart) wrapping >=1 word char
    counts. These tests assert the QUOTED-PHRASE predicate directly (not the
    full lexical-shape OR) so they isolate exactly this signal."""

    # --- Double quotes DO signal a phrase ---
    def test_straight_double_quoted_phrase_signals(self) -> None:
        assert _query_has_quoted_phrase('"release_command runs alembic"') is True

    def test_smart_double_quoted_phrase_signals(self) -> None:
        assert _query_has_quoted_phrase("“Retry-After header”") is True

    # --- Single quotes / apostrophes NEVER signal a phrase (the whole class) ---
    def test_balanced_single_quoted_phrase_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("look up the 'retry backoff' section") is False

    def test_single_quoted_phrase_with_internal_apostrophe_does_not_signal(self) -> None:
        # codex's false-CLOSE edge -- now uniformly False (never signals).
        assert _query_has_quoted_phrase("'don't retry'") is False

    def test_open_single_quote_no_close_does_not_signal(self) -> None:
        # codex's false-OPEN edge -- now uniformly False (never signals).
        assert _query_has_quoted_phrase("'don't retry") is False

    def test_whats_contraction_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("what's the rate limit behavior") is False

    def test_dont_contraction_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("why don't the retries fire") is False

    def test_its_contraction_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("where it's stored and how") is False

    def test_smart_apostrophe_contraction_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("what’s the cache eviction policy") is False

    def test_plural_possessive_apostrophe_does_not_signal(self) -> None:
        assert _query_has_quoted_phrase("how are the users' records purged") is False

    # --- Full-predicate consequences ---
    def test_whats_contraction_gates_closed_end_to_end(self) -> None:
        # The exact regression the blocker names -- no digit/identifier token
        # either, so the whole predicate stays closed.
        assert query_has_lexical_shape("what's the rate limit behavior") is False

    def test_double_quote_still_opens_full_predicate(self) -> None:
        assert query_has_lexical_shape('"release_command runs alembic"') is True

    def test_single_quote_around_identifier_still_opens_via_identifier_path(self) -> None:
        # Nothing legit is lost: a single-quoted IDENTIFIER still opens the gate
        # -- not via the quoted-phrase signal, but via the underscore/identifier
        # predicate. Assert BOTH facts to prove the routing.
        query = "look up 'API_KEY_HMAC_SECRET' handling"
        assert _query_has_quoted_phrase(query) is False
        assert query_has_lexical_shape(query) is True


# ===========================================================================
# Layer 1b: bm25_top1_is_discriminative (gate predicate #2) — exact IDF math
# ===========================================================================


class TestBm25Top1IsDiscriminative:
    """Six candidate "sections" -- deliberately mirrors design-memo section 1's
    "system"/"calling" false-signal example (common term, low IDF) against a
    rare, discriminative term (appears in exactly one section)."""

    _CANDIDATE_TEXTS = [
        "the system calling convention uses stack frames",  # section 0 (decoy, common terms)
        "the system calling thread pool handles requests",  # section 1 (decoy, common terms)
        "webhook retry backoff uses jitter and calling code",  # section 2
        "rate limiter enforces a token bucket per calling key",  # section 3
        "cache invalidation runs on ttl expiry for stale entries",  # section 4 (rare terms)
        "the system logs every calling attempt for audit",  # section 5 (decoy, common terms)
    ]

    def test_common_term_across_most_sections_is_not_discriminative(self) -> None:
        # "system" and "calling" each appear in >=4 of 6 sections -- exactly
        # the design memo's false-signal case. top1_score below the floor,
        # so only the IDF check is exercised.
        query = "how does it stop someone calling too often"
        top1_text = self._CANDIDATE_TEXTS[0]  # a decoy, not the true rate-limit answer
        result = bm25_top1_is_discriminative(
            query,
            top1_text,
            self._CANDIDATE_TEXTS,
            top1_score=0.3,
            idf_tau=0.8,
            score_floor=5.0,
        )
        assert result is False

        # Sanity-check the IDF math directly: "calling" appears in 5/6 docs.
        tokenized = [bm25_tokenize(t) for t in self._CANDIDATE_TEXTS]
        idf_calling = bm25_idf("calling", tokenized)
        assert idf_calling < 0.8, f"expected low IDF for a near-universal term, got {idf_calling}"

    def test_rare_discriminative_term_gates_open(self) -> None:
        # "invalidation" appears in exactly 1/6 sections -- high IDF.
        query = "cache invalidation strategy"
        top1_text = self._CANDIDATE_TEXTS[4]
        result = bm25_top1_is_discriminative(
            query,
            top1_text,
            self._CANDIDATE_TEXTS,
            top1_score=0.3,
            idf_tau=0.8,
            score_floor=5.0,
        )
        assert result is True

        tokenized = [bm25_tokenize(t) for t in self._CANDIDATE_TEXTS]
        idf_invalidation = bm25_idf("invalidation", tokenized)
        assert (
            idf_invalidation >= 0.8
        ), f"expected high IDF for a 1-of-6 term, got {idf_invalidation}"

    def test_score_floor_alone_opens_gate_even_without_idf_match(self) -> None:
        # No query term overlaps top1_text at all (IDF check can't fire),
        # but the raw BM25 score already clears the floor.
        result = bm25_top1_is_discriminative(
            "zzz_no_overlap_query",
            "totally unrelated text",
            self._CANDIDATE_TEXTS,
            top1_score=12.0,
            idf_tau=0.8,
            score_floor=5.0,
        )
        assert result is True

    def test_empty_query_terms_never_gate_open(self) -> None:
        result = bm25_top1_is_discriminative(
            "   ",
            self._CANDIDATE_TEXTS[0],
            self._CANDIDATE_TEXTS,
            top1_score=0.3,
            idf_tau=0.8,
            score_floor=5.0,
        )
        assert result is False

    def test_idf_formula_matches_bm25_idf_directly(self) -> None:
        """Cross-check: the gate's IDF check must use the SAME smoothed IDF
        formula as bm25_idf, not a hand-rolled approximation."""
        tokenized_docs = [bm25_tokenize(t) for t in self._CANDIDATE_TEXTS]
        n = len(tokenized_docs)
        df = sum(1 for doc in tokenized_docs if "invalidation" in doc)
        expected = math.log(1 + (n - df + 0.5) / (df + 0.5))
        assert bm25_idf("invalidation", tokenized_docs) == pytest.approx(expected)


class TestBlocker2GateUsesPrefixStemLikeBm25Scorer:
    """BLOCKER-2 LOCK (2026-07-08 codex review): the match-quality predicate
    must judge "did top-1 match a query term" with the SAME >=8-char prefix
    stemming the BM25 scorer uses. An exact-token check would MISS a real BM25
    hit (query ``authentication`` vs a doc's ``authenticate``) and wrongly CLOSE
    the gate on a valid lexical win."""

    # 6 sections; the rare, discriminative term "authenticate" appears in
    # exactly one section, so query "authentication" (>=8 chars, shares the
    # 8-char "authenti" prefix) is a genuine BM25 hit there.
    _CANDIDATE_TEXTS = [
        "the module handles authenticate flows for inbound requests",  # gold
        "billing runs through the merchant of record integration",
        "the cache stores entries with a ttl and hnsw index",
        "webhooks retry with an exponential backoff schedule",
        "rate limits protect the endpoint from abusive callers",
        "logging captures every request for later audit review",
    ]

    def test_prefix_stem_match_opens_gate_where_exact_membership_would_close(self) -> None:
        query = "authentication"
        gold_text = self._CANDIDATE_TEXTS[0]

        # Precondition: an EXACT-token check (the pre-fix behavior) WOULD close
        # the gate -- "authentication" is not literally a token in gold_text.
        assert "authentication" not in bm25_tokenize(gold_text)
        # And BM25 genuinely scores gold_text > 0 for this query (prefix stem).
        assert bm25_scores(query, [gold_text])[0] > 0.0

        # The fixed predicate must OPEN (top1_score below floor so only the
        # prefix-stem IDF check runs).
        assert (
            bm25_top1_is_discriminative(
                query,
                gold_text,
                self._CANDIDATE_TEXTS,
                top1_score=0.3,
                idf_tau=0.8,
                score_floor=5.0,
            )
            is True
        )

    def test_gate_predicate_matches_bm25_scorer_prefix_behavior(self) -> None:
        """The gate's "matched?" decision must agree with whether the BM25
        scorer actually scored the term > 0 on the top-1 doc -- for BOTH a
        prefix-stem hit and a genuine miss."""
        gold_text = self._CANDIDATE_TEXTS[0]

        # Prefix-stem hit: scorer > 0 AND gate opens.
        assert bm25_scores("authentication", [gold_text])[0] > 0.0
        assert bm25_top1_is_discriminative(
            "authentication",
            gold_text,
            self._CANDIDATE_TEXTS,
            top1_score=0.3,
            idf_tau=0.8,
            score_floor=5.0,
        )

        # Genuine miss: a rare term absent from gold_text -> scorer == 0 AND
        # gate stays closed (no phantom match).
        assert bm25_scores("kubernetes", [gold_text])[0] == 0.0
        assert not bm25_top1_is_discriminative(
            "kubernetes",
            gold_text,
            self._CANDIDATE_TEXTS,
            top1_score=0.3,
            idf_tau=0.8,
            score_floor=5.0,
        )


# ===========================================================================
# Layer 1c: _gate_should_fuse_g (the OR-combinator, module-level in
# semantic_compressor.py) -- model-free (no SemanticNode.embedding read).
# ===========================================================================


class _StubNode:
    """Minimal stand-in for SemanticNode -- the gate only reads .text."""

    def __init__(self, text: str) -> None:
        self.text = text


class TestGateShouldFuseGCombinator:
    def _import(self):
        from src.semantic_compressor import _gate_should_fuse_g

        return _gate_should_fuse_g

    def test_query_shape_alone_opens_gate_regardless_of_bm25(self) -> None:
        gate = self._import()
        candidate_nodes = [("n1", _StubNode("some unrelated text"))]
        # bm25_ranked deliberately empty/weak -- query-shape predicate must
        # short-circuit True without even needing bm25_ranked.
        assert gate("API_KEY_HMAC_SECRET", [], candidate_nodes) is True

    def test_no_lexical_shape_and_empty_bm25_gates_closed(self) -> None:
        gate = self._import()
        candidate_nodes = [("n1", _StubNode("some unrelated text"))]
        assert gate("how does it stop someone calling too often", [], candidate_nodes) is False

    def test_no_lexical_shape_but_discriminative_bm25_top1_opens_gate(self) -> None:
        gate = self._import()
        candidate_nodes = [
            ("n1", _StubNode("cache invalidation runs on ttl expiry for stale entries")),
            ("n2", _StubNode("the system calling convention uses stack frames")),
            ("n3", _StubNode("the system calling thread pool handles requests")),
            ("n4", _StubNode("webhook retry backoff uses jitter and calling code")),
            ("n5", _StubNode("rate limiter enforces a token bucket per calling key")),
            ("n6", _StubNode("the system logs every calling attempt for audit")),
        ]
        bm25_ranked = [("n1", 0.3)]  # weak score, but "invalidation" is rare
        assert gate("cache invalidation strategy", bm25_ranked, candidate_nodes) is True

    def test_no_lexical_shape_and_common_term_bm25_top1_gates_closed(self) -> None:
        gate = self._import()
        candidate_nodes = [
            ("n1", _StubNode("the system calling convention uses stack frames")),
            ("n2", _StubNode("the system calling thread pool handles requests")),
            ("n3", _StubNode("webhook retry backoff uses jitter and calling code")),
            ("n4", _StubNode("rate limiter enforces a token bucket per calling key")),
            ("n5", _StubNode("cache invalidation runs on ttl expiry for stale entries")),
            ("n6", _StubNode("the system logs every calling attempt for audit")),
        ]
        bm25_ranked = [("n1", 0.3)]
        assert (
            gate("how does it stop someone calling too often", bm25_ranked, candidate_nodes)
            is False
        )

    def test_documented_known_risk_digit_shaped_lexical_trap_still_opens_gate(self) -> None:
        """Design-memo-documented trade-off, NOT a bug: a lexical_trap query
        that happens to be digit-shaped (e.g. a bare port number that appears
        MORE often in a decoy section) still gates OPEN via predicate #1,
        because query-shape is evaluated independently of which candidate
        BM25 actually ranks first. This is why the memo calls lexical_trap
        "adversarial for BM25 -- the class that punishes an over-eager gate."
        The per-class harness measurement is what proves whether this
        trade-off costs a regression in practice."""
        gate = self._import()
        candidate_nodes = [
            ("decoy", _StubNode("the decoy section mentions 8000 8000 8000 repeatedly")),
            ("gold", _StubNode("the gold section mentions 8000 exactly once")),
        ]
        # BM25 top-1 is the WRONG (decoy) node -- gate still opens on shape.
        bm25_ranked = [("decoy", 9.0), ("gold", 1.0)]
        assert gate("8000", bm25_ranked, candidate_nodes) is True


# ===========================================================================
# Layer 2: core correctness contract at the SemanticCompressor level.
# Path G === Path A (gate closed) / Path G === Path C (gate open).
# Mirrors tests/test_f11_path_c.py's model-free _FakeEmbedder pattern.
# ===========================================================================


class _FakeEmbedder:
    """Deterministic hash-based embeddings -- reproducible without models."""

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


def _make_compressor_with_chunks(chunks: Dict[str, Dict[str, Any]], embedder: _FakeEmbedder):
    from src.semantic_compressor import SemanticCompressor, SemanticNode

    compressor = object.__new__(SemanticCompressor)
    compressor.model = embedder
    compressor.chunks = {}
    for node_id, data in chunks.items():
        node = object.__new__(SemanticNode)
        node.text = data["text"]
        node.importance = data.get("importance", 0.5)
        node.metadata = {"tokens": len(data["text"].split())}
        node.embedding = embedder.encode([data["text"]])[0]
        compressor.chunks[node_id] = node
    compressor.graphs = {}
    return compressor


_GATE_CLOSED_CHUNKS = {
    "doc_a_n1": {"text": "the system calling convention uses stack frames"},
    "doc_a_n2": {"text": "the system calling thread pool handles requests"},
    "doc_a_n3": {"text": "webhook retry backoff uses jitter and calling code"},
    "doc_a_n4": {"text": "rate limiter enforces a token bucket per calling key"},
    "doc_a_n5": {"text": "the system logs every calling attempt for audit"},
}
_GATE_CLOSED_QUERY = "how does it stop someone calling too often"

_GATE_OPEN_CHUNKS = {
    "doc_b_n1": {"text": "authentication resolves via a Clerk session JWT"},
    "doc_b_n2": {"text": "billing runs through Polar as merchant of record"},
    "doc_b_n3": {"text": "the semantic cache uses API_KEY_HMAC_SECRET for hashing"},
    "doc_b_n4": {"text": "webhooks retry with exponential backoff starting at 250 ms"},
    "doc_b_n5": {"text": "rate limits raise RateLimitExceededError with Retry-After"},
}
_GATE_OPEN_QUERY = "API_KEY_HMAC_SECRET"


def _set_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    import src.semantic_compressor as _sc

    monkeypatch.setattr(_sc, "F11_RANKER_PATH", path, raising=True)


class TestPathGMatchesPathAWhenGateClosed:
    def test_gate_closed_query_g_equals_a(self, fake_embedder, monkeypatch) -> None:
        compressor_a = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)
        compressor_g = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)

        _set_path(monkeypatch, "a")
        results_a = compressor_a.search_semantic_with_scores(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )

        _set_path(monkeypatch, "g")
        results_g = compressor_g.search_semantic_with_scores(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )

        assert results_g == results_a, (
            "Path G must degrade EXACTLY to Path A's dense-only ranking when "
            "the gate stays closed (pure-paraphrase / no lexical signal)"
        )


class TestPathGMatchesPathCWhenGateOpen:
    def test_gate_open_query_g_equals_c(self, fake_embedder, monkeypatch) -> None:
        compressor_a = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        compressor_c = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        compressor_g = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)

        _set_path(monkeypatch, "a")
        results_a = compressor_a.search_semantic_with_scores(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )

        _set_path(monkeypatch, "c")
        results_c = compressor_c.search_semantic_with_scores(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )

        _set_path(monkeypatch, "g")
        results_g = compressor_g.search_semantic_with_scores(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )

        assert results_g == results_c, (
            "Path G must fuse IDENTICALLY to Path C's BM25+RRF hybrid when "
            "the gate opens (identifier / numeric / quoted-phrase query)"
        )
        # Confirm this test actually exercises fusion -- RRF's rank-based
        # scores are numerically distinct from raw cosine similarity, so a
        # gate-always-closed bug (silently falling back to Path A for both
        # "c" and "g") would make results_g/_c equal results_a instead.
        assert results_g != results_a, (
            "Path G/C output is identical to Path A's dense-only ranking -- "
            "the gate-open / unconditional-fusion branch was not actually "
            "exercised (RRF fusion produces different scores than raw cosine)"
        )


class TestPathAAndPathCRemainUntouched:
    """Guard against the new dispatch branch accidentally changing Path A or
    Path C behavior for callers that never pass F11_RANKER_PATH=g."""

    def test_path_a_default_unaffected_by_new_dispatch(self, fake_embedder, monkeypatch) -> None:
        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "a")
        results = compressor.search_semantic_with_scores(_GATE_OPEN_QUERY, file_id="doc_b", top_k=5)
        # Path A is dense-only -- cosine similarity is bounded to [-1, 1].
        assert all(-1.0 - 1e-6 <= score <= 1.0 + 1e-6 for _, score in results)

    def test_path_c_unconditional_fusion_unaffected_by_new_dispatch(
        self, fake_embedder, monkeypatch
    ) -> None:
        compressor_a = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)
        compressor_c = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)

        _set_path(monkeypatch, "a")
        results_a = compressor_a.search_semantic_with_scores(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )

        _set_path(monkeypatch, "c")
        # Even a gate-closed-shaped query (paraphrase) must still fuse under
        # Path C -- Path C has no gate, by design (that's the HOLD bug).
        results_c = compressor_c.search_semantic_with_scores(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )
        assert results_c != results_a, (
            "Path C must remain unconditional fusion regardless of query shape "
            "-- if this now matches Path A exactly, the new Path G gate leaked "
            "into Path C's dispatch branch"
        )


class TestBlocker3ScoreTypeLabelReflectsWhatRan:
    """BLOCKER-3 LOCK (2026-07-08 codex review): the reported score_type must
    reflect what the ranker ACTUALLY ran. Under Path G that is PER-CALL --
    "rrf" when the gate fused, "cosine" when it fell back to dense. Path "a" /
    "c" labels are unchanged. Exercised via the typed ranker (the single source
    of truth the handler + retrieve_evidence read)."""

    def test_typed_ranker_labels_path_a_cosine(self, fake_embedder, monkeypatch) -> None:
        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "a")
        _results, score_type = compressor.search_semantic_with_scores_typed(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )
        assert score_type == "cosine"

    def test_typed_ranker_labels_path_c_rrf(self, fake_embedder, monkeypatch) -> None:
        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "c")
        _results, score_type = compressor.search_semantic_with_scores_typed(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )
        assert score_type == "rrf"

    def test_typed_ranker_labels_path_g_rrf_when_gate_fuses(
        self, fake_embedder, monkeypatch
    ) -> None:
        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "g")
        # _GATE_OPEN_QUERY = "API_KEY_HMAC_SECRET" -> identifier shape -> gate opens.
        results, score_type = compressor.search_semantic_with_scores_typed(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )
        assert score_type == "rrf", "gate fused, so label must be rrf"

        # And it must agree with what Path C (unconditional fusion) produces.
        _set_path(monkeypatch, "c")
        compressor_c = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        results_c = compressor_c.search_semantic_with_scores(
            _GATE_OPEN_QUERY, file_id="doc_b", top_k=5
        )
        assert results == results_c

    def test_typed_ranker_labels_path_g_cosine_when_gate_closes(
        self, fake_embedder, monkeypatch
    ) -> None:
        compressor = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "g")
        # _GATE_CLOSED_QUERY is a pure paraphrase -> gate closes -> dense-only.
        results, score_type = compressor.search_semantic_with_scores_typed(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )
        assert score_type == "cosine", "gate closed, so label must be cosine"

        # And it must equal Path A's dense-only output.
        _set_path(monkeypatch, "a")
        compressor_a = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)
        results_a = compressor_a.search_semantic_with_scores(
            _GATE_CLOSED_QUERY, file_id="doc_a", top_k=5
        )
        assert results == results_a

    def test_plain_wrapper_return_shape_unchanged(self, fake_embedder, monkeypatch) -> None:
        """The public search_semantic_with_scores must still return a bare list
        of (node_id, score) tuples (NOT the (list, str) typed shape)."""
        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        _set_path(monkeypatch, "g")
        results = compressor.search_semantic_with_scores(_GATE_OPEN_QUERY, file_id="doc_b", top_k=5)
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    @pytest.mark.asyncio
    async def test_handler_wire_score_type_rrf_when_path_g_gate_fuses(
        self, fake_embedder, monkeypatch
    ) -> None:
        """End-to-end at the handler: under Path G, a gate-opening query must
        make the wire response report score_type == 'rrf' (both top-level and
        per-result), not the hardcoded-cosine bug."""
        import src.semantic_compressor as _sc
        import src.handlers.compression_handlers as _ch
        from src.handlers.compression_handlers import handle_search_semantic

        monkeypatch.setattr(_sc, "F11_RANKER_PATH", "g", raising=True)
        monkeypatch.setattr(_ch, "F11_RANKER_PATH", "g", raising=True)

        compressor = _make_compressor_with_chunks(_GATE_OPEN_CHUNKS, fake_embedder)
        compressor._generate_summary = lambda text, max_length=100: text[:max_length]
        compressor._access_tracker = None
        context = {"compressor": compressor}
        args = {"query": _GATE_OPEN_QUERY, "file_id": "doc_b", "top_k": 3}

        raw = await handle_search_semantic(context, args)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        assert payload["score_type"] == "rrf"
        for result in payload["results"]:
            assert result["score_type"] == "rrf"

    @pytest.mark.asyncio
    async def test_handler_wire_score_type_cosine_when_path_g_gate_closes(
        self, fake_embedder, monkeypatch
    ) -> None:
        """End-to-end at the handler: under Path G, a pure-paraphrase query
        (gate closes -> dense-only) must report score_type == 'cosine'."""
        import src.semantic_compressor as _sc
        import src.handlers.compression_handlers as _ch
        from src.handlers.compression_handlers import handle_search_semantic

        monkeypatch.setattr(_sc, "F11_RANKER_PATH", "g", raising=True)
        monkeypatch.setattr(_ch, "F11_RANKER_PATH", "g", raising=True)

        compressor = _make_compressor_with_chunks(_GATE_CLOSED_CHUNKS, fake_embedder)
        compressor._generate_summary = lambda text, max_length=100: text[:max_length]
        compressor._access_tracker = None
        context = {"compressor": compressor}
        args = {"query": _GATE_CLOSED_QUERY, "file_id": "doc_a", "top_k": 3}

        raw = await handle_search_semantic(context, args)
        payload = json.loads(raw) if isinstance(raw, str) else raw
        assert payload["score_type"] == "cosine"
        for result in payload["results"]:
            assert result["score_type"] == "cosine"
