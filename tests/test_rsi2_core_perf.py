"""Round 1 RSI-2 performance & byte-identical / <1e-12 equivalence tests."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.cache_economics import (
    ANTHROPIC_SONNET_4_6,
    OPENAI_GPT5,
    ProviderCacheProfile,
    _SEGMENT_CACHE_MAX_SIZE,
    _SEGMENT_COUNT_CACHES,
    _get_encoder,
    bill_prefix_turn,
)
from src.semantic_compressor import SemanticCompressor, SemanticNode
from src.similarity import cosine_similarity
from src.token_estimation import (
    _ENCODER_COUNT_CACHES,
    _TOKEN_CACHE_MAX_SIZE,
    TokenEstimator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ref_sequential_gemini(text: str) -> int:
    if not text:
        return 0
    if len(text) > 100_000:
        return len(text) // 4
    tokens = 0.0
    for ch in text:
        tokens += 0.25 if ord(ch) <= 127 else 1.3
    return int(tokens)


def _ref_bill_prefix_turn(
    prev_segments: list[str],
    curr_segments: list[str],
    profile: ProviderCacheProfile,
) -> dict[str, float | int]:
    prefix_segments = 0
    for prev_seg, curr_seg in zip(prev_segments, curr_segments):
        if prev_seg == curr_seg:
            prefix_segments += 1
        else:
            break

    enc = _get_encoder()
    prev_toks: list[int] = []
    for seg in prev_segments:
        if seg:
            prev_toks.extend(enc.encode(seg))
    curr_toks: list[int] = []
    for seg in curr_segments:
        if seg:
            curr_toks.extend(enc.encode(seg))

    lcp_tokens = 0
    for p_tok, c_tok in zip(prev_toks, curr_toks):
        if p_tok == c_tok:
            lcp_tokens += 1
        else:
            break

    total_tokens = len(curr_toks)
    cached_tokens = lcp_tokens if lcp_tokens >= profile.persistent_tier_tokens else 0
    fresh_tokens = total_tokens - cached_tokens

    rate_per_tok = profile.input_rate_per_1m / 1_000_000.0
    cache_read_dollars = cached_tokens * rate_per_tok * profile.cache_read_mult
    cache_write_dollars = fresh_tokens * rate_per_tok * profile.cache_write_mult
    net_billed_dollars = cache_read_dollars + cache_write_dollars
    uncached_dollars = total_tokens * rate_per_tok
    dollars_saved_vs_uncached = uncached_dollars - net_billed_dollars

    return {
        "cached_prefix_segments": prefix_segments,
        "lcp_tokens": lcp_tokens,
        "cached_tokens": cached_tokens,
        "fresh_tokens": fresh_tokens,
        "total_tokens": total_tokens,
        "cache_read_dollars": round(cache_read_dollars, 8),
        "cache_write_dollars": round(cache_write_dollars, 8),
        "net_billed_dollars": round(net_billed_dollars, 8),
        "dollars_saved_vs_uncached": round(dollars_saved_vs_uncached, 8),
    }


def _make_node(node_id: str, embedding, text: str = "sample text") -> SemanticNode:
    return SemanticNode(
        node_id=node_id,
        text=text,
        embedding=np.asarray(embedding, dtype=np.float64),
        importance=0.5,
        metadata={"tokens": 2, "position": 0, "entities": []},
    )


def test_cold_boot_server_skips_heavy_ml_imports() -> None:
    """Importing worker-owned modules and src.server must not import sklearn/scipy/torch/ST."""
    script_modules = """
import sys
import src.afm  # noqa: F401
import src.similarity  # noqa: F401
import src.embeddings  # noqa: F401
import src.embeddings_tfidf  # noqa: F401
import src.adaptive_rate_allocator  # noqa: F401
import src.semantic_compressor_retrieval  # noqa: F401
import src.token_estimation  # noqa: F401
import src.cache_economics  # noqa: F401
import src.extractive_baseline  # noqa: F401
import src.response_formatter  # noqa: F401

heavy = ("sklearn", "scipy", "torch", "sentence_transformers")
loaded = [
    name
    for name in sys.modules
    if any(name == pkg or name.startswith(pkg + ".") for pkg in heavy)
]
assert not loaded, f"Unexpected eager heavy imports: {loaded}"
assert src.response_formatter._estimator_instance is None
print("CLEAN_MODULES_OK")
"""
    proc_mod = subprocess.run(
        [sys.executable, "-c", script_modules],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc_mod.returncode == 0, f"stderr: {proc_mod.stderr}"
    assert "CLEAN_MODULES_OK" in proc_mod.stdout

    script_server = """
import sys
import src.server  # noqa: F401

heavy = ("sklearn", "scipy", "torch", "sentence_transformers")
loaded = [
    name
    for name in sys.modules
    if any(name == pkg or name.startswith(pkg + ".") for pkg in heavy)
]
assert not loaded, f"Unexpected eager heavy imports in src.server: {loaded}"
print("CLEAN_SERVER_OK")
"""
    proc_srv = subprocess.run(
        [sys.executable, "-c", script_server],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=120,
    )
    assert proc_srv.returncode == 0, f"stderr: {proc_srv.stderr}"
    assert "CLEAN_SERVER_OK" in proc_srv.stdout


def test_similarity_matches_sklearn_within_1e12() -> None:
    """Dense pure-NumPy cosine_similarity matches sklearn < 1e-12 on all edge cases."""
    sklearn_pairwise = pytest.importorskip("sklearn.metrics.pairwise")
    sklearn_cos = sklearn_pairwise.cosine_similarity

    rng = np.random.default_rng(42)
    x_rand = rng.normal(size=(12, 32))
    y_rand = rng.normal(size=(7, 32))

    # Y is None self-similarity
    np.testing.assert_allclose(cosine_similarity(x_rand), sklearn_cos(x_rand), atol=1e-12, rtol=0.0)
    # Two 2D matrices
    np.testing.assert_allclose(
        cosine_similarity(x_rand, y_rand), sklearn_cos(x_rand, y_rand), atol=1e-12, rtol=0.0
    )

    # 1D, 0D, 3D, empty, non-finite, and mismatched feature dims raise ValueError (matching sklearn)
    v1 = rng.normal(size=(32,))
    for bad in (
        v1,
        np.array(1.0),
        np.ones((1, 2, 3)),
        np.empty((0, 4)),
        np.empty((2, 0)),
        np.array([[1.0, np.nan]]),
        np.array([[1.0, np.inf]]),
    ):
        with pytest.raises(ValueError):
            cosine_similarity(bad)
    with pytest.raises(ValueError):
        cosine_similarity(np.ones((2, 3)), np.ones((2, 4)))

    # Overflow-safe (1e308) and underflow-safe (1e-300) self-cosine == 1.0
    extreme = np.array([[1e308, 1e308], [1e-300, 1e-300]], dtype=np.float64)
    np.testing.assert_allclose(
        cosine_similarity(extreme), np.ones((2, 2), dtype=np.float64), atol=1e-12, rtol=0.0
    )

    # float32 output dtype preservation (matches sklearn)
    x_f32 = rng.normal(size=(4, 8)).astype(np.float32)
    y_f32 = rng.normal(size=(3, 8)).astype(np.float32)
    assert cosine_similarity(x_f32).dtype == np.float32
    assert cosine_similarity(x_f32, y_f32).dtype == np.float32
    assert cosine_similarity(x_f32, y_rand[:, :8]).dtype == np.float64

    # Zero-norm and sub-1e-12 tiny-norm vectors (Witness W1: must NOT clip at 1e-12!)
    edge = np.array(
        [
            [0.0, 0.0, 0.0],
            [1e-15, 2e-15, -2e-15],
            [1.0, 2.0, -2.0],
            [-3e-15, 4e-15, 0.0],
        ],
        dtype=np.float64,
    )
    got_edge = cosine_similarity(edge)
    ref_edge = sklearn_cos(edge)
    np.testing.assert_allclose(got_edge, ref_edge, atol=1e-12, rtol=0.0)
    # Specifically verify tiny non-zero vectors keep unit cosine with collinear scaled vectors
    assert got_edge[1, 2] == pytest.approx(1.0, abs=1e-12)
    assert got_edge[0, 1] == 0.0
    assert got_edge[0, 0] == 0.0

    # Even smaller non-zero vector (5e-16) stays exact unit-normalized without 1e-12 clip
    ultra_tiny = np.array([[-3e-16, 4e-16, 0.0], [-3.0, 4.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(
        cosine_similarity(ultra_tiny), np.ones((2, 2), dtype=np.float64), atol=1e-12, rtol=0.0
    )


def test_similarity_sparse_delegates_to_sklearn() -> None:
    """Scipy sparse matrices still work via lazy sklearn delegation."""
    scipy_sparse = pytest.importorskip("scipy.sparse")
    sklearn_pairwise = pytest.importorskip("sklearn.metrics.pairwise")

    mat = scipy_sparse.csr_matrix([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    got = cosine_similarity(mat)
    ref = sklearn_pairwise.cosine_similarity(mat)
    np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0.0)


def test_estimate_gemini_matches_sequential_reference() -> None:
    """estimate_gemini fast path is 100% exact vs sequential loop on ASCII & non-ASCII."""
    import random

    est = TokenEstimator()
    cases = [
        "",
        "a",
        "ab",
        "abc",
        "abcd",
        "hello world!",
        "A" * 99_999,
        "A" * 100_000,
        "A" * 100_001,
        "hi\u4e2d",
        "prefix \u00e9\u00e0\u00fc middle 123 \u65e5\u672c\u8a9e",
        "def f():  # em-dash \u2014 spinner \u280b\n    return 1\n",
    ]
    # Pure non-ASCII k=1..300 covering all Witness W7 IEEE-754 integer boundary counts (incl. k=30)
    cases.extend(["\u4e2d" * k for k in range(1, 301)])
    # Mixed ASCII + non-ASCII hitting exact/near-integer thresholds (e.g. 2 ASCII + 5 non-ASCII = 7.0)
    cases.extend(["a" * (k % 11) + "\u4e2d" * k for k in range(1, 120)])

    rng = random.Random(20261127)
    alphabet = ["a", " ", "\n", "x", "0", "\u2014", "\u280b", "\u4e2d", "\u00e9", "\U0001f680"]
    for _ in range(250):
        length = rng.randint(1, 400)
        s = "".join(rng.choice(alphabet) for _ in range(length))
        cases.append(s)

    for text in cases:
        assert est.estimate_gemini(text) == _ref_sequential_gemini(text), f"Mismatch for {text!r}"


def test_token_estimator_shared_encoder_and_bounded_cache(monkeypatch) -> None:
    """TokenEstimator uses digest-only per-encoder cache, honors monkeypatches, and is thread-safe."""
    import src.token_estimation as te

    est1 = TokenEstimator()
    est2 = TokenEstimator()
    if est1._encoder is not None:
        assert est1._encoder is est2._encoder

    # Monkeypatching _load_tiktoken_encoder must not return stale cached encoder
    fake_enc = MagicMock()
    fake_enc.encode.side_effect = lambda s: list(range(len(s) + 7))
    monkeypatch.setattr(te, "_load_tiktoken_encoder", lambda: fake_enc)

    est_patched = TokenEstimator()
    assert est_patched._encoder is fake_enc
    secret_text = "SECRET_TENANT_ALPHA_abc"
    assert est_patched.count_tokens(secret_text) == len(secret_text) + 7
    assert est_patched.count_tokens(secret_text) == len(secret_text) + 7
    # Cached for len <= 65536
    assert fake_enc.encode.call_count == 1

    # Cache stores ONLY (32-byte digest, int len) -> int; zero raw tenant text retained
    enc_cache = _ENCODER_COUNT_CACHES[fake_enc]
    assert len(enc_cache) == 1
    for (digest, length), count in enc_cache.items():
        assert isinstance(digest, bytes) and len(digest) == 32
        assert length == len(secret_text)
        assert isinstance(count, int)
        assert secret_text.encode("utf-8") not in digest

    # > 65536 chars bypasses the bounded cache so multi-MB strings are not pinned
    huge = "x" * 70_000
    assert est_patched.count_tokens(huge) == 70_007
    assert est_patched.count_tokens(huge) == 70_007
    assert fake_enc.encode.call_count == 3

    # Bounded to _TOKEN_CACHE_MAX_SIZE (256)
    assert _TOKEN_CACHE_MAX_SIZE == 256
    for i in range(300):
        est_patched.count_tokens(f"bounded_probe_{i}")
    assert len(_ENCODER_COUNT_CACHES[fake_enc]) <= _TOKEN_CACHE_MAX_SIZE

    # Concurrent calls + swapped encoder stay consistent
    fake_enc2 = MagicMock()
    fake_enc2.encode.side_effect = lambda s: list(range(len(s) + 100))
    monkeypatch.setattr(te, "_load_tiktoken_encoder", lambda: fake_enc2)
    est_swapped = TokenEstimator()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(est_swapped.count_tokens, ["concurrent_msg"] * 24))
    assert all(r == len("concurrent_msg") + 100 for r in results)


def test_bill_prefix_turn_matches_direct_concatenation_lcp() -> None:
    """bill_prefix_turn matches full concatenation LCP with digest-only segment count cache."""
    profiles = [
        ANTHROPIC_SONNET_4_6,
        OPENAI_GPT5,
        ProviderCacheProfile("low_tier", 2.0, 1.25, 0.1, persistent_tier_tokens=2),
    ]
    pairs = [
        (["ab", "", "cd"], ["ab", "cd"]),
        (["", "alpha beta", "", "gamma delta"], ["alpha beta", "gamma epsilon"]),
        (["alpha beta", "gamma delta", "tail1"], ["alpha beta", "gamma delta", "tail2"]),
        (["", ""], ["", ""]),
        ([], ["new segment"]),
        (["old segment"], []),
        (["hello world foo"], ["hello world bar"]),
    ]

    for profile in profiles:
        for prev, curr in pairs:
            assert bill_prefix_turn(prev, curr, profile) == _ref_bill_prefix_turn(
                prev, curr, profile
            )

    # Large segment > 65536 chars must bypass segment cache and still match reference
    assert _SEGMENT_CACHE_MAX_SIZE == 256
    enc = _get_encoder()
    _SEGMENT_COUNT_CACHES.pop(enc, None)
    huge_seg = "token " * 12_000  # 72_000 chars > 65_536
    secret_seg = "CONFIDENTIAL_PREFIX_SEGMENT_42"
    got = bill_prefix_turn([huge_seg, secret_seg, "a"], [huge_seg, secret_seg, "b"], OPENAI_GPT5)
    ref = _ref_bill_prefix_turn(
        [huge_seg, secret_seg, "a"], [huge_seg, secret_seg, "b"], OPENAI_GPT5
    )
    assert got == ref
    seg_cache = _SEGMENT_COUNT_CACHES.get(enc, {})
    assert len(seg_cache) <= 1
    for (digest, length), count in seg_cache.items():
        assert isinstance(digest, bytes) and len(digest) == 32
        assert length == len(secret_seg)
        assert isinstance(count, int)


def test_batched_retrieval_matches_per_node_reference_including_nan_and_ragged(monkeypatch) -> None:
    """_max_dense_cosine and search_semantic_with_scores_typed match per-node cosine + edge cases."""
    comp = SemanticCompressor()
    query_vec = np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float64)
    monkeypatch.setattr(comp.model, "encode", lambda texts: [query_vec])

    comp.chunks = {
        "doc_n0": _make_node("doc_n0", [1.0, 0.5, -0.25, 0.75]),
        "doc_n1": _make_node("doc_n1", [0.0, 1.0, 0.0, 0.0]),
        "doc_n2": _make_node("doc_n2", [-1.0, -0.5, 0.25, -0.75]),
        "doc_n3": _make_node("doc_n3", [0.0, 0.0, 0.0, 0.0]),
        "other_n0": _make_node("other_n0", [0.2, 0.4, 0.6, 0.8]),
    }

    # Reference per-node calculations
    doc_nodes = [(nid, node) for nid, node in comp.chunks.items() if nid.startswith("doc_")]
    ref_max = max(
        float(cosine_similarity([query_vec], [node.embedding])[0][0]) for _, node in doc_nodes
    )
    assert comp._max_dense_cosine("q", file_id="doc") == pytest.approx(ref_max, abs=1e-12)

    ranked, score_type = comp.search_semantic_with_scores_typed("q", file_id="doc", top_k=4)
    assert score_type == "cosine"
    ref_ranked = sorted(
        (
            (nid, float(cosine_similarity([query_vec], [node.embedding])[0][0]))
            for nid, node in doc_nodes
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    assert [nid for nid, _ in ranked] == [nid for nid, _ in ref_ranked]
    for (_, s_got), (_, s_ref) in zip(ranked, ref_ranked):
        assert s_got == pytest.approx(s_ref, abs=1e-12)

    # Partial NaN similarity: NaN must be ignored if a finite similarity exists
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.semantic_compressor_retrieval.cosine_similarity",
            lambda q, m: np.array([[np.nan, 0.42, -0.1, 0.0]], dtype=np.float64),
        )
        assert comp._max_dense_cosine("q", file_id="doc") == pytest.approx(0.42, abs=1e-12)

    # All-NaN similarities -> -1.0 without RuntimeWarning
    comp.chunks = {"nan_n0": _make_node("nan_n0", [np.nan, np.nan, np.nan, np.nan])}
    with monkeypatch.context() as mp:
        mp.setattr(
            "src.semantic_compressor_retrieval.cosine_similarity",
            lambda q, m: np.array([[np.nan]], dtype=np.float64),
        )
        assert comp._max_dense_cosine("q", file_id="nan") == -1.0

    # Ragged embedding shapes fallback cleanly to per-node loop
    comp.chunks = {
        "rag_n0": _make_node("rag_n0", [1.0, 0.0, 0.0, 0.0]),
        "rag_n1": _make_node("rag_n1", [0.0, 1.0, 0.0, 0.0]),
    }
    # Force one node embedding to a 2D shape so np.asarray raises inhomogeneous ValueError
    comp.chunks["rag_n1"].embedding = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    monkeypatch.setattr(
        "src.semantic_compressor_retrieval.cosine_similarity",
        lambda x, y: np.array([[0.8 if np.asarray(y).ndim == 2 else 0.2]]),
    )
    assert comp._max_dense_cosine("q", file_id="rag") == pytest.approx(0.8, abs=1e-12)
    rag_ranked, _ = comp.search_semantic_with_scores_typed("q", file_id="rag", top_k=2)
    assert len(rag_ranked) == 2


def test_chunked_retrieval_multiblock_exact_ties_and_nan_row(monkeypatch) -> None:
    """Chunked <=256-row retrieval preserves float32 parity, excludes NaNs, and clamps block size."""
    import src.semantic_compressor_retrieval as ret_mod

    comp = SemanticCompressor()
    rng = np.random.default_rng(7)
    query_vec = np.array([1.0, 0.25, -0.5, 0.5], dtype=np.float32)
    monkeypatch.setattr(comp.model, "encode", lambda texts: [query_vec])

    chunks: dict[str, SemanticNode] = {}
    tie_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    for i in range(600):
        emb = tie_vec.copy() if i in (10, 260, 520) else rng.normal(size=(4,)).astype(np.float32)
        if i in (0, 300):
            emb = np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)
        if i == 599:
            emb = query_vec.copy()  # Exact top-1 valid candidate after NaN rows
        chunks[f"big_n{i}"] = SemanticNode(
            node_id=f"big_n{i}",
            text=f"chunk {i}",
            embedding=emb,
            importance=0.5,
            metadata={"tokens": 2, "position": i, "entities": []},
        )
    comp.chunks = chunks

    # Override _SIMILARITY_BLOCK_SIZE > 256 and spy on batch sizes to prove hard <=256 clamp
    monkeypatch.setattr(ret_mod, "_SIMILARITY_BLOCK_SIZE", 1000)
    seen_batch_rows: list[int] = []
    real_cosine = ret_mod.cosine_similarity

    def _spy_cosine(x, y):
        seen_batch_rows.append(np.asarray(y).shape[0])
        return real_cosine(x, y)

    monkeypatch.setattr(ret_mod, "cosine_similarity", _spy_cosine)

    # Reference per-node float32 scores (excluding non-finite rows)
    ref_pairs = []
    for nid, node in chunks.items():
        if not np.isfinite(node.embedding).all():
            continue
        sim = float(cosine_similarity([query_vec], [node.embedding])[0][0])
        ref_pairs.append((nid, sim))
    ref_max = max(s for _, s in ref_pairs)
    assert comp._max_dense_cosine("q", file_id="big") == pytest.approx(ref_max, abs=1e-7)
    assert seen_batch_rows and max(seen_batch_rows) <= 256

    seen_batch_rows.clear()
    ref_pairs.sort(key=lambda x: x[1], reverse=True)
    got_ranked, score_type = comp.search_semantic_with_scores_typed("q", file_id="big", top_k=25)
    assert score_type == "cosine"
    assert seen_batch_rows and max(seen_batch_rows) <= 256
    assert all(np.isfinite(score) for _, score in got_ranked)
    assert "big_n0" not in [nid for nid, _ in got_ranked]
    assert "big_n300" not in [nid for nid, _ in got_ranked]
    assert got_ranked[0][0] == "big_n599"
    assert [nid for nid, _ in got_ranked] == [nid for nid, _ in ref_pairs[:25]]
    for (_, g_score), (_, r_score) in zip(got_ranked, ref_pairs[:25]):
        assert g_score == pytest.approx(r_score, abs=1e-7)

    # Mixed float32/float64 block with float32 query preserves exact per-node float32 precision
    mixed_nodes = [
        SemanticNode(
            node_id="m_f32",
            text="f32",
            embedding=np.array([0.3, 0.7, -0.2, 0.4], dtype=np.float32),
            importance=0.5,
            metadata={"tokens": 1, "position": 0, "entities": []},
        ),
        SemanticNode(
            node_id="m_f64",
            text="f64",
            embedding=np.array([0.3, 0.7, -0.2, 0.4], dtype=np.float64),
            importance=0.5,
            metadata={"tokens": 1, "position": 1, "entities": []},
        ),
        SemanticNode(
            node_id="m_nan",
            text="nan",
            embedding=np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32),
            importance=0.5,
            metadata={"tokens": 1, "position": 2, "entities": []},
        ),
    ]
    mixed_scores = comp._batched_dense_cosine(query_vec, mixed_nodes)
    assert mixed_scores[0] == float(
        cosine_similarity([query_vec], [mixed_nodes[0].embedding])[0][0]
    )
    assert mixed_scores[1] == float(
        cosine_similarity([query_vec], [mixed_nodes[1].embedding])[0][0]
    )
    assert np.isnan(mixed_scores[2])

    # Explicit [1, 1] float32 counterexample against sklearn's float32 intermediate rounding
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

    monkeypatch.setattr(ret_mod, "cosine_similarity", sk_cosine)
    q_ones = np.array([1.0, 1.0], dtype=np.float32)
    ones_nodes = [
        SemanticNode(
            node_id="ones_f32",
            text="f32",
            embedding=np.array([1.0, 1.0], dtype=np.float32),
            importance=0.5,
            metadata={"tokens": 1, "position": 0, "entities": []},
        ),
        SemanticNode(
            node_id="ones_f64",
            text="f64",
            embedding=np.array([1.0, 1.0], dtype=np.float64),
            importance=0.5,
            metadata={"tokens": 1, "position": 1, "entities": []},
        ),
    ]
    sk_mixed_scores = comp._batched_dense_cosine(q_ones, ones_nodes)
    assert sk_mixed_scores[0] == float(sk_cosine([q_ones], [ones_nodes[0].embedding])[0][0])
    assert sk_mixed_scores[1] == float(sk_cosine([q_ones], [ones_nodes[1].embedding])[0][0])


def test_afm_and_embeddings_broken_optional_package_fallback(monkeypatch) -> None:
    """find_spec non-None with broken runtime import falls back cleanly in afm and embeddings."""
    import builtins
    import src.afm as afm_mod
    import src.embeddings as emb_mod

    # 1. afm.FocusManager falls back to HashingEmbedder when sentence_transformers import breaks
    real_import = builtins.__import__

    def _broken_st_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise OSError("broken torch/sentence_transformers shared library")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(afm_mod, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.delitem(afm_mod.__dict__, "SentenceTransformer", raising=False)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", _broken_st_import)
    fm = afm_mod.FocusManager()
    assert isinstance(fm.embedder, afm_mod.HashingEmbedder)
    monkeypatch.setattr(builtins, "__import__", real_import)

    # 2. embeddings.EmbeddingManager handles broken sklearn without corrupting _tfidf_manager
    emb_mod.EmbeddingManager.reset_for_testing()
    try:
        mgr = emb_mod.EmbeddingManager(tier=emb_mod.EmbeddingTier.TFIDF, enable_cache=False)
        monkeypatch.setattr(emb_mod, "TFIDF_AVAILABLE", True)
        monkeypatch.setattr(emb_mod, "ONNX_AVAILABLE", False)
        monkeypatch.setattr(emb_mod, "SentenceTransformer", None)

        def _broken_tfidf():
            raise ImportError("simulated broken sklearn C extension")

        monkeypatch.setattr(emb_mod, "TFIDFEmbeddingManager", _broken_tfidf)
        with pytest.raises(RuntimeError, match="broken sklearn|All embedding tiers failed"):
            mgr.encode(["hello world"], tier=emb_mod.EmbeddingTier.TFIDF)
        assert mgr._tfidf_manager is None
    finally:
        emb_mod.EmbeddingManager.reset_for_testing()
