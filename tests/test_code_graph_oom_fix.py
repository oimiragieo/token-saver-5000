"""Block-wise code-graph edge-building OOM fix (task #236 rank11).

Mirrors tests/test_graph_oom_fix.py (the #30 fix for SemanticCompressor) but
targets CodeSemanticCompressor._build_similarity_edges.

Before this fix, CodeSemanticCompressor.ingest_code_file() built the code
similarity graph via a full sklearn cosine_similarity(embeddings) call
(materialising an N x N matrix) + an uncapped
`for i in range(N): for j in range(i+1, N)` double loop over ALL chunks --
O(N^2) memory and time with no ceiling.

Two receipts:
  A. Edge-equivalence -- the block-wise helper produces the EXACT same edge
     set as the reference full-matrix cosine_similarity approach, INCLUDING
     when embeddings are NOT pre-normalised (a code-embedding model such as
     microsoft/codebert-base is not guaranteed to emit unit-norm vectors the
     way many sentence-transformers text models do -- the helper must
     L2-normalise internally to stay behaviour-preserving).
  B. Memory-bound -- the static helper's peak allocation does not scale
     quadratically with N.

Model-free: _build_similarity_edges is a @staticmethod over numpy arrays only
(no model load / HF cache), per token-saver-5000/CLAUDE.md model-free testing.

    cd token-saver-5000 && PYTHONPATH=. pytest tests/test_code_graph_oom_fix.py -v
"""

import tracemalloc

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# On UNFIXED code these symbols do not exist -> ImportError at collection,
# which is this module's "fails on current code" receipt.
from src.code_compressor import (
    CodeSemanticCompressor,
    _MAX_GRAPH_CHUNKS,
    _SIMILARITY_BLOCK_SIZE,
)


def _emb(n, dim=768, seed=42, normalised=True):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    if not normalised:
        return raw
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def _reference_edges(embeddings, chunk_ids, threshold):
    """Ground truth: the ORIGINAL full-matrix approach, weights rounded to
    5dp to stay clear of float32 ULP noise between the two code paths."""
    sim = cosine_similarity(embeddings)
    n = len(chunk_ids)
    return {
        (chunk_ids[i], chunk_ids[j], round(float(sim[i][j]), 5))
        for i in range(n)
        for j in range(i + 1, n)
        if sim[i][j] > threshold
    }


def _blockwise_edges(
    embeddings,
    chunk_ids,
    threshold,
    block_size=_SIMILARITY_BLOCK_SIZE,
    max_chunks=_MAX_GRAPH_CHUNKS,
):
    raw = CodeSemanticCompressor._build_similarity_edges(
        embeddings=embeddings,
        chunk_ids=chunk_ids,
        similarity_threshold=threshold,
        block_size=block_size,
        max_chunks=max_chunks,
    )
    return {(src, dst, round(w, 5)) for src, dst, w in raw}


class TestCodeBlockWiseEdgeEquivalence:
    def test_edges_match_reference_normalised(self):
        n = 30
        e = _emb(n, seed=1, normalised=True)
        ids = [f"file.py::chunk_{i}" for i in range(n)]
        ref = _reference_edges(e, ids, 0.70)
        assert _blockwise_edges(e, ids, 0.70) == ref

    def test_edges_match_reference_NON_normalised(self):
        # The CodeBERT case: raw (non-unit-norm) embeddings. A bare dot
        # product would diverge here; the helper must normalise internally.
        n = 25
        e = _emb(n, seed=2, normalised=False)
        assert not np.allclose(np.linalg.norm(e, axis=1), 1.0)
        ids = [f"file.py::chunk_{i}" for i in range(n)]
        ref = _reference_edges(e, ids, 0.10)
        assert _blockwise_edges(e, ids, 0.10) == ref

    def test_edges_match_reference_threshold_zero(self):
        n = 20
        e = _emb(n, dim=64, seed=7)
        ids = [f"x::n{i}" for i in range(n)]
        assert _blockwise_edges(e, ids, 0.0) == _reference_edges(e, ids, 0.0)

    def test_edges_match_reference_block_size_1(self):
        n = 25
        e = _emb(n, dim=64, seed=21)
        ids = [f"z::n{i}" for i in range(n)]
        assert _blockwise_edges(e, ids, 0.3, block_size=1) == _reference_edges(e, ids, 0.3)

    def test_max_chunks_ceiling_limits_edges(self):
        # max_chunks < N (the pathological-file case): only the first
        # max_chunks chunks participate in dense edges.
        n, cap = 60, 20
        e = _emb(n, dim=64, seed=77)
        ids = [f"c::n{i}" for i in range(n)]
        bw = _blockwise_edges(e, ids, 0.0, max_chunks=cap)
        for src, dst, _w in bw:
            assert int(src.split("::n")[1]) < cap
            assert int(dst.split("::n")[1]) < cap

    def test_single_chunk_returns_no_edges(self):
        e = _emb(1, dim=64, seed=5)
        assert (
            CodeSemanticCompressor._build_similarity_edges(
                embeddings=e, chunk_ids=["only::n0"], similarity_threshold=0.5
            )
            == []
        )


class TestCodeGraphBuildingMemoryBound:
    def test_peak_memory_scales_subquadratically(self):
        dim = 768
        tracemalloc.start()
        CodeSemanticCompressor._build_similarity_edges(
            embeddings=_emb(500, dim=dim),
            chunk_ids=[f"n{i}" for i in range(500)],
            similarity_threshold=0.5,
            block_size=256,
            max_chunks=_MAX_GRAPH_CHUNKS,
        )
        _c, peak_500 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        CodeSemanticCompressor._build_similarity_edges(
            embeddings=_emb(1000, dim=dim),
            chunk_ids=[f"n{i}" for i in range(1000)],
            similarity_threshold=0.5,
            block_size=256,
            max_chunks=_MAX_GRAPH_CHUNKS,
        )
        _c, peak_1000 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Doubling N must NOT quadruple peak (O(N^2)). Generous 4x guard.
        assert peak_1000 < 4 * peak_500, (
            f"N=500->{peak_500 // 1024}KB N=1000->{peak_1000 // 1024}KB "
            f"ratio {peak_1000 / peak_500:.1f}x (expected <4x) -- possible O(N^2) regression"
        )

    def test_blockwise_below_full_matrix(self):
        dim, n = 768, 1200
        e = _emb(n, dim=dim, seed=99)
        tracemalloc.start()
        cosine_similarity(e)
        _c, peak_full = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        CodeSemanticCompressor._build_similarity_edges(
            embeddings=e,
            chunk_ids=[f"n{i}" for i in range(n)],
            similarity_threshold=0.5,
            block_size=256,
            max_chunks=_MAX_GRAPH_CHUNKS,
        )
        _c, peak_bw = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak_bw < peak_full
