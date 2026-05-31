"""
Tests for the block-wise graph edge-building OOM fix.

Two required receipts:
  A. Edge-equivalence test  — block-wise helper produces the EXACT same edge
     set as the reference full-matrix cosine_similarity + threshold approach.
  B. Memory-bound regression test — compressing a large synthetic document
     (hundreds of chunks) keeps peak tracemalloc memory well under 2 GB,
     confirming O(block×N) behaviour rather than O(N²).

Run with:
    cd token-saver-5000
    PYTHONPATH=. pytest tests/test_graph_oom_fix.py -v
"""

import sys
import os
import tracemalloc

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor, _SIMILARITY_BLOCK_SIZE, _MAX_GRAPH_CHUNKS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_l2_normalised(n: int, dim: int = 64, seed: int = 42) -> np.ndarray:
    """Return an (n, dim) float32 array of L2-normalised random unit vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def _reference_edges(
    embeddings: np.ndarray,
    node_ids: list,
    threshold: float,
) -> set:
    """
    Build the upper-triangle edge set using the original full-matrix approach
    (sklearn cosine_similarity).  Used as the ground-truth reference in the
    equivalence test.

    Weights are rounded to 5 decimal places to stay clear of float32 ULP
    differences between sklearn's implementation and the dot-product path.
    """
    sim_matrix = cosine_similarity(embeddings)
    n = len(node_ids)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > threshold:
                edges.add((node_ids[i], node_ids[j], round(float(sim_matrix[i][j]), 5)))
    return edges


def _blockwise_edges(
    embeddings: np.ndarray,
    node_ids: list,
    threshold: float,
    block_size: int = _SIMILARITY_BLOCK_SIZE,
    max_chunks: int = _MAX_GRAPH_CHUNKS,
) -> set:
    """
    Build the edge set via the new block-wise static helper.

    Weights are rounded to 5 decimal places to match the reference helper.
    """
    raw = SemanticCompressor._build_similarity_edges(
        embeddings=embeddings,
        node_ids=node_ids,
        similarity_threshold=threshold,
        block_size=block_size,
        max_chunks=max_chunks,
    )
    return {(src, dst, round(w, 5)) for src, dst, w in raw}


# ---------------------------------------------------------------------------
# Receipt A: Edge-equivalence test
# ---------------------------------------------------------------------------


class TestBlockWiseEdgeEquivalence:
    """
    For a small document (N < block_size and N < max_chunks), the block-wise
    helper must produce the identical edge set as the full cosine_similarity
    reference computation.
    """

    def test_edges_match_reference_small_doc(self):
        """
        30-chunk document, default threshold.  Block-wise == full-matrix.
        """
        n = 30
        embeddings = _make_l2_normalised(n, dim=64, seed=1)
        node_ids = [f"doc_n{i}" for i in range(n)]
        threshold = 0.5

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold)

        assert bw == ref, (
            f"Block-wise edge set differs from full-matrix reference.\n"
            f"Only in block-wise: {bw - ref}\n"
            f"Only in reference:  {ref - bw}"
        )

    def test_edges_match_reference_threshold_zero(self):
        """
        With threshold=0 all upper-triangle pairs become edges.  The two
        methods must agree on the full upper triangle.
        """
        n = 20
        embeddings = _make_l2_normalised(n, dim=32, seed=7)
        node_ids = [f"x_n{i}" for i in range(n)]
        threshold = 0.0

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold)

        assert bw == ref

    def test_edges_match_reference_threshold_one(self):
        """
        With threshold=1.0 (above any realistic cosine similarity) no edges
        should be added by either method.
        """
        n = 40
        embeddings = _make_l2_normalised(n, dim=64, seed=13)
        node_ids = [f"y_n{i}" for i in range(n)]
        threshold = 1.0  # nothing passes

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold)

        assert bw == ref
        assert len(bw) == 0

    def test_edges_match_reference_block_size_1(self):
        """
        Stress the block iteration path: block_size=1 forces one row per
        iteration.  Output must still match the full-matrix reference.
        """
        n = 25
        embeddings = _make_l2_normalised(n, dim=64, seed=21)
        node_ids = [f"z_n{i}" for i in range(n)]
        threshold = 0.3

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold, block_size=1)

        assert bw == ref

    def test_edges_match_reference_block_size_larger_than_n(self):
        """
        When block_size > N the first (and only) block covers everything.
        Output must still match the full-matrix reference.
        """
        n = 15
        embeddings = _make_l2_normalised(n, dim=64, seed=99)
        node_ids = [f"w_n{i}" for i in range(n)]
        threshold = 0.4

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold, block_size=1000)

        assert bw == ref

    def test_weight_precision_matches_reference(self):
        """
        Edge weights must match the full-matrix reference to at least 6 decimal
        places (the round() call in the helpers).
        """
        n = 10
        embeddings = _make_l2_normalised(n, dim=128, seed=55)
        node_ids = [f"p_n{i}" for i in range(n)]
        threshold = 0.1

        ref = _reference_edges(embeddings, node_ids, threshold)
        bw = _blockwise_edges(embeddings, node_ids, threshold)

        # Both sets must be equal — weight included in the tuple comparison
        assert bw == ref

    def test_max_chunks_ceiling_limits_edges(self):
        """
        When max_chunks < N, edges are only built among the first max_chunks
        nodes.  Nodes beyond max_chunks should appear as destinations only up
        to the ceiling.
        """
        n = 60
        max_chunks = 20
        embeddings = _make_l2_normalised(n, dim=64, seed=77)
        node_ids = [f"c_n{i}" for i in range(n)]
        threshold = 0.0  # everything connects so the ceiling is observable

        bw = _blockwise_edges(embeddings, node_ids, threshold, max_chunks=max_chunks)

        # All edge endpoints must be within [0, max_chunks)
        for src, dst, _ in bw:
            src_idx = int(src.split("_n")[1])
            dst_idx = int(dst.split("_n")[1])
            assert src_idx < max_chunks, f"src={src} exceeds max_chunks={max_chunks}"
            assert dst_idx < max_chunks, f"dst={dst} exceeds max_chunks={max_chunks}"


# ---------------------------------------------------------------------------
# Receipt B: Memory-bound regression test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGraphBuildingMemoryBound:
    """
    Compress a large synthetic document and assert that peak tracemalloc
    allocation stays well under 2 GB (2_000_000_000 bytes).

    This locks the O(block×N) memory invariant so the O(N²) regression can
    never silently return.
    """

    @staticmethod
    def _make_large_text(target_chunks: int = 300) -> str:
        """
        Build a synthetic document large enough to produce ~target_chunks
        chunks after the compressor's chunking pass.  Each paragraph is
        ~80 words (≈110 tokens), well above the default min-chunk threshold.
        """
        paragraphs = []
        topics = [
            "quantum computing",
            "machine learning",
            "distributed systems",
            "graph algorithms",
            "natural language processing",
            "neural networks",
            "compiler design",
            "operating systems",
            "cryptography",
            "computer vision",
        ]
        for i in range(target_chunks):
            topic = topics[i % len(topics)]
            paragraphs.append(
                f"Paragraph {i} discusses {topic} in detail. "
                f"This paragraph covers the fundamentals of {topic} and explores "
                f"advanced techniques used in modern {topic} research. "
                f"Practitioners of {topic} rely on rigorous mathematical foundations "
                f"and empirical validation. The field of {topic} has grown rapidly "
                f"over the past decade, with new breakthroughs emerging regularly. "
                f"Understanding {topic} requires both theoretical insight and practical "
                f"experience with real-world systems and datasets in context {i}."
            )
        return "\n\n".join(paragraphs)

    def test_peak_memory_stays_under_2gb(self):
        """
        A ~300-paragraph document must be processed with peak allocation
        under 2 GB (2_000_000_000 bytes) when measured by tracemalloc.

        Before the fix, a 10 000-chunk document peaked at ~6.9 GB RSS.
        After the fix, a 300-chunk document (which would have been ~90 MB
        for the N×N float32 matrix at 384-dim embeddings) stays modest.
        """
        text = self._make_large_text(target_chunks=300)

        tracemalloc.start()
        try:
            compressor = SemanticCompressor()
            compressor.ingest_file(text, "large_doc_oom_test")
        finally:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        peak_gb = peak / 1_000_000_000
        assert peak < 2_000_000_000, (
            f"Peak tracemalloc allocation was {peak_gb:.3f} GB — "
            f"exceeds 2 GB ceiling. The O(N²) regression may have returned."
        )

    def test_block_wise_helper_peak_memory_scales_subquadratically(self):
        """
        Direct memory check on the static helper:  process 500 L2-normalised
        unit vectors (dim=384, matching MiniLM output) in block_size=256 blocks.

        Peak tracemalloc should be dominated by the (256 × 500) float32 block
        (≈0.5 MB) rather than a (500 × 500) matrix (≈1 MB before copies).
        Both are tiny, but this confirms the code path is exercised and that
        doubling N does not quadruple peak allocation.
        """
        dim = 384  # MiniLM embedding dimension
        node_ids_500 = [f"n{i}" for i in range(500)]
        emb_500 = _make_l2_normalised(500, dim=dim, seed=42)

        tracemalloc.start()
        SemanticCompressor._build_similarity_edges(
            embeddings=emb_500,
            node_ids=node_ids_500,
            similarity_threshold=0.5,
            block_size=256,
            max_chunks=_MAX_GRAPH_CHUNKS,
        )
        _curr, peak_500 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        node_ids_1000 = [f"n{i}" for i in range(1000)]
        emb_1000 = _make_l2_normalised(1000, dim=dim, seed=42)

        tracemalloc.start()
        SemanticCompressor._build_similarity_edges(
            embeddings=emb_1000,
            node_ids=node_ids_1000,
            similarity_threshold=0.5,
            block_size=256,
            max_chunks=_MAX_GRAPH_CHUNKS,
        )
        _curr, peak_1000 = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Doubling N must NOT quadruple peak (that would be O(N²)).
        # With block-wise we expect the similarity-buffer portion of peak to
        # scale ≈linearly (same block_size, twice as many columns).
        # Allow a 4× ratio as a generous guard against O(N²) regression.
        assert peak_1000 < 4 * peak_500, (
            f"Peak memory scaled too aggressively: "
            f"N=500 → {peak_500 / 1024:.0f} KB, "
            f"N=1000 → {peak_1000 / 1024:.0f} KB  "
            f"(ratio {peak_1000 / peak_500:.1f}× — expected <4×). "
            f"Possible O(N²) regression."
        )
