"""
Output-equivalence + speedup receipts for the vectorized MMR skeleton selector
(roadmap A2).

The query-guided branch of ``SemanticCompressor._select_skeleton_nodes`` used to
run a per-pair ``sklearn.cosine_similarity([cand], [sel])`` Python loop — O(N²)
cosine calls with one tiny BLAS dispatch each. The vectorized rewrite L2-normalises
the node-embedding matrix ONCE and computes:

  * relevance = E_norm @ q_norm          (one matmul instead of N cosine calls)
  * redundancy via a running max-sim vector updated by a single
    ``E_norm @ E_selected`` column per pick.

OUTPUT-EQUIVALENCE GATE (load-bearing): the vectorized selector MUST pick the SAME
node ids in the SAME order as the original loop on a battery of diverse fixtures.
A reference re-implementation of the ORIGINAL greedy loop lives here as the ground
truth; ``_select_skeleton_nodes`` (the production method) is compared against it.

Run with:
    cd token-saver-5000
    PYTHONPATH=. pytest tests/test_mmr_vectorization_equivalence.py -v
"""

import os
import sys
import time

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor, SemanticNode

# ---------------------------------------------------------------------------
# Reference: the ORIGINAL per-pair cosine_similarity greedy loop.
# Copied verbatim (logic-wise) from the pre-A2 implementation so the test pins
# the exact selection sequence the production method must reproduce.
# ---------------------------------------------------------------------------


def _reference_select_skeleton_nodes(
    compressor: SemanticCompressor,
    file_nodes,
    num_skeleton,
    query=None,
    redundancy_penalty=0.2,
    priority_scores=None,
    importance_override=None,
):
    """Pre-A2 reference: per-pair sklearn cosine_similarity, greedy MMR."""
    if num_skeleton <= 0 or not file_nodes:
        return set(), []

    override = importance_override or {}

    def _imp(node_id, node):
        return override.get(node_id, node.importance)

    if not query or not query.strip():
        ranked = sorted(file_nodes, key=lambda item: _imp(item[0], item[1]), reverse=True)
        chosen = [node_id for node_id, _ in ranked[:num_skeleton]]
        return set(chosen), chosen

    query_embedding = compressor.model.encode([query])[0]
    importance_scores = {node_id: _imp(node_id, node) for node_id, node in file_nodes}
    relevance_scores = {
        node_id: float(cosine_similarity([query_embedding], [node.embedding])[0][0])
        for node_id, node in file_nodes
    }

    importance_norm = compressor._normalize_scores(importance_scores)
    relevance_norm = compressor._normalize_scores(relevance_scores)
    priority_norm = (
        compressor._normalize_scores(priority_scores)
        if priority_scores
        else {node_id: 0.0 for node_id, _ in file_nodes}
    )

    hybrid_scores = {
        node_id: 0.25 * importance_norm.get(node_id, 0.0)
        + 0.55 * relevance_norm.get(node_id, 0.0)
        + 0.20 * priority_norm.get(node_id, 0.0)
        for node_id, _ in file_nodes
    }

    selected = []
    selected_set = set()
    candidate_ids = [node_id for node_id, _ in file_nodes]
    node_lookup = {node_id: node for node_id, node in file_nodes}

    while len(selected) < num_skeleton and candidate_ids:
        best_id = None
        best_score = float("-inf")
        for candidate_id in candidate_ids:
            candidate_score = hybrid_scores[candidate_id]
            if selected:
                max_similarity = max(
                    float(
                        cosine_similarity(
                            [node_lookup[candidate_id].embedding],
                            [node_lookup[selected_id].embedding],
                        )[0][0]
                    )
                    for selected_id in selected
                )
                candidate_score -= redundancy_penalty * max_similarity
            if candidate_score > best_score:
                best_score = candidate_score
                best_id = candidate_id
        if best_id is None:
            break
        selected.append(best_id)
        selected_set.add(best_id)
        candidate_ids.remove(best_id)

    return selected_set, selected


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


class _FakeModel:
    """Deterministic encoder: maps a query string to a fixed unit vector."""

    def __init__(self, query_vec):
        self._query_vec = np.asarray(query_vec, dtype=np.float32)

    def encode(self, texts, **kwargs):
        return np.array([self._query_vec for _ in texts], dtype=np.float32)


def _make_nodes(embeddings, importances, prefix="n"):
    nodes = []
    for i, (emb, imp) in enumerate(zip(embeddings, importances)):
        nid = f"{prefix}{i}"
        nodes.append(
            (
                nid,
                SemanticNode(
                    node_id=nid,
                    text=f"node {i}",
                    embedding=np.asarray(emb, dtype=np.float32),
                    importance=float(imp),
                    metadata={"tokens": 5, "entities": [], "position": i},
                ),
            )
        )
    return nodes


def _random_unit(n, dim, seed):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return raw / norms


def _build_compressor(query_vec):
    c = SemanticCompressor.__new__(SemanticCompressor)
    c.model = _FakeModel(query_vec)
    return c


# A battery of ≥10 diverse fixtures: varied node counts, dims, query/no-query,
# MMR lambda (redundancy_penalty), seeds.
_FIXTURES = [
    # (name, n, dim, seed, num_skeleton, redundancy_penalty, with_query, with_priority)
    ("tiny_q", 5, 8, 1, 3, 0.2, True, False),
    ("small_q", 12, 16, 2, 5, 0.2, True, False),
    ("medium_q", 40, 32, 3, 12, 0.2, True, False),
    ("large_q", 120, 64, 4, 30, 0.2, True, False),
    ("lambda_low", 30, 32, 5, 10, 0.05, True, False),
    ("lambda_high", 30, 32, 6, 10, 0.8, True, False),
    ("lambda_zero", 25, 16, 7, 8, 0.0, True, False),
    ("priority", 35, 32, 8, 10, 0.3, True, True),
    ("no_query", 30, 16, 9, 10, 0.2, False, False),
    ("all_selected", 8, 16, 10, 8, 0.2, True, False),
    ("more_than_nodes", 6, 16, 11, 20, 0.2, True, False),
    ("wide_dim", 50, 384, 12, 15, 0.2, True, False),
]


def _params(fix):
    name, n, dim, seed, num_skel, pen, with_q, with_prio = fix
    embeddings = _random_unit(n, dim, seed)
    rng = np.random.default_rng(seed + 1000)
    importances = rng.uniform(0.0, 1.0, size=n)
    nodes = _make_nodes(embeddings, importances, prefix=f"{name}_")
    query = "find the relevant content" if with_q else None
    query_vec = _random_unit(1, dim, seed + 2000)[0]
    priority = None
    if with_prio:
        priority = {nid: float(rng.uniform(0.0, 1.0)) for nid, _ in nodes}
    return name, nodes, num_skel, query, pen, priority, query_vec


# ---------------------------------------------------------------------------
# Receipt A: selection-sequence equivalence on the battery of fixtures
# ---------------------------------------------------------------------------


class TestMMRSelectionEquivalence:
    @pytest.mark.parametrize("fix", _FIXTURES, ids=[f[0] for f in _FIXTURES])
    def test_selection_sequence_matches_reference(self, fix):
        name, nodes, num_skel, query, pen, priority, query_vec = _params(fix)
        compressor = _build_compressor(query_vec)

        ref_set, ref_order = _reference_select_skeleton_nodes(
            compressor,
            nodes,
            num_skel,
            query=query,
            redundancy_penalty=pen,
            priority_scores=priority,
        )

        # Production method returns a set; assert SET equivalence (the public
        # contract) AND, via the ordered helper below, sequence equivalence.
        prod_set = compressor._select_skeleton_nodes(
            nodes,
            num_skel,
            query=query,
            redundancy_penalty=pen,
            priority_scores=priority,
        )

        assert prod_set == ref_set, (
            f"[{name}] selected SET differs.\n"
            f"  only in prod: {prod_set - ref_set}\n"
            f"  only in ref:  {ref_set - prod_set}"
        )

        # Sequence equivalence: the ordered production path (private helper)
        # must produce the identical pick order as the reference greedy loop.
        prod_order = compressor._select_skeleton_nodes_ordered(
            nodes,
            num_skel,
            query=query,
            redundancy_penalty=pen,
            priority_scores=priority,
        )
        assert (
            prod_order == ref_order
        ), f"[{name}] selected ORDER differs.\n  ref:  {ref_order}\n  prod: {prod_order}"

    def test_empty_and_degenerate_inputs(self):
        compressor = _build_compressor(_random_unit(1, 8, 1)[0])
        assert compressor._select_skeleton_nodes([], 5, query="x") == set()
        nodes = _make_nodes(_random_unit(3, 8, 2), [0.1, 0.2, 0.3])
        assert compressor._select_skeleton_nodes(nodes, 0, query="x") == set()


# ---------------------------------------------------------------------------
# Receipt B: speedup lock — vectorized path ≥5× faster on a 200-node fixture
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestMMRSpeedup:
    def test_vectorized_is_at_least_5x_faster_on_200_nodes(self):
        n, dim = 200, 384
        embeddings = _random_unit(n, dim, seed=99)
        rng = np.random.default_rng(123)
        importances = rng.uniform(0.0, 1.0, size=n)
        nodes = _make_nodes(embeddings, importances, prefix="speed_")
        query_vec = _random_unit(1, dim, seed=999)[0]
        compressor = _build_compressor(query_vec)
        num_skel = 50
        query = "performance benchmark query"

        # Warm up (model.encode, BLAS) so timing reflects the algorithm.
        _ = compressor._select_skeleton_nodes(nodes, num_skel, query=query)
        _ref_set, _ref_order = _reference_select_skeleton_nodes(
            compressor, nodes, num_skel, query=query
        )

        reps = 3
        t0 = time.perf_counter()
        for _ in range(reps):
            _reference_select_skeleton_nodes(compressor, nodes, num_skel, query=query)
        ref_time = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            compressor._select_skeleton_nodes(nodes, num_skel, query=query)
        vec_time = (time.perf_counter() - t0) / reps

        speedup = ref_time / vec_time if vec_time > 0 else float("inf")
        assert speedup >= 5.0, (
            f"Vectorized MMR only {speedup:.1f}× faster (ref={ref_time * 1000:.1f}ms, "
            f"vec={vec_time * 1000:.1f}ms) — expected ≥5×."
        )
