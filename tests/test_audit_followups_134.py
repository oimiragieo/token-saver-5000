"""Audit follow-up #134 regression locks (2026-06-24).

Three lower-severity engine findings from the 2026-06-24 audit:
1. find_duplicates used rsplit('_', 1) instead of the canonical
   extract_file_id_from_node, so two functions in the SAME code file
   (file::symbol with underscores in the symbol) were compared as if
   cross-file -> false-positive duplicates.
2. _max_dense_cosine encoded the query with no finiteness guard -> a NaN/Inf
   query embedding propagated NaN into the sufficiency gate.
3. _select_skeleton_nodes_ordered had a zero-norm guard but NOT a finiteness
   guard; a NaN norm is truthy, so q_unit (and the whole relevance vector)
   went NaN, silently corrupting MMR ranking.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor, SemanticNode


def _node(node_id, vec, importance=0.0):
    return SemanticNode(
        node_id=node_id,
        text=node_id,
        embedding=np.array(vec, dtype=float),
        importance=importance,
    )


def test_find_duplicates_same_file_code_nodes_not_compared():
    """Two functions in the SAME file must NOT be flagged as duplicates, even
    when their symbol names contain underscores (rsplit mis-extracted the file
    id). A genuine cross-file duplicate is still detected."""
    c = SemanticCompressor()
    c.chunks = {
        "main.py::process_data": _node("main.py::process_data", [1.0, 0.0, 0.0]),
        "main.py::handle_request": _node("main.py::handle_request", [1.0, 0.0, 0.0]),
        "other.py::foo": _node("other.py::foo", [1.0, 0.0, 0.0]),
    }
    dups = c.find_duplicates(threshold=0.95)
    pairs = {frozenset((d["node_a"], d["node_b"])) for d in dups if "warning" not in d}

    # Same-file pair must be skipped (was a false positive before the fix).
    assert frozenset(("main.py::process_data", "main.py::handle_request")) not in pairs
    # Cross-file identical embeddings are still genuine duplicates.
    assert frozenset(("main.py::process_data", "other.py::foo")) in pairs


def test_max_dense_cosine_nan_query_returns_zero(monkeypatch):
    """A degenerate (NaN) query embedding yields 0.0 relevance, not NaN."""
    c = SemanticCompressor()
    c.chunks = {"d_n0": _node("d_n0", [1.0, 0.0, 0.0])}
    monkeypatch.setattr(c.model, "encode", lambda texts: [np.array([np.nan, np.nan, np.nan])])

    result = c._max_dense_cosine("any query text")

    assert result == 0.0
    assert not np.isnan(result)


def test_select_skeleton_ordered_nan_query_falls_back_to_importance(monkeypatch):
    """A NaN query embedding falls back to importance-only ordering instead of
    propagating NaN through the MMR relevance vector."""
    c = SemanticCompressor()
    file_nodes = [
        ("f_n0", _node("f_n0", [1.0, 0.0, 0.0], importance=0.1)),
        ("f_n1", _node("f_n1", [0.0, 1.0, 0.0], importance=0.9)),
        ("f_n2", _node("f_n2", [0.0, 0.0, 1.0], importance=0.5)),
    ]
    monkeypatch.setattr(c.model, "encode", lambda texts: [np.array([np.nan, np.nan, np.nan])])

    picked = c._select_skeleton_nodes_ordered(file_nodes, num_skeleton=2, query="x")

    # No NaN crash; importance-ordered fallback: f_n1 (0.9) then f_n2 (0.5).
    assert picked == ["f_n1", "f_n2"]
