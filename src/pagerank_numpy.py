"""NumPy PageRank for environments without SciPy (Docker ONNX-only image)."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def pagerank_numpy(
    graph: Any,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> Dict[str, float]:
    """Compute PageRank with a dense NumPy power iteration (no SciPy).

    Matches NetworkX's undirected handling: undirected edges become
    bidirectional. Edge ``weight`` attributes are honored when present.
    """
    nodes = list(graph)
    n = len(nodes)
    if n == 0:
        return {}

    index = {node: i for i, node in enumerate(nodes)}
    # Column-stochastic transition matrix M[v, u] = weight(u→v) / outweight(u)
    M = np.zeros((n, n), dtype=np.float64)
    if graph.is_directed():
        edges = graph.edges(data=True)
        for u, v, data in edges:
            M[index[v], index[u]] += float(data.get("weight", 1.0))
    else:
        for u, v, data in graph.edges(data=True):
            w = float(data.get("weight", 1.0))
            M[index[v], index[u]] += w
            M[index[u], index[v]] += w

    col_sums = M.sum(axis=0)
    dangling = col_sums == 0
    safe_sums = col_sums.copy()
    safe_sums[dangling] = 1.0
    M /= safe_sums

    x = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = np.full(n, 1.0 / n, dtype=np.float64)
    dangling_mask = dangling.astype(np.float64)

    for _ in range(max_iter):
        x_last = x
        dangling_sum = float(dangling_mask @ x)
        x = alpha * (M @ x + dangling_sum * teleport) + (1.0 - alpha) * teleport
        if np.abs(x - x_last).sum() < n * tol:
            break

    return {nodes[i]: float(x[i]) for i in range(n)}


def compute_pagerank(graph: Any, **kwargs: Any) -> Dict[str, float]:
    """Prefer NetworkX (SciPy) PageRank; fall back to NumPy when SciPy is absent."""
    try:
        import networkx as nx

        return nx.pagerank(graph, **kwargs)
    except ImportError:
        # nx.pagerank imports scipy internally; Docker ONNX image has networkx but not scipy.
        return pagerank_numpy(graph, **kwargs)
