"""Tests for PageRank SciPy fallback (Docker ONNX runtime has no scipy)."""

from unittest.mock import patch

import networkx as nx

from src.pagerank_numpy import compute_pagerank, pagerank_numpy


def test_pagerank_numpy_small_graph() -> None:
    graph = nx.path_graph(4)
    scores = pagerank_numpy(graph)
    assert len(scores) == 4
    assert sum(scores.values()) > 0.99


def test_compute_pagerank_falls_back_when_scipy_missing() -> None:
    graph = nx.path_graph(3)

    def _raise_scipy(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'scipy'")

    with patch("networkx.pagerank", _raise_scipy):
        scores = compute_pagerank(graph)

    assert len(scores) == 3
    assert abs(sum(scores.values()) - 1.0) < 1e-5


def test_compute_pagerank_uses_networkx_when_available() -> None:
    graph = nx.path_graph(3)
    with patch("networkx.pagerank", return_value={"a": 0.5, "b": 0.3, "c": 0.2}) as mock_pr:
        graph = nx.path_graph(3)
        nodes = list(graph.nodes)
        mock_pr.return_value = {n: 1.0 / len(nodes) for n in nodes}
        scores = compute_pagerank(graph)
        mock_pr.assert_called_once()
        assert len(scores) == 3
