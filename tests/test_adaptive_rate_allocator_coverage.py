from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from src.adaptive_rate_allocator import (
    AdaptiveRateAllocator,
    ContextWindowAdapter,
    MultiLevelSemanticEncoder,
)


@dataclass
class _Chunk:
    text: str
    importance: float


class _DummyCompressor:
    def __init__(self):
        self.skeleton_ratio = 0.2
        self.graphs = {"doc": nx.Graph()}
        self.graphs["doc"].add_nodes_from(["doc_n0", "doc_n1", "doc_n2", "doc_n3", "doc_n4"])
        self.chunks = {
            "doc_n0": _Chunk("main topic alpha", 0.9),
            "doc_n1": _Chunk("supporting beta", 0.8),
            "doc_n2": _Chunk("detail gamma", 0.5),
            "doc_n3": _Chunk("detail delta", 0.4),
            "doc_n4": _Chunk("detail epsilon", 0.3),
        }

    def read_skeleton(self, file_id: str) -> str:
        return f"skeleton:{file_id}:{self.skeleton_ratio:.2f}"

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        return text[:max_length]


def test_adaptive_allocator_complexity_and_selection():
    torch.manual_seed(1234)
    np.random.seed(1234)
    allocator = AdaptiveRateAllocator(num_rate_levels=5, temperature=1.0)

    g = nx.Graph()
    g.add_nodes_from(["n1", "n2", "n3"])
    g.add_edges_from([("n1", "n2"), ("n2", "n3")])

    complexity = allocator.calculate_complexity_score(g)
    assert 0.0 <= complexity <= 1.0
    assert allocator.calculate_complexity_score(nx.Graph()) == 0.0

    logits = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    soft, idx = allocator.gumbel_softmax_rate_selection(logits, hard=False)
    assert soft.shape[0] == 5
    assert 0 <= idx < 5

    hard, idx2 = allocator.gumbel_softmax_rate_selection(logits, hard=True)
    assert hard.shape[0] == 5
    assert 0 <= idx2 < 5

    ratio, diag = allocator(g, available_context_tokens=20000, max_context_tokens=100000)
    assert 0.09 <= ratio <= 0.31
    assert "complexity" in diag
    assert "selection_probs" in diag


def test_context_window_adapter_and_multilevel_encoder():
    compressor = _DummyCompressor()
    adapter = ContextWindowAdapter(compressor)
    out = adapter.adapt_to_context_window("doc", available_tokens=5000, max_tokens=100000)
    assert "[CONTEXT WINDOW ADAPTATION]" in out
    assert "skeleton:doc" in out

    encoder = MultiLevelSemanticEncoder(compressor)
    levels = encoder.encode_multilevel("doc", available_tokens=10000)
    assert set(levels.keys()) == {"main", "auxiliary", "detail", "available_tokens"}
    assert levels["available_tokens"] == 10000

    summary = encoder.generate_adaptive_skeleton("doc", available_tokens=2000)
    assert "MULTI-LEVEL SEMANTIC SKELETON" in summary
    assert "Context budget:" in summary
    assert "Included:" in summary
