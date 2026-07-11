"""#187 rerank wire: the rerank stage reorders / fails safe (model-free).

The dispatch integration (RERANK_ENABLED default-OFF -> byte-identical retrieval)
is covered by the EXISTING search suite (test_f11_path_c / test_f11_gated_fusion /
test_semantic_compressor_unit), which all run with the flag off. Here we unit-test
the rerank stage itself with a MOCK scorer -- no model load, no HF cache -- proving
the pool reorders by the injected cross-encoder scores and never breaks retrieval.
"""

from __future__ import annotations

from src.semantic_compressor import SemanticCompressor


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text


def _bare() -> SemanticCompressor:
    return object.__new__(SemanticCompressor)


def test_rerank_pool_reorders_by_injected_scorer():
    svc = _bare()
    svc.chunks = {"n1": _Chunk("doc about apples"), "n2": _Chunk("doc about oranges")}
    ranked = [("n1", 0.8), ("n2", 0.5)]  # retrieval order: n1 above n2
    # Mock cross-encoder scores n2 higher -> n2 must be promoted to the top.
    svc._rerank_scorer = lambda query, docs: [0.1, 0.9]
    out = svc._rerank_pool("q", ranked)
    assert [nid for nid, _ in out] == ["n2", "n1"]


def test_rerank_pool_is_fail_safe_on_scorer_error():
    svc = _bare()
    svc.chunks = {"n1": _Chunk("a"), "n2": _Chunk("b")}
    ranked = [("n1", 0.8), ("n2", 0.5)]

    def _boom(query, docs):
        raise RuntimeError("model unavailable")

    svc._rerank_scorer = _boom
    # Must fall back to the input retrieval order, never raise.
    assert svc._rerank_pool("q", ranked) == ranked


def test_rerank_pool_single_candidate_is_noop():
    svc = _bare()
    svc.chunks = {"n1": _Chunk("only")}
    ranked = [("n1", 0.9)]
    # <2 candidates -> rerank_candidates short-circuits; scorer never called.
    svc._rerank_scorer = lambda q, d: [0.0]
    assert svc._rerank_pool("q", ranked) == ranked
