"""Model-free unit tests for the recall-gated rerank stage (#187).

No model load / no HF cache: the cross-encoder scorer is a mock callable, so the
pure gate + reorder logic is tested in isolation (the ONNX model plugs in at
wiring time). See ``compression-engine-sota`` skill (model-free-test the pure fn).
"""

from __future__ import annotations

from src.reranker_gate import RerankConfig, rerank_candidates

# Candidate = (id, retrieval_score, text), ordered best-first by retrieval_score.
_TEXT = lambda c: c[2]  # noqa: E731
_SCORE = lambda c: c[1]  # noqa: E731


def _run(query, cands, scorer, **cfg):
    return rerank_candidates(
        query,
        cands,
        text_of=_TEXT,
        score_of=_SCORE,
        scorer=scorer,
        config=RerankConfig(**cfg),
    )


def _ids(cands):
    return [c[0] for c in cands]


def test_disabled_is_noop():
    cands = [("a", 0.9, "x"), ("b", 0.1, "y")]
    out, did = _run("q", cands, lambda q, d: [1.0, 2.0], enabled=False)
    assert out is cands and did is False


def test_fewer_than_two_candidates_is_noop():
    cands = [("a", 0.9, "x")]
    out, did = _run("q", cands, lambda q, d: [9.0], enabled=True)
    assert out == cands and did is False


def test_empty_query_is_noop():
    cands = [("a", 0.9, "x"), ("b", 0.1, "y")]
    out, did = _run("", cands, lambda q, d: [1.0, 2.0], enabled=True)
    assert out is cands and did is False


def test_reorders_by_rerank_score():
    # Retrieval order a,b,c but the cross-encoder says c is most relevant.
    cands = [("a", 0.9, "ta"), ("b", 0.5, "tb"), ("c", 0.1, "tc")]
    out, did = _run("q", cands, lambda q, d: [0.2, 0.5, 0.9], enabled=True)
    assert did is True
    assert _ids(out) == ["c", "b", "a"]


def test_pool_truncation_keeps_tail_in_place():
    cands = [("a", 0.9, "ta"), ("b", 0.8, "tb"), ("c", 0.7, "tc")]
    # Only the top 2 are reranked (and reversed); c stays as the tail.
    out, did = _run("q", cands, lambda q, d: [0.1, 0.9], enabled=True, pool_size=2)
    assert did is True
    assert _ids(out) == ["b", "a", "c"]


def test_confidence_skip_fires_on_separated_top1():
    # top1 (0.95) - top2 (0.10) = 0.85 > margin 0.5 -> skip reranking.
    cands = [("a", 0.95, "ta"), ("b", 0.10, "tb")]
    out, did = _run("q", cands, lambda q, d: [0.0, 9.0], enabled=True, confidence_skip_margin=0.5)
    assert did is False
    assert _ids(out) == ["a", "b"]


def test_confidence_skip_does_not_fire_on_close_scores():
    # top1 (0.55) - top2 (0.50) = 0.05 < margin 0.5 -> rerank runs.
    cands = [("a", 0.55, "ta"), ("b", 0.50, "tb")]
    out, did = _run("q", cands, lambda q, d: [0.0, 9.0], enabled=True, confidence_skip_margin=0.5)
    assert did is True
    assert _ids(out) == ["b", "a"]


def test_scorer_wrong_length_fails_safe():
    cands = [("a", 0.9, "ta"), ("b", 0.5, "tb")]
    out, did = _run("q", cands, lambda q, d: [0.1], enabled=True)  # 1 score for 2 docs
    assert did is False
    assert out is cands


def test_scorer_receives_query_and_texts():
    seen = {}

    def scorer(q, docs):
        seen["q"] = q
        seen["docs"] = list(docs)
        return [1.0, 2.0]

    cands = [("a", 0.9, "alpha"), ("b", 0.5, "beta")]
    _run("my-query", cands, scorer, enabled=True)
    assert seen["q"] == "my-query"
    assert seen["docs"] == ["alpha", "beta"]


def test_ties_preserve_retrieval_order():
    # Equal rerank scores -> stable: retrieval order a,b,c preserved.
    cands = [("a", 0.9, "ta"), ("b", 0.8, "tb"), ("c", 0.7, "tc")]
    out, did = _run("q", cands, lambda q, d: [0.5, 0.5, 0.5], enabled=True)
    assert did is True
    assert _ids(out) == ["a", "b", "c"]
