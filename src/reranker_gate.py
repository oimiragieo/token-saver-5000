"""Recall-gated cross-encoder rerank stage (#187, WORLD-CLASS #1).

Second-stage reranking for the retrieval path: an embedding first-pass returns a
candidate pool, then a cross-encoder scores each ``(query, candidate)`` pair and
re-orders by that relevance. Per the 2026 SOTA (see the ``compression-engine-sota``
skill + LongLLMLingua coarse-to-fine): reranking only helps when the candidate
pool's recall materially exceeds the top-k recall -- i.e. there is a *recall gap*
to recover. When the first-stage retrieval is already confident (a well-separated
top-1), reranking adds latency without quality.

This module owns the QUERY-TIME concern: the reorder + a cheap confidence-skip
heuristic. The OFFLINE "should we enable reranking for this workload at all"
decision is the #266-corpus Pareto verification (a labelled recall@pool-vs-recall@k
measurement), not something computable per-query.

The cross-encoder scorer is INJECTED as a plain callable ``(query, [docs]) ->
[scores]`` so this pure logic is unit-tested model-free (mock scorer, no model
load, no HF cache). The ONNX cross-encoder (e.g. granite-embedding-reranker-r2 or
ms-marco-MiniLM) plugs in at wiring time behind a default-OFF flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence, TypeVar

T = TypeVar("T")


class RerankScorer(Protocol):
    """A cross-encoder relevance scorer: query + candidate texts -> per-pair scores.

    Contract: returns exactly one float per input document, higher = more relevant.
    """

    def __call__(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RerankConfig:
    """Recall-gated rerank knobs. Ships default-OFF (enablement discipline)."""

    enabled: bool = False
    # Rerank at most the top-N retrieved candidates (cross-encoders cost one
    # forward pass PER pair, so a bounded pool keeps CPU latency predictable).
    pool_size: int = 50
    # Confidence skip: if the top-1 retrieval score exceeds #2 by more than this
    # margin, the first stage is already confident and reranking rarely helps
    # (there is no recall gap to recover). 0.0 disables the skip (always rerank
    # the pool when enabled).
    confidence_skip_margin: float = 0.0


def rerank_candidates(
    query: str,
    candidates: list[T],
    *,
    text_of: Callable[[T], str],
    score_of: Callable[[T], float],
    scorer: RerankScorer,
    config: RerankConfig,
) -> tuple[list[T], bool]:
    """Recall-gated cross-encoder reorder. Returns ``(ordered_candidates, did_rerank)``.

    ``candidates`` must be ordered best-first by the first-stage retrieval score.
    ``text_of`` / ``score_of`` are accessors so this is agnostic to the engine's
    candidate representation (node id / tuple / dataclass).

    No-op (returns the input unchanged + ``False``) when: reranking is disabled,
    the query is empty, there are fewer than 2 candidates, or the confidence-skip
    fires. Otherwise the top ``pool_size`` are scored by the injected cross-encoder
    and reordered by that score (descending, stable on ties); any tail beyond the
    pool is appended unchanged. A scorer that returns the wrong number of scores is
    a contract violation and fails SAFE (no reorder) rather than corrupting order.
    """
    if not config.enabled or not query or len(candidates) < 2:
        return candidates, False

    if config.confidence_skip_margin > 0.0:
        top1 = score_of(candidates[0])
        top2 = score_of(candidates[1])
        if (top1 - top2) > config.confidence_skip_margin:
            return candidates, False

    pool_size = max(0, config.pool_size)
    pool = candidates[:pool_size]
    tail = candidates[pool_size:]
    if len(pool) < 2:
        return candidates, False

    scores = list(scorer(query, [text_of(c) for c in pool]))
    if len(scores) != len(pool):
        # Fail safe: a broken scorer must never silently corrupt retrieval order.
        return candidates, False

    # Sort indices by score so the candidate objects themselves are never compared
    # (they may be unorderable); Python's sort is stable, preserving retrieval
    # order among equal rerank scores.
    order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
    reordered = [pool[i] for i in order]
    return reordered + tail, True
