"""Regression lock: query-adaptive skeleton budget must never collapse below
the proportional floor (2026-07-06, architecture plan Move 5 / plan item MF6).

``_generate_skeleton``'s query-adaptive branch (``semantic_compressor.py``,
inside the ``if query and len(file_nodes) > 1:`` block) used to compute::

    num_skeleton = sum(1 for r in per_node_ratios if r >= effective_ratio)
    num_skeleton = max(1, num_skeleton)

``compute_section_ratios`` keeps the *average* ratio at ``effective_ratio``,
but when query relevance concentrates heavily on one or two sections, most
per-node ratios sink toward the 0.05 floor — so the count-based budget can
collapse to as few as 1 node regardless of document size, silently discarding
the caller's requested ``skeleton_ratio``. This is the SAME bug class as the
2026-07-06 dead ``skeleton_ratio`` knob: a caller who asked for more
compression (or less) gets ignored.

The fix clamps the query-adaptive count to never undercut the proportional
floor already computed earlier in the same function
(``max(1, int(len(file_nodes) * effective_ratio))``).
"""

from __future__ import annotations

from unittest.mock import patch

import src.query_adaptive as query_adaptive_module
from src.semantic_compressor import SemanticCompressor

_TOPICS = (
    "Authentication",
    "Billing",
    "Resilience",
    "Caching",
    "Webhooks",
    "Migrations",
    "Telemetry",
    "RateLimiting",
    "Scheduling",
    "Search",
)


def _build_ten_topic_doc() -> str:
    """A ~10-section doc with H2 boundaries — deterministic node count across
    embedding tiers (mirrors ``_build_multi_node_doc`` in
    ``test_read_skeleton_auto_fidelity.py``). Uses genuinely distinct topic
    WORDS (not just an index number) so intra-doc dedup / SemToken don't
    collapse the sections into one near-duplicate node — a purely
    index-templated doc ("Sentence {k} about topic {i}...") embeds as
    near-identical across sections and collapses to 1 node, which would make
    this test indistinguishable from the bug it's proving a fix for."""
    sections = []
    for topic in _TOPICS:
        body = " ".join(
            f"Sentence {k} about the {topic} subsystem behavior and design "
            f"rationale that is unique to {topic} number {k}."
            for k in range(20)
        )
        sections.append(f"## {topic}\n\n{body}")
    return "# Overview\n\n" + "\n\n".join(sections)


def _anchor_count(skeleton_text: str) -> int:
    return skeleton_text.count("[ANCHOR]")


class TestQueryAdaptiveBudgetNeverCollapsesBelowFloor:
    def test_concentrated_relevance_does_not_collapse_budget_to_one(self):
        # Fixed skeleton_ratio (not "auto") so the proportional floor is a
        # known, stable number for this test.
        compressor = SemanticCompressor(skeleton_ratio=0.2)
        file_id = "budget_floor_doc"
        compressor.ingest_file(_build_ten_topic_doc(), file_id=file_id)

        file_nodes = [(nid, compressor.chunks[nid]) for nid in compressor.graphs[file_id].nodes()]
        n = len(file_nodes)
        floor = max(1, int(n * compressor.skeleton_ratio))
        assert n >= 8, f"expected ~10 nodes from the 10-topic doc, got {n}"
        assert floor >= 2, (
            "test requires a document large enough for the proportional floor "
            f"to exceed 1 node (got floor={floor} for n={n}) — otherwise the "
            "collapsed-to-1 bug is indistinguishable from correct behavior"
        )

        # Force the bug's trigger condition: relevance concentrated almost
        # entirely on ONE section, driving every other per-node ratio toward
        # the 0.05 floor so only 1 node clears effective_ratio (0.2).
        concentrated_ratios = [0.9] + [0.01] * (n - 1)

        # selection_strategy="pagerank" skips the separate COMI coarse filter
        # (which would itself shrink file_nodes) so this test isolates ONLY
        # the query-adaptive budget-collapse code path.
        with patch.object(
            query_adaptive_module,
            "compute_section_ratios",
            return_value=concentrated_ratios,
        ):
            skeleton = compressor._generate_skeleton(
                file_id, query="topic 0 subsystem behavior", selection_strategy="pagerank"
            )

        anchors = _anchor_count(skeleton.skeleton_text)
        assert anchors >= floor, (
            f"query-adaptive budget collapsed to {anchors} anchor(s) despite a "
            f"proportional floor of {floor} for {n} nodes at skeleton_ratio="
            f"{compressor.skeleton_ratio} — a concentrated-relevance query must "
            "not silently shrink the requested skeleton size below its floor."
        )

    def test_uniform_relevance_is_unaffected_by_the_clamp(self):
        """Sanity check: when relevance IS roughly uniform (the common case),
        the clamp is a no-op — the query-adaptive count already meets the
        floor on its own, so this must not artificially inflate normal
        results."""
        compressor = SemanticCompressor(skeleton_ratio=0.2)
        file_id = "budget_floor_uniform_doc"
        compressor.ingest_file(_build_ten_topic_doc(), file_id=file_id)

        file_nodes = [(nid, compressor.chunks[nid]) for nid in compressor.graphs[file_id].nodes()]
        n = len(file_nodes)
        uniform_ratios = [0.2] * n  # every node exactly at effective_ratio

        with patch.object(
            query_adaptive_module, "compute_section_ratios", return_value=uniform_ratios
        ):
            skeleton = compressor._generate_skeleton(
                file_id, query="topic 0 subsystem behavior", selection_strategy="pagerank"
            )

        anchors = _anchor_count(skeleton.skeleton_text)
        # sum(1 for r in uniform_ratios if r >= 0.2) == n, and the floor is
        # max(1, int(n*0.2)) <= n, so the clamp (max of the two) must equal n
        # here — not silently truncated or inflated further.
        assert anchors == n
