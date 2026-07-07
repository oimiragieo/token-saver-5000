"""Regression lock: `ingest_context`'s `skeleton_ratio` param must actually
change output (2026-07-06, architecture plan Move 5 / knob-honesty fix).

Before this fix, ``handle_ingest`` read ``args["skeleton_ratio"]`` into a
``CompressionAdvisor.estimate_compression`` preview call and then discarded
it — the real compression always used the shared singleton compressor's
instance default (``"auto"`` in production, ``0.2`` on a bare
``SemanticCompressor()``). An agent asking for MORE or LESS compression got a
silent no-op — the same failure class as the activation-honesty bug the
2026-07-02 sprint fixed for small-doc estimates.

The fix threads the (validated) caller-supplied ``skeleton_ratio`` through as
a per-file_id override on the compressor (``SemanticCompressor.
set_file_skeleton_ratio`` / ``_resolve_skeleton_ratio``) — safe on the shared
MCP singleton because it's keyed by file_id, not a blanket instance mutation
(that pattern is only safe on REST's per-thread compressor).

These tests exercise the REAL ``handle_ingest`` handler against a REAL
``SemanticCompressor`` (via the shared ``handler_context`` fixture) — no
compressor mocking — because the whole point is proving the ratio actually
reaches the engine's output, not just that a setter got called.
"""

from __future__ import annotations

import pytest

from src.handlers.compression_handlers import handle_ingest

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


def _multi_section_doc(n_sections: int = 10) -> str:
    """Uses genuinely distinct topic WORDS (not just an index number) so
    intra-doc dedup / SemToken don't collapse the sections into one
    near-duplicate node — see test_query_adaptive_budget_floor.py for the
    same lesson learned the hard way."""
    sections = []
    for topic in _TOPICS[:n_sections]:
        body = " ".join(
            f"Sentence {k} about the {topic} subsystem behavior and design "
            f"rationale that is unique to {topic} number {k}."
            for k in range(20)
        )
        sections.append(f"## {topic}\n\n{body}")
    return "# Overview\n\n" + "\n\n".join(sections)


def _count_anchors(skeleton_text: str) -> int:
    return skeleton_text.count("[ANCHOR]")


class TestSkeletonRatioValidation:
    """Invalid values are rejected with a clear error, not silently ignored."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_ratio", [0.0, -0.1, 1.5, "banana", True, False])
    async def test_invalid_skeleton_ratio_rejected(self, handler_context, bad_ratio):
        args = {
            "file_id": "bad_ratio_doc",
            "text": _multi_section_doc(3),
            "skeleton_ratio": bad_ratio,
        }
        with pytest.raises(ValueError, match="skeleton_ratio"):
            await handle_ingest(handler_context, args)

    @pytest.mark.asyncio
    async def test_auto_string_is_accepted(self, handler_context):
        args = {
            "file_id": "auto_ratio_doc",
            "text": _multi_section_doc(3),
            "skeleton_ratio": "auto",
        }
        # Must not raise.
        await handle_ingest(handler_context, args)

    @pytest.mark.asyncio
    async def test_valid_float_is_accepted(self, handler_context):
        args = {
            "file_id": "valid_ratio_doc",
            "text": _multi_section_doc(3),
            "skeleton_ratio": 0.5,
        }
        await handle_ingest(handler_context, args)


class TestSkeletonRatioTakesEffect:
    """LOAD-BEARING: proves skeleton_ratio is no longer a silent no-op."""

    @pytest.mark.asyncio
    async def test_higher_ratio_yields_more_anchors_than_lower_ratio(self, handler_context):
        doc = _multi_section_doc(10)

        await handle_ingest(
            handler_context, {"file_id": "ratio_low", "text": doc, "skeleton_ratio": 0.1}
        )
        await handle_ingest(
            handler_context, {"file_id": "ratio_high", "text": doc, "skeleton_ratio": 0.9}
        )

        compressor = handler_context["compressor"]
        low_skeleton = compressor._generate_skeleton("ratio_low")
        high_skeleton = compressor._generate_skeleton("ratio_high")

        low_anchors = _count_anchors(low_skeleton.skeleton_text)
        high_anchors = _count_anchors(high_skeleton.skeleton_text)

        assert low_anchors < high_anchors, (
            f"skeleton_ratio=0.1 produced {low_anchors} anchors, "
            f"skeleton_ratio=0.9 produced {high_anchors} anchors — the ratio "
            "must materially change output; if these are equal the knob is "
            "still a silent no-op."
        )
        # Same source document -> same total node count either way.
        assert low_skeleton.total_nodes == high_skeleton.total_nodes

    @pytest.mark.asyncio
    async def test_override_is_scoped_to_its_own_file_id(self, handler_context):
        """A ratio requested for one document must not leak onto a sibling
        document ingested on the same (shared) compressor instance."""
        doc = _multi_section_doc(10)

        await handle_ingest(
            handler_context, {"file_id": "scoped_a", "text": doc, "skeleton_ratio": 0.9}
        )
        await handle_ingest(handler_context, {"file_id": "scoped_b", "text": doc})

        compressor = handler_context["compressor"]
        assert compressor._file_skeleton_ratio_overrides.get("scoped_a") == 0.9
        assert "scoped_b" not in compressor._file_skeleton_ratio_overrides

    @pytest.mark.asyncio
    async def test_omitted_ratio_sets_no_override_and_uses_instance_default(self, handler_context):
        doc = _multi_section_doc(10)
        await handle_ingest(handler_context, {"file_id": "ratio_omitted", "text": doc})

        compressor = handler_context["compressor"]
        assert "ratio_omitted" not in compressor._file_skeleton_ratio_overrides

        skeleton = compressor._generate_skeleton("ratio_omitted")
        expected_floor = max(1, int(skeleton.total_nodes * compressor.skeleton_ratio))
        assert _count_anchors(skeleton.skeleton_text) == expected_floor
