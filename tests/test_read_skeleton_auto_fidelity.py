"""Regression locks for the read_skeleton production-default fidelity fix.

Context: the live MCP ``read_skeleton`` path built the compressor with a fixed
``skeleton_ratio=0.2``.  The node-budget floor ``max(1, int(N * ratio))`` floors
to **1 node** for any document with <= 5 semantic nodes at 0.2 — so a ~2k-token /
5-node prose doc surfaced 1 ANCHOR + 4 content-free ``[HIDDEN]`` lines (dropping
~80%).  The fix flips the *production* MCP build default to ``"auto"`` (the
engine's existing adaptive curve, which returns 0.8 for <8k-token docs → ~4/5
nodes), renders the already-computed hidden-node summary on ``[HIDDEN]`` lines,
and keeps ``0.2`` reachable as an explicit aggressive mode.

These tests are written BEFORE the implementation (TDD Red phase).

CRITICAL: the GTM benchmark (tests/test_gtm_benchmarks.py) constructs
``SemanticCompressor()`` with the *constructor default* and MUST stay unchanged.
The constructor default therefore stays at 0.2 (locked by
tests/test_adaptive_compression.py::test_default_uses_0_2).  Only the production
MCP build path (server_factory_service.code_adapter_config) flips to "auto".
"""

from __future__ import annotations

from src.semantic_compressor import SemanticCompressor, compute_adaptive_ratio


def _build_multi_node_doc() -> str:
    """A ~1.4k-token doc that deterministically yields ~6 semantic nodes.

    Uses H2 section boundaries so the chunker splits one node per section
    regardless of the embedding tier (TF-IDF in CI merges plain paragraphs
    aggressively, which would make node count tier-dependent and the test
    flaky).  No verdict/finding markers, so it stays plain (baseline) content.
    The total stays well under the 8k-token adaptive boundary → auto ratio 0.8.
    """
    sections = []
    for topic in (
        "Authentication",
        "Billing",
        "Resilience",
        "Caching",
        "Webhooks",
    ):
        body = " ".join(
            f"Sentence {k} about the {topic} subsystem behavior and design "
            f"rationale that is unique to {topic} number {k}."
            for k in range(20)
        )
        sections.append(f"## {topic}\n\n{body}")
    return "# System Overview\n\n" + "\n\n".join(sections)


# Resolved at import: a multi-node prose-like doc (~6 nodes, ~1.4k tokens).
_FIVE_NODE_PROSE = _build_multi_node_doc()


def _count_anchor_nodes(skeleton_text: str) -> int:
    return skeleton_text.count("[ANCHOR]")


def _count_hidden_nodes(skeleton_text: str) -> int:
    return skeleton_text.count("[HIDDEN]")


class TestProductionReadSkeletonIsFaithful:
    """The production MCP read_skeleton path surfaces most of a small doc."""

    def test_production_factory_default_is_auto(self):
        """code_adapter_config (the production MCP build) uses skeleton_ratio='auto'."""
        from src.semantic_modulator.app.server_factory_service import (
            ServerFactoryService,
        )

        config = ServerFactoryService.code_adapter_config(preload_code_model=False)
        assert config["skeleton_ratio"] == "auto", (
            "Production MCP compressor must build with adaptive 'auto' ratio so "
            "small/medium docs are faithful, not floored to 1 node."
        )

    def test_small_prose_doc_surfaces_majority_of_nodes(self):
        """A multi-node small doc through the production ('auto') path shows >= 80% nodes.

        This is the exact bug the fix closes: at the old fixed 0.2 the same doc
        floored to 1 ANCHOR + N-1 HIDDEN.  At 'auto' (0.8 for <8k tokens) the
        engine surfaces the majority of nodes (faithful-but-denser).
        """
        compressor = SemanticCompressor(skeleton_ratio="auto")
        result = compressor.ingest_file(_FIVE_NODE_PROSE, "faithful_prose")
        assert result is not None
        skeleton = compressor._generate_skeleton("faithful_prose")

        total = skeleton.total_nodes
        anchors = _count_anchor_nodes(skeleton.skeleton_text)

        # The doc must produce several nodes for this test to be meaningful.
        assert total >= 5, f"expected >= 5 nodes for the fixture, got {total}"
        # Faithful: the adaptive 0.8 curve surfaces >= 80% of nodes (int floor),
        # i.e. at least 4 nodes — NOT the single-node floor of the old default.
        expected_min = max(4, int(total * 0.8))
        assert anchors >= expected_min, (
            f"auto ratio should surface >= 80% of nodes (>= {expected_min}); "
            f"got {anchors}/{total} anchors. skeleton:\n{skeleton.skeleton_text}"
        )

    def test_old_fixed_ratio_would_have_floored_aggressively(self):
        """Characterization: the OLD fixed 0.2 floors the same doc to ~1 anchor.

        Documents the regression the fix closes (NOT the new production default)
        and proves 'auto' surfaces strictly more than the old 0.2 default on the
        identical document.
        """
        aggressive = SemanticCompressor(skeleton_ratio=0.2)
        aggressive.ingest_file(_FIVE_NODE_PROSE, "aggressive_prose")
        aggr_skeleton = aggressive._generate_skeleton("aggressive_prose")
        aggr_total = aggr_skeleton.total_nodes
        aggr_anchors = _count_anchor_nodes(aggr_skeleton.skeleton_text)

        auto = SemanticCompressor(skeleton_ratio="auto")
        auto.ingest_file(_FIVE_NODE_PROSE, "auto_prose")
        auto_skeleton = auto._generate_skeleton("auto_prose")
        auto_anchors = _count_anchor_nodes(auto_skeleton.skeleton_text)

        # 0.2 floors to int(N*0.2) which is 1 for a 5-6 node doc.
        assert aggr_anchors == max(
            1, int(aggr_total * 0.2)
        ), f"0.2 should floor to int(N*0.2) anchors; got {aggr_anchors}/{aggr_total}"
        # auto must surface strictly more than the old aggressive default.
        assert (
            auto_anchors > aggr_anchors
        ), f"auto ({auto_anchors}) must surface more than 0.2 ({aggr_anchors})"


class TestExplicitAggressiveModePreserved:
    """0.2 stays reachable as an explicit aggressive ratio (mode not lost)."""

    def test_explicit_0_2_still_floors_small_doc(self):
        """Passing skeleton_ratio=0.2 explicitly yields the aggressive 1-node form."""
        compressor = SemanticCompressor(skeleton_ratio=0.2)
        assert compressor.skeleton_ratio == 0.2
        compressor.ingest_file(_FIVE_NODE_PROSE, "explicit_aggressive")
        skeleton = compressor._generate_skeleton("explicit_aggressive")
        total = skeleton.total_nodes
        anchors = _count_anchor_nodes(skeleton.skeleton_text)
        # Aggressive: the doc floors to int(N*0.2) anchors (1 for a 5-6 node doc).
        assert anchors == max(1, int(total * 0.2)), (
            f"explicit 0.2 must remain the aggressive floored form; " f"got {anchors}/{total}"
        )

    def test_adapter_accepts_explicit_aggressive_ratio(self):
        """CodeCompressionAdapter still accepts an explicit float ratio."""
        from src.code_compression_adapter import CodeCompressionAdapter

        adapter = CodeCompressionAdapter(skeleton_ratio=0.2)
        assert adapter.skeleton_ratio == 0.2
        assert adapter._text_compressor.skeleton_ratio == 0.2


class TestHiddenLineCarriesSummary:
    """Hidden nodes render the already-computed summary (feature d)."""

    def test_hidden_line_includes_node_summary(self):
        """A [HIDDEN] line surfaces a non-empty content summary, not just a pointer."""
        # Aggressive ratio guarantees multiple HIDDEN nodes on a multi-node doc.
        compressor = SemanticCompressor(skeleton_ratio=0.2)
        compressor.ingest_file(_FIVE_NODE_PROSE, "hidden_summary_doc")
        skeleton = compressor._generate_skeleton("hidden_summary_doc")

        lines = skeleton.skeleton_text.splitlines()
        hidden_lines = [ln for ln in lines if "[HIDDEN]" in ln]
        assert hidden_lines, f"expected at least one [HIDDEN] line:\n{skeleton.skeleton_text}"

        # Skeleton-Version 2 (2026-07-01): the verbose "Detail hidden (use
        # modulate_region to expand)" phrase was hoisted to the header ONCE (ratio
        # ceiling fix); each hidden node line is now "[node_id] [HIDDEN] - {summary}".
        # The line must still carry a content summary after the [HIDDEN] marker.
        def _carries_summary(line: str) -> bool:
            idx = line.find("[HIDDEN]")
            if idx < 0:
                return False
            tail = line[idx + len("[HIDDEN]") :].lstrip(" -").strip()
            return len(tail) > 0

        assert any(_carries_summary(ln) for ln in hidden_lines), (
            "Every hidden line is the old content-free placeholder; the "
            "already-computed summary was discarded.\n"
            f"{skeleton.skeleton_text}"
        )

    def test_hidden_line_preserves_drilldown_pointer(self):
        """The hidden line still references modulate_region for drill-down."""
        compressor = SemanticCompressor(skeleton_ratio=0.2)
        compressor.ingest_file(_FIVE_NODE_PROSE, "hidden_pointer_doc")
        skeleton = compressor._generate_skeleton("hidden_pointer_doc")
        assert (
            "[HIDDEN]" in skeleton.skeleton_text
        ), f"expected hidden nodes at aggressive ratio:\n{skeleton.skeleton_text}"
        assert (
            "modulate_region" in skeleton.skeleton_text
        ), "hidden lines must keep the modulate_region drill-down pointer"


class TestAdaptiveCurveHasNoCliff:
    """Audit (feature 3): the adaptive curve transitions gradually, no cliff."""

    def test_curve_is_monotonically_non_increasing(self):
        """Larger corpora never keep MORE nodes than smaller ones."""
        sizes = [0, 1000, 7999, 8000, 20000, 31999, 32000, 60000, 99999, 100000, 500000]
        ratios = [compute_adaptive_ratio(s) for s in sizes]
        for prev, cur in zip(ratios, ratios[1:]):
            assert cur <= prev, f"curve not monotone: {prev} -> {cur}"

    def test_no_single_step_exceeds_first_step(self):
        """No tier transition drops more than the first (0.8 -> 0.5) step.

        Confirms there is NO '0.8 -> 0.1 with nothing between' cliff: the
        intermediate 0.5 and 0.2 tiers keep every step <= 0.3.
        """
        boundaries = [7999, 8000, 31999, 32000, 99999, 100000]
        ratios = [compute_adaptive_ratio(b) for b in boundaries]
        # Steps occur at index pairs (1,0)?? compute deltas across the boundary jumps.
        deltas = [
            compute_adaptive_ratio(7999) - compute_adaptive_ratio(8000),  # 0.8 -> 0.5
            compute_adaptive_ratio(31999) - compute_adaptive_ratio(32000),  # 0.5 -> 0.2
            compute_adaptive_ratio(99999) - compute_adaptive_ratio(100000),  # 0.2 -> 0.1
        ]
        assert all(d >= 0 for d in deltas), f"non-monotone deltas: {deltas}"
        first_step = deltas[0]
        assert all(d <= first_step + 1e-9 for d in deltas), (
            f"a later tier step exceeds the first (0.8->0.5) step: {deltas} "
            f"(boundary ratios sampled: {ratios})"
        )
