"""Audit 2026-06-24 CRITICAL regression lock.

The COMI coarse pass (triggered by a query on a document with >3 nodes) keeps
only the top ~50% of nodes by query relevance and reassigns ``file_nodes`` to
that subset. Anchors are merged into ``skeleton_nodes`` separately, but that set
only controls [ANCHOR]/[HIDDEN] *labelling* — the render loop iterates
``file_nodes``. So an explicitly-anchored node that COMI dropped from
``file_nodes`` was absent from the rendered skeleton entirely, silently
violating the evidence-aware / "always keep this region" anchor contract.

Fix: ``semantic_compressor.py`` unions dropped anchors back into the COMI-kept
set before rebuilding ``file_nodes`` (~line 1326).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor


# H2 section boundaries force one node per section regardless of embedding tier
# (TF-IDF in CI merges plain paragraphs aggressively, which would make node
# count tier-dependent and the test flaky). Mirrors _build_multi_node_doc in
# test_read_skeleton_auto_fidelity.py. 5 sections + H1 => ~6 nodes, ~1.4k tokens.
def _build_sectioned_doc() -> str:
    sections = []
    for topic in ("Authentication", "Billing", "Resilience", "Caching", "Webhooks"):
        body = " ".join(
            f"Sentence {k} about the {topic} subsystem behavior and design "
            f"rationale that is unique to {topic} number {k}."
            for k in range(20)
        )
        sections.append(f"## {topic}\n\n{body}")
    return "# System Overview\n\n" + "\n\n".join(sections)


SECTIONED_DOC = _build_sectioned_doc()
QUERY = "Authentication login session token credentials and identity verification"


def _extract_anchor_ids(skeleton_text: str):
    anchors = []
    for line in skeleton_text.splitlines():
        if "[ANCHOR]" in line and line.startswith("["):
            anchors.append(line.split("]")[0][1:])
    return anchors


def test_comi_coarse_pass_never_drops_explicit_anchor():
    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(SECTIONED_DOC, "anchor_comi_doc")

    node_ids = list(compressor.graphs["anchor_comi_doc"].nodes())
    # COMI only runs with >3 nodes; assert the precondition so a chunking change
    # that merged nodes below the threshold fails loudly instead of silently
    # not exercising the bug.
    assert len(node_ids) > 3, f"expected >3 nodes for the COMI path, got {len(node_ids)}"

    # The lowest-relevance node to the quantum query is guaranteed to fall in
    # the COMI bottom 50% (coarse_keep == max(2, n // 2) keeps only the top
    # half), so anchoring it exercises exactly the dropped-anchor path.
    ranked = compressor.search_semantic_with_scores(
        QUERY, file_id="anchor_comi_doc", top_k=len(node_ids)
    )
    assert ranked, "search returned no ranked nodes"
    lowest_node = ranked[-1][0]

    resp = compressor._generate_skeleton(
        "anchor_comi_doc",
        query=QUERY,
        anchor_node_ids={lowest_node},
    )
    anchors = _extract_anchor_ids(resp.skeleton_text)
    assert lowest_node in anchors, (
        f"anchored low-relevance node {lowest_node!r} was dropped by the COMI "
        f"coarse pass and never rendered; anchors present: {anchors}"
    )
