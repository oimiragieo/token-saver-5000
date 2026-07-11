"""TDD coverage for query-guided and evidence-aware compression paths."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor

TEST_DOCUMENT = """
Quantum error correction protects fragile qubits from decoherence.
Surface codes use repeated stabilizer checks to detect errors.
Logical qubits require many physical qubits in fault-tolerant systems.

Pasta sauce improves with slow simmering and fresh basil.
Tomatoes, garlic, and olive oil form the core flavor base.
Salt balance and acid adjustment are critical near the end of cooking.
"""


def _extract_anchor_ids(skeleton_text: str):
    anchors = []
    for line in skeleton_text.splitlines():
        if "[ANCHOR]" in line and line.startswith("["):
            anchors.append(line.split("]")[0][1:])
    return anchors


def test_read_skeleton_query_guided_adds_selection_metadata():
    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(TEST_DOCUMENT, "query_doc")

    skeleton = compressor.read_skeleton("query_doc", query="basil tomato sauce")
    assert "Selection mode: QUERY_GUIDED" in skeleton
    assert "Query: basil tomato sauce" in skeleton


def test_query_guided_header_skeleton_count_matches_rendered_anchors():
    """#287: the query_guided header must report the ACTUAL anchor count, which is
    always <= total nodes. Previously it echoed the target `num_skeleton` sized off
    the full node set while file_nodes was narrowed to the query subset, producing
    impossible headers like 'Total nodes: 19 | Skeleton nodes: 30'."""
    import re

    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(TEST_DOCUMENT, "query_doc")
    skeleton = compressor.read_skeleton("query_doc", query="basil tomato sauce")

    m = re.search(r"Total nodes:\s*(\d+)\s*\|\s*Skeleton nodes:\s*(\d+)", skeleton)
    assert m, skeleton
    total, skel = int(m.group(1)), int(m.group(2))
    assert skel <= total, f"Skeleton nodes ({skel}) > Total nodes ({total})\n{skeleton}"
    anchor_count = len(_extract_anchor_ids(skeleton))
    assert (
        skel == anchor_count
    ), f"header claims {skel} skeleton nodes but {anchor_count} anchors rendered"


def test_query_guided_anchors_include_top_semantic_hit():
    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(TEST_DOCUMENT, "query_doc")

    top_hit = compressor.search_semantic_with_scores(
        "basil tomato garlic",
        file_id="query_doc",
        top_k=1,
    )[0][0]
    skeleton = compressor.read_skeleton("query_doc", query="basil tomato garlic")
    anchor_ids = _extract_anchor_ids(skeleton)
    assert top_hit in anchor_ids


def test_retrieve_evidence_detects_insufficient_query_signal():
    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(TEST_DOCUMENT, "query_doc")

    result = compressor.retrieve_evidence(
        query="xylophonic hypercube gibberish terms",
        file_id="query_doc",
        top_k=2,
        min_similarity=0.95,
    )

    assert result.sufficient is False
    assert result.used_expanded_search is True
    assert "insufficient" in result.message.lower()


def test_read_skeleton_evidence_aware_includes_status_and_skeleton():
    compressor = SemanticCompressor(skeleton_ratio=0.34)
    compressor.ingest_file(TEST_DOCUMENT, "query_doc")

    text = compressor.read_skeleton_evidence_aware(
        file_id="query_doc",
        query="surface code qubits",
        top_k=2,
        min_similarity=0.1,
    )

    assert "=== EVIDENCE STATUS:" in text
    assert "=== SEMANTIC SKELETON: query_doc ===" in text
