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
