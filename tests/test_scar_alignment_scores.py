"""Targeted tests for SCAR alignment score behavior."""

import pytest

from src.scar_compressor import SCAREnhancedCompressor
from src.semantic_compressor import SemanticCompressor

TEST_DOC = """
Service retries are bounded and use exponential backoff.
Errors include timeout, dependency failure, and invalid payload.
Observability includes traces, logs, and request identifiers.
"""


def test_search_with_alignment_scores_are_normalized():
    base = SemanticCompressor()
    base.ingest_file(TEST_DOC, "scar_norm_doc")

    scar = SCAREnhancedCompressor(
        base_compressor=base,
        use_learnable_compression=True,
        use_alignment_guidance=True,
    )
    results = scar.search_with_alignment(
        query="retry and timeout behavior",
        file_id="scar_norm_doc",
        top_k=3,
        alignment_weight=0.5,
    )

    assert results
    for _, score in results:
        assert 0.0 <= score <= 1.0


def test_search_with_alignment_rejects_invalid_weight():
    base = SemanticCompressor()
    base.ingest_file(TEST_DOC, "scar_weight_doc")
    scar = SCAREnhancedCompressor(
        base_compressor=base,
        use_learnable_compression=False,
        use_alignment_guidance=False,
    )

    with pytest.raises(ValueError, match="alignment_weight"):
        scar.search_with_alignment(
            query="retry behavior",
            file_id="scar_weight_doc",
            top_k=3,
            alignment_weight=1.2,
        )
