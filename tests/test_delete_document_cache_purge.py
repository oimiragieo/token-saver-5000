import pytest
from unittest.mock import MagicMock
from src.semantic_compressor import SemanticCompressor
from src.code_compression_adapter import CodeCompressionAdapter


def test_semantic_compressor_purge_document_caches():
    """Verify purge_document_caches clears baseline skeletons, stats, and pagerank caches."""
    compressor = object.__new__(SemanticCompressor)
    compressor._baseline_skeleton_cache = {"doc_1": "skeleton_1", "doc_2": "skeleton_2"}
    compressor._baseline_skeleton_stats = {
        "doc_1": {"skeleton_tokens": 10, "ratio": 0.5},
        "doc_2": {"skeleton_tokens": 20, "ratio": 0.4},
    }
    compressor._pagerank_cache = {
        "pagerank_doc_1_abc123": {"node1": 0.5},
        "pagerank_doc_2_def456": {"node2": 0.8},
    }

    # Call purge_document_caches
    assert hasattr(compressor, "purge_document_caches"), "purge_document_caches missing on SemanticCompressor"
    compressor.purge_document_caches("doc_1")

    assert "doc_1" not in compressor._baseline_skeleton_cache
    assert "doc_1" not in compressor._baseline_skeleton_stats
    assert "pagerank_doc_1_abc123" not in compressor._pagerank_cache

    # doc_2 preserved
    assert "doc_2" in compressor._baseline_skeleton_cache
    assert "doc_2" in compressor._baseline_skeleton_stats
    assert "pagerank_doc_2_def456" in compressor._pagerank_cache


def test_code_compression_adapter_deletes_document_caches():
    """Verify delete_document_from_memory on CodeCompressionAdapter purges underlying caches."""
    tc = object.__new__(SemanticCompressor)
    tc.chunks = {}
    tc.graphs = {}
    tc.file_metadata = {}
    tc._file_skeleton_ratio_overrides = {}
    tc._baseline_skeleton_cache = {"doc_1": "skeleton_1"}
    tc._baseline_skeleton_stats = {"doc_1": {"skeleton_tokens": 10, "ratio": 0.5}}
    tc._pagerank_cache = {"pagerank_doc_1_abc": {"node1": 0.5}}

    adapter = object.__new__(CodeCompressionAdapter)
    adapter._text_compressor = tc
    adapter._code_compressor = None
    adapter._code_file_ids = set()

    adapter.delete_document_from_memory("doc_1")

    assert "doc_1" not in tc._baseline_skeleton_cache
    assert "doc_1" not in tc._baseline_skeleton_stats
    assert "pagerank_doc_1_abc" not in tc._pagerank_cache
