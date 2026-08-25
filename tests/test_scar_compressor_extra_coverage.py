"""scar compressor — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
        "ace_framework": MagicMock(),
        "focus_manager": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_semantic_node(text="test", importance=0.5, embedding=None):
    node = MagicMock()
    node.text = text
    node.importance = importance
    node.embedding = embedding if embedding is not None else np.random.rand(384).astype(np.float32)
    node.metadata = {"tokens": 10, "position": 0, "entities": []}
    return node


def _make_code_chunk(
    name="func", chunk_type="function", code="def f(): pass", docstring="", start_line=1, end_line=5
):
    chunk = MagicMock()
    chunk.name = name
    chunk.chunk_type = chunk_type
    chunk.code = code
    chunk.docstring = docstring
    chunk.start_line = start_line
    chunk.end_line = end_line
    return chunk


class TestScarCompressor:
    def test_preservation_loss(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        original = torch.randn(2, 8)
        recon = torch.randn(2, 8)
        loss = comp.compute_preservation_loss(original, recon)
        assert loss.item() > 0

    def test_compress_batch_numpy(self):
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        data = np.random.randn(3, 8).astype(np.float32)
        result = comp.compress_batch(data)
        assert result.shape == (3, 4)

    def test_forward_with_reconstruction(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        emb = torch.randn(2, 8)
        compressed, recon = comp(emb, return_reconstruction=True)
        assert recon is not None
        assert compressed.shape == (2, 4)

    def test_forward_without_reconstruction(self):
        import torch
        from src.scar_compressor import LearnableSemanticCompressor as LearnableCompressor

        comp = LearnableCompressor(input_dim=8, compressed_dim=4)
        emb = torch.randn(2, 8)
        compressed, recon = comp(emb, return_reconstruction=False)
        assert recon is None


class TestSCAREnhancedCompressor:
    """Cover SCAR compressor alignment and compression paths."""

    def test_alignment_with_projection(self):
        """Cover lines 200-232 - alignment with projection enabled."""
        pytest.importorskip("torch")
        import torch
        from src.scar_compressor import SemanticAlignmentModule

        module = SemanticAlignmentModule(embedding_dim=16)
        source = torch.randn(16)
        target = torch.randn(16)
        loss, metrics = module(source, target, use_projection=True)
        assert "cosine_similarity_projected" in metrics
        assert "l2_distance" in metrics

    def test_alignment_without_projection(self):
        """Cover lines 200-232 - alignment without projection."""
        pytest.importorskip("torch")
        import torch
        from src.scar_compressor import SemanticAlignmentModule

        module = SemanticAlignmentModule(embedding_dim=16)
        source = torch.randn(16)
        target = torch.randn(16)
        loss, metrics = module(source, target, use_projection=False)
        assert metrics["cosine_similarity_projected"] == metrics["cosine_similarity_original"]

    def test_compute_alignment_score_with_projection(self):
        """Cover line 264."""
        pytest.importorskip("torch")
        from src.scar_compressor import SemanticAlignmentModule

        module = SemanticAlignmentModule(embedding_dim=16)
        sources = np.random.rand(5, 16).astype(np.float32)
        query = np.random.rand(16).astype(np.float32)
        scores = module.compute_alignment_score(sources, query, use_projection=True)
        assert scores.shape == (5,)

    def test_compress_embeddings_disabled(self):
        """Cover line 349 - compression disabled."""
        pytest.importorskip("torch")
        import torch
        from src.scar_compressor import SCAREnhancedCompressor

        mock_compressor = MagicMock()
        mock_compressor.model.get_sentence_embedding_dimension.return_value = 16
        comp = SCAREnhancedCompressor(
            base_compressor=mock_compressor,
            use_learnable_compression=False,
            use_alignment_guidance=False,
        )
        embeddings = torch.randn(5, 16)
        result = comp.compress_embeddings(embeddings)
        assert torch.equal(result, embeddings)

    def test_search_with_alignment_skips_file(self):
        """Cover line 390 - skip nodes from other files."""
        pytest.importorskip("torch")
        from src.scar_compressor import SCAREnhancedCompressor

        mock_compressor = MagicMock()
        mock_compressor.model.get_sentence_embedding_dimension.return_value = 16
        mock_compressor.model.encode.return_value = [np.random.rand(16)]
        node = MagicMock()
        node.embedding = np.random.rand(16)
        mock_compressor.chunks = {"other_n0": node}

        comp = SCAREnhancedCompressor(
            base_compressor=mock_compressor,
            use_learnable_compression=False,
            use_alignment_guidance=False,
        )

        with patch("sklearn.metrics.pairwise.cosine_similarity", return_value=[[0.9]]):
            results = comp.search_with_alignment("test", file_id="doc1", top_k=5)
            assert len(results) == 0

    def test_adaptive_modulate_fidelity_levels(self):
        """Cover lines 460-461, 466-467."""
        pytest.importorskip("torch")
        from src.scar_compressor import SCAREnhancedCompressor

        mock_compressor = MagicMock()
        mock_compressor.model.get_sentence_embedding_dimension.return_value = 16
        mock_compressor.model.encode.return_value = [np.random.rand(16)]
        mock_compressor.modulate_region.return_value = "content"

        node1 = MagicMock()
        node1.embedding = np.random.rand(16)
        node2 = MagicMock()
        node2.embedding = np.random.rand(16)
        mock_compressor.chunks = {"doc_n0": node1, "doc_n1": node2}

        comp = SCAREnhancedCompressor(
            base_compressor=mock_compressor,
            use_learnable_compression=False,
            use_alignment_guidance=False,
        )

        with patch.object(
            comp, "search_with_alignment", return_value=[("doc_n0", 0.9), ("doc_n1", 0.3)]
        ):
            result = comp.adaptive_modulate("test query", file_id="doc")
            assert "SCAR ADAPTIVE" in result
