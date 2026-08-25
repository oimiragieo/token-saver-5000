"""multimodal compressor — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
import sys


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


class TestMultimodalCompressor:
    def test_encode_image_no_encoder(self):
        from src.multimodal_compressor import MultiModalCompressor

        mc = MultiModalCompressor.__new__(MultiModalCompressor)
        mc.image_encoder = None
        result = mc._encode_image(b"fake_image_data")
        assert result is None

    @pytest.mark.skipif(not _has_pillow(), reason="Pillow not installed")
    def test_encode_image_exception(self):
        from src.multimodal_compressor import MultiModalCompressor

        mc = MultiModalCompressor.__new__(MultiModalCompressor)
        mc.image_encoder = MagicMock()

        with patch("PIL.Image.open", side_effect=Exception("bad image")):
            result = mc._encode_image(b"bad data")
        assert result is None


class TestMultimodalCompressor_boost4:
    """Cover multimodal compressor edge cases."""

    def test_encode_image_no_encoder(self):
        """Cover line 147-148 - image encoder not available."""
        from src.multimodal_compressor import MultiModalCompressor

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_mgr.get_text_embedder.return_value = MagicMock(
                get_sentence_embedding_dimension=lambda: 384
            )
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor(use_clip_for_images=False)
            result = comp._encode_image(b"fake_image_data")
            assert result is None

    def test_encode_image_exception(self):
        """Cover lines 149-150 - image encoding exception."""
        from src.multimodal_compressor import MultiModalCompressor

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_encoder = MagicMock(get_sentence_embedding_dimension=lambda: 384)
            mock_mgr.get_text_embedder.return_value = mock_encoder
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor(use_clip_for_images=False)
            comp.image_encoder = MagicMock()
            with patch("src.multimodal_compressor.Image", create=True) as mock_pil:
                mock_pil.open.side_effect = Exception("bad image")
                with patch.dict(
                    sys.modules,
                    {
                        "PIL": MagicMock(),
                        "PIL.Image": MagicMock(open=MagicMock(side_effect=Exception("bad"))),
                    },
                ):
                    comp._encode_image(b"bad_data")
                    # Should return None on error

    def test_clip_load_exception(self):
        """Cover lines 105-110 - CLIP load failure."""
        from src.multimodal_compressor import MultiModalCompressor

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_mgr.get_text_embedder.return_value = MagicMock(
                get_sentence_embedding_dimension=lambda: 384
            )
            mock_mgr.get_image_embedder.side_effect = Exception("CLIP not available")
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor(use_clip_for_images=True)
            assert comp.image_encoder is None

    def test_codebert_encoder(self):
        """Cover line 98 - use_codebert_for_code."""
        from src.multimodal_compressor import MultiModalCompressor

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_mgr.get_text_embedder.return_value = MagicMock(
                get_sentence_embedding_dimension=lambda: 384
            )
            mock_mgr.get_code_embedder.return_value = MagicMock()
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor(use_codebert_for_code=True)
            assert comp.code_encoder == mock_mgr.get_code_embedder.return_value

    def test_generate_summary_with_code_and_image_nodes(self):
        """Cover lines 411, 425, 430-440 - summary with code, image nodes."""
        import networkx as nx
        from src.multimodal_compressor import MultiModalCompressor, ModalityType, MultiModalNode

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_mgr.get_text_embedder.return_value = MagicMock(
                get_sentence_embedding_dimension=lambda: 384
            )
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor()

            graph = nx.Graph()
            # Add text nodes
            for i in range(4):
                nid = f"proj_text_{i}"
                graph.add_node(nid)
                comp.nodes[nid] = MultiModalNode(
                    node_id=nid,
                    content=f"text content {i}",
                    modality=ModalityType.TEXT,
                    embedding=np.random.rand(384),
                    importance=0.5,
                    metadata={},
                )
            # Add code nodes
            for i in range(4):
                nid = f"proj_code_{i}"
                graph.add_node(nid)
                comp.nodes[nid] = MultiModalNode(
                    node_id=nid,
                    content=f"def func_{i}(): pass",
                    modality=ModalityType.CODE,
                    embedding=np.random.rand(384),
                    importance=0.6,
                    metadata={"file": f"file_{i}.py"},
                )
            # Add image nodes
            for i in range(4):
                nid = f"proj_img_{i}"
                graph.add_node(nid)
                comp.nodes[nid] = MultiModalNode(
                    node_id=nid,
                    content="x" * 2048,
                    modality=ModalityType.IMAGE,
                    embedding=np.random.rand(384),
                    importance=0.4,
                    metadata={"file": f"img_{i}.png"},
                )
            # Add cross-modal edge
            graph.add_edge("proj_text_0", "proj_code_0", connection_type="cross_modal", weight=0.7)
            comp.graphs["proj"] = graph

            result = comp.generate_multimodal_summary("proj")
            assert "CODE" in result
            assert "IMAGE" in result

    def test_unknown_query_type(self):
        """Cover line 333 - unknown query type in search."""
        from src.multimodal_compressor import MultiModalCompressor

        with patch("src.multimodal_compressor.EmbeddingManager") as mock_em:
            mock_mgr = MagicMock()
            mock_encoder = MagicMock(get_sentence_embedding_dimension=lambda: 384)
            mock_encoder.encode.return_value = [np.random.rand(384)]
            mock_mgr.get_text_embedder.return_value = mock_encoder
            mock_em.return_value = mock_mgr
            comp = MultiModalCompressor()
            comp.nodes = {}
            result = comp.search_cross_modal("hello", query_type="unknown")
            assert result == []
