"""experimental handlers — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
import sys
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


class TestExperimentalHandlers:
    """Cover experimental handler error/edge paths."""

    @pytest.mark.asyncio
    async def test_toon_encode_exception(self):
        """Cover lines 137-139 - TOON encode exception."""
        from src.handlers.experimental_handlers import handle_toon_encode

        ctx = _make_mock_context()
        with patch(
            "src.handlers.experimental_handlers._get_toon_serializer", side_effect=Exception("boom")
        ):
            result = await handle_toon_encode(ctx, {"data": {"key": "value"}})
            parsed = json.loads(result)
            assert "error" in parsed
            assert parsed["experimental"] is True

    @pytest.mark.asyncio
    async def test_toon_decode_exception(self):
        """Cover lines 190-192 - TOON decode exception."""
        from src.handlers.experimental_handlers import handle_toon_decode

        ctx = _make_mock_context()
        # Force exception during parsing
        result = await handle_toon_decode(ctx, {"toon_input": None})
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_toon_decode_with_continuation_line(self):
        """Cover line 170 - parsing lines with colon but no dash."""
        from src.handlers.experimental_handlers import handle_toon_decode

        ctx = _make_mock_context()
        toon = "- item1\nkey: value\n- item2"
        result = await handle_toon_decode(ctx, {"toon_input": toon})
        parsed = json.loads(result)
        assert "data" in parsed

    @pytest.mark.asyncio
    async def test_scar_compress_no_embeddings(self):
        """Cover line 253 - chunk doesn't match doc_id prefix."""
        from src.handlers.experimental_handlers import handle_scar_compress

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        ctx["compressor"].chunks = {"other_n0": MagicMock()}
        result = await handle_scar_compress(ctx, {"doc_id": "doc1"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "No embeddings" in parsed["error"]

    @pytest.mark.asyncio
    async def test_scar_compress_import_error(self):
        """Cover line 300 - SCAR import error."""
        from src.handlers.experimental_handlers import handle_scar_compress

        ctx = _make_mock_context()
        ctx["compressor"].graphs = {"doc1": MagicMock()}
        node = MagicMock()
        node.embedding = np.random.rand(384).tolist()
        ctx["compressor"].chunks = {"doc1_n0": node}
        with patch(
            "src.handlers.experimental_handlers._get_scar_compressor",
            side_effect=ImportError("no torch"),
        ):
            result = await handle_scar_compress(ctx, {"doc_id": "doc1"})
            parsed = json.loads(result)
            assert "error" in parsed

    @pytest.mark.asyncio
    async def test_scar_get_stats_no_pytorch(self):
        """Cover lines 327-328 - PyTorch not available path."""
        from src.handlers.experimental_handlers import handle_scar_get_stats

        ctx = _make_mock_context()
        with patch.dict(sys.modules, {"torch": None}):
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args: (
                    (_ for _ in ()).throw(ImportError())
                    if name == "torch"
                    else __import__(name, *args)
                ),
            ):
                result = await handle_scar_get_stats(ctx, {})
                parsed = json.loads(result)
                assert parsed["pytorch_available"] is False

    @pytest.mark.asyncio
    async def test_scar_get_stats_exception(self):
        """Cover lines 350-352 - outer exception."""
        from src.handlers.experimental_handlers import handle_scar_get_stats

        ctx = _make_mock_context()
        with patch(
            "src.handlers.experimental_handlers.json.dumps",
            side_effect=[Exception("boom"), '{"error": "x"}'],
        ):
            await handle_scar_get_stats(ctx, {})

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _has_pillow(), reason="Pillow not installed")
    async def test_multimodal_ingest_with_images(self):
        """Cover lines 456-457 - image paths added."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        ctx = _make_mock_context()
        # Make path_validator.validate return the path string (not a MagicMock)
        ctx["path_validator"].validate = MagicMock(side_effect=lambda p: p)
        mock_compressor = MagicMock()
        mock_compressor.ingest_mixed_content.return_value = {"node_count": 3}
        with patch(
            "src.handlers.experimental_handlers._get_multimodal_compressor",
            return_value=mock_compressor,
        ):
            with patch("os.path.exists", return_value=True):
                result = await handle_multimodal_ingest(
                    ctx,
                    {
                        "doc_id": "mixed1",
                        "text_content": "hello",
                        "code_content": "def f(): pass",
                        "image_paths": ["/tmp/img.png"],
                    },
                )
                parsed = json.loads(result)
                assert "image" in parsed["content_types"]

    @pytest.mark.asyncio
    async def test_multimodal_ingest_exception(self):
        """Cover lines 476-478 - generic exception."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        ctx = _make_mock_context()
        with patch(
            "src.handlers.experimental_handlers._get_multimodal_compressor",
            side_effect=Exception("fail"),
        ):
            result = await handle_multimodal_ingest(
                ctx,
                {
                    "doc_id": "doc1",
                    "text_content": "hello",
                },
            )
            parsed = json.loads(result)
            assert "error" in parsed

    @pytest.mark.asyncio
    async def test_get_multimodal_compressor_lazy(self):
        """Cover lines 55-57 - lazy import."""
        from src.handlers.experimental_handlers import _get_multimodal_compressor

        with patch(
            "src.handlers.experimental_handlers.MultiModalCompressor", create=True
        ) as mock_cls:
            with patch.dict(
                sys.modules, {"src.multimodal_compressor": MagicMock(MultiModalCompressor=mock_cls)}
            ):
                with patch("src.multimodal_compressor.MultiModalCompressor", mock_cls, create=True):
                    _get_multimodal_compressor(use_clip=True)

    @pytest.mark.asyncio
    async def test_lazy_import_helpers(self):
        """Cover lines 488-490, 495-497 - lazy imports."""
        from src.handlers.experimental_handlers import _get_compression_verifier

        with patch("src.compression_verifier.CompressionVerifier", create=True) as mock_cv:
            with patch.dict(
                sys.modules, {"src.compression_verifier": MagicMock(CompressionVerifier=mock_cv)}
            ):
                _get_compression_verifier()

    @pytest.mark.asyncio
    async def test_evidence_stats_exception(self):
        """Cover lines 672-674 - evidence stats exception."""
        from src.handlers.experimental_handlers import handle_get_evidence_stats

        ctx = _make_mock_context()
        with patch(
            "src.handlers.experimental_handlers._get_evidence_store", side_effect=Exception("boom")
        ):
            result = await handle_get_evidence_stats(ctx, {})
            parsed = json.loads(result)
            assert "error" in parsed
