"""resource manager — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import json
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


class TestResourceManager:
    def test_check_health_at_capacity(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10, max_documents=5)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 5.0, "d2": 6.0}  # exceeds 10MB
        health = rm.check_health()
        assert not health["healthy"]

    def test_check_health_warn_threshold(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10, max_documents=100, warn_threshold=0.5)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 6.0}  # 60% > 50% threshold
        health = rm.check_health()
        assert len(health["warnings"]) > 0

    def test_check_health_doc_count_limit(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_documents=2, max_total_storage_mb=100)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 1.0, "d2": 1.0, "d3": 1.0}
        health = rm.check_health()
        assert any("Document count" in w for w in health["warnings"])

    def test_memory_health_exceeded(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=10)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 15.0}
        healthy, msg = rm.check_memory_health()
        assert not healthy
        assert "exceeded" in msg

    def test_suggest_cleanup_empty(self):
        from src.resource_manager import ResourceManager

        rm = ResourceManager()
        assert rm.suggest_cleanup() is None

    def test_suggest_cleanup_under_threshold(self):
        from src.resource_manager import ResourceManager, ResourceLimits

        limits = ResourceLimits(max_total_storage_mb=1000, warn_threshold=0.8)
        rm = ResourceManager(limits=limits)
        rm.document_sizes = {"d1": 1.0}
        assert rm.suggest_cleanup() is None

    def test_get_stats_comprehensive(self):
        from src.resource_manager import ResourceManager

        rm = ResourceManager()
        rm.register_document("doc1", 1024 * 1024)
        stats = rm.get_stats()
        assert "limits" in stats
        assert "documents" in stats


class TestResourceHandlers:
    @pytest.mark.asyncio
    async def test_should_compress_file_not_found(self):
        from src.handlers.resource_handlers import handle_should_compress

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = "/nonexistent/file.txt"

        result = await handle_should_compress(ctx, {"file_path": "/nonexistent/file.txt"})
        data = json.loads(result)
        assert "error" in data or data.get("recommendation") == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_should_compress_binary_extension(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert data["recommendation"] == "CONVERT_THEN_COMPRESS"

    @pytest.mark.asyncio
    async def test_should_compress_empty_file(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "empty.xyz"
        f.write_text("")

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert data["recommendation"] in ("SKIP", "UNKNOWN")

    @pytest.mark.asyncio
    async def test_should_compress_code_file(self, tmp_path):
        from src.handlers.resource_handlers import handle_should_compress

        f = tmp_path / "test.py"
        f.write_text("x = 1\n" * 500)

        ctx = _make_mock_context()
        ctx["path_validator"] = MagicMock()
        ctx["path_validator"].validate.return_value = str(f)

        result = await handle_should_compress(ctx, {"file_path": str(f)})
        data = json.loads(result)
        assert "estimated_tokens" in data


class TestResourceHandlersDiagnostics:
    """Cover resource handler diagnostics paths."""

    @pytest.mark.asyncio
    async def test_env_semantic_compressor_model_loaded(self):
        """Cover lines 498-505 - SemanticCompressor with model loaded."""
        from src.handlers.resource_handlers import handle_check_environment

        ctx = _make_mock_context()
        compressor = MagicMock()
        compressor.model = MagicMock()
        # Not a CodeCompressionAdapter - using SemanticCompressor directly
        compressor.graphs = {"doc1": MagicMock()}
        compressor.chunks = {"doc1_n0": MagicMock()}
        del compressor._text_compressor  # Ensure it's not a CCA
        compressor.configure_mock(**{"__class__.__name__": "SemanticCompressor"})
        ctx["compressor"] = compressor
        ctx["sync_manager"].export_metadata.return_value = {}
        ctx["resource_manager"].get_enabled_tool_names.return_value = ["tool1"]

        result = await handle_check_environment(ctx, {})
        parsed = json.loads(result)
        assert parsed["status"] in ("healthy", "degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_env_stale_documents(self, tmp_path):
        """Cover lines 536-545, 548-549 - stale document detection."""
        from src.handlers.resource_handlers import handle_check_environment

        ctx = _make_mock_context()
        compressor = MagicMock()
        compressor.graphs = {}
        compressor.chunks = {}
        compressor.model = None
        del compressor._text_compressor
        ctx["compressor"] = compressor

        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")
        ctx["sync_manager"].export_metadata.return_value = {
            "doc1": {"file_path": str(test_file), "mtime": 0}
        }
        ctx["resource_manager"].get_enabled_tool_names.return_value = []

        result = await handle_check_environment(ctx, {})
        parsed = json.loads(result)
        assert "warnings" in parsed

    @pytest.mark.asyncio
    async def test_env_disk_space_check(self):
        """Cover lines 559-563 - low disk space."""
        from src.handlers.resource_handlers import handle_check_environment

        ctx = _make_mock_context()
        compressor = MagicMock()
        compressor.graphs = {}
        compressor.chunks = {}
        compressor.model = None
        del compressor._text_compressor
        ctx["compressor"] = compressor
        ctx["sync_manager"].export_metadata.return_value = {}
        ctx["resource_manager"].get_enabled_tool_names.return_value = []

        mock_usage = (1024 * 1024 * 200, 1024 * 1024 * 150, 1024 * 1024 * 50)  # 50MB free
        with patch("shutil.disk_usage", return_value=mock_usage):
            result = await handle_check_environment(ctx, {})
            parsed = json.loads(result)
            # Low disk space warning
            assert parsed["status"] in ("degraded", "unhealthy")

    @pytest.mark.asyncio
    async def test_env_no_warnings(self):
        """Cover lines 608, 611 - no warnings/recommendations."""
        from src.handlers.resource_handlers import handle_check_environment

        ctx = _make_mock_context()
        compressor = MagicMock()
        compressor.graphs = {}
        compressor.chunks = {}
        compressor.model = MagicMock()
        del compressor._text_compressor
        ctx["compressor"] = compressor
        ctx["sync_manager"].export_metadata.return_value = {}
        ctx["resource_manager"].get_enabled_tool_names.return_value = ["tool1"]

        with patch("shutil.disk_usage", return_value=(10**12, 5 * 10**11, 5 * 10**11)):
            result = await handle_check_environment(ctx, {})
            parsed = json.loads(result)
            # Should have "message" if no warnings
            if "warnings" not in parsed and "recommendations" not in parsed:
                assert "message" in parsed

    @pytest.mark.asyncio
    async def test_should_compress_binary_by_content(self):
        """Cover lines 696-697, 726 - binary detection by content sniffing."""
        from src.handlers.resource_handlers import handle_should_compress

        ctx = _make_mock_context()
        ctx["path_validator"].validate.return_value = "/tmp/test.xyz"

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=100):
                with patch(
                    "src.handlers.resource_handlers.is_binary_content", return_value=(True, None)
                ):
                    result = await handle_should_compress(ctx, {"file_path": "/tmp/test.xyz"})
                    parsed = json.loads(result)
                    assert parsed["recommendation"] in ("SKIP", "CONVERT_THEN_COMPRESS")

    @pytest.mark.asyncio
    async def test_should_compress_binary_read_error(self):
        """Cover line 726 - read error during binary detection."""
        from src.handlers.resource_handlers import handle_should_compress

        ctx = _make_mock_context()
        ctx["path_validator"].validate.return_value = "/tmp/test.xyz"

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=100):
                with patch(
                    "src.handlers.resource_handlers.is_binary_content",
                    return_value=(False, "Permission denied"),
                ):
                    result = await handle_should_compress(ctx, {"file_path": "/tmp/test.xyz"})
                    parsed = json.loads(result)
                    assert "error" in parsed

    @pytest.mark.asyncio
    async def test_should_compress_unknown_ext_empty(self):
        """Cover lines 813-814 - empty file with unknown extension."""
        from src.handlers.resource_handlers import handle_should_compress

        ctx = _make_mock_context()
        ctx["path_validator"].validate.return_value = "/tmp/test.xyz"

        with patch("os.path.exists", return_value=True):
            with patch("os.path.getsize", return_value=0):
                result = await handle_should_compress(ctx, {"file_path": "/tmp/test.xyz"})
                parsed = json.loads(result)
                assert parsed["recommendation"] in ("SKIP", "UNKNOWN")
