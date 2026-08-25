"""batch manager — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock, patch
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


class TestBatchManager:
    def test_progress_zero_total(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=0, completed=0, successful=0, failed=0)
        assert p.progress_percentage == 0.0

    def test_progress_zero_completed(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=5, completed=0, successful=0, failed=0)
        assert p.success_rate == 0.0

    def test_progress_str(self):
        from src.batch_manager import BatchProgress

        p = BatchProgress(total=10, completed=5, successful=3, failed=2)
        s = str(p)
        assert "50.0%" in s

    def test_progress_callback_error(self):
        from src.batch_manager import BatchProgressTracker

        tracker = BatchProgressTracker(
            total=2,
            on_progress=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Should not raise
        tracker.update(True, "doc1")

    @pytest.mark.asyncio
    async def test_batch_ingest_from_files_missing_file(self, tmp_path):
        from src.batch_manager import batch_ingest_from_files

        compressor = MagicMock()
        compressor.ingest_text = MagicMock(return_value=MagicMock())

        results = await batch_ingest_from_files(compressor, [str(tmp_path / "nonexistent.txt")])
        assert results == []  # empty list since no valid docs

    @pytest.mark.asyncio
    async def test_batch_ingest_from_files_read_error(self, tmp_path):
        from src.batch_manager import batch_ingest_from_files

        f = tmp_path / "test.txt"
        f.write_text("content")
        compressor = MagicMock()

        with patch("builtins.open", side_effect=IOError("read error")):
            results = await batch_ingest_from_files(compressor, [str(f)])
        assert results == []
