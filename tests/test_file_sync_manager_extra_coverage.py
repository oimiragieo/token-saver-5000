"""file sync manager — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock
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


class TestFileSyncManager:
    def test_register_file_relative_path_raises(self):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()
        with pytest.raises(ValueError, match="Security violation"):
            fsm.register_file("doc1", "content", "relative/path.txt")

    def test_get_file_diff_no_metadata(self):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()
        assert fsm.get_file_diff("nonexistent") is None

    def test_get_file_diff_no_version_manager(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        result = fsm.get_file_diff("doc1")
        assert "WARN" in result

    def test_get_file_diff_with_version_manager(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        vm = MagicMock()
        vm.diff_with_current_file.return_value = "no changes"
        result = fsm.get_file_diff("doc1", version_manager=vm)
        assert result == "no changes"

    def test_get_file_diff_exception(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        vm = MagicMock()
        vm.diff_with_current_file.side_effect = Exception("boom")
        result = fsm.get_file_diff("doc1", version_manager=vm)
        assert result is None

    def test_check_sync_mtime_changed_content_same(self, tmp_path):
        from src.file_sync_manager import FileSyncManager

        fsm = FileSyncManager()

        f = tmp_path / "test.txt"
        f.write_text("content")
        fsm.register_file("doc1", str(f), "content")

        # Simulate mtime change
        fsm.file_metadata["doc1"].mtime = 0

        result = fsm.check_file_sync("doc1")
        assert result["in_sync"] is True
