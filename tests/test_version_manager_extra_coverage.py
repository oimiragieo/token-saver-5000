"""version manager — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import os
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


class TestVersionManager:
    def _make_vm(self, tmp_path):
        from src.version_manager import VersionManager

        return VersionManager(storage_dir=str(tmp_path / "versions"))

    def test_add_version_relative_path_raises(self, tmp_path):
        vm = self._make_vm(tmp_path)
        with pytest.raises(ValueError, match="Security violation"):
            vm.add_version("doc1", "content", "abc123", file_path="relative/path.txt")

    def test_get_latest_content_none(self, tmp_path):
        vm = self._make_vm(tmp_path)
        assert vm.get_latest_content("nonexistent") is None

    def test_get_latest_content_exists(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "hello", "abc789", file_path=None)
        assert vm.get_latest_content("doc1") == "hello"

    def test_diff_no_cached_version(self, tmp_path):
        vm = self._make_vm(tmp_path)
        result = vm.diff_with_current_file("doc1")
        assert "No cached versions" in result

    def test_diff_specified_version_not_found(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        result = vm.diff_with_current_file("doc1", cached_version_id=999)
        assert "not found" in result

    def test_diff_no_file_path(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        result = vm.diff_with_current_file("doc1")
        assert "No source file path" in result

    def test_diff_no_content(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "", "empty123", file_path=os.path.abspath(__file__))
        result = vm.diff_with_current_file("doc1")
        assert "empty" in result

    def test_diff_file_not_on_disk(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=str(tmp_path / "missing.txt"))
        result = vm.diff_with_current_file("doc1")
        assert "not found" in result

    def test_diff_permission_error(self, tmp_path):
        vm = self._make_vm(tmp_path)
        f = tmp_path / "perm.txt"
        f.write_text("original")
        vm.add_version("doc1", "content", "abc123", file_path=str(f))

        with patch("builtins.open", side_effect=PermissionError("denied")):
            result = vm.diff_with_current_file("doc1")
        assert "Permission denied" in result

    def test_diff_unicode_error(self, tmp_path):
        vm = self._make_vm(tmp_path)
        f = tmp_path / "enc.txt"
        f.write_text("original")
        vm.add_version("doc1", "content", "abc123", file_path=str(f))

        with patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")):
            result = vm.diff_with_current_file("doc1")
        assert "Encoding error" in result

    @pytest.mark.asyncio
    async def test_delete_versions_async(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "content", "abc123", file_path=None)
        await vm.delete_versions_async("doc1")
        assert "doc1" not in vm.versions

    def test_save_all_versions_no_doc(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm._save_all_versions("nonexistent")  # should not raise

    def test_load_index_corrupt_file(self, tmp_path):
        vm = self._make_vm(tmp_path)
        bad_file = vm.storage_dir / "bad.json"
        bad_file.write_text("NOT JSON")
        vm._load_index()  # should not raise

    def test_get_stats(self, tmp_path):
        vm = self._make_vm(tmp_path)
        vm.add_version("doc1", "hello world", "abc456", file_path=None)
        stats = vm.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_versions"] == 1
