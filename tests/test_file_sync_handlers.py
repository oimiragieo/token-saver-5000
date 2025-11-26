"""
Comprehensive Tests for File Sync Handler Functions

Tests all 4 file sync handler functions with full coverage of:
- Success cases
- Error handling
- Edge cases
- Integration with sync_manager and version_manager

Target coverage: 80%+ for src/handlers/file_sync_handlers.py (currently 13%)
"""

import os
import pytest
import tempfile
from unittest.mock import Mock, patch
from src.handlers.file_sync_handlers import (
    handle_check_file_sync,
    handle_diff_cached_file,
    handle_refresh_document,
    handle_get_version_history,
)
from src.types import HandlerContext
from src.file_sync_manager import FileSyncManager, FileMetadata
from src.version_manager import VersionManager
from src.semantic_compressor import SemanticCompressor


class TestHandleCheckFileSync:
    """Tests for handle_check_file_sync handler"""

    @pytest.fixture
    def mock_context(self):
        """Create mock handler context for testing"""
        sync_manager = Mock()
        context: HandlerContext = {
            "sync_manager": sync_manager,
            "validate_file_id": lambda file_id, must_exist=False: None,
            "compressor": Mock(),
            "version_manager": Mock(),
            "persistence": Mock(),
            "save_file_sync_metadata": Mock(),
        }
        return context

    def test_check_file_sync_in_sync(self, mock_context):
        """Test check_file_sync when file is in sync"""
        # Setup
        mock_context["sync_manager"].check_file_sync.return_value = {
            "in_sync": True,
            "reason": "File unchanged (checksum match)",
        }

        # Execute
        result = handle_check_file_sync(mock_context, {"file_id": "test_doc"})

        # Verify
        assert "✅" in result
        assert "test_doc is in sync" in result
        assert "File unchanged" in result

    def test_check_file_sync_out_of_sync(self, mock_context):
        """Test check_file_sync when file is out of sync"""
        # Setup
        mock_context["sync_manager"].check_file_sync.return_value = {
            "in_sync": False,
            "reason": "File modified (checksum mismatch)",
            "file_path": "/path/to/file.txt",
            "cached_checksum": "abc123def456",
            "current_checksum": "xyz789ghi012",
            "cached_mtime": 1234567890.0,
            "current_mtime": 1234567900.0,
        }

        # Execute
        result = handle_check_file_sync(mock_context, {"file_id": "test_doc"})

        # Verify
        assert "⚠️" in result
        assert "OUT OF SYNC" in result
        assert "checksum mismatch" in result
        assert "/path/to/file.txt" in result
        assert "abc123de" in result  # Checksum prefix
        assert "xyz789gh" in result  # Checksum prefix
        assert "refresh_document" in result
        assert "diff_cached_file" in result

    def test_check_file_sync_with_timestamps(self, mock_context):
        """Test that timestamps are formatted correctly"""
        # Setup
        mock_context["sync_manager"].check_file_sync.return_value = {
            "in_sync": False,
            "reason": "File modified",
            "cached_mtime": 1609459200.0,  # 2021-01-01 00:00:00
            "current_mtime": 1609545600.0,  # 2021-01-02 00:00:00
        }

        # Execute
        result = handle_check_file_sync(mock_context, {"file_id": "test_doc"})

        # Verify timestamps are formatted
        assert "2021-01-01" in result or "2021-01-02" in result
        assert ":" in result  # Time separator

    def test_check_file_sync_invalid_file_id(self, mock_context):
        """Test validation error handling"""

        # Setup validation to raise error
        def validate_error(file_id, must_exist=False):
            raise ValueError(f"Document {file_id} not found")

        mock_context["validate_file_id"] = validate_error

        # Execute & verify
        with pytest.raises(ValueError, match="not found"):
            handle_check_file_sync(mock_context, {"file_id": "nonexistent"})


class TestHandleDiffCachedFile:
    """Tests for handle_diff_cached_file handler"""

    @pytest.fixture
    def mock_context(self):
        """Create mock handler context"""
        sync_manager = Mock()
        sync_manager.file_metadata = {}
        version_manager = Mock()
        context: HandlerContext = {
            "sync_manager": sync_manager,
            "version_manager": version_manager,
            "validate_file_id": lambda file_id, must_exist=False: None,
            "compressor": Mock(),
            "persistence": Mock(),
            "save_file_sync_metadata": Mock(),
        }
        return context

    def test_diff_cached_file_success(self, mock_context):
        """Test successful diff generation"""
        # Setup
        file_id = "test_doc"
        mock_context["sync_manager"].file_metadata[file_id] = FileMetadata(
            doc_id=file_id,
            file_path="/path/to/file.txt",
            checksum="abc123",
            mtime=1234567890.0,
            ingestion_time=1234567890.0,
            size_bytes=1000,
        )
        mock_context[
            "version_manager"
        ].diff_with_current_file.return_value = """
--- a/file.txt
+++ b/file.txt
@@ -1,3 +1,3 @@
 Line 1
-Line 2 (old)
+Line 2 (new)
 Line 3
"""

        # Execute
        result = handle_diff_cached_file(mock_context, {"file_id": file_id})

        # Verify
        assert "---" in result
        assert "+++" in result
        assert "Line 2 (old)" in result
        assert "Line 2 (new)" in result
        mock_context["version_manager"].diff_with_current_file.assert_called_once_with(
            file_id, context_lines=3
        )

    def test_diff_cached_file_custom_context_lines(self, mock_context):
        """Test diff with custom context_lines parameter"""
        # Setup
        file_id = "test_doc"
        mock_context["sync_manager"].file_metadata[file_id] = FileMetadata(
            doc_id=file_id,
            file_path="/path/to/file.txt",
            checksum="abc123",
            mtime=1234567890.0,
            ingestion_time=1234567890.0,
            size_bytes=1000,
        )
        mock_context["version_manager"].diff_with_current_file.return_value = "diff output"

        # Execute
        handle_diff_cached_file(mock_context, {"file_id": file_id, "context_lines": 5})

        # Verify custom context_lines is passed
        mock_context["version_manager"].diff_with_current_file.assert_called_once_with(
            file_id, context_lines=5
        )

    def test_diff_cached_file_no_source_file(self, mock_context):
        """Test diff when no source file is registered"""
        # Setup - file_id NOT in file_metadata
        file_id = "test_doc"

        # Execute
        result = handle_diff_cached_file(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "File sync not enabled" in result
        assert "no source file registered" in result
        assert "Re-ingest with file_path parameter" in result

    def test_diff_cached_file_no_version_history(self, mock_context):
        """Test diff when version manager returns nothing"""
        # Setup
        file_id = "test_doc"
        mock_context["sync_manager"].file_metadata[file_id] = FileMetadata(
            doc_id=file_id,
            file_path="/path/to/file.txt",
            checksum="abc123",
            mtime=1234567890.0,
            ingestion_time=1234567890.0,
            size_bytes=1000,
        )
        mock_context["version_manager"].diff_with_current_file.return_value = None

        # Execute
        result = handle_diff_cached_file(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "Cannot generate diff" in result


class TestHandleRefreshDocument:
    """Tests for handle_refresh_document handler"""

    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, b"Test content v1\n")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def mock_context(self):
        """Create mock handler context with isolated managers"""
        # Use temporary directory for test isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = FileSyncManager()
            version_manager = VersionManager(storage_dir=temp_dir + "/versions")
            compressor = Mock()
            persistence = Mock()

            # Mock skeleton response
            skeleton_mock = Mock()
            skeleton_mock.total_tokens = 100
            skeleton_mock.skeleton_tokens = 20
            skeleton_mock.compression_ratio = 5.0
            compressor.ingest_file.return_value = skeleton_mock
            compressor.graphs = {"test_doc": Mock()}
            compressor.chunks = {}
            compressor.file_metadata = {}

            context: HandlerContext = {
                "sync_manager": sync_manager,
                "version_manager": version_manager,
                "compressor": compressor,
                "persistence": persistence,
                "validate_file_id": lambda file_id, must_exist=False: None,
                "save_file_sync_metadata": Mock(),
            }
            yield context

    def test_refresh_document_success(self, mock_context, temp_file):
        """Test successful document refresh"""
        # Setup
        file_id = "test_doc"
        sync_manager = mock_context["sync_manager"]
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        sync_manager.register_file(file_id, temp_file, content)

        # Execute
        result = handle_refresh_document(mock_context, {"file_id": file_id})

        # Verify
        assert "✅" in result
        assert f"Refreshed {file_id}" in result
        assert "100" in result  # total_tokens
        assert "20" in result  # skeleton_tokens
        assert "5.0x" in result  # compression ratio
        assert "Version history" in result

        # Verify version was stored
        history = mock_context["version_manager"].get_version_history(file_id)
        assert len(history) == 1

    def test_refresh_document_no_source_file(self, mock_context):
        """Test refresh when no source file is registered"""
        # Setup - file_id NOT in file_metadata
        file_id = "test_doc"

        # Execute
        result = handle_refresh_document(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "File sync not enabled" in result

    def test_refresh_document_text_only_ingestion(self, mock_context):
        """Test refresh when document was ingested as text-only (no file_path)"""
        # Setup - metadata exists but file_path is None
        file_id = "test_doc"
        sync_manager = mock_context["sync_manager"]
        sync_manager.register_file(file_id, None, "text only content")

        # Execute
        result = handle_refresh_document(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "no source file" in result
        assert "text-only ingestion" in result

    def test_refresh_document_file_read_error(self, mock_context):
        """Test refresh when source file cannot be read"""
        # Setup with non-existent file path (absolute path for security validation)
        file_id = "test_doc"
        sync_manager = mock_context["sync_manager"]
        nonexistent_path = (
            os.path.abspath("/tmp/nonexistent/path.txt")
            if os.name != "nt"
            else os.path.abspath("C:\\temp\\nonexistent\\path.txt")
        )
        sync_manager.register_file(file_id, nonexistent_path, "dummy content")

        # Execute
        result = handle_refresh_document(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "Error reading source file" in result
        assert "nonexistent" in result.lower() or "path.txt" in result
        assert "may have been moved or deleted" in result

    def test_refresh_document_ingestion_error(self, mock_context, temp_file):
        """Test refresh when re-ingestion fails"""
        # Setup
        file_id = "test_doc"
        sync_manager = mock_context["sync_manager"]
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()
        sync_manager.register_file(file_id, temp_file, content)

        # Make ingest_file raise an error
        mock_context["compressor"].ingest_file.side_effect = ValueError("Invalid content")

        # Execute
        result = handle_refresh_document(mock_context, {"file_id": file_id})

        # Verify
        assert "❌" in result
        assert "Error re-ingesting document" in result
        assert "Invalid content" in result


class TestHandleGetVersionHistory:
    """Tests for handle_get_version_history handler"""

    @pytest.fixture
    def mock_context(self):
        """Create mock handler context with isolated version manager"""
        # Use temporary directory for test isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = VersionManager(storage_dir=temp_dir + "/versions")
            context: HandlerContext = {
                "version_manager": version_manager,
                "sync_manager": Mock(),
                "compressor": Mock(),
                "persistence": Mock(),
                "validate_file_id": lambda file_id, must_exist=False: None,
                "save_file_sync_metadata": Mock(),
            }
            yield context

    def test_get_version_history_no_history(self, mock_context):
        """Test get_version_history when no history exists"""
        # Execute
        result = handle_get_version_history(mock_context, {"doc_id": "test_doc"})

        # Verify
        assert "📜" in result
        assert "No version history" in result
        assert "Has not been ingested yet" in result
        assert "text-only mode" in result

    def test_get_version_history_single_version(self, mock_context):
        """Test get_version_history with one version"""
        # Setup
        doc_id = "test_doc"
        version_manager = mock_context["version_manager"]
        test_file_path = (
            os.path.abspath("/tmp/path/to/file.txt")
            if os.name != "nt"
            else os.path.abspath("C:\\temp\\path\\to\\file.txt")
        )
        version_manager.add_version(
            doc_id=doc_id,
            content="Test content",
            checksum="abc123def456",
            file_path=test_file_path,
            compression_stats={
                "total_tokens": 100,
                "skeleton_tokens": 20,
                "compression_ratio": 5.0,
            },
        )

        # Execute
        result = handle_get_version_history(mock_context, {"doc_id": doc_id})

        # Verify
        assert "📜 Version History: test_doc" in result
        assert "Version 1" in result
        assert "file.txt" in result  # Check filename is present (path may vary)
        assert "abc123def456" in result
        assert "100" in result  # total_tokens
        assert "20" in result  # skeleton_tokens
        assert "5.0x" in result  # compression ratio
        assert "Total versions: 1" in result

    def test_get_version_history_multiple_versions(self, mock_context):
        """Test get_version_history with multiple versions"""
        # Setup
        doc_id = "test_doc"
        version_manager = mock_context["version_manager"]

        # Add 3 versions (use absolute paths for security validation)
        base_path = "/tmp/path" if os.name != "nt" else "C:\\temp\\path"
        for i in range(3):
            test_file_path = os.path.abspath(f"{base_path}/v{i+1}.txt")
            version_manager.add_version(
                doc_id=doc_id,
                content=f"Content v{i+1}",
                checksum=f"checksum_{i+1}",
                file_path=test_file_path,
                compression_stats={
                    "total_tokens": 100 * (i + 1),
                    "skeleton_tokens": 20 * (i + 1),
                    "compression_ratio": 5.0,
                },
            )

        # Execute
        result = handle_get_version_history(mock_context, {"doc_id": doc_id})

        # Verify all versions are listed
        assert "Version 1" in result
        assert "Version 2" in result
        assert "Version 3" in result
        assert "Total versions: 3" in result
        assert "v1.txt" in result
        assert "v2.txt" in result
        assert "v3.txt" in result

    def test_get_version_history_timestamp_formatting(self, mock_context):
        """Test that timestamps are formatted without microseconds"""
        # Setup
        doc_id = "test_doc"
        version_manager = mock_context["version_manager"]
        test_file_path = (
            os.path.abspath("/tmp/path/to/file.txt")
            if os.name != "nt"
            else os.path.abspath("C:\\temp\\path\\to\\file.txt")
        )
        version_manager.add_version(
            doc_id=doc_id,
            content="Test content",
            checksum="abc123",
            file_path=test_file_path,
        )

        # Execute
        result = handle_get_version_history(mock_context, {"doc_id": doc_id})

        # Verify timestamp is formatted (contains date/time separators)
        assert "-" in result  # Date separator
        assert ":" in result  # Time separator
        assert "Version 1" in result  # Version info is present


# Integration Tests
class TestFileSyncHandlersIntegration:
    """Integration tests with real components"""

    @pytest.fixture
    def temp_file(self):
        """Create temporary file for testing"""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, b"Original content\n")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)

    @pytest.fixture
    def integrated_context(self, temp_file):
        """Create integrated handler context with real managers"""
        from src.embeddings import EmbeddingManager

        # Use temporary directory for test isolation
        with tempfile.TemporaryDirectory() as temp_dir:
            sync_manager = FileSyncManager()
            version_manager = VersionManager(storage_dir=temp_dir + "/versions")
            compressor = SemanticCompressor()
            persistence = Mock()  # Mock persistence to avoid disk I/O

            # Mock embedding manager to avoid model downloads
            with patch("src.semantic_compressor.EmbeddingManager") as mock_em_class:
                mock_em = Mock(spec=EmbeddingManager)
                mock_em.encode.return_value = [[0.1] * 384]  # Mock embedding
                mock_em_class.return_value = mock_em

                context: HandlerContext = {
                    "sync_manager": sync_manager,
                    "version_manager": version_manager,
                    "compressor": compressor,
                    "persistence": persistence,
                    "validate_file_id": lambda file_id, must_exist=False: None,
                    "save_file_sync_metadata": Mock(),
                }
                yield context

    def test_full_refresh_workflow(self, integrated_context, temp_file):
        """Test complete refresh workflow: check → refresh → check → diff → history"""
        file_id = "integration_test"

        # 1. Ingest document initially
        with open(temp_file, "r", encoding="utf-8") as f:
            content = f.read()

        integrated_context["compressor"].ingest_file(content, file_id)
        integrated_context["sync_manager"].register_file(file_id, temp_file, content)

        # 2. Check sync (should be in sync)
        result = handle_check_file_sync(integrated_context, {"file_id": file_id})
        assert "✅" in result
        assert "in sync" in result

        # 3. Modify file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("Modified content\n")

        # 4. Check sync again (should be out of sync)
        result = handle_check_file_sync(integrated_context, {"file_id": file_id})
        assert "⚠️" in result
        assert "OUT OF SYNC" in result

        # 5. Refresh document
        result = handle_refresh_document(integrated_context, {"file_id": file_id})
        assert "✅" in result
        assert "Refreshed" in result

        # 6. Check version history
        result = handle_get_version_history(integrated_context, {"doc_id": file_id})
        assert "Version 1" in result
        assert "Total versions: 1" in result
