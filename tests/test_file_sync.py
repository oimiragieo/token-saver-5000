"""
Test File Sync and Version Management Systems

Tests for:
- FileSyncManager: Track file changes and detect staleness
- VersionManager: Maintain version history and diffs
"""

import os
import tempfile
import threading
import time

import pytest

from src.file_sync_manager import FileSyncManager
from src.version_manager import VersionManager


class TestFileSyncManager:
    """Test FileSyncManager functionality"""

    def setup_method(self):
        """Initialize sync manager for each test"""
        self.sync_manager = FileSyncManager()

    def test_register_file_with_path(self):
        """Test registering a file with source path"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Test content for sync")
            temp_path = f.name

        try:
            # Register file
            metadata = self.sync_manager.register_file(
                doc_id="test_doc", file_path=temp_path, content="Test content for sync"
            )

            # Verify metadata
            assert metadata.doc_id == "test_doc"
            assert metadata.file_path == temp_path
            assert metadata.checksum is not None
            assert metadata.mtime is not None
            assert metadata.size_bytes > 0

            # Verify it's in the manager
            assert "test_doc" in self.sync_manager.file_metadata

        finally:
            os.unlink(temp_path)

    def test_register_file_text_only(self):
        """Test registering text without source file"""
        metadata = self.sync_manager.register_file(
            doc_id="text_only", file_path=None, content="Just text, no file"
        )

        assert metadata.doc_id == "text_only"
        assert metadata.file_path is None
        assert metadata.checksum is not None
        assert metadata.mtime is None
        assert metadata.size_bytes == len("Just text, no file".encode("utf-8"))

    def test_check_file_sync_unchanged(self):
        """Test sync check for unchanged file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original content")
            temp_path = f.name

        try:
            self.sync_manager.register_file("unchanged_doc", temp_path, "Original content")

            # Check sync status
            status = self.sync_manager.check_file_sync("unchanged_doc")

            assert status["in_sync"] is True
            assert "unchanged" in status["reason"].lower()
            assert status["has_source_file"] is True

        finally:
            os.unlink(temp_path)

    def test_check_file_sync_modified(self):
        """Test sync check for modified file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original content")
            temp_path = f.name

        try:
            # Register original
            self.sync_manager.register_file("modified_doc", temp_path, "Original content")

            # Wait a bit to ensure mtime changes
            time.sleep(0.1)

            # Modify file
            with open(temp_path, "w") as f:
                f.write("Modified content")

            # Check sync status
            status = self.sync_manager.check_file_sync("modified_doc")

            assert status["in_sync"] is False
            assert "changed" in status["reason"].lower()
            assert status["has_source_file"] is True
            assert "current_checksum" in status
            assert "cached_checksum" in status
            assert status["current_checksum"] != status["cached_checksum"]

        finally:
            os.unlink(temp_path)

    def test_check_file_sync_deleted(self):
        """Test sync check for deleted file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Content")
            temp_path = f.name

        # Register file
        self.sync_manager.register_file("deleted_doc", temp_path, "Content")

        # Delete file
        os.unlink(temp_path)

        # Check sync status
        status = self.sync_manager.check_file_sync("deleted_doc")

        assert status["in_sync"] is False
        assert "deleted" in status["reason"].lower()
        assert status["has_source_file"] is False

    def test_check_file_sync_text_only(self):
        """Test sync check for text-only document"""
        self.sync_manager.register_file("text_doc", None, "Text content")

        status = self.sync_manager.check_file_sync("text_doc")

        assert status["in_sync"] is True
        assert "text-only" in status["reason"].lower()
        assert status["has_source_file"] is False

    def test_check_file_sync_unregistered(self):
        """Test sync check for unregistered document"""
        status = self.sync_manager.check_file_sync("nonexistent")

        assert status["in_sync"] is False
        assert "not registered" in status["reason"].lower()
        assert status["has_source_file"] is False

    def test_get_stale_documents(self):
        """Test getting list of stale documents"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f1:
            f1.write("Content 1")
            temp1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f2:
            f2.write("Content 2")
            temp2 = f2.name

        try:
            # Register two files
            self.sync_manager.register_file("doc1", temp1, "Content 1")
            self.sync_manager.register_file("doc2", temp2, "Content 2")

            # Modify one
            time.sleep(0.1)
            with open(temp1, "w") as f:
                f.write("Modified content 1")

            # Get stale docs
            stale = self.sync_manager.get_stale_documents()

            assert "doc1" in stale
            assert "doc2" not in stale

        finally:
            os.unlink(temp1)
            os.unlink(temp2)

    def test_get_sync_summary(self):
        """Test getting sync summary for all documents"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Content")
            temp_path = f.name

        try:
            # Register files with different states
            self.sync_manager.register_file("file_doc", temp_path, "Content")
            self.sync_manager.register_file("text_doc", None, "Text only")

            # Modify the file
            time.sleep(0.1)
            with open(temp_path, "w") as f:
                f.write("Modified")

            summary = self.sync_manager.get_sync_summary()

            assert summary["total_documents"] == 2
            assert summary["out_of_sync"] == 1
            assert summary["no_source_file"] == 1
            assert len(summary["details"]) == 2

        finally:
            os.unlink(temp_path)

    def test_update_metadata(self):
        """Test updating metadata after re-ingestion"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original")
            temp_path = f.name

        try:
            # Initial registration
            meta1 = self.sync_manager.register_file("update_doc", temp_path, "Original")
            checksum1 = meta1.checksum

            # Update metadata with new content
            time.sleep(0.1)
            self.sync_manager.update_metadata("update_doc", temp_path, "Updated")

            # Verify updated
            meta2 = self.sync_manager.file_metadata["update_doc"]
            assert meta2.checksum != checksum1

        finally:
            os.unlink(temp_path)

    def test_remove_metadata(self):
        """Test removing metadata"""
        self.sync_manager.register_file("remove_doc", None, "Content")

        assert "remove_doc" in self.sync_manager.file_metadata

        self.sync_manager.remove_metadata("remove_doc")

        assert "remove_doc" not in self.sync_manager.file_metadata

    def test_export_import_metadata(self):
        """Test exporting and importing metadata"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Content")
            temp_path = f.name

        try:
            # Register some files
            self.sync_manager.register_file("doc1", temp_path, "Content")
            self.sync_manager.register_file("doc2", None, "Text")

            # Export
            exported = self.sync_manager.export_metadata()

            assert len(exported) == 2
            assert "doc1" in exported
            assert "doc2" in exported

            # Create new manager and import
            new_manager = FileSyncManager()
            new_manager.import_metadata(exported)

            assert len(new_manager.file_metadata) == 2
            assert new_manager.file_metadata["doc1"].checksum == exported["doc1"]["checksum"]

        finally:
            os.unlink(temp_path)

    def test_checksum_calculation(self):
        """Test MD5 checksum calculation"""
        content = "Test content for checksum"
        checksum1 = self.sync_manager._calculate_checksum(content)

        # Same content should give same checksum
        checksum2 = self.sync_manager._calculate_checksum(content)
        assert checksum1 == checksum2

        # Different content should give different checksum
        checksum3 = self.sync_manager._calculate_checksum("Different content")
        assert checksum1 != checksum3

    def test_lru_eviction_on_max_entries(self):
        """Test automatic LRU eviction when max_entries exceeded (v0.4.2)"""
        # Create sync manager with max_entries=5
        sync = FileSyncManager(max_entries=5)

        # Register 7 files
        for i in range(7):
            time.sleep(0.01)  # Ensure different ingestion times
            sync.register_file(f"doc{i}", None, f"Content {i}")

        # Should only have last 5 entries (2 evicted)
        assert len(sync.file_metadata) == 5

        # First 2 docs should be evicted (oldest ingestion times)
        assert "doc0" not in sync.file_metadata
        assert "doc1" not in sync.file_metadata

        # Last 5 should remain
        for i in range(2, 7):
            assert f"doc{i}" in sync.file_metadata

    def test_lru_eviction_unlimited(self):
        """Test that max_entries=0 disables LRU eviction (v0.4.2)"""
        sync = FileSyncManager(max_entries=0)

        # Register many files
        for i in range(20):
            sync.register_file(f"doc{i}", None, f"Content {i}")

        # All 20 should be kept (no eviction)
        assert len(sync.file_metadata) == 20

    def test_lru_eviction_stats(self):
        """Test that get_stats shows LRU eviction info (v0.4.2)"""
        sync = FileSyncManager(max_entries=10)

        # Register 8 files (below limit)
        for i in range(8):
            sync.register_file(f"doc{i}", None, f"Content {i}")

        stats = sync.get_stats()

        assert stats["total_entries"] == 8
        assert stats["max_entries_limit"] == 10
        assert stats["approaching_limit"] is False  # 8/10 = 80% < 90%

        # Add 2 more to reach 100% of limit
        sync.register_file("doc8", None, "Content 8")
        sync.register_file("doc9", None, "Content 9")

        stats = sync.get_stats()
        assert stats["total_entries"] == 10
        assert stats["approaching_limit"] is True  # 10/10 = 100% >= 90%

    def test_lru_eviction_preserves_newest(self):
        """Test that LRU eviction keeps newest entries (v0.4.2)"""
        sync = FileSyncManager(max_entries=3)

        # Register 5 files with distinguishable ingestion times
        for i in range(5):
            time.sleep(0.01)
            sync.register_file(f"doc{i}", None, f"Content {i}")

        # Should have only last 3 (newest)
        assert len(sync.file_metadata) == 3
        assert "doc2" in sync.file_metadata
        assert "doc3" in sync.file_metadata
        assert "doc4" in sync.file_metadata

    def test_get_stats_empty(self):
        """Test get_stats on empty sync manager (v0.4.2)"""
        sync = FileSyncManager(max_entries=100)
        stats = sync.get_stats()

        assert stats["total_entries"] == 0
        assert stats["max_entries_limit"] == 100
        assert stats["oldest_entry_time"] is None
        assert stats["newest_entry_time"] is None
        assert stats["approaching_limit"] is False

    def test_get_stats_with_entries(self):
        """Test get_stats with file entries (v0.4.2)"""
        sync = FileSyncManager(max_entries=100)

        # Register files with different characteristics
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("A" * 1000)  # 1KB file
            temp_path = f.name

        try:
            # File with source path
            sync.register_file("doc1", temp_path, "A" * 1000)
            # Text-only
            sync.register_file("doc2", None, "B" * 500)

            stats = sync.get_stats()

            assert stats["total_entries"] == 2
            assert stats["entries_with_source_file"] == 1
            assert stats["text_only_entries"] == 1
            assert stats["total_size_mb"] > 0
            assert stats["oldest_entry_time"] is not None
            assert stats["newest_entry_time"] is not None

        finally:
            os.unlink(temp_path)


class TestVersionManager:
    """Test VersionManager functionality"""

    def setup_method(self):
        """Initialize version manager with temp storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.version_manager = VersionManager(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up temp storage"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_add_version(self):
        """Test adding a new version"""
        version = self.version_manager.add_version(
            doc_id="test_doc",
            content="First version content",
            checksum="abc123",
            file_path="/path/to/file.txt",
            metadata={"author": "test"},
            compression_stats={"ratio": 5.0},
        )

        assert version.version_id == 1
        assert version.doc_id == "test_doc"
        assert version.content == "First version content"
        assert version.checksum == "abc123"
        assert version.file_path == "/path/to/file.txt"

    def test_add_multiple_versions(self):
        """Test adding multiple versions of same document"""
        v1 = self.version_manager.add_version("doc", "Version 1", "check1")
        v2 = self.version_manager.add_version("doc", "Version 2", "check2")
        v3 = self.version_manager.add_version("doc", "Version 3", "check3")

        assert v1.version_id == 1
        assert v2.version_id == 2
        assert v3.version_id == 3
        assert len(self.version_manager.versions["doc"]) == 3

    def test_get_version(self):
        """Test getting specific version"""
        self.version_manager.add_version("doc", "V1", "c1")
        self.version_manager.add_version("doc", "V2", "c2")
        self.version_manager.add_version("doc", "V3", "c3")

        v2 = self.version_manager.get_version("doc", 2)

        assert v2 is not None
        assert v2.version_id == 2
        assert v2.content == "V2"

    def test_get_version_invalid(self):
        """Test getting invalid version"""
        self.version_manager.add_version("doc", "V1", "c1")

        # Non-existent doc
        assert self.version_manager.get_version("nonexistent", 1) is None

        # Invalid version number
        assert self.version_manager.get_version("doc", 99) is None
        assert self.version_manager.get_version("doc", 0) is None

    def test_get_latest_version(self):
        """Test getting latest version"""
        self.version_manager.add_version("doc", "V1", "c1")
        self.version_manager.add_version("doc", "V2", "c2")
        self.version_manager.add_version("doc", "V3", "c3")

        latest = self.version_manager.get_latest_version("doc")

        assert latest is not None
        assert latest.version_id == 3
        assert latest.content == "V3"

    def test_get_version_history(self):
        """Test getting version history summary"""
        self.version_manager.add_version("doc", "V1" * 100, "c1")
        self.version_manager.add_version("doc", "V2" * 200, "c2")

        history = self.version_manager.get_version_history("doc")

        assert len(history) == 2
        assert history[0]["version_id"] == 1
        assert history[1]["version_id"] == 2
        assert history[0]["checksum"] == "c1"
        assert "content_length" in history[0]
        assert "timestamp" in history[0]

    def test_diff_versions(self):
        """Test diffing between versions"""
        content1 = "Line 1\nLine 2\nLine 3\n"
        content2 = "Line 1\nModified Line 2\nLine 3\n"

        self.version_manager.add_version("doc", content1, "c1")
        self.version_manager.add_version("doc", content2, "c2")

        diff = self.version_manager.diff_versions("doc", 1, 2)

        assert diff is not None
        assert "-Line 2" in diff
        assert "+Modified Line 2" in diff

    def test_diff_versions_invalid(self):
        """Test diffing with invalid versions"""
        self.version_manager.add_version("doc", "V1", "c1")

        # Invalid version numbers
        assert self.version_manager.diff_versions("doc", 1, 99) is None
        assert self.version_manager.diff_versions("nonexistent", 1, 2) is None

    def test_diff_with_current_file(self):
        """Test diffing cached version with current file"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original content")
            temp_path = f.name

        try:
            # Add version with original content
            self.version_manager.add_version("doc", "Original content", "c1", file_path=temp_path)

            # Modify file on disk
            with open(temp_path, "w") as f:
                f.write("Modified content on disk")

            # Get diff
            diff = self.version_manager.diff_with_current_file("doc")

            assert diff is not None
            assert "-Original content" in diff
            assert "+Modified content on disk" in diff

        finally:
            os.unlink(temp_path)

    def test_diff_with_current_file_no_changes(self):
        """Test diff when file hasn't changed"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Same content")
            temp_path = f.name

        try:
            self.version_manager.add_version("doc", "Same content", "c1", file_path=temp_path)

            diff = self.version_manager.diff_with_current_file("doc")

            assert diff is not None
            assert "No differences" in diff

        finally:
            os.unlink(temp_path)

    def test_get_all_documents(self):
        """Test getting all document IDs"""
        self.version_manager.add_version("doc1", "Content", "c1")
        self.version_manager.add_version("doc2", "Content", "c2")
        self.version_manager.add_version("doc3", "Content", "c3")

        all_docs = self.version_manager.get_all_documents()

        assert len(all_docs) == 3
        assert "doc1" in all_docs
        assert "doc2" in all_docs
        assert "doc3" in all_docs

    def test_delete_versions(self):
        """Test deleting all versions of a document"""
        self.version_manager.add_version("doc", "V1", "c1")
        self.version_manager.add_version("doc", "V2", "c2")

        assert "doc" in self.version_manager.versions

        self.version_manager.delete_versions("doc")

        assert "doc" not in self.version_manager.versions

    def test_get_stats(self):
        """Test getting storage statistics"""
        self.version_manager.add_version("doc1", "A" * 1000, "c1")
        self.version_manager.add_version("doc1", "B" * 2000, "c2")
        self.version_manager.add_version("doc2", "C" * 500, "c3")

        stats = self.version_manager.get_stats()

        assert stats["total_documents"] == 2
        assert stats["total_versions"] == 3
        assert stats["total_size_mb"] > 0
        assert stats["avg_versions_per_doc"] == 1.5

    def test_persistence_save_and_load(self):
        """Test that versions persist to disk and reload"""
        # Add versions
        self.version_manager.add_version("doc1", "Content 1", "c1")
        self.version_manager.add_version("doc1", "Content 2", "c2")

        # Create new manager with same storage dir
        new_manager = VersionManager(storage_dir=self.temp_dir)

        # Should load versions from disk
        assert "doc1" in new_manager.versions
        assert len(new_manager.versions["doc1"]) == 2

        latest = new_manager.get_latest_version("doc1")
        assert latest.content == "Content 2"

    def test_automatic_pruning_on_add(self):
        """Test automatic pruning when adding versions beyond max_versions (v0.4.2)"""
        # Create version manager with max_versions=5
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=5)

        # Add 8 versions
        for i in range(8):
            vm.add_version("doc", f"Version {i}", f"checksum{i}")

        # Should only have last 5 versions (3 were auto-pruned)
        assert len(vm.versions["doc"]) == 5

        # First version should be version 4 (versions 0-2 pruned)
        first_version = vm.versions["doc"][0]
        assert first_version.version_id == 4
        assert first_version.content == "Version 3"

        # Last version should be version 8
        last_version = vm.versions["doc"][-1]
        assert last_version.version_id == 8
        assert last_version.content == "Version 7"

    def test_automatic_pruning_persists_to_disk(self):
        """Test that automatic pruning removes versions from disk (v0.4.2)"""
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=3)

        # Add 5 versions
        for i in range(5):
            vm.add_version("test_doc", f"Content {i}", f"check{i}")

        # Should only have 3 versions in memory
        assert len(vm.versions["test_doc"]) == 3

        # Create new manager to verify disk state
        vm2 = VersionManager(storage_dir=self.temp_dir)

        # Should load only 3 versions from disk (not 5)
        assert len(vm2.versions["test_doc"]) == 3
        assert vm2.versions["test_doc"][0].version_id == 3
        assert vm2.versions["test_doc"][-1].version_id == 5

    def test_manual_pruning_single_document(self):
        """Test manual pruning for a single document (v0.4.2)"""
        # Use unlimited initially to accumulate versions, then manually prune
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=0)

        # Add 15 versions (no auto-pruning)
        for i in range(15):
            vm.add_version("doc1", f"V{i}", f"c{i}")

        # Should have all 15 versions
        assert len(vm.versions["doc1"]) == 15

        # Change limit and manually prune
        vm.max_versions = 10
        pruned = vm.prune_old_versions("doc1")

        assert pruned["doc1"] == 5  # 5 versions pruned
        assert len(vm.versions["doc1"]) == 10
        assert vm.versions["doc1"][0].version_id == 6
        assert vm.versions["doc1"][-1].version_id == 15

    def test_manual_pruning_all_documents(self):
        """Test manual pruning for all documents (v0.4.2)"""
        # Use unlimited initially to accumulate versions
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=0)

        # Add versions to multiple docs (no auto-pruning)
        for i in range(10):
            vm.add_version("doc1", f"V{i}", f"c{i}")
        for i in range(7):
            vm.add_version("doc2", f"V{i}", f"c{i}")
        for i in range(3):
            vm.add_version("doc3", f"V{i}", f"c{i}")

        # Change limit and prune all docs
        vm.max_versions = 5
        pruned = vm.prune_old_versions()

        assert pruned["doc1"] == 5  # 10 → 5
        assert pruned["doc2"] == 2  # 7 → 5
        assert "doc3" not in pruned  # 3 < 5, no pruning needed

        assert len(vm.versions["doc1"]) == 5
        assert len(vm.versions["doc2"]) == 5
        assert len(vm.versions["doc3"]) == 3

    def test_pruning_with_unlimited_max_versions(self):
        """Test that max_versions=0 disables pruning (v0.4.2)"""
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=0)

        # Add many versions
        for i in range(20):
            vm.add_version("doc", f"V{i}", f"c{i}")

        # All 20 should be kept (no pruning)
        assert len(vm.versions["doc"]) == 20

        # Manual prune should do nothing
        pruned = vm.prune_old_versions()
        assert len(pruned) == 0

    def test_pruning_stats(self):
        """Test that get_stats shows pruning info (v0.4.2)"""
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=5)

        # Add versions that exceed limit
        for i in range(10):
            vm.add_version("doc1", "A" * 1000, f"c{i}")

        # Add versions within limit
        for i in range(3):
            vm.add_version("doc2", "B" * 500, f"c{i}")

        stats = vm.get_stats()

        assert stats["max_versions_limit"] == 5
        assert stats["docs_needing_pruning"] == 0  # Already auto-pruned
        assert stats["potential_savings_mb"] == 0  # Already pruned

    def test_pruning_memory_efficiency(self):
        """Test that pruning actually reduces memory usage (v0.4.2)"""
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=10)

        # Add 50 versions with significant content
        for i in range(50):
            content = f"Version {i} " * 100  # ~1.3KB per version
            vm.add_version("doc", content, f"checksum{i}")

        # After auto-pruning, should have only 10 versions
        assert len(vm.versions["doc"]) == 10

        # Calculate size of kept versions
        total_size = sum(len(v.content.encode("utf-8")) for v in vm.versions["doc"])

        # Should be roughly 10 versions × 1.3KB = ~13KB
        # (much less than 50 × 1.3KB = ~65KB)
        assert total_size < 20000  # 20KB threshold

    def test_pruning_preserves_version_ids(self):
        """Test that pruning preserves correct version_id numbering (v0.4.2)"""
        vm = VersionManager(storage_dir=self.temp_dir, max_versions=3)

        # Add 7 versions
        for i in range(7):
            vm.add_version("doc", f"Content {i}", f"check{i}")

        # Should have versions 5, 6, 7 (version_id, not indices)
        assert len(vm.versions["doc"]) == 3
        assert vm.versions["doc"][0].version_id == 5
        assert vm.versions["doc"][1].version_id == 6
        assert vm.versions["doc"][2].version_id == 7

        # Content should match version_id
        assert "Content 4" in vm.versions["doc"][0].content
        assert "Content 5" in vm.versions["doc"][1].content
        assert "Content 6" in vm.versions["doc"][2].content


class TestACEContextManager:
    """Tests for ACEContextManager LRU eviction (v0.4.2)"""

    def test_basic_add_and_get(self):
        """Test basic add and get operations"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=5)

        # Add contexts
        manager["ctx1"] = {"data": "value1"}
        manager["ctx2"] = {"data": "value2"}

        # Retrieve contexts
        assert manager["ctx1"]["data"] == "value1"
        assert manager["ctx2"]["data"] == "value2"

        # Check length
        assert len(manager) == 2

    def test_automatic_lru_eviction(self):
        """Test automatic LRU eviction when max_contexts exceeded"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=3)

        # Add 5 contexts
        for i in range(5):
            manager[f"ctx{i}"] = {"data": f"value{i}"}

        # Should only have last 3 contexts (ctx2, ctx3, ctx4)
        assert len(manager) == 3
        assert "ctx0" not in manager
        assert "ctx1" not in manager
        assert "ctx2" in manager
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_access_tracking_moves_to_end(self):
        """Test that accessing a context marks it as recently used"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=3)

        # Add 3 contexts
        manager["ctx1"] = {"data": "v1"}
        manager["ctx2"] = {"data": "v2"}
        manager["ctx3"] = {"data": "v3"}

        # Access ctx1 (should move to end)
        _ = manager["ctx1"]

        # Add ctx4 (should evict ctx2, not ctx1)
        manager["ctx4"] = {"data": "v4"}

        assert len(manager) == 3
        assert "ctx1" in manager  # Accessed recently, kept
        assert "ctx2" not in manager  # Oldest, evicted
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_update_existing_key_moves_to_end(self):
        """Test that updating an existing key moves it to end"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=3)

        # Add 3 contexts
        manager["ctx1"] = {"data": "v1"}
        manager["ctx2"] = {"data": "v2"}
        manager["ctx3"] = {"data": "v3"}

        # Update ctx1 (should move to end)
        manager["ctx1"] = {"data": "updated"}

        # Add ctx4 (should evict ctx2, not ctx1)
        manager["ctx4"] = {"data": "v4"}

        assert len(manager) == 3
        assert "ctx1" in manager  # Updated recently, kept
        assert manager["ctx1"]["data"] == "updated"
        assert "ctx2" not in manager  # Oldest, evicted
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_unlimited_mode(self):
        """Test that max_contexts=0 disables eviction"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=0)

        # Add many contexts
        for i in range(20):
            manager[f"ctx{i}"] = {"data": f"value{i}"}

        # All 20 should be kept (no eviction)
        assert len(manager) == 20
        assert "ctx0" in manager
        assert "ctx19" in manager

    def test_get_stats(self):
        """Test get_stats method"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=5)

        # Add 3 contexts
        manager["ctx1"] = {"data": "v1"}
        manager["ctx2"] = {"data": "v2"}
        manager["ctx3"] = {"data": "v3"}

        stats = manager.get_stats()

        assert stats["total_contexts"] == 3
        assert stats["max_contexts_limit"] == 5
        assert "ctx1" in stats["context_ids"]
        assert "ctx2" in stats["context_ids"]
        assert "ctx3" in stats["context_ids"]
        assert stats["approaching_limit"] is False  # 3/5 = 60% < 90%

    def test_approaching_limit_warning(self):
        """Test approaching_limit warning (> 90% capacity)"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=10)

        # Add 9 contexts (90% of limit - NOT approaching yet)
        for i in range(9):
            manager[f"ctx{i}"] = {"data": f"value{i}"}

        stats = manager.get_stats()
        assert stats["approaching_limit"] is False  # 9/10 = 90% (not > 90%)

        # Add one more context (100% - definitely approaching)
        manager["ctx10"] = {"data": "value10"}
        stats = manager.get_stats()
        assert stats["approaching_limit"] is True  # 10/10 = 100% (> 90%)

    def test_eviction_order(self):
        """Test that eviction happens in correct LRU order"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=5)

        # Add 5 contexts
        for i in range(5):
            manager[f"ctx{i}"] = {"order": i}

        # Access ctx0 and ctx2 (move them to end)
        _ = manager["ctx0"]
        _ = manager["ctx2"]

        # Add 3 more contexts (should evict ctx1, ctx3, ctx4 in that order)
        manager["ctx5"] = {"order": 5}
        manager["ctx6"] = {"order": 6}
        manager["ctx7"] = {"order": 7}

        # Should have: ctx0, ctx2, ctx5, ctx6, ctx7
        assert len(manager) == 5
        assert "ctx0" in manager  # Accessed
        assert "ctx1" not in manager  # Evicted first
        assert "ctx2" in manager  # Accessed
        assert "ctx3" not in manager  # Evicted second
        assert "ctx4" not in manager  # Evicted third
        assert "ctx5" in manager
        assert "ctx6" in manager
        assert "ctx7" in manager

    def test_stats_unlimited_mode(self):
        """Test get_stats with unlimited mode"""
        from src.server import ACEContextManager

        manager = ACEContextManager(max_contexts=0)

        # Add contexts
        for i in range(5):
            manager[f"ctx{i}"] = {"data": f"value{i}"}

        stats = manager.get_stats()

        assert stats["total_contexts"] == 5
        assert stats["max_contexts_limit"] == "unlimited"
        assert stats["approaching_limit"] is False  # No limit to approach


class TestIntegration:
    """Integration tests for FileSyncManager + VersionManager"""

    def setup_method(self):
        """Initialize both managers"""
        self.temp_dir = tempfile.mkdtemp()
        self.sync_manager = FileSyncManager()
        self.version_manager = VersionManager(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_workflow(self):
        """Test complete workflow: register, version, modify, detect, diff"""
        # Create a file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Original document content")
            temp_path = f.name

        try:
            # Step 1: Register file with sync manager
            sync_meta = self.sync_manager.register_file(
                "workflow_doc", temp_path, "Original document content"
            )

            # Step 2: Add version
            self.version_manager.add_version(
                "workflow_doc", "Original document content", sync_meta.checksum, temp_path
            )

            # Verify initial sync
            sync_status = self.sync_manager.check_file_sync("workflow_doc")
            assert sync_status["in_sync"] is True

            # Step 3: Modify file on disk
            time.sleep(0.1)
            with open(temp_path, "w") as f:
                f.write("Modified document content")

            # Step 4: Detect staleness
            sync_status = self.sync_manager.check_file_sync("workflow_doc")
            assert sync_status["in_sync"] is False

            # Step 5: Get diff using version manager
            diff = self.version_manager.diff_with_current_file("workflow_doc")
            assert "-Original document content" in diff
            assert "+Modified document content" in diff

            # Step 6: Re-ingest (simulate)
            new_meta = self.sync_manager.register_file(
                "workflow_doc", temp_path, "Modified document content"
            )
            self.version_manager.add_version(
                "workflow_doc", "Modified document content", new_meta.checksum, temp_path
            )

            # Verify sync restored
            sync_status = self.sync_manager.check_file_sync("workflow_doc")
            assert sync_status["in_sync"] is True

            # Verify version history
            history = self.version_manager.get_version_history("workflow_doc")
            assert len(history) == 2

            # Diff between versions
            version_diff = self.version_manager.diff_versions("workflow_doc", 1, 2)
            assert "-Original document content" in version_diff

        finally:
            os.unlink(temp_path)


class TestThreadSafety:
    """Test thread safety of FileSyncManager"""

    def setup_method(self):
        """Initialize sync manager for each test"""
        self.sync_manager = FileSyncManager()
        self.errors = []  # Track errors from threads

    def test_concurrent_register_file(self):
        """Test concurrent file registration doesn't cause race conditions"""

        def register_files(thread_id, count):
            """Register multiple files from a thread"""
            try:
                for i in range(count):
                    doc_id = f"thread{thread_id}_doc{i}"
                    content = f"Content from thread {thread_id}, document {i}"
                    self.sync_manager.register_file(doc_id, None, content)
            except Exception as e:
                self.errors.append(f"Thread {thread_id}: {e}")

        # Spawn 10 threads, each registering 20 files
        threads = []
        for thread_id in range(10):
            t = threading.Thread(target=register_files, args=(thread_id, 20))
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Should have no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"

        # Should have 200 documents registered (10 threads × 20 docs)
        assert len(self.sync_manager.file_metadata) == 200

    def test_concurrent_check_sync(self):
        """Test concurrent sync checking doesn't cause race conditions"""

        # Pre-register some files
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name
            f.write("Test content")

        try:
            for i in range(50):
                self.sync_manager.register_file(
                    f"doc{i}",
                    temp_path if i % 2 == 0 else None,
                    "Test content",
                )

            def check_sync_repeatedly(thread_id, iterations):
                """Check sync status repeatedly from a thread"""
                try:
                    for _ in range(iterations):
                        for i in range(50):
                            self.sync_manager.check_file_sync(f"doc{i}")
                except Exception as e:
                    self.errors.append(f"Thread {thread_id}: {e}")

            # Spawn 5 threads, each checking sync 100 times
            threads = []
            for thread_id in range(5):
                t = threading.Thread(target=check_sync_repeatedly, args=(thread_id, 100))
                threads.append(t)
                t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # Should have no errors
            assert len(self.errors) == 0, f"Errors occurred: {self.errors}"

        finally:
            os.unlink(temp_path)

    def test_concurrent_mixed_operations(self):
        """Test concurrent mixed operations (register, check, remove, export)"""

        def register_worker(thread_id):
            """Register files"""
            try:
                for i in range(10):
                    self.sync_manager.register_file(
                        f"register_{thread_id}_{i}", None, f"Content {i}"
                    )
            except Exception as e:
                self.errors.append(f"Register thread {thread_id}: {e}")

        def check_worker(thread_id):
            """Check file sync"""
            try:
                for i in range(10):
                    self.sync_manager.check_file_sync(f"check_doc_{i}")
            except Exception as e:
                self.errors.append(f"Check thread {thread_id}: {e}")

        def remove_worker(thread_id):
            """Remove metadata"""
            try:
                for i in range(10):
                    self.sync_manager.remove_metadata(f"remove_{thread_id}_{i}")
            except Exception as e:
                self.errors.append(f"Remove thread {thread_id}: {e}")

        def export_worker(thread_id):
            """Export metadata"""
            try:
                for _ in range(10):
                    self.sync_manager.export_metadata()
            except Exception as e:
                self.errors.append(f"Export thread {thread_id}: {e}")

        def get_summary_worker(thread_id):
            """Get sync summary"""
            try:
                for _ in range(10):
                    self.sync_manager.get_sync_summary()
            except Exception as e:
                self.errors.append(f"Summary thread {thread_id}: {e}")

        # Pre-populate some data
        for i in range(20):
            self.sync_manager.register_file(f"check_doc_{i}", None, f"Content {i}")

        # Spawn mixed threads
        threads = []

        # 3 register threads
        for i in range(3):
            t = threading.Thread(target=register_worker, args=(i,))
            threads.append(t)

        # 2 check threads
        for i in range(2):
            t = threading.Thread(target=check_worker, args=(i,))
            threads.append(t)

        # 2 remove threads
        for i in range(2):
            t = threading.Thread(target=remove_worker, args=(i,))
            threads.append(t)

        # 2 export threads
        for i in range(2):
            t = threading.Thread(target=export_worker, args=(i,))
            threads.append(t)

        # 2 summary threads
        for i in range(2):
            t = threading.Thread(target=get_summary_worker, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no errors
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"

    def test_concurrent_iteration_safety(self):
        """Test that iteration doesn't fail when dict is modified concurrently"""

        # Pre-register files
        for i in range(100):
            self.sync_manager.register_file(f"doc{i}", None, f"Content {i}")

        def modifier_thread():
            """Add and remove files while others iterate"""
            try:
                for i in range(50):
                    self.sync_manager.register_file(f"new_doc{i}", None, "New content")
                    self.sync_manager.remove_metadata(f"doc{i}")
            except Exception as e:
                self.errors.append(f"Modifier: {e}")

        def iterator_thread(thread_id):
            """Iterate over files"""
            try:
                for _ in range(20):
                    self.sync_manager.get_stale_documents()
                    self.sync_manager.get_sync_summary()
                    self.sync_manager.export_metadata()
            except Exception as e:
                self.errors.append(f"Iterator {thread_id}: {e}")

        threads = []

        # 1 modifier thread
        t = threading.Thread(target=modifier_thread)
        threads.append(t)

        # 5 iterator threads
        for i in range(5):
            t = threading.Thread(target=iterator_thread, args=(i,))
            threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Should have no errors (especially no "dictionary changed size during iteration")
        assert len(self.errors) == 0, f"Errors occurred: {self.errors}"


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
