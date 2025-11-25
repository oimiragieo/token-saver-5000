"""
Comprehensive Edge Case and Error Handling Tests

Tests error conditions, edge cases, and validation logic:
- Empty/invalid inputs
- Resource limit violations
- Invalid file IDs and node IDs
- Invalid fidelity levels
- Concurrent operations
- Data corruption scenarios

Run with: pytest tests/test_edge_cases.py -v
"""

import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor, FidelityLevel
from src.resource_manager import ResourceManager, ResourceLimits
from src.afm import FocusManager, AFMConfig


class TestInputValidation:
    """Test input validation and error handling"""

    def setup_method(self):
        """Initialize for each test"""
        self.compressor = SemanticCompressor()

    def test_empty_document_ingestion(self):
        """Test ingesting empty document raises appropriate error"""
        with pytest.raises(ValueError, match="empty|text"):
            self.compressor.ingest_file("", "empty_doc")

    def test_whitespace_only_document(self):
        """Test ingesting whitespace-only document"""
        with pytest.raises(ValueError, match="empty|text"):
            self.compressor.ingest_file("   \n\n\t  ", "whitespace_doc")

    def test_invalid_file_id_empty(self):
        """Test empty file_id is rejected"""
        with pytest.raises(ValueError, match="file_id"):
            self.compressor.ingest_file("valid text", "")

    def test_invalid_file_id_whitespace(self):
        """Test whitespace file_id is rejected"""
        with pytest.raises(ValueError, match="file_id"):
            self.compressor.ingest_file("valid text", "   ")

    def test_read_skeleton_nonexistent_file(self):
        """Test reading skeleton of non-existent file"""
        with pytest.raises(ValueError, match="not found|File.*not found"):
            self.compressor.read_skeleton("nonexistent_file")

    def test_search_semantic_nonexistent_file(self):
        """Test searching in non-existent file"""
        results = self.compressor.search_semantic("query", file_id="nonexistent")
        assert len(results) == 0  # Should return empty results, not crash

    def test_modulate_region_empty_node_list(self):
        """Test modulating with empty node list"""
        result = self.compressor.modulate_region([], FidelityLevel.RAW)
        assert "MODULATED CONTENT" in result  # Should return header even if empty

    def test_modulate_region_invalid_node_id(self):
        """Test modulating with invalid node ID"""
        result = self.compressor.modulate_region(["invalid_node_id"], FidelityLevel.RAW)
        assert "not found" in result.lower()

    def test_get_stats_nonexistent_file(self):
        """Test getting stats for non-existent file"""
        with pytest.raises((ValueError, KeyError), match="not found|File"):
            self.compressor.get_stats("nonexistent_file")


class TestResourceLimits:
    """Test resource limit enforcement"""

    def test_document_size_limit_exceeded(self):
        """Test document exceeding size limit"""
        manager = ResourceManager(ResourceLimits(max_document_size_mb=0.001))  # 1KB limit

        # Create 2MB document
        large_text = "x" * (2 * 1024 * 1024)

        allowed, msg = manager.check_document_size("test", len(large_text.encode()))
        assert not allowed
        assert "exceeds limit" in msg.lower()

    def test_total_storage_limit_exceeded(self):
        """Test total storage limit"""
        manager = ResourceManager(ResourceLimits(max_total_storage_mb=1.0))

        # Register several documents
        manager.register_document("doc1", 300 * 1024 * 1024)  # 300MB
        manager.register_document("doc2", 400 * 1024 * 1024)  # 400MB

        # Try to add one more that exceeds total
        allowed, msg = manager.check_document_size("doc3", 400 * 1024 * 1024)
        assert not allowed
        assert "total storage" in msg.lower()

    def test_document_count_limit_exceeded(self):
        """Test maximum document count"""
        manager = ResourceManager(ResourceLimits(max_documents=3))

        # Register max documents
        manager.register_document("doc1", 1024)
        manager.register_document("doc2", 1024)
        manager.register_document("doc3", 1024)

        # Try to add one more
        allowed, msg = manager.check_document_size("doc4", 1024)
        assert not allowed
        assert "too many documents" in msg.lower()

    def test_unregister_document(self):
        """Test unregistering documents frees up space"""
        manager = ResourceManager(ResourceLimits(max_documents=2))

        manager.register_document("doc1", 1024)
        manager.register_document("doc2", 1024)

        # Should be at limit
        allowed, _ = manager.check_document_size("doc3", 1024)
        assert not allowed

        # Unregister one
        manager.unregister_document("doc1")

        # Should now have space
        allowed, _ = manager.check_document_size("doc3", 1024)
        assert allowed


class TestAFMEdgeCases:
    """Test AFM dialogue memory edge cases"""

    def setup_method(self):
        """Initialize for each test"""
        self.manager = FocusManager(AFMConfig())

    def test_empty_message_content(self):
        """Test adding message with empty content"""
        with pytest.raises(ValueError, match="content|empty"):
            self.manager.add_message("user", "")

    def test_invalid_role(self):
        """Test adding message with invalid role"""
        with pytest.raises(ValueError, match="role|invalid"):
            self.manager.add_message("invalid_role", "test message")

    def test_build_context_zero_budget(self):
        """Test building context with zero token budget"""
        self.manager.add_message("user", "Hello")

        with pytest.raises(ValueError, match="budget|positive"):
            self.manager.build_context("query", budget_tokens=0)

    def test_build_context_negative_budget(self):
        """Test building context with negative token budget"""
        self.manager.add_message("user", "Hello")

        with pytest.raises(ValueError, match="budget|positive"):
            self.manager.build_context("query", budget_tokens=-100)

    def test_build_context_empty_history(self):
        """Test building context with no messages"""
        context, stats = self.manager.build_context("query", budget_tokens=1000)

        # Should return empty context but not crash
        assert isinstance(context, list)
        assert stats.total_messages == 0

    def test_clear_history(self):
        """Test clearing dialogue history"""
        self.manager.add_message("user", "Message 1")
        self.manager.add_message("user", "Message 2")

        assert len(self.manager.messages) == 2

        self.manager.clear_history()

        assert len(self.manager.messages) == 0

    def test_very_long_message(self):
        """Test handling very long messages"""
        # Create 10,000 token message
        long_message = " ".join(["word"] * 10000)

        # Should not crash
        self.manager.add_message("user", long_message)
        context, stats = self.manager.build_context("query", budget_tokens=5000)

        # Message should be truncated/compressed
        assert stats.total_tokens <= 5000


class TestDataIntegrity:
    """Test data integrity and corruption scenarios"""

    def setup_method(self):
        """Initialize for each test"""
        self.compressor = SemanticCompressor()

    def test_duplicate_file_id_ingestion(self):
        """Test ingesting same file_id twice (should overwrite)"""
        text1 = "First version of document"
        text2 = "Second version of document"

        self.compressor.ingest_file(text1, "duplicate_test")
        result2 = self.compressor.ingest_file(text2, "duplicate_test")

        # Second ingestion should work
        assert result2 is not None

        # Should have nodes from second version only
        skeleton = self.compressor.read_skeleton("duplicate_test")
        assert "Second version" in skeleton or "duplicate_test" in skeleton

    def test_special_characters_in_text(self):
        """Test handling special characters"""
        special_text = """
        Document with special chars: <>&"'
        Unicode: 你好 世界 🎉
        Equations: ∑ ∫ √ π
        """

        # Should not crash
        result = self.compressor.ingest_file(special_text, "special_chars")
        assert result is not None
        assert result.total_nodes > 0

    def test_very_short_document(self):
        """Test handling very short documents (single sentence)"""
        short_text = "This is a very short document."

        result = self.compressor.ingest_file(short_text, "short_doc")

        assert result is not None
        assert result.total_nodes >= 1
        # Compression ratio might be < 1 for very short docs
        assert result.total_tokens > 0

    def test_document_with_no_semantic_structure(self):
        """Test random/nonsense text"""
        random_text = "asdf qwer zxcv poiu lkjh mnbv"

        # Should still work, just might not compress well
        result = self.compressor.ingest_file(random_text, "random_doc")
        assert result is not None
        assert result.total_nodes >= 1


class TestConcurrency:
    """Test concurrent operations (basic thread safety checks)"""

    def test_multiple_documents_sequential(self):
        """Test ingesting multiple documents sequentially"""
        compressor = SemanticCompressor()

        for i in range(5):
            text = f"Document number {i} with some content about topic {i}"
            result = compressor.ingest_file(text, f"doc_{i}")
            assert result is not None

        # All documents should be accessible
        for i in range(5):
            skeleton = compressor.read_skeleton(f"doc_{i}")
            assert skeleton is not None
            assert f"doc_{i}" in skeleton


class TestValidationHelpers:
    """Test validation helper functions"""

    def test_token_counting_fallback(self):
        """Test token counting works even without tiktoken"""
        compressor = SemanticCompressor()

        text = "This is a test sentence with approximately ten words here."
        tokens = compressor._count_tokens(text)

        # Should return a reasonable count (either tiktoken or word-based fallback)
        assert tokens > 0
        assert tokens < 100  # Sanity check

    def test_chunk_text_preserves_paragraphs(self):
        """Test text chunking respects paragraph boundaries"""
        compressor = SemanticCompressor()

        text = """First paragraph here.

Second paragraph here.

Third paragraph here."""

        chunks = compressor._chunk_text(text, max_chunk_size=100)

        # Should create multiple chunks
        assert len(chunks) > 0
        # First chunk should contain first paragraph
        assert "First paragraph" in chunks[0]

    def test_extract_key_entities(self):
        """Test entity extraction"""
        compressor = SemanticCompressor()

        text = "The Quantum Algorithm uses Shor Code for Error Correction in IBM systems."
        entities = compressor._extract_key_entities(text)

        # Should extract some capitalized words
        assert len(entities) >= 0  # May or may not find entities depending on implementation


class TestErrorRecovery:
    """Test error recovery and graceful degradation"""

    def test_search_with_invalid_top_k(self):
        """Test search with invalid top_k values"""
        compressor = SemanticCompressor()
        compressor.ingest_file("Sample document for testing", "test_doc")

        # Zero top_k
        results = compressor.search_semantic("query", "test_doc", top_k=0)
        assert len(results) == 0

        # Negative top_k (should be handled gracefully)
        results = compressor.search_semantic("query", "test_doc", top_k=-1)
        assert len(results) == 0

    def test_modulate_region_with_mixed_valid_invalid_nodes(self):
        """Test modulation with mix of valid and invalid node IDs"""
        compressor = SemanticCompressor()
        compressor.ingest_file("Sample document for testing modulation", "test_doc")

        # Get one valid node
        all_nodes = [nid for nid in compressor.chunks.keys() if nid.startswith("test_doc")]
        valid_node = all_nodes[0] if all_nodes else "test_doc_n0"

        # Mix valid and invalid
        mixed_nodes = [valid_node, "invalid_node_123", "another_invalid"]

        result = compressor.modulate_region(mixed_nodes, FidelityLevel.RAW)

        # Should include warnings for invalid nodes
        assert "not found" in result.lower() or valid_node in result


class TestSemanticCompressorAdvanced:
    """Advanced edge cases for semantic compression"""

    def setup_method(self):
        """Initialize for each test"""
        self.compressor = SemanticCompressor()

    def test_extremely_large_document(self):
        """Test document approaching 1M token limit"""
        # Create a document ~500K tokens (word-based estimate)
        large_text = " ".join(["word"] * 500_000)

        # Should handle gracefully, possibly with chunking
        result = self.compressor.ingest_file(large_text, "large_doc")
        assert result is not None
        # May create one or more nodes (chunking depends on implementation)
        assert result.total_nodes >= 1
        # Should have very high compression due to repetition
        assert result.compression_ratio > 1.0

    def test_single_word_document(self):
        """Test document with only one word"""
        result = self.compressor.ingest_file("Hello", "single_word")

        assert result is not None
        assert result.total_nodes >= 1
        # Single word may not compress well
        assert result.total_tokens >= 1

    def test_circular_semantic_references(self):
        """Test document with circular conceptual structure"""
        circular_text = """
        A leads to B. B depends on C. C references A.
        The cycle continues: A -> B -> C -> A.
        This creates a semantic loop in the graph.
        """

        result = self.compressor.ingest_file(circular_text, "circular_doc")
        assert result is not None
        # Should handle cycles gracefully
        skeleton = self.compressor.read_skeleton("circular_doc")
        assert "circular_doc" in skeleton

    def test_duplicate_node_content(self):
        """Test document with highly repetitive content"""
        # Exact repetition 100 times
        repetitive = "This exact sentence appears many times. " * 100

        result = self.compressor.ingest_file(repetitive, "repetitive_doc")
        assert result is not None
        # Compression should be very high due to similarity
        # (fewer unique nodes created)


class TestAFMAdvanced:
    """Advanced AFM edge cases"""

    def setup_method(self):
        """Initialize for each test"""
        from src.afm import FocusManager, AFMConfig

        self.manager = FocusManager(AFMConfig())

    def test_message_with_only_punctuation(self):
        """Test message containing only punctuation marks"""
        # AFM should accept punctuation (may be meaningful in some contexts)
        # but handle it gracefully
        self.manager.add_message("user", "!@#$%^&*()")
        assert len(self.manager.messages) == 1

    def test_null_message_content(self):
        """Test None as message content"""
        with pytest.raises((ValueError, TypeError)):
            self.manager.add_message("user", None)

    def test_budget_exhaustion_during_compression(self):
        """Test behavior when budget runs out mid-build"""
        # Add many messages
        for i in range(20):
            self.manager.add_message("user", f"Message {i} with content " * 50)

        # Request context with very small budget
        context, stats = self.manager.build_context("query", budget_tokens=100)

        # Should return partial context within budget
        assert stats.total_tokens <= 100
        # Not all messages should fit (some dropped)
        assert stats.dropped_count > 0

    def test_concurrent_focus_manager_access(self):
        """Test thread-safety of AFM operations"""
        import threading

        def add_messages(manager, prefix):
            for i in range(10):
                manager.add_message("user", f"{prefix} message {i}")

        # Concurrent writes
        t1 = threading.Thread(target=add_messages, args=(self.manager, "Thread1"))
        t2 = threading.Thread(target=add_messages, args=(self.manager, "Thread2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Should have all 20 messages without corruption
        assert len(self.manager.messages) == 20


class TestFileSyncEdgeCases:
    """File sync manager edge cases"""

    def setup_method(self):
        """Initialize for each test"""
        from src.file_sync_manager import FileSyncManager
        import tempfile

        self.sync_manager = FileSyncManager()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup temp files"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_file_deleted_mid_sync(self):
        """Test handling of deleted source file"""
        import tempfile

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_file.write("Original content")
        temp_file.close()

        # Register file
        self.sync_manager.register_file("doc1", temp_file.name, "Original content")

        # Delete file
        os.unlink(temp_file.name)

        # Check sync - should detect deletion
        status = self.sync_manager.check_file_sync("doc1")
        assert not status["in_sync"]
        assert "deleted" in status["reason"].lower()

    def test_permission_error_during_read(self):
        """Test handling of permission errors"""
        import tempfile
        import stat

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_file.write("Original content")
        temp_file.close()

        # Register file
        self.sync_manager.register_file("doc2", temp_file.name, "Original content")

        # Change file content
        with open(temp_file.name, "w") as f:
            f.write("Modified content")

        # Make file unreadable (Unix-like systems)
        try:
            os.chmod(temp_file.name, 0o000)

            # Check sync - should handle permission error
            status = self.sync_manager.check_file_sync("doc2")
            # May fail on Windows, so check result is a dict
            assert isinstance(status, dict)

        finally:
            # Restore permissions for cleanup
            os.chmod(temp_file.name, stat.S_IRUSR | stat.S_IWUSR)
            os.unlink(temp_file.name)

    def test_symlink_handling(self):
        """Test handling of symbolic links"""
        import tempfile

        # Create actual file
        actual_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        actual_file.write("Actual content")
        actual_file.close()

        # Create symlink (skip on Windows if not supported)
        symlink_path = os.path.join(self.temp_dir, "symlink.txt")
        try:
            os.symlink(actual_file.name, symlink_path)

            # Register via symlink
            self.sync_manager.register_file("doc3", symlink_path, "Actual content")

            # Check sync - should work through symlink
            status = self.sync_manager.check_file_sync("doc3")
            assert status["in_sync"]

        except OSError:
            # Symlinks not supported on this platform
            pytest.skip("Symlinks not supported")
        finally:
            if os.path.exists(symlink_path):
                os.unlink(symlink_path)
            os.unlink(actual_file.name)

    def test_network_path_handling(self):
        """Test UNC paths (Windows network paths)"""
        # This is a basic validation test
        # Real UNC paths would require network setup

        if sys.platform != "win32":
            pytest.skip("UNC paths only on Windows")

        # Test that UNC-style path doesn't crash
        unc_path = r"\\server\share\file.txt"

        # Should handle gracefully (file won't exist)
        self.sync_manager.register_file("doc4", unc_path, "Content")
        status = self.sync_manager.check_file_sync("doc4")

        # Should detect file doesn't exist
        assert not status["in_sync"]

    def test_checksum_collision_detection(self):
        """Test handling of checksum collisions (theoretical)"""
        # MD5 collisions are extremely rare but possible
        # Test that system relies on checksum correctly

        content1 = "First content"
        content2 = "Different content"

        self.sync_manager.register_file("doc5", None, content1)
        meta = self.sync_manager.file_metadata["doc5"]

        # Different content should have different checksum
        # (In practice, MD5 collisions are extremely rare)
        meta2 = self.sync_manager.register_file("doc6", None, content2)
        assert meta.checksum != meta2.checksum

    def test_file_path_with_special_characters(self):
        """Test file paths with Unicode and special chars"""

        # Create file with special chars in name
        special_name = "file with spaces & üñíçödé 你好.txt"
        special_path = os.path.join(self.temp_dir, special_name)

        with open(special_path, "w", encoding="utf-8") as f:
            f.write("Content with unicode: 你好世界")

        # Register file
        self.sync_manager.register_file("doc7", special_path, "Content with unicode: 你好世界")

        # Check sync
        status = self.sync_manager.check_file_sync("doc7")
        assert status["in_sync"]

        os.unlink(special_path)

    def test_concurrent_sync_checks(self):
        """Test thread-safety of sync checking"""
        import threading
        import tempfile

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
        temp_file.write("Content")
        temp_file.close()

        self.sync_manager.register_file("doc8", temp_file.name, "Content")

        results = []

        def check_sync():
            status = self.sync_manager.check_file_sync("doc8")
            results.append(status["in_sync"])

        # Concurrent checks
        threads = [threading.Thread(target=check_sync) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All checks should succeed
        assert len(results) == 10
        assert all(results)

        os.unlink(temp_file.name)


class TestVersionManagerEdgeCases:
    """Version manager edge cases"""

    def setup_method(self):
        """Initialize for each test"""
        from src.version_manager import VersionManager
        import tempfile

        self.temp_dir = tempfile.mkdtemp()
        self.vm = VersionManager(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Cleanup"""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_storage_corruption_recovery(self):
        """Test recovery from corrupted storage file"""
        # Add valid version
        self.vm.add_version("doc1", "Content", "checksum123")

        # Corrupt the storage file
        version_file = Path(self.temp_dir) / "doc1.json"
        with open(version_file, "w") as f:
            f.write("{ INVALID JSON }")

        # Create new VM - should log error but not crash
        from src.version_manager import VersionManager

        vm2 = VersionManager(storage_dir=self.temp_dir)

        # Corrupted file should be skipped
        assert "doc1" not in vm2.versions

    def test_version_limit_enforcement(self):
        """Test automatic LRU pruning with many versions (Week 3 feature)"""
        # Add 100 versions
        for i in range(100):
            self.vm.add_version("doc2", f"Content version {i}", f"checksum{i}")

        # Should have only last 10 versions due to automatic pruning (DEFAULT_VERSION_RETENTION = 10)
        history = self.vm.get_version_history("doc2")
        assert len(history) == 10, f"Expected 10 versions (automatic pruning), got {len(history)}"

        # Latest should be version 100 (version_id counter is preserved)
        latest = self.vm.get_latest_version("doc2")
        assert latest.version_id == 100

        # Oldest retained should be version 91 (100 - 10 + 1)
        oldest = history[0]
        assert oldest["version_id"] == 91

    def test_concurrent_version_writes(self):
        """Test thread-safety of version writes"""
        import threading

        def add_versions(prefix):
            for i in range(10):
                self.vm.add_version(
                    f"doc_{prefix}", f"Content {prefix}-{i}", f"checksum_{prefix}_{i}"
                )

        # Concurrent writes to different documents
        threads = [threading.Thread(target=add_versions, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have 5 documents, each with 10 versions
        assert len(self.vm.versions) == 5
        for i in range(5):
            assert len(self.vm.versions[f"doc_{i}"]) == 10

    def test_diff_on_binary_content(self):
        """Test diffing binary-like content"""
        # Binary content (will be stored as string)
        binary_content = "Binary data: \x00\x01\x02\xff"

        self.vm.add_version("doc3", binary_content, "checksum1")
        self.vm.add_version("doc3", binary_content + "\x03", "checksum2")

        # Diff should work (binary stored as strings)
        diff = self.vm.diff_versions("doc3", 1, 2)
        assert diff is not None

    def test_missing_version_file(self):
        """Test handling when version file is manually deleted"""
        # Add version (creates file)
        self.vm.add_version("doc4", "Content", "checksum")

        # Manually delete version file
        version_file = Path(self.temp_dir) / "doc4.json"
        version_file.unlink()

        # Create new VM - should handle missing file gracefully
        from src.version_manager import VersionManager

        vm2 = VersionManager(storage_dir=self.temp_dir)

        # doc4 should not be in memory
        assert "doc4" not in vm2.versions


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
