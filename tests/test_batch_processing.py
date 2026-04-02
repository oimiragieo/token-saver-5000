"""
Tests for batch processing functionality (v0.6.0)

This module tests the BatchCompressionManager, batch_ingest_documents MCP tool,
and related batch processing features added in v0.6.0.

Test Categories:
- Basic batch ingestion
- Concurrent operations
- Error isolation
- Progress tracking
- Retry functionality
- Bounded concurrency
- Performance characteristics
- MCP tool integration
"""

import pytest
from src.batch_manager import (
    BatchCompressionManager,
    BatchDocument,
    BatchProgressTracker,
    batch_ingest_from_dict,
)
from src.semantic_compressor import SemanticCompressor
from src.handlers import compression_handlers

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def compressor():
    """Create a SemanticCompressor instance for testing."""
    return SemanticCompressor()


@pytest.fixture
def sample_documents():
    """Create sample documents for batch testing."""
    return [
        BatchDocument(
            file_id="doc1",
            text="Quantum computing uses qubits for parallel computation through superposition.",
            metadata={"topic": "quantum"},
        ),
        BatchDocument(
            file_id="doc2",
            text="Machine learning models learn from training data to make predictions.",
            metadata={"topic": "ml"},
        ),
        BatchDocument(
            file_id="doc3",
            text="Blockchain technology enables decentralized consensus through proof of work.",
            metadata={"topic": "blockchain"},
        ),
    ]


@pytest.fixture
def handler_context(compressor):
    """Create a handler context for MCP tool testing."""
    from src.resource_manager import ResourceManager
    from src.file_sync_manager import FileSyncManager
    from src.version_manager import VersionManager
    from src.persistence import PersistenceManager

    return {
        "compressor": compressor,
        "afm": None,
        "sync_manager": FileSyncManager(),
        "version_manager": VersionManager(),
        "ace_contexts": {},
        "resource_manager": ResourceManager(),
        "persistence": PersistenceManager(),
        "retrieval_history": {},
    }


# ===========================
# Basic Batch Ingestion Tests
# ===========================


class TestBasicBatchIngestion:
    """Test basic batch ingestion functionality."""

    @pytest.mark.asyncio
    async def test_batch_ingest_three_documents(self, compressor, sample_documents):
        """Test basic batch ingestion with 3 documents."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        results = await manager.compress_batch(sample_documents)

        # Verify all documents succeeded
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all(r.result is not None for r in results)

        # Verify results have expected attributes
        for result in results:
            assert result.file_id in ["doc1", "doc2", "doc3"]
            assert result.processing_time > 0
            assert result.result.compression_ratio > 0

    @pytest.mark.asyncio
    async def test_batch_ingest_empty_list(self, compressor):
        """Test batch ingestion with empty document list."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        results = await manager.compress_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_ingest_single_document(self, compressor):
        """Test batch ingestion with a single document."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        documents = [
            BatchDocument(
                file_id="single_doc",
                text="Neural networks consist of interconnected layers of artificial neurons.",
                metadata={},
            )
        ]

        results = await manager.compress_batch(documents)

        assert len(results) == 1
        assert results[0].success
        assert results[0].file_id == "single_doc"


# ===========================
# Concurrent Operations Tests
# ===========================


class TestConcurrentOperations:
    """Test concurrent batch processing behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_batch_execution(self, compressor):
        """Test that documents are processed concurrently."""
        import time

        manager = BatchCompressionManager(compressor, max_concurrent=4)

        # Create 10 documents
        documents = [
            BatchDocument(
                file_id=f"doc_{i}",
                text=f"Document {i}: This is a test document about topic {i}.",
                metadata={"index": i},
            )
            for i in range(10)
        ]

        start_time = time.time()
        results = await manager.compress_batch(documents)
        elapsed = time.time() - start_time

        # Verify all succeeded
        assert len(results) == 10
        assert all(r.success for r in results)

        # Concurrent execution should be faster than sequential
        # (Though exact speedup depends on CPU and async scheduler)
        assert elapsed < 30.0  # Should complete in reasonable time

    @pytest.mark.asyncio
    async def test_max_concurrent_respected(self, compressor):
        """Test that max_concurrent limit is respected."""
        # Test with max_concurrent=1 (sequential)
        manager_sequential = BatchCompressionManager(compressor, max_concurrent=1)

        documents = [BatchDocument(f"doc_{i}", f"Test document {i}.", {}) for i in range(3)]

        results = await manager_sequential.compress_batch(documents)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_max_concurrent_8(self, compressor):
        """Test batch processing with maximum concurrency (8)."""
        manager = BatchCompressionManager(compressor, max_concurrent=8)

        documents = [BatchDocument(f"doc_{i}", f"High concurrency test {i}.", {}) for i in range(8)]

        results = await manager.compress_batch(documents)

        assert len(results) == 8
        assert all(r.success for r in results)


# ===========================
# Error Isolation Tests
# ===========================


class TestErrorIsolation:
    """Test error isolation in batch processing."""

    @pytest.mark.asyncio
    async def test_mixed_success_failure(self, compressor):
        """Test that failures don't block successful ingestions."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        documents = [
            BatchDocument("doc1", "Valid document 1", {}),
            BatchDocument("doc2", "", {}),  # Invalid: empty text
            BatchDocument("doc3", "Valid document 2", {}),
            BatchDocument("doc4", "x", {}),  # Valid: single character accepted
            BatchDocument("doc5", "Valid document 3", {}),
        ]

        results = await manager.compress_batch(documents)

        assert len(results) == 5

        # Verify successful documents
        successful = [r for r in results if r.success]
        assert len(successful) == 4
        assert {r.file_id for r in successful} == {"doc1", "doc3", "doc4", "doc5"}

        # Verify failed documents
        failed = [r for r in results if not r.success]
        assert len(failed) == 1
        assert all(r.error is not None for r in failed)

    @pytest.mark.asyncio
    async def test_all_failures(self, compressor):
        """Test batch where all documents fail."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        documents = [
            BatchDocument("doc1", "", {}),  # Empty
            BatchDocument("doc2", "  ", {}),  # Whitespace
            BatchDocument("doc3", "   ", {}),  # Whitespace
        ]

        results = await manager.compress_batch(documents)

        assert len(results) == 3
        assert all(not r.success for r in results)
        assert all(r.error is not None for r in results)


# ===========================
# Progress Tracking Tests
# ===========================


class TestProgressTracking:
    """Test progress tracking functionality."""

    @pytest.mark.asyncio
    async def test_progress_tracker_updates(self, compressor):
        """Test that progress tracker updates correctly."""
        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.completed)

        manager = BatchCompressionManager(compressor, max_concurrent=4)

        documents = [BatchDocument(f"doc_{i}", f"Test doc {i}.", {}) for i in range(5)]

        results = await manager.compress_batch(documents, on_progress=on_progress)

        # Verify progress updates
        assert len(progress_updates) == 5
        assert progress_updates == [1, 2, 3, 4, 5]

        # Verify results
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_progress_percentage_calculation(self, compressor):
        """Test progress percentage calculation."""
        tracker = BatchProgressTracker(total=10)

        # Initial state
        assert tracker.progress.progress_percentage == 0.0

        # After 5 completions
        for _ in range(5):
            tracker.update(success=True)

        assert tracker.progress.progress_percentage == 50.0

        # After all completions
        for _ in range(5):
            tracker.update(success=True)

        assert tracker.progress.progress_percentage == 100.0

    @pytest.mark.asyncio
    async def test_success_rate_calculation(self, compressor):
        """Test success rate calculation."""
        tracker = BatchProgressTracker(total=10)

        # 7 successes, 3 failures
        for _ in range(7):
            tracker.update(success=True)
        for _ in range(3):
            tracker.update(success=False)

        assert tracker.progress.success_rate == 70.0


# ===========================
# Retry Functionality Tests
# ===========================


class TestRetryFunctionality:
    """Test batch retry functionality."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, compressor):
        """Test that failed documents are retried."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        # Mix of valid and invalid documents
        documents = [
            BatchDocument("doc1", "Valid document 1", {}),
            BatchDocument("doc2", "", {}),  # Will fail
        ]

        results, failed_docs = await manager.compress_batch_with_retry(documents, max_retries=2)

        # doc1 should succeed, doc2 should fail even after retries
        assert len(results) == 2
        assert len(failed_docs) == 1
        assert failed_docs[0].file_id == "doc2"


# ===========================
# Utility Function Tests
# ===========================


class TestUtilityFunctions:
    """Test batch processing utility functions."""

    @pytest.mark.asyncio
    async def test_batch_ingest_from_dict(self, compressor):
        """Test batch_ingest_from_dict utility function."""
        documents_dict = {
            "doc1": "Quantum entanglement enables quantum communication.",
            "doc2": "Deep learning uses neural networks with multiple layers.",
            "doc3": "Distributed systems achieve fault tolerance through replication.",
        }

        results = await batch_ingest_from_dict(compressor, documents_dict)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert {r.file_id for r in results} == {"doc1", "doc2", "doc3"}

    @pytest.mark.asyncio
    async def test_batch_ingest_with_metadata(self, compressor):
        """Test batch_ingest_from_dict with metadata."""
        documents_dict = {"doc1": "Test content"}
        metadata_dict = {"doc1": {"author": "Alice", "date": "2025-01-01"}}

        results = await batch_ingest_from_dict(compressor, documents_dict, metadata_dict)

        assert len(results) == 1
        assert results[0].success


# ===========================
# MCP Tool Integration Tests
# ===========================


class TestMCPToolIntegration:
    """Test batch_ingest_documents MCP tool."""

    @pytest.mark.asyncio
    async def test_handle_batch_ingest_success(self, handler_context):
        """Test handle_batch_ingest with valid documents."""
        args = {
            "documents": [
                {"file_id": "doc1", "text": "Quantum physics document", "metadata": {}},
                {"file_id": "doc2", "text": "Machine learning document", "metadata": {}},
            ],
            "max_concurrent": 4,
        }

        result = await compression_handlers.handle_batch_ingest(handler_context, args)

        # Verify JSON response
        import json

        response = json.loads(result)
        assert response["total"] == 2
        assert response["successful"] == 2
        assert response["failed"] == 0

    @pytest.mark.asyncio
    async def test_handle_batch_ingest_validation_error(self, handler_context):
        """Test handle_batch_ingest with invalid arguments."""
        args = {
            "documents": [],  # Empty list
        }

        with pytest.raises(Exception) as exc_info:
            await compression_handlers.handle_batch_ingest(handler_context, args)

        assert "documents" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_handle_batch_ingest_max_concurrent_validation(self, handler_context):
        """Test max_concurrent validation."""
        args = {
            "documents": [{"file_id": "doc1", "text": "Test"}],
            "max_concurrent": 10,  # Invalid: max is 8
        }

        with pytest.raises(ValueError) as exc_info:
            await compression_handlers.handle_batch_ingest(handler_context, args)

        assert "max_concurrent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_batch_ingest_with_failures(self, handler_context):
        """Test handle_batch_ingest with some failures."""
        args = {
            "documents": [
                {"file_id": "doc1", "text": "Valid document"},
                {"file_id": "doc2", "text": ""},  # Empty text - will fail
            ]
        }

        result = await compression_handlers.handle_batch_ingest(handler_context, args)

        import json

        response = json.loads(result)
        assert response["total"] == 2
        assert response["successful"] == 1
        assert response["failed"] == 1
        assert "failed_file_ids" in response
        assert "doc2" in response["failed_file_ids"]


# ===========================
# Performance Tests
# ===========================


class TestPerformanceCharacteristics:
    """Test performance characteristics of batch processing."""

    @pytest.mark.asyncio
    async def test_batch_faster_than_sequential(self, compressor):
        """Test that batch processing overlaps work rather than fully serializing."""
        import time

        documents = [
            BatchDocument(f"doc_{i}", f"Performance test document {i}.", {}) for i in range(10)
        ]

        manager_batch = BatchCompressionManager(compressor, max_concurrent=4)
        start_batch = time.time()
        results = await manager_batch.compress_batch(documents)
        batch_time = time.time() - start_batch
        total_reported_work = sum(result.processing_time for result in results)

        assert all(result.success for result in results)
        assert batch_time < 30.0
        assert batch_time < total_reported_work, (
            f"Batch wall-clock time ({batch_time:.2f}s) should be lower than "
            f"the sum of per-document work ({total_reported_work:.2f}s) when "
            f"bounded concurrency overlaps execution"
        )

    @pytest.mark.asyncio
    async def test_processing_time_tracking(self, compressor):
        """Test that processing times are tracked correctly."""
        manager = BatchCompressionManager(compressor, max_concurrent=4)

        documents = [BatchDocument("doc1", "Test document for timing.", {})]

        results = await manager.compress_batch(documents)

        assert len(results) == 1
        assert results[0].processing_time > 0
        assert results[0].processing_time < 10.0  # Should be reasonable
