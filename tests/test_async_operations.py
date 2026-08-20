"""
Tests for async operations in semantic compressor and handlers.

This module validates the async encoding wrapper, concurrent operations,
and timeout prevention added in v0.5.0 Phase 1.
"""

import asyncio
import numpy as np
import pytest
import time
from src.semantic_compressor import SemanticCompressor
from src.handlers import compression_handlers, mcp_core


@pytest.fixture
def compressor():
    """Create a SemanticCompressor instance for testing."""
    return SemanticCompressor()


@pytest.fixture
def handler_context(compressor):
    """Create a handler context for testing."""
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


class TestAsyncEncoding:
    """Test async encoding wrapper behavior."""

    @pytest.mark.asyncio
    async def test_encode_async_returns_embeddings(self, compressor):
        """Test that _encode_async returns valid embeddings."""
        texts = ["Hello world", "Test document", "Async encoding"]

        embeddings = await compressor._encode_async(texts)

        # Verify embeddings shape
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # all-MiniLM-L6-v2 embedding size

        # Verify embeddings are normalized (cosine similarity ready)
        import numpy as np

        norms = np.linalg.norm(embeddings, axis=1)
        assert all(0.99 < norm < 1.01 for norm in norms)

    @pytest.mark.asyncio
    async def test_encode_async_nonblocking(self, compressor):
        """Test that _encode_async doesn't block the event loop."""
        texts = ["Document " + str(i) for i in range(100)]

        # Start async encoding
        encode_task = asyncio.create_task(compressor._encode_async(texts))

        # Run a quick task concurrently
        await asyncio.sleep(0.01)  # Should complete before encoding

        # Wait for encoding to finish
        embeddings = await encode_task

        assert embeddings.shape[0] == 100


class TestConcurrentOperations:
    """Test concurrent ingest operations."""

    @pytest.mark.asyncio
    async def test_concurrent_ingests(self, compressor):
        """Test that multiple ingests can run concurrently."""
        docs = [
            ("doc1", "The quantum computer uses qubits for superposition."),
            ("doc2", "Machine learning models learn from training data."),
            ("doc3", "Blockchain technology enables decentralized consensus."),
        ]

        # Run concurrent ingests
        tasks = [
            compressor.ingest_file_async(text, file_id, {"test": "concurrent"})
            for file_id, text in docs
        ]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        # Verify all ingests succeeded
        assert len(results) == 3
        for result in results:
            # SkeletonResponse is a dataclass, check for attributes
            assert hasattr(result, "skeleton_text")
            assert hasattr(result, "file_id")

        # Verify documents are stored (chunks use node_ids like "doc1_n0")
        assert any(nid.startswith("doc1_") for nid in compressor.chunks)
        assert any(nid.startswith("doc2_") for nid in compressor.chunks)
        assert any(nid.startswith("doc3_") for nid in compressor.chunks)

        # Concurrent execution should be faster than sequential
        # (Though with max_workers=1, they'll run sequentially in thread pool)
        print(f"Concurrent ingest time: {elapsed:.2f}s")


class TestHandlerAsyncConversion:
    """Test that handlers are properly converted to async."""

    @pytest.mark.asyncio
    async def test_handle_ingest_async(self, handler_context):
        """Test that handle_ingest is async and works correctly."""
        args = {
            "text": "Quantum computing leverages superposition and entanglement.",
            "file_id": "test_async_doc",
            "metadata": {"test": "async_handler"},
        }

        result = await compression_handlers.handle_ingest(handler_context, args)

        # Verify result is JSON string
        assert isinstance(result, str)
        assert "skeleton" in result
        assert "test_async_doc" in result

    @pytest.mark.asyncio
    async def test_route_tool_call_async(self, handler_context):
        """Test that route_tool_call is async and routes correctly."""
        args = {
            "text": "Neural networks consist of interconnected layers.",
            "file_id": "route_test_doc",
        }

        result = await mcp_core.route_tool_call("ingest_context", args, handler_context)

        # Verify routing worked
        assert isinstance(result, str)
        assert "skeleton" in result
        assert "route_test_doc" in result


class TestTimeoutPrevention:
    """Test that async operations prevent MCP timeouts."""

    @pytest.mark.asyncio
    async def test_large_document_responsiveness(self, compressor):
        """Test that large document ingestion doesn't block event loop."""
        # Create a large document (1000 chunks)
        large_text = "\n\n".join(
            [
                f"Section {i}: This is a test paragraph about topic {i}. "
                f"It contains information about {i} and related concepts."
                for i in range(500)
            ]
        )

        # Start ingestion
        ingest_task = asyncio.create_task(compressor.ingest_file_async(large_text, "large_doc", {}))

        # Verify event loop remains responsive during ingestion
        for _ in range(10):
            await asyncio.sleep(0.1)  # Should not block

        # Wait for ingestion to complete
        result = await ingest_task

        # SkeletonResponse is a dataclass, check for attributes
        assert hasattr(result, "skeleton_text")
        assert result.file_id == "large_doc"
        # Chunks use node_ids like "large_doc_n0", not doc_ids
        assert any(nid.startswith("large_doc_") for nid in compressor.chunks)


class TestAsyncErrorHandling:
    """Test error handling in async context."""

    @pytest.mark.asyncio
    async def test_concurrent_ingest_error_isolation(self, compressor):
        """Test that errors in one concurrent ingest don't affect others."""
        # Mix valid and invalid ingests
        tasks = [
            compressor.ingest_file_async("Valid document 1", "doc1", {}),
            compressor.ingest_file_async("", "doc2", {}),  # Empty text (invalid)
            compressor.ingest_file_async("Valid document 2", "doc3", {}),
        ]

        # Gather with return_exceptions to prevent failure propagation
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify valid ingests succeeded (SkeletonResponse is a dataclass)
        assert hasattr(results[0], "skeleton_text")
        assert hasattr(results[2], "skeleton_text")

        # Verify invalid ingest raised error
        assert isinstance(results[1], (ValueError, Exception))

    @pytest.mark.asyncio
    async def test_encode_async_error_handling(self, compressor):
        """An encoder error must PROPAGATE through the async wrapper.

        DO NOT depend on the ambient model to reject None. `SemanticCompressor`
        draws `model` from a process-wide singleton `EmbeddingManager`, so a
        test that ran earlier and replaced it with a Mock makes `None` perfectly
        acceptable -- and this assertion then fails in the full suite while
        passing alone and in its own file. That is exactly how it behaved: green
        in isolation, red only in CI's complete run.

        The property worth pinning is the WRAPPER's: `_encode_async` offloads to
        a ThreadPoolExecutor, and an exception raised inside that thread must
        surface to the awaiting caller rather than being swallowed. Driving it
        with an encoder that raises deterministically tests that, and cannot be
        neutralised by whatever the singleton currently holds.
        """
        sentinel = TypeError("encoder rejected the input")

        class _RaisingModel:
            def encode(self, *_args, **_kwargs):
                raise sentinel

        original = compressor.model
        compressor.model = _RaisingModel()
        try:
            with pytest.raises(TypeError) as caught:
                await compressor._encode_async(["anything"])
            assert caught.value is sentinel, (
                "a different TypeError surfaced - the wrapper replaced the "
                "encoder's error instead of propagating it"
            )
        finally:
            compressor.model = original

        # NON-VACUITY: the same call SUCCEEDS with a working encoder, so the
        # arm above cannot be passing because _encode_async is simply broken.
        class _WorkingModel:
            def encode(self, texts, *_args, **_kwargs):
                return np.zeros((len(texts), 8), dtype=np.float32)

        compressor.model = _WorkingModel()
        try:
            out = await compressor._encode_async(["anything"])
            assert out.shape == (1, 8), f"working encoder returned {out.shape}"
        finally:
            compressor.model = original


class TestPerformanceCharacteristics:
    """Test performance characteristics of async operations."""

    @pytest.mark.asyncio
    async def test_async_overhead_minimal(self, compressor):
        """Test that async wrapper adds minimal overhead."""
        texts = ["Test document for performance measurement."]

        # Measure sync performance (direct call)
        start_sync = time.time()
        sync_result = compressor.model.encode(texts, show_progress_bar=False)
        sync_time = time.time() - start_sync

        # Measure async performance
        start_async = time.time()
        async_result = await compressor._encode_async(texts)
        async_time = time.time() - start_async

        # Async overhead should be < 200ms (includes ThreadPoolExecutor setup)
        overhead = async_time - sync_time
        print(f"Sync: {sync_time:.4f}s, Async: {async_time:.4f}s, Overhead: {overhead:.4f}s")
        assert overhead < 0.2  # 200ms tolerance (includes executor initialization)

        # Results should be identical (within floating point precision)
        import numpy as np

        assert np.allclose(sync_result, async_result, rtol=1e-5)

    @pytest.mark.asyncio
    async def test_health_check_responsiveness(self, handler_context):
        """Test that health check responds quickly during compression."""
        # Start a large compression operation
        large_text = "\n\n".join([f"Paragraph {i}" for i in range(200)])
        compress_task = asyncio.create_task(
            handler_context["compressor"].ingest_file_async(large_text, "bg_doc", {})
        )

        # Issue health check during compression
        await asyncio.sleep(0.01)  # Let compression start

        start_health = time.time()
        health_result = await compression_handlers.handle_get_stats(handler_context, {})
        health_time = time.time() - start_health

        # Health check should respond in < 100ms
        assert health_time < 0.1
        assert isinstance(health_result, str)
        assert "Total files ingested" in health_result or "total_documents" in health_result

        # Wait for compression to finish
        await compress_task
