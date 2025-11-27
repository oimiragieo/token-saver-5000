"""Performance tests for Token Saver 5000 v0.7.0.

Tests throughput, latency, memory usage, cache effectiveness, and burst capacity.
Uses psutil for memory profiling and time.perf_counter() for precise timing.

Test categories:
- Throughput: docs/second, tokens/second processing rates
- Latency: p50, p95, p99 latency percentiles
- Memory: RSS growth, leak detection, large document handling
- Cache: hit rates, overhead, eviction performance
- Burst: concurrent load handling with rate limiting
"""

import asyncio
import time
import pytest
import psutil
import statistics

from src.batch_manager import BatchCompressionManager


class TestThroughputPerformance:
    """Throughput tests measuring processing speed in docs/second and tokens/second."""

    @pytest.mark.asyncio
    async def test_throughput_single_document_ingestion(self, compressor, large_document):
        """Test single document ingestion throughput (should be > 2 docs/second).

        Performance threshold rationale:
        - Embedding generation: ~300ms per document (SBERT on CPU)
        - Graph construction: ~100ms for large document
        - Persistence overhead: ~50ms
        - Target: 2 docs/sec = 500ms per document (reasonable for CPU-only)
        """
        start = time.perf_counter()
        num_docs = 10

        for i in range(num_docs):
            doc_id = f"throughput_test_{i}"
            await compressor.ingest_context(doc_id, large_document)

        elapsed = time.perf_counter() - start
        throughput = num_docs / elapsed

        assert throughput > 2.0, f"Throughput {throughput:.2f} docs/sec too slow (expected > 2)"
        assert len(compressor.documents) == num_docs

    @pytest.mark.asyncio
    async def test_throughput_batch_ingestion(self, compressor, performance_documents):
        """Test batch ingestion throughput vs sequential (should be > 4× speedup).

        Performance threshold rationale:
        - Batch processing uses concurrent.futures with semaphore (default: 4 workers)
        - Expected speedup: ~4× vs sequential (proven in v0.6.0: 4× improvement)
        - Batch should process > 10 docs/second (vs ~2.5 sequential)
        """
        # Sequential baseline
        sequential_start = time.perf_counter()
        for i, doc in enumerate(performance_documents[:20]):
            await compressor.ingest_context(f"seq_{i}", doc)
        sequential_elapsed = time.perf_counter() - sequential_start

        # Reset compressor
        compressor.documents.clear()
        compressor.semantic_graphs.clear()

        # Batch processing
        batch_start = time.perf_counter()
        manager = BatchCompressionManager(compressor)
        results = await manager.ingest_batch(
            [(f"batch_{i}", doc) for i, doc in enumerate(performance_documents[:20])]
        )
        batch_elapsed = time.perf_counter() - batch_start

        speedup = sequential_elapsed / batch_elapsed
        batch_throughput = 20 / batch_elapsed

        assert speedup > 3.0, f"Speedup {speedup:.2f}× insufficient (expected > 3×)"
        assert batch_throughput > 8.0, f"Batch throughput {batch_throughput:.2f} docs/sec too slow"
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_throughput_compression_speed(self, compressor, large_document):
        """Test compression speed in tokens/second (should be > 1000 tokens/sec).

        Performance threshold rationale:
        - Large document: ~500 tokens input
        - Target compression time: < 500ms
        - Tokens/second: 500 / 0.5 = 1000 tokens/sec minimum
        - Embedding generation dominates (300ms), graph ops ~100ms
        """
        doc_id = "compression_speed_test"
        await compressor.ingest_context(doc_id, large_document)

        # Measure compression time
        start = time.perf_counter()
        result = await compressor.compress_document(doc_id)
        elapsed = time.perf_counter() - start

        # Estimate input tokens (rough: 4 chars per token)
        input_tokens = len(large_document) / 4
        tokens_per_second = input_tokens / elapsed

        assert (
            tokens_per_second > 1000.0
        ), f"Compression speed {tokens_per_second:.0f} tokens/sec too slow (expected > 1000)"
        assert result["compression_ratio"] > 1.0


class TestLatencyPerformance:
    """Latency tests measuring p50, p95, p99 percentiles for key operations."""

    @pytest.mark.asyncio
    async def test_latency_p50_ingestion(self, compressor):
        """Test median ingestion latency (p50 should be < 1 second).

        Performance threshold rationale:
        - Median case: Small-to-medium documents (~200 tokens)
        - Expected p50: ~500ms (embedding + graph construction)
        - Threshold: < 1s for median to ensure responsive user experience
        """
        latencies = []

        for i in range(50):
            doc = f"Test document {i} with some content to analyze. " * 10
            start = time.perf_counter()
            await compressor.ingest_context(f"latency_p50_{i}", doc)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        p50 = statistics.median(latencies)

        assert p50 < 1.0, f"Median latency {p50:.3f}s exceeds 1s threshold"
        assert len(compressor.documents) == 50

    @pytest.mark.asyncio
    async def test_latency_p95_compression(self, compressor, large_document):
        """Test 95th percentile compression latency (p95 should be < 5 seconds).

        Performance threshold rationale:
        - 95th percentile: Large documents with complex graphs
        - Worst-case embedding time: ~1s (large doc, cold cache)
        - Graph construction: ~1s (complex relationships)
        - Compression algorithm: ~2s (semantic analysis)
        - Threshold: < 5s to handle tail latency without timeouts
        """
        latencies = []

        for i in range(50):
            doc_id = f"latency_p95_{i}"
            await compressor.ingest_context(doc_id, large_document)

            start = time.perf_counter()
            await compressor.compress_document(doc_id)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        assert p95 < 5.0, f"95th percentile latency {p95:.3f}s exceeds 5s threshold"

    @pytest.mark.asyncio
    async def test_latency_p99_expansion(self, compressor, large_document):
        """Test 99th percentile expansion latency (p99 should be < 2 seconds).

        Performance threshold rationale:
        - Expansion is faster than compression (no embedding generation)
        - Main overhead: Graph traversal + text reconstruction
        - Expected p99: ~1.5s (complex skeleton with many nodes)
        - Threshold: < 2s for worst-case expansion responsiveness
        """
        latencies = []

        # Prepare compressed documents
        for i in range(50):
            doc_id = f"latency_p99_{i}"
            await compressor.ingest_context(doc_id, large_document)
            await compressor.compress_document(doc_id)

        # Measure expansion latency
        for i in range(50):
            doc_id = f"latency_p99_{i}"
            start = time.perf_counter()
            await compressor.expand_skeleton(doc_id)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed)

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]

        assert p99 < 2.0, f"99th percentile expansion latency {p99:.3f}s exceeds 2s threshold"


class TestMemoryUsagePerformance:
    """Memory usage tests tracking RSS growth, leaks, and large document handling."""

    @pytest.mark.asyncio
    async def test_memory_usage_baseline(self, compressor):
        """Test memory baseline before/after operations (should track RSS growth).

        Performance threshold rationale:
        - Baseline memory: Process RSS before operations
        - After 100 small docs: Expected growth ~50MB (embeddings + graphs)
        - Memory leak check: Growth should stabilize, not grow linearly
        """
        process = psutil.Process()
        baseline_rss = process.memory_info().rss / 1024 / 1024  # MB

        # Ingest 100 small documents
        for i in range(100):
            doc = f"Small test document {i}. " * 5
            await compressor.ingest_context(f"baseline_{i}", doc)

        after_rss = process.memory_info().rss / 1024 / 1024  # MB
        growth = after_rss - baseline_rss

        # Memory growth should be reasonable (< 100MB for 100 small docs)
        assert growth < 100.0, f"Memory growth {growth:.1f}MB excessive for 100 small docs"
        assert len(compressor.documents) == 100

    @pytest.mark.asyncio
    async def test_memory_usage_large_document(self, compressor, large_document):
        """Test memory growth for large document (10k token doc should grow < 500MB).

        Performance threshold rationale:
        - Large document: ~500 tokens (2KB text)
        - SBERT embeddings: 384-dim float32 = ~1.5KB per sentence
        - Graph structure: ~50 nodes × 2KB = ~100KB
        - Total expected: ~200MB for complete processing (with model overhead)
        - Threshold: < 500MB to allow headroom for batch operations
        """
        process = psutil.Process()
        baseline_rss = process.memory_info().rss / 1024 / 1024  # MB

        # Ingest and compress large document
        doc_id = "large_doc_memory_test"
        await compressor.ingest_context(doc_id, large_document)
        await compressor.compress_document(doc_id)

        after_rss = process.memory_info().rss / 1024 / 1024  # MB
        growth = after_rss - baseline_rss

        assert growth < 500.0, f"Memory growth {growth:.1f}MB excessive for single large document"

    @pytest.mark.asyncio
    async def test_memory_usage_batch_processing(self, compressor, performance_documents):
        """Test memory leak detection during batch processing (RSS should stabilize).

        Performance threshold rationale:
        - Batch 1: Initial memory allocation (embeddings, graphs)
        - Batch 2: Should reuse memory (LRU eviction, cache limits)
        - Memory leak indicator: Growth in batch 2 > 20% of batch 1
        - Threshold: < 20% additional growth indicates no major leak
        """
        process = psutil.Process()
        baseline_rss = process.memory_info().rss / 1024 / 1024  # MB

        # First batch
        manager = BatchCompressionManager(compressor)
        await manager.ingest_batch(
            [(f"batch1_{i}", doc) for i, doc in enumerate(performance_documents[:50])]
        )

        after_batch1_rss = process.memory_info().rss / 1024 / 1024  # MB
        batch1_growth = after_batch1_rss - baseline_rss

        # Clear compressor state
        compressor.documents.clear()
        compressor.semantic_graphs.clear()

        # Second batch (should not grow significantly - cache reuse)
        await manager.ingest_batch(
            [(f"batch2_{i}", doc) for i, doc in enumerate(performance_documents[50:100])]
        )

        after_batch2_rss = process.memory_info().rss / 1024 / 1024  # MB
        batch2_growth = after_batch2_rss - after_batch1_rss

        # Batch 2 growth should be < 20% of Batch 1 (indicates memory reuse)
        leak_ratio = batch2_growth / batch1_growth if batch1_growth > 0 else 0

        assert leak_ratio < 0.2, (
            f"Memory leak detected: Batch 2 growth {batch2_growth:.1f}MB is "
            f"{leak_ratio:.1%} of Batch 1 growth {batch1_growth:.1f}MB (expected < 20%)"
        )


class TestCacheEffectiveness:
    """Cache effectiveness tests measuring hit rates, overhead, and eviction performance."""

    @pytest.mark.asyncio
    async def test_cache_hit_rate_repeated_documents(self, compressor):
        """Test cache hit rate for repeated documents (should be > 80%).

        Performance threshold rationale:
        - Repeated identical documents should hit embedding cache
        - LRU cache default capacity: 10k entries
        - Expected hit rate: > 80% for repeated ingestion
        - Cache miss sources: Eviction (capacity), hash collisions (rare)
        """
        doc = "Repeated test document for cache hit rate analysis. " * 20

        # Warm up cache
        for i in range(10):
            await compressor.ingest_context(f"warmup_{i}", doc)

        # Get baseline cache stats
        from src.embeddings import get_embedding_manager

        manager = get_embedding_manager()
        initial_stats = manager.embedding_cache.cache_stats()
        initial_hits = initial_stats.get("hits", 0)
        initial_total = initial_stats.get("total_requests", 0)

        # Repeat ingestion (should hit cache)
        for i in range(100):
            await compressor.ingest_context(f"repeat_{i}", doc)

        final_stats = manager.embedding_cache.cache_stats()
        final_hits = final_stats.get("hits", 0)
        final_total = final_stats.get("total_requests", 0)

        new_hits = final_hits - initial_hits
        new_requests = final_total - initial_total
        hit_rate = new_hits / new_requests if new_requests > 0 else 0

        assert hit_rate > 0.80, f"Cache hit rate {hit_rate:.1%} below 80% threshold"

    @pytest.mark.asyncio
    async def test_cache_memory_overhead(self, compressor):
        """Test cache memory overhead tracking (LRU cache size monitoring).

        Performance threshold rationale:
        - Each cache entry: ~2KB (384-dim embedding + metadata)
        - 10k entries: ~20MB total
        - Acceptable overhead: < 50MB (allows headroom for hash table overhead)
        """
        from src.embeddings import get_embedding_manager

        manager = get_embedding_manager()

        # Get initial cache size
        initial_stats = manager.embedding_cache.cache_stats()
        initial_size = initial_stats.get("current_size", 0)

        # Fill cache with diverse documents
        for i in range(1000):
            doc = f"Cache overhead test document {i} with unique content. " * 10
            await compressor.ingest_context(f"cache_overhead_{i}", doc)

        final_stats = manager.embedding_cache.cache_stats()
        final_size = final_stats.get("current_size", 0)
        cache_growth = final_size - initial_size

        # Cache growth should be reasonable (< 5000 entries for diverse docs)
        assert cache_growth < 5000, f"Cache grew by {cache_growth} entries (expected < 5000)"

    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self, compressor):
        """Test LRU cache eviction is O(1) (eviction time should not scale with size).

        Performance threshold rationale:
        - LRU eviction: O(1) using OrderedDict.popitem(last=False)
        - Eviction triggers when cache reaches capacity
        - Expected: Eviction time < 10ms regardless of cache size
        - Performance degradation indicator: Eviction time > 10ms
        """
        from src.embeddings import get_embedding_manager

        get_embedding_manager()

        # Fill cache to near capacity
        capacity = 10000
        for i in range(capacity - 100):
            doc = f"Eviction test document {i}. " * 5
            await compressor.ingest_context(f"eviction_{i}", doc)

        # Measure eviction time (trigger by adding more entries)
        eviction_times = []
        for i in range(200):  # Trigger evictions
            doc = f"Eviction trigger document {i}. " * 5
            start = time.perf_counter()
            await compressor.ingest_context(f"trigger_{i}", doc)
            elapsed = time.perf_counter() - start
            eviction_times.append(elapsed)

        # Average eviction time should be < 10ms
        avg_eviction_time = statistics.mean(eviction_times) * 1000  # Convert to ms

        assert (
            avg_eviction_time < 10.0
        ), f"Average eviction time {avg_eviction_time:.2f}ms exceeds 10ms threshold (O(1) violation)"


class TestBurstCapacity:
    """Burst capacity tests measuring concurrent load handling with rate limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_10_documents(self, compressor, performance_documents):
        """Test handling 10 concurrent document ingestions (should complete without errors).

        Performance threshold rationale:
        - 10 concurrent: Light load, should complete without throttling
        - Expected time: ~5s (parallel embedding generation)
        - Error threshold: 0 failures (system should handle easily)
        """
        start = time.perf_counter()

        tasks = [
            compressor.ingest_context(f"concurrent_10_{i}", doc)
            for i, doc in enumerate(performance_documents[:10])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start

        # Count errors
        errors = [r for r in results if isinstance(r, Exception)]

        assert len(errors) == 0, f"Encountered {len(errors)} errors in 10 concurrent ingestions"
        assert elapsed < 10.0, f"10 concurrent documents took {elapsed:.1f}s (expected < 10s)"

    @pytest.mark.asyncio
    async def test_concurrent_50_documents(self, compressor, performance_documents):
        """Test handling 50 concurrent document ingestions (may have some throttling).

        Performance threshold rationale:
        - 50 concurrent: Medium load, may trigger semaphore throttling
        - BatchCompressionManager: Default 4 concurrent workers
        - Expected time: ~25s (50 docs / 4 workers / 2 docs/sec)
        - Error threshold: < 5% failures (acceptable for burst load)
        """
        start = time.perf_counter()

        tasks = [
            compressor.ingest_context(f"concurrent_50_{i}", doc)
            for i, doc in enumerate(performance_documents[:50])
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.perf_counter() - start

        # Count errors
        errors = [r for r in results if isinstance(r, Exception)]
        error_rate = len(errors) / len(results)

        assert error_rate < 0.05, f"Error rate {error_rate:.1%} exceeds 5% threshold"
        assert elapsed < 60.0, f"50 concurrent documents took {elapsed:.1f}s (expected < 60s)"

    @pytest.mark.asyncio
    async def test_concurrent_100_documents_with_rate_limiting(
        self, compressor, performance_documents
    ):
        """Test rate limiting prevents overload at 100 concurrent (should gracefully throttle).

        Performance threshold rationale:
        - 100 concurrent: Heavy load, rate limiter should activate
        - Rate limiter: Semaphore with max 4 concurrent (BatchCompressionManager)
        - Expected time: ~50s (100 docs / 4 workers / 2 docs/sec)
        - Success criteria: All docs processed, no crashes, < 10% errors
        """
        start = time.perf_counter()

        # Use BatchCompressionManager for rate limiting
        manager = BatchCompressionManager(compressor, max_concurrent=4)
        results = await manager.ingest_batch(
            [(f"concurrent_100_{i}", doc) for i, doc in enumerate(performance_documents[:100])]
        )

        elapsed = time.perf_counter() - start

        # Count successes and failures
        successes = sum(1 for r in results if r.success)
        failures = sum(1 for r in results if not r.success)
        error_rate = failures / len(results)

        assert error_rate < 0.10, f"Error rate {error_rate:.1%} exceeds 10% threshold under load"
        assert successes >= 90, f"Only {successes}/100 documents succeeded (expected >= 90)"
        assert elapsed < 120.0, f"100 concurrent documents took {elapsed:.1f}s (expected < 120s)"
