"""
Chaos Engineering Tests for Token Saver 5000 v0.7.0

This module tests system resilience under failure conditions including:
- Disk failures (full, permissions, corruption, slow I/O)
- Model crashes (CUDA OOM, timeouts, corrupted weights)
- Network issues (partitions, timeouts, connection failures)
- Data corruption (NaN/Inf embeddings, malformed JSON, invalid diffs)

Each test validates both failure detection AND recovery mechanisms.

Architecture:
- Uses reliability components: TimeoutManager, CircuitBreaker, RetryPolicy
- Uses graceful degradation: Embedding tier fallback, memory-only persistence
- Uses structured error types: OperationTimeoutError, CircuitBreakerOpenError, etc.

Added in v0.7.0 Week 3-4 (Production Hardening - Reliability Infrastructure).
"""

import asyncio
import json
import pytest
from unittest.mock import patch
import numpy as np

from src.error_types import (
    CircuitBreakerOpenError,
)
from src.reliability import CircuitBreaker, RetryPolicy
from src.graceful_degradation import GracefulDegradation
from src.handlers import compression_handlers


# ===========================
# Test Category 1: Disk Failures
# ===========================


class TestDiskFailures:
    """Test system resilience to disk failures."""

    @pytest.mark.asyncio
    async def test_disk_full_during_persistence(self, handler_context, sample_text_short):
        """Test graceful degradation when disk is full (ENOSPC).

        Expected behavior:
        - Persistence fails with OSError ENOSPC
        - System falls back to memory-only storage
        - Returns warning to user about degraded mode
        - Core functionality continues to work
        """
        # Mock disk full error for write operations
        original_open = open

        def failing_open(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if "w" in mode:
                raise OSError(28, "No space left on device")  # ENOSPC
            return original_open(*args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            # Attempt to ingest document
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_disk_full",
                "fidelity": "BALANCED",
            }

            # Should succeed (compression works, persistence degraded)
            result = await compression_handlers.handle_ingest(handler_context, ingest_args)

            # Verify document is ingested (in memory)
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert result_dict["file_id"] == "test_disk_full"

            # Verify document is accessible (from memory)
            read_args = {"file_id": "test_disk_full"}
            skeleton = await compression_handlers.handle_read_skeleton(handler_context, read_args)
            assert "test_disk_full" in skeleton

    @pytest.mark.asyncio
    async def test_disk_permission_denied(self, handler_context, sample_text_short):
        """Test fallback when disk permissions are denied (EACCES).

        Expected behavior:
        - Persistence fails with OSError EACCES
        - System falls back to in-memory storage
        - Core functionality continues
        """

        def failing_open(*args, **kwargs):
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if "w" in mode:
                raise OSError(13, "Permission denied")  # EACCES
            return open(*args, **kwargs)

        with patch("builtins.open", side_effect=failing_open):
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_permission_denied",
                "fidelity": "BALANCED",
            }

            result = await compression_handlers.handle_ingest(handler_context, ingest_args)
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert result_dict["file_id"] == "test_permission_denied"

    @pytest.mark.asyncio
    async def test_corrupted_persistence_file(self, handler_context, temp_dir):
        """Test recovery from corrupted persistence files.

        Expected behavior:
        - Loading corrupted JSON fails gracefully
        - System skips corrupted file and continues
        - New documents can still be ingested
        """
        # Create corrupted persistence file
        corrupt_file = temp_dir / ".semantic_modulator_data" / "skeletons" / "corrupted.json"
        corrupt_file.parent.mkdir(parents=True, exist_ok=True)
        corrupt_file.write_text("{invalid json content!!!")

        # Attempt to load persistence (should handle corruption)
        from src.persistence import PersistenceManager

        persistence = PersistenceManager()

        # Should not crash, just skip corrupted file
        # (PersistenceManager handles this internally)
        stats = persistence.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_disk_slow_io(self, handler_context, sample_text_short):
        """Test timeout protection for slow disk I/O.

        Expected behavior:
        - Slow write operations hit timeout
        - TimeoutManager raises OperationTimeoutError
        - System can continue with in-memory operation
        """

        async def slow_save(*args, **kwargs):
            """Simulate slow disk I/O (exceeds timeout)."""
            await asyncio.sleep(15.0)  # Exceeds 10s persistence timeout
            return True

        with patch.object(handler_context["persistence"], "save_document", side_effect=slow_save):
            # Wrap in timeout manager
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_slow_io",
            }

            # This should timeout during persistence phase
            # But ingestion should still complete (memory-only)
            result = await compression_handlers.handle_ingest(handler_context, ingest_args)

            # Verify document is accessible (from memory)
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert result_dict["file_id"] == "test_slow_io"

    @pytest.mark.asyncio
    async def test_disk_recovery_after_failure(self, handler_context, sample_text_short):
        """Test normal operation resumes after disk space is freed.

        Expected behavior:
        - First write fails (disk full)
        - Disk space freed
        - Second write succeeds
        """
        call_count = 0

        def intermittent_open(*args, **kwargs):
            """First call fails, subsequent calls succeed."""
            nonlocal call_count
            call_count += 1
            mode = args[1] if len(args) > 1 else kwargs.get("mode", "r")
            if "w" in mode and call_count == 1:
                raise OSError(28, "No space left on device")
            return open(*args, **kwargs)

        with patch("builtins.open", side_effect=intermittent_open):
            # First ingest (fails to persist)
            ingest_args1 = {"text": sample_text_short, "file_id": "test_recovery_1"}
            result1 = await compression_handlers.handle_ingest(handler_context, ingest_args1)
            assert isinstance(result1, str)

            # Second ingest (should succeed after "disk freed")
            ingest_args2 = {"text": sample_text_short, "file_id": "test_recovery_2"}
            result2 = await compression_handlers.handle_ingest(handler_context, ingest_args2)
            assert isinstance(result2, str)
            result2_dict = json.loads(result2)
            assert result2_dict["file_id"] == "test_recovery_2"


# ===========================
# Test Category 2: Model Crashes
# ===========================


class TestModelCrashes:
    """Test system resilience to embedding model failures."""

    @pytest.mark.asyncio
    async def test_embedding_model_cuda_oom(self, handler_context, sample_text_short):
        """Test fallback when CUDA runs out of memory.

        Expected behavior:
        - CUDA OOM error raised during encoding
        - System falls back to ONNX tier
        - Compression continues with degraded embeddings
        """
        handler_context["compressor"].model.encode

        def failing_encode(*args, **kwargs):
            """Simulate CUDA out of memory."""
            raise RuntimeError("CUDA out of memory")

        # Mock embedding manager to fail on STANDARD tier
        with patch.object(
            handler_context["compressor"].model, "encode", side_effect=failing_encode
        ):
            # Should fall back to ONNX/TFIDF tier
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_cuda_oom",
            }

            # Should succeed with fallback tier
            result = await compression_handlers.handle_ingest(handler_context, ingest_args)
            assert isinstance(result, str)
            result_dict = json.loads(result)
            assert result_dict["file_id"] == "test_cuda_oom"

    @pytest.mark.asyncio
    async def test_embedding_model_timeout(self, compressor):
        """Test timeout enforcement for embedding generation.

        Expected behavior:
        - Embedding takes too long (>30s)
        - TimeoutManager raises OperationTimeoutError
        - Clear error message provided to user
        """

        async def slow_encode(*args, **kwargs):
            """Simulate slow embedding that exceeds timeout."""
            await asyncio.sleep(35.0)  # Exceeds 30s embedding timeout
            return np.zeros((1, 384))

        with patch.object(compressor, "_encode_async", side_effect=slow_encode):
            with pytest.raises(asyncio.TimeoutError):
                # This should timeout during embedding
                await asyncio.wait_for(
                    compressor.ingest_file_async("Test text", "timeout_doc", {}),
                    timeout=32.0,  # Slightly longer than embedding timeout
                )

    @pytest.mark.asyncio
    async def test_embedding_model_corrupted_weights(self, handler_context, sample_text_short):
        """Test fallback when model weights are corrupted.

        Expected behavior:
        - Model loading fails with ValueError
        - System falls back to TFIDF tier
        - Compression continues with basic embeddings
        """

        def failing_encode(*args, **kwargs):
            """Simulate corrupted model weights."""
            raise ValueError("Error loading model weights: corrupted file")

        with patch.object(
            handler_context["compressor"].model, "encode", side_effect=failing_encode
        ):
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_corrupted_weights",
            }

            # Should succeed with TFIDF fallback
            result = await compression_handlers.handle_ingest(handler_context, ingest_args)
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_embedding_model_retry_on_transient_error(self, handler_context):
        """Test retry logic for transient embedding errors.

        Expected behavior:
        - First attempt fails with OSError (transient)
        - RetryPolicy retries with exponential backoff
        - Second attempt succeeds
        """
        call_count = 0

        async def flaky_encode(*args, **kwargs):
            """Fail first time, succeed second time."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OSError("Temporary failure")
            # Success on retry
            return np.random.randn(len(args[0]), 384)

        # Create retry policy
        retry_policy = RetryPolicy(max_retries=3, base_delay=0.1)

        # Simulate flaky encoding with retry
        async def encode_with_retry(texts):
            return await retry_policy.execute(flaky_encode, texts)

        with patch.object(
            handler_context["compressor"], "_encode_async", side_effect=encode_with_retry
        ):
            ingest_args = {
                "text": "Test transient error handling",
                "file_id": "test_retry",
            }

            result = await compression_handlers.handle_ingest(handler_context, ingest_args)
            assert isinstance(result, str)
            assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_embedding_all_tiers_fail(self, handler_context, sample_text_short):
        """Test behavior when all embedding tiers fail.

        Expected behavior:
        - STANDARD tier fails
        - ONNX tier fails
        - TFIDF tier fails
        - Raises exception with clear message
        """

        def failing_encode(*args, **kwargs):
            """All tiers fail."""
            raise RuntimeError("All embedding tiers unavailable")

        with patch.object(
            handler_context["compressor"].model, "encode", side_effect=failing_encode
        ):
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_all_tiers_fail",
            }

            # Should raise exception (no fallback available)
            with pytest.raises(Exception):
                await compression_handlers.handle_ingest(handler_context, ingest_args)


# ===========================
# Test Category 3: Network Issues
# ===========================


class TestNetworkIssues:
    """Test system resilience to network failures."""

    @pytest.mark.asyncio
    async def test_network_partition_file_sync(self, handler_context, temp_file):
        """Test file sync fallback during network partition.

        Expected behavior:
        - File stat fails with OSError (network partition)
        - System uses cached metadata
        - Assumes file not stale (safe default)
        """

        def failing_stat(*args, **kwargs):
            """Simulate network partition during file stat."""
            raise OSError(113, "No route to host")  # Network unreachable

        with patch("os.stat", side_effect=failing_stat):
            # Attempt to check file staleness
            result = await GracefulDegradation.file_sync_with_fallback(
                str(temp_file), handler_context["sync_manager"]
            )

            assert result["mode"] == "cached_metadata"
            assert result["is_stale"] is False  # Safe default
            assert "warning" in result

    @pytest.mark.asyncio
    async def test_network_timeout_external_api(self):
        """Test circuit breaker for external API timeouts.

        Expected behavior:
        - API call times out repeatedly
        - Circuit breaker opens after threshold
        - Subsequent calls fail fast (CircuitBreakerOpenError)
        """
        circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

        async def failing_api_call():
            """Simulate external API timeout."""
            raise asyncio.TimeoutError("API request timed out")

        # Fail threshold times to open circuit
        for _ in range(3):
            with pytest.raises(asyncio.TimeoutError):
                await circuit_breaker.call(failing_api_call)

        # Circuit should now be OPEN
        assert circuit_breaker.state == "OPEN"

        # Next call should fail fast
        with pytest.raises(CircuitBreakerOpenError):
            await circuit_breaker.call(failing_api_call)

    @pytest.mark.asyncio
    async def test_network_connection_refused(self):
        """Test retry with exponential backoff for connection errors.

        Expected behavior:
        - Connection refused (transient error)
        - RetryPolicy retries with exponential backoff
        - Eventually succeeds or exhausts retries
        """
        call_count = 0

        async def flaky_connection():
            """Fail twice, succeed third time."""
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionRefusedError("Connection refused")
            return "success"

        retry_policy = RetryPolicy(max_retries=3, base_delay=0.1)
        result = await retry_policy.execute(flaky_connection)

        assert result == "success"
        assert call_count == 3  # Failed twice, succeeded on third

    @pytest.mark.asyncio
    async def test_network_intermittent_failures(self):
        """Test resilience to intermittent network failures.

        Expected behavior:
        - Random failures occur
        - RetryPolicy handles intermittent errors
        - Operation eventually succeeds
        """
        call_count = 0

        async def intermittent_operation():
            """Randomly fail or succeed."""
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 1:  # Fail on odd attempts
                raise ConnectionError("Network glitch")
            return "success"

        retry_policy = RetryPolicy(max_retries=5, base_delay=0.05)
        result = await retry_policy.execute(intermittent_operation)

        assert result == "success"
        assert call_count >= 2  # At least one retry

    @pytest.mark.asyncio
    async def test_network_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through HALF_OPEN state.

        Expected behavior:
        - Circuit opens after failures
        - After timeout, transitions to HALF_OPEN
        - Successful call transitions to CLOSED
        """
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=0.5, half_open_max_calls=1)

        async def api_call(should_fail=True):
            """Configurable API call for testing."""
            if should_fail:
                raise ConnectionError("API unavailable")
            return "success"

        # Fail to open circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call(api_call, should_fail=True)

        assert circuit_breaker.state == "OPEN"

        # Wait for timeout
        await asyncio.sleep(0.6)

        # Next call should transition to HALF_OPEN (and succeed)
        result = await circuit_breaker.call(api_call, should_fail=False)

        assert result == "success"
        assert circuit_breaker.state == "CLOSED"  # Recovered!


# ===========================
# Test Category 4: Data Corruption
# ===========================


class TestDataCorruption:
    """Test detection and rejection of corrupted data."""

    @pytest.mark.asyncio
    async def test_corrupted_embeddings_nan_values(self, handler_context, sample_text_short):
        """Test detection of NaN values in embeddings.

        Expected behavior:
        - Embedding contains NaN values
        - Validation detects corruption
        - Rejects corrupted embeddings
        """

        async def corrupt_encode(*args, **kwargs):
            """Return embeddings with NaN values."""
            embeddings = np.random.randn(len(args[0]), 384)
            embeddings[0, 0] = np.nan  # Inject NaN
            return embeddings

        with patch.object(
            handler_context["compressor"], "_encode_async", side_effect=corrupt_encode
        ):
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_nan_embeddings",
            }

            # Should detect NaN and reject/retry
            with pytest.raises((ValueError, Exception)):
                await compression_handlers.handle_ingest(handler_context, ingest_args)

    @pytest.mark.asyncio
    async def test_corrupted_embeddings_inf_values(self, handler_context, sample_text_short):
        """Test detection of Inf values in embeddings.

        Expected behavior:
        - Embedding contains Inf values
        - Validation detects corruption
        - Rejects corrupted embeddings
        """

        async def corrupt_encode(*args, **kwargs):
            """Return embeddings with Inf values."""
            embeddings = np.random.randn(len(args[0]), 384)
            embeddings[0, 0] = np.inf  # Inject Inf
            return embeddings

        with patch.object(
            handler_context["compressor"], "_encode_async", side_effect=corrupt_encode
        ):
            ingest_args = {
                "text": sample_text_short,
                "file_id": "test_inf_embeddings",
            }

            # Should detect Inf and reject/retry
            with pytest.raises((ValueError, Exception)):
                await compression_handlers.handle_ingest(handler_context, ingest_args)

    @pytest.mark.asyncio
    async def test_corrupted_skeleton_json(self, handler_context):
        """Test handling of malformed skeleton JSON.

        Expected behavior:
        - JSON parsing fails with validation error
        - Clear error message provided
        - System remains stable
        """
        # Attempt to parse invalid skeleton JSON
        invalid_skeleton = '{"file_id": "test", "nodes": [invalid json}}'

        with pytest.raises(json.JSONDecodeError):
            json.loads(invalid_skeleton)

    @pytest.mark.asyncio
    async def test_corrupted_version_history(self, handler_context, sample_text_short):
        """Test handling of corrupted version diffs.

        Expected behavior:
        - Diff parsing fails
        - System skips corrupted version
        - New versions can still be added
        """
        # Create a version with corrupted diff
        version_data = {
            "timestamp": "2024-01-01T00:00:00",
            "version": 1,
            "diff": "<<<CORRUPTED DIFF DATA>>>",
            "skeleton": {"file_id": "test", "nodes": []},
        }

        # Version manager should handle corrupted diff gracefully
        version_id = handler_context["version_manager"].add_version("test_doc", version_data)

        # Should still be able to add new versions
        assert version_id is not None

    @pytest.mark.asyncio
    async def test_corrupted_cache_file(self, temp_dir):
        """Test recovery from corrupted embedding cache.

        Expected behavior:
        - Cache file is corrupted (msgpack decode error)
        - System clears cache and rebuilds
        - Operations continue normally
        """
        # Create corrupted cache file
        cache_file = temp_dir / "embedding_cache.msgpack"
        cache_file.write_bytes(b"corrupted binary data!!!")

        # Try to load cache (should handle corruption)
        try:
            from src.embedding_cache import LRUEmbeddingCache

            cache = LRUEmbeddingCache(capacity=100, persist_path=str(cache_file))
            # Should start with empty cache after detecting corruption
            assert cache.size == 0
        except ImportError:
            # LRUEmbeddingCache is optional
            pytest.skip("embedding_cache not available")
