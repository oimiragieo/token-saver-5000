"""
Comprehensive Tests for Reliability Infrastructure

Tests for TimeoutManager, CircuitBreaker, RetryPolicy, and RateLimiter.

Coverage:
- Timeout enforcement and configuration
- Circuit breaker state transitions
- Retry logic with exponential backoff
- Rate limiting with token bucket algorithm
- Graceful degradation fallback strategies

Total: 30+ comprehensive reliability tests
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.error_types import (
    CircuitBreakerOpenError,
    OperationTimeoutError,
    RateLimitExceededError,
    RetryExhaustedError,
)
from src.graceful_degradation import GracefulDegradation
from src.rate_limiter import RateLimiter, configure_rate_limiter, get_rate_limiter
from src.reliability import CircuitBreaker, RetryPolicy, TimeoutManager

# =============================================================================
# TimeoutManager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_timeout_manager_successful_operation():
    """Test TimeoutManager allows successful operation within timeout."""

    async def fast_operation():
        await asyncio.sleep(0.1)
        return "success"

    result = await TimeoutManager.with_timeout(
        fast_operation(),
        operation="test",
        timeout=1.0,
    )

    assert result == "success"


@pytest.mark.asyncio
async def test_timeout_manager_timeout_exceeded():
    """Test TimeoutManager raises OperationTimeoutError when timeout exceeded."""

    async def slow_operation():
        await asyncio.sleep(2.0)
        return "too slow"

    with pytest.raises(OperationTimeoutError) as exc_info:
        await TimeoutManager.with_timeout(
            slow_operation(),
            operation="test",
            timeout=0.5,
        )

    assert exc_info.value.operation == "test"
    assert exc_info.value.timeout == 0.5


@pytest.mark.asyncio
async def test_timeout_manager_configured_timeout():
    """Test TimeoutManager uses configured timeout for operation type."""
    TimeoutManager.configure_timeout("custom_op", 0.3)

    async def operation():
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(OperationTimeoutError) as exc_info:
        await TimeoutManager.with_timeout(operation(), operation="custom_op")

    assert exc_info.value.operation == "custom_op"
    assert exc_info.value.timeout == 0.3


@pytest.mark.asyncio
async def test_timeout_manager_explicit_timeout_override():
    """Test explicit timeout overrides configured timeout."""
    TimeoutManager.configure_timeout("test_op", 10.0)

    async def operation():
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(OperationTimeoutError):
        await TimeoutManager.with_timeout(
            operation(),
            operation="test_op",
            timeout=0.2,  # Override configured timeout
        )


# =============================================================================
# CircuitBreaker Tests
# =============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_closed_state():
    """Test CircuitBreaker starts in CLOSED state and allows requests."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

    async def successful_operation():
        return "success"

    result = await breaker.call(successful_operation)
    assert result == "success"
    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    """Test CircuitBreaker opens after failure threshold exceeded."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)

    call_count = 0

    async def failing_operation():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Service unavailable")

    # Fail 3 times (threshold)
    for i in range(3):
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

    assert breaker.state == "OPEN"
    assert breaker.failure_count == 3
    assert call_count == 3


@pytest.mark.asyncio
async def test_circuit_breaker_rejects_when_open():
    """Test CircuitBreaker rejects requests when OPEN."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=10.0)

    async def failing_operation():
        raise RuntimeError("Fail")

    # Trigger circuit breaker to OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

    # Should now reject with CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError) as exc_info:
        await breaker.call(failing_operation)

    assert "OPEN" in str(exc_info.value)


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_after_timeout():
    """Test CircuitBreaker transitions to HALF_OPEN after timeout."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.5)

    async def failing_operation():
        raise RuntimeError("Fail")

    async def successful_operation():
        return "success"

    # Trigger OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

    assert breaker.state == "OPEN"

    # Wait for timeout
    await asyncio.sleep(0.6)

    # Should transition to HALF_OPEN and allow request
    result = await breaker.call(successful_operation)
    assert result == "success"
    assert breaker.state == "CLOSED"  # Success in HALF_OPEN closes circuit


@pytest.mark.asyncio
async def test_circuit_breaker_reopens_on_half_open_failure():
    """Test CircuitBreaker reopens if HALF_OPEN request fails."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=0.3)

    async def failing_operation():
        raise RuntimeError("Still failing")

    # Trigger OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

    # Wait for timeout to HALF_OPEN
    await asyncio.sleep(0.4)

    # Failure in HALF_OPEN should reopen circuit
    with pytest.raises(RuntimeError):
        await breaker.call(failing_operation)

    assert breaker.state == "OPEN"


@pytest.mark.asyncio
async def test_circuit_breaker_reset():
    """Test CircuitBreaker reset() restores CLOSED state."""
    breaker = CircuitBreaker(failure_threshold=2)

    async def failing_operation():
        raise RuntimeError("Fail")

    # Trigger OPEN
    for _ in range(2):
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

    assert breaker.state == "OPEN"

    # Reset
    breaker.reset()

    assert breaker.state == "CLOSED"
    assert breaker.failure_count == 0


# =============================================================================
# RetryPolicy Tests
# =============================================================================


@pytest.mark.asyncio
async def test_retry_policy_success_no_retry():
    """Test RetryPolicy succeeds on first attempt (no retry needed)."""
    policy = RetryPolicy(max_retries=3)

    call_count = 0

    async def successful_operation():
        nonlocal call_count
        call_count += 1
        return "success"

    result = await policy.execute(successful_operation)
    assert result == "success"
    assert call_count == 1  # Only called once


@pytest.mark.asyncio
async def test_retry_policy_success_after_retries():
    """Test RetryPolicy succeeds after transient failures."""
    policy = RetryPolicy(max_retries=3, base_delay=0.1)

    call_count = 0

    async def flaky_operation():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise OSError("Transient error")
        return "success"

    start = time.time()
    result = await policy.execute(flaky_operation)
    elapsed = time.time() - start

    assert result == "success"
    assert call_count == 3
    assert elapsed >= 0.1  # At least one retry delay


@pytest.mark.asyncio
async def test_retry_policy_exhausted():
    """Test RetryPolicy raises RetryExhaustedError after max retries."""
    policy = RetryPolicy(max_retries=2, base_delay=0.05)

    call_count = 0

    async def always_failing():
        nonlocal call_count
        call_count += 1
        raise OSError("Always fails")

    with pytest.raises(RetryExhaustedError) as exc_info:
        await policy.execute(always_failing)

    assert exc_info.value.max_retries == 2
    assert call_count == 3  # Initial attempt + 2 retries


@pytest.mark.asyncio
async def test_retry_policy_exponential_backoff():
    """Test RetryPolicy uses exponential backoff."""
    policy = RetryPolicy(
        max_retries=3,
        base_delay=0.1,
        backoff_factor=2.0,
    )

    call_times = []

    async def failing_operation():
        call_times.append(time.time())
        raise OSError("Fail")

    with pytest.raises(RetryExhaustedError):
        await policy.execute(failing_operation)

    # Verify exponential backoff delays
    delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]

    # Delays should be approximately: 0.1, 0.2, 0.4
    assert len(delays) == 3
    assert delays[0] >= 0.1
    assert delays[1] >= 0.2
    assert delays[2] >= 0.4


@pytest.mark.asyncio
async def test_retry_policy_non_retryable_exception():
    """Test RetryPolicy doesn't retry non-retryable exceptions."""
    policy = RetryPolicy(max_retries=3)

    call_count = 0

    async def non_retryable_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Not retryable")

    with pytest.raises(ValueError):
        await policy.execute(non_retryable_error)

    assert call_count == 1  # No retries for ValueError


# =============================================================================
# RateLimiter Tests
# =============================================================================


@pytest.mark.asyncio
async def test_rate_limiter_within_capacity():
    """Test RateLimiter allows requests within capacity."""
    limiter = RateLimiter(rate=10.0, capacity=10, name="test")

    # Should acquire 5 tokens immediately
    result = await limiter.acquire(tokens=5)
    assert result is True

    stats = limiter.get_stats()
    assert stats["current_tokens"] == 5.0


@pytest.mark.asyncio
async def test_rate_limiter_burst_capacity():
    """Test RateLimiter allows burst up to capacity."""
    limiter = RateLimiter(rate=5.0, capacity=20, name="test")

    # Burst: Acquire full capacity
    result = await limiter.acquire(tokens=20)
    assert result is True

    stats = limiter.get_stats()
    assert stats["current_tokens"] == 0.0


@pytest.mark.asyncio
async def test_rate_limiter_refill():
    """Test RateLimiter refills tokens over time."""
    limiter = RateLimiter(rate=10.0, capacity=20, name="test")

    # Consume all tokens
    await limiter.acquire(tokens=20)
    assert limiter.tokens == 0.0

    # Wait 0.5 seconds: Should refill 5 tokens (10 tokens/sec * 0.5 sec)
    await asyncio.sleep(0.5)

    # Trigger refill by checking stats
    await limiter.acquire(tokens=1)
    assert limiter.tokens >= 3.0  # At least 4 tokens refilled


@pytest.mark.asyncio
async def test_rate_limiter_blocking_wait():
    """Test RateLimiter blocks and waits for tokens."""
    limiter = RateLimiter(rate=10.0, capacity=5, name="test")

    # Consume all tokens
    await limiter.acquire(tokens=5)

    # Request more tokens: Should block and wait
    start = time.time()
    await limiter.acquire(tokens=5)
    elapsed = time.time() - start

    # Should wait approximately 0.5 seconds (5 tokens / 10 tokens per sec)
    assert elapsed >= 0.4


@pytest.mark.asyncio
async def test_rate_limiter_non_blocking_reject():
    """Test RateLimiter rejects when non-blocking and tokens unavailable."""
    limiter = RateLimiter(rate=5.0, capacity=10, name="test")

    # Consume all tokens
    await limiter.acquire(tokens=10)

    # Non-blocking: Should reject
    with pytest.raises(RateLimitExceededError):
        await limiter.acquire(tokens=5, blocking=False)

    stats = limiter.get_stats()
    assert stats["rejected_requests"] == 1


@pytest.mark.asyncio
async def test_rate_limiter_statistics():
    """Test RateLimiter provides accurate statistics."""
    limiter = RateLimiter(rate=10.0, capacity=10, name="test_stats")

    # Make requests
    await limiter.acquire(tokens=3)
    await limiter.acquire(tokens=2)

    with pytest.raises(RateLimitExceededError):
        await limiter.acquire(tokens=10, blocking=False)

    stats = limiter.get_stats()
    assert stats["total_requests"] == 3
    assert stats["rejected_requests"] == 1
    assert stats["rejection_rate_percent"] > 0


# =============================================================================
# Global Rate Limiter Tests
# =============================================================================


def test_get_rate_limiter():
    """Test get_rate_limiter() returns configured limiter."""
    limiter = get_rate_limiter("ingest")
    assert limiter.name == "ingest"
    assert limiter.rate == 10.0


def test_get_rate_limiter_not_configured():
    """Test get_rate_limiter() raises KeyError for unknown operation."""
    with pytest.raises(KeyError):
        get_rate_limiter("unknown_operation")


def test_configure_rate_limiter():
    """Test configure_rate_limiter() adds new limiter."""
    configure_rate_limiter("custom_op", rate=15.0, capacity=30)

    limiter = get_rate_limiter("custom_op")
    assert limiter.rate == 15.0
    assert limiter.capacity == 30


# =============================================================================
# Graceful Degradation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_graceful_degradation_embedding_fallback():
    """Test GracefulDegradation.embed_with_fallback() falls back on failure."""

    # Mock embeddings manager with tier fallback
    embeddings_manager = AsyncMock()

    call_count = {"STANDARD": 0, "ONNX": 0, "TFIDF": 0}

    async def mock_encode(texts, tier):
        call_count[tier] += 1
        if tier == "STANDARD":
            raise RuntimeError("STANDARD tier failed")
        elif tier == "ONNX":
            raise RuntimeError("ONNX tier failed")
        else:  # TFIDF
            return [[0.1, 0.2, 0.3]]  # Success

    embeddings_manager.encode = mock_encode

    result = await GracefulDegradation.embed_with_fallback(
        ["test text"],
        embeddings_manager,
        preferred_tier="STANDARD",
    )

    # Should try STANDARD, ONNX, then succeed with TFIDF
    assert call_count["STANDARD"] == 1
    assert call_count["ONNX"] == 1
    assert call_count["TFIDF"] == 1
    assert result == [[0.1, 0.2, 0.3]]


@pytest.mark.asyncio
async def test_graceful_degradation_persistence_fallback():
    """Test GracefulDegradation.persist_with_fallback() falls back to memory."""

    # Mock persistence manager that fails
    persistence_manager = AsyncMock()
    persistence_manager.save_document.side_effect = OSError("Disk full")

    result = await GracefulDegradation.persist_with_fallback(
        "doc_id",
        {"data": "test"},
        persistence_manager,
    )

    assert result["success"] is False
    assert result["mode"] == "memory"
    assert "warning" in result


@pytest.mark.asyncio
async def test_graceful_degradation_file_sync_fallback():
    """Test GracefulDegradation.file_sync_with_fallback() uses cached metadata."""

    # Mock file sync manager that fails stat check
    file_sync_manager = AsyncMock()
    file_sync_manager.check_staleness.side_effect = OSError("Network unreachable")

    result = await GracefulDegradation.file_sync_with_fallback(
        "/path/to/file.txt",
        file_sync_manager,
    )

    assert result["is_stale"] is False  # Assumes not stale if can't check
    assert result["mode"] == "cached_metadata"
    assert "warning" in result


# =============================================================================
# Integration Tests (Multiple Components)
# =============================================================================


@pytest.mark.asyncio
async def test_timeout_with_retry_integration():
    """Test TimeoutManager and RetryPolicy work together."""
    # Configure OperationTimeoutError as retryable
    policy = RetryPolicy(
        max_retries=2,
        base_delay=0.1,
        retryable_exceptions=(OSError, TimeoutError, asyncio.TimeoutError, OperationTimeoutError),
    )

    call_count = 0

    async def slow_then_fast():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(2.0)  # Will timeout
        return "success"

    async def operation_with_timeout():
        return await TimeoutManager.with_timeout(
            slow_then_fast(),
            operation="test",
            timeout=0.5,
        )

    # First attempt times out, second succeeds
    result = await policy.execute(operation_with_timeout)
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_circuit_breaker_with_retry_integration():
    """Test CircuitBreaker and RetryPolicy work together."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=0.3)
    policy = RetryPolicy(max_retries=5, base_delay=0.05)

    call_count = 0

    async def intermittent_failure():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise OSError("Transient failure")
        return "success"

    result = await policy.execute(lambda: breaker.call(intermittent_failure))
    assert result == "success"
    assert call_count == 3
