"""
Reliability Infrastructure for Production MCP Server

This module provides production-ready reliability patterns including:
- Configurable timeouts to prevent server hangs
- Circuit breakers to prevent cascading failures
- Retry logic with exponential backoff for transient errors

Architecture:
- TimeoutManager: Wraps async operations with configurable timeouts
- CircuitBreaker: Prevents cascading failures with CLOSED/OPEN/HALF_OPEN states
- RetryPolicy: Handles transient errors with exponential backoff

Best Practices:
1. All async operations should have timeouts (prevents indefinite hangs)
2. External dependencies should use circuit breakers (fail fast on outages)
3. Transient errors should be retried (improves resilience)

Usage:
    from src.reliability import TimeoutManager, CircuitBreaker, RetryPolicy

    # Timeout example
    result = await TimeoutManager.with_timeout(
        async_operation(), operation="embedding_generation"
    )

    # Circuit breaker example
    breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
    result = await breaker.call(risky_function, arg1, arg2)

    # Retry example
    policy = RetryPolicy(max_retries=3, base_delay=1.0)
    result = await policy.execute(flaky_function, arg1, arg2)
"""

import asyncio
import time
from typing import Any, Callable, Dict, Optional, Tuple

from .error_types import (
    CircuitBreakerOpenError,
    OperationTimeoutError,
    RetryExhaustedError,
)
from .structured_logging import get_logger

logger = get_logger(__name__)


class TimeoutManager:
    """
    Configurable timeout enforcement for all async operations.

    Prevents server hangs by wrapping async operations with timeouts.
    Timeouts are configurable per operation type (embedding, compression, etc.).

    Example:
        result = await TimeoutManager.with_timeout(
            slow_async_operation(),
            operation="embedding_generation"
        )
    """

    # Timeout configuration (seconds)
    TIMEOUT_CONFIG: Dict[str, float] = {
        "embedding_generation": 30.0,  # 30s per embedding batch
        "graph_construction": 60.0,  # 1m for large documents
        "compression": 120.0,  # 2m for complex graphs
        "persistence": 10.0,  # 10s for disk I/O
        "file_sync": 5.0,  # 5s for file stat operations
        "default": 30.0,  # Default timeout for unknown operations
    }

    @staticmethod
    async def with_timeout(coro, operation: str, timeout: Optional[float] = None) -> Any:
        """
        Wrap async operation with configured timeout.

        Args:
            coro: Async coroutine to execute
            operation: Operation type (used to lookup timeout)
            timeout: Optional explicit timeout (overrides configured value)

        Returns:
            Result of coroutine execution

        Raises:
            OperationTimeoutError: If operation exceeds timeout
        """
        # Get timeout from config or use explicit value
        if timeout is None:
            timeout = TimeoutManager.TIMEOUT_CONFIG.get(
                operation, TimeoutManager.TIMEOUT_CONFIG["default"]
            )

        try:
            logger.debug(
                "timeout_started",
                operation=operation,
                timeout_seconds=timeout,
            )
            result = await asyncio.wait_for(coro, timeout=timeout)
            logger.debug("timeout_completed", operation=operation)
            return result

        except asyncio.TimeoutError:
            logger.error(
                "operation_timeout",
                operation=operation,
                timeout_seconds=timeout,
            )
            raise OperationTimeoutError(operation, timeout)

    @staticmethod
    def configure_timeout(operation: str, timeout: float) -> None:
        """
        Configure timeout for specific operation type.

        Args:
            operation: Operation type to configure
            timeout: Timeout in seconds

        Example:
            TimeoutManager.configure_timeout("embedding_generation", 45.0)
        """
        TimeoutManager.TIMEOUT_CONFIG[operation] = timeout
        logger.info(
            "timeout_configured",
            operation=operation,
            timeout_seconds=timeout,
        )


class CircuitBreaker:
    """
    Prevent cascading failures with circuit breaker pattern.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject all requests (fail fast)
    - HALF_OPEN: Test if service recovered, allow one request

    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        result = await breaker.call(external_service_call, arg1, arg2)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of consecutive failures before opening
            timeout: Seconds to wait before transitioning from OPEN to HALF_OPEN
            half_open_max_calls: Max calls allowed in HALF_OPEN state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.half_open_calls = 0

        logger.info(
            "circuit_breaker_initialized",
            failure_threshold=failure_threshold,
            timeout_seconds=timeout,
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function execution

        Raises:
            CircuitBreakerOpenError: If circuit breaker is OPEN
            Original exception: If function fails
        """
        # Check circuit breaker state
        if self.state == "OPEN":
            # Check if timeout has elapsed
            if time.time() - self.last_failure_time < self.timeout:
                logger.warning(
                    "circuit_breaker_blocked",
                    state="OPEN",
                    failure_count=self.failure_count,
                )
                raise CircuitBreakerOpenError(
                    component=func.__name__,
                    failure_count=self.failure_count,
                )

            # Transition to HALF_OPEN
            self.state = "HALF_OPEN"
            self.half_open_calls = 0
            logger.info(
                "circuit_breaker_state_change",
                old_state="OPEN",
                new_state="HALF_OPEN",
            )

        # HALF_OPEN: Limit concurrent calls
        if self.state == "HALF_OPEN":
            if self.half_open_calls >= self.half_open_max_calls:
                logger.warning(
                    "circuit_breaker_half_open_limit",
                    half_open_calls=self.half_open_calls,
                    limit=self.half_open_max_calls,
                )
                raise CircuitBreakerOpenError(
                    component=func.__name__,
                    failure_count=self.failure_count,
                )
            self.half_open_calls += 1

        try:
            # Execute function
            result = await func(*args, **kwargs)

            # Success: Update state
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count += 1
                logger.info(
                    "circuit_breaker_state_change",
                    old_state="HALF_OPEN",
                    new_state="CLOSED",
                    reason="success_after_half_open",
                )
            else:
                self.success_count += 1

            return result

        except Exception as e:
            # Failure: Update state
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                if self.state != "OPEN":
                    self.state = "OPEN"
                    logger.error(
                        "circuit_breaker_state_change",
                        old_state="CLOSED/HALF_OPEN",
                        new_state="OPEN",
                        failure_count=self.failure_count,
                        threshold=self.failure_threshold,
                    )

            logger.error(
                "circuit_breaker_failure",
                state=self.state,
                failure_count=self.failure_count,
                error_type=type(e).__name__,
            )

            raise

    def get_state(self) -> Dict[str, Any]:
        """
        Get current circuit breaker state.

        Returns:
            Dictionary with state, failure_count, success_count
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
        }

    def reset(self) -> None:
        """Reset circuit breaker to CLOSED state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        logger.info("circuit_breaker_reset")


# Transient errors that should trigger retry
TRANSIENT_ERRORS = (
    OSError,  # Disk I/O errors (disk full, permissions)
    TimeoutError,  # Network timeouts
    asyncio.TimeoutError,  # Async timeouts
    ConnectionError,  # Network connection errors
    # Add more transient errors as needed
)


class RetryPolicy:
    """
    Configurable retry with exponential backoff.

    Handles transient errors by retrying with increasing delays.
    Uses exponential backoff to avoid overwhelming failing services.

    Example:
        policy = RetryPolicy(max_retries=3, base_delay=1.0)
        result = await policy.execute(flaky_function, arg1, arg2)
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        retryable_exceptions: Optional[Tuple[type, ...]] = None,
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay before first retry (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Multiplier for exponential backoff
            retryable_exceptions: Tuple of exceptions to retry (default: TRANSIENT_ERRORS)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.retryable_exceptions = retryable_exceptions or TRANSIENT_ERRORS

        logger.debug(
            "retry_policy_initialized",
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_factor=backoff_factor,
        )

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Result of function execution

        Raises:
            RetryExhaustedError: If all retries exhausted
            Original exception: If non-retryable exception occurs
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Success: Log if retried
                if attempt > 0:
                    logger.info(
                        "retry_success",
                        function=func.__name__,
                        attempt=attempt,
                        total_retries=self.max_retries,
                    )

                return result

            except self.retryable_exceptions as e:
                last_exception = e

                # Check if we should retry
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff
                    delay = min(
                        self.base_delay * (self.backoff_factor**attempt),
                        self.max_delay,
                    )

                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_retries=self.max_retries,
                        delay_seconds=delay,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

                    await asyncio.sleep(delay)
                else:
                    # All retries exhausted
                    logger.error(
                        "retry_exhausted",
                        function=func.__name__,
                        max_retries=self.max_retries,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )
                    raise RetryExhaustedError(
                        operation=func.__name__,
                        max_retries=self.max_retries,
                        last_exception=e,
                    )

            except Exception as e:
                # Non-retryable exception, fail immediately
                logger.error(
                    "non_retryable_error",
                    function=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise

        # Should never reach here
        raise RetryExhaustedError(
            operation=func.__name__,
            max_retries=self.max_retries,
            last_exception=last_exception,
        )


# Pre-configured retry policies for common operations
RETRY_POLICIES: Dict[str, RetryPolicy] = {
    "embedding_generation": RetryPolicy(max_retries=2, base_delay=1.0),
    "persistence": RetryPolicy(max_retries=3, base_delay=0.5),
    "file_sync": RetryPolicy(max_retries=2, base_delay=1.0),
}
