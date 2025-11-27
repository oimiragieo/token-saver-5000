"""
Custom Exception Types for Reliability Infrastructure

This module defines custom exception types for the reliability infrastructure,
including timeout errors, circuit breaker errors, retry exhaustion, and rate
limiting errors.

Architecture:
- Clear exception hierarchy with base ReliabilityError
- Specific exceptions for different failure modes
- Detailed error messages with context

Usage:
    from src.error_types import OperationTimeoutError, CircuitBreakerOpenError

    raise OperationTimeoutError("embedding_generation", timeout=30.0)
    raise CircuitBreakerOpenError("persistence", failure_count=5)
"""


class ReliabilityError(Exception):
    """Base exception for all reliability infrastructure errors."""

    pass


class OperationTimeoutError(ReliabilityError):
    """Raised when an operation exceeds its configured timeout."""

    def __init__(self, operation: str, timeout: float):
        self.operation = operation
        self.timeout = timeout
        super().__init__(f"{operation} exceeded {timeout}s timeout")


class CircuitBreakerOpenError(ReliabilityError):
    """Raised when circuit breaker is in OPEN state (too many failures)."""

    def __init__(self, component: str, failure_count: int = None):
        self.component = component
        self.failure_count = failure_count
        message = f"Circuit breaker for {component} is OPEN"
        if failure_count:
            message += f" ({failure_count} consecutive failures)"
        super().__init__(message)


class RetryExhaustedError(ReliabilityError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, operation: str, max_retries: int, last_exception: Exception = None):
        self.operation = operation
        self.max_retries = max_retries
        self.last_exception = last_exception
        message = f"{operation} failed after {max_retries} retries"
        if last_exception:
            message += f": {type(last_exception).__name__}: {last_exception}"
        super().__init__(message)


class RateLimitExceededError(ReliabilityError):
    """Raised when rate limit is exceeded (too many requests)."""

    def __init__(self, operation: str, rate: float, wait_time: float = None):
        self.operation = operation
        self.rate = rate
        self.wait_time = wait_time
        message = f"Rate limit exceeded for {operation} (limit: {rate} requests/sec)"
        if wait_time:
            message += f", retry in {wait_time:.1f}s"
        super().__init__(message)


class GracefulDegradationError(ReliabilityError):
    """Raised when graceful degradation fallback is being used."""

    def __init__(self, component: str, fallback: str, reason: str = None):
        self.component = component
        self.fallback = fallback
        self.reason = reason
        message = f"{component} degraded to {fallback}"
        if reason:
            message += f": {reason}"
        super().__init__(message)
