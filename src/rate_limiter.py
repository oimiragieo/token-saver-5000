"""
Rate Limiting Infrastructure for Resource Protection

This module provides token bucket rate limiting to prevent resource exhaustion
from unbounded concurrent requests.

Architecture:
- Token bucket algorithm for smooth rate limiting
- Async-safe with asyncio locks
- Configurable rate and burst capacity
- Automatic token refill based on elapsed time

Best Practices:
1. Apply rate limiting to resource-intensive operations
2. Configure rate based on system capacity
3. Set burst capacity for handling traffic spikes
4. Monitor rate limit hits for capacity planning

Usage:
    from src.rate_limiter import RateLimiter, RATE_LIMITERS

    # Global rate limiter
    await RATE_LIMITERS["ingest"].acquire()

    # Custom rate limiter
    limiter = RateLimiter(rate=10.0, capacity=20)
    await limiter.acquire(tokens=5)
"""

import asyncio
import time
from typing import Dict

from .error_types import RateLimitExceededError
from .structured_logging import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API-style operations.

    Implements token bucket algorithm:
    - Tokens refill at constant rate
    - Requests consume tokens
    - Burst capacity allows temporary spikes
    - Blocks when tokens unavailable

    Example:
        limiter = RateLimiter(rate=10.0, capacity=20)
        await limiter.acquire()  # Acquire 1 token
        await limiter.acquire(tokens=5)  # Acquire 5 tokens
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        name: str = "unnamed",
    ):
        """
        Initialize token bucket rate limiter.

        Args:
            rate: Tokens per second (refill rate)
            capacity: Maximum bucket capacity (burst limit)
            name: Name for logging purposes
        """
        self.rate = rate  # Tokens per second
        self.capacity = capacity  # Bucket capacity
        self.name = name

        self.tokens = float(capacity)  # Start with full bucket
        self.last_update = time.time()
        self.lock = asyncio.Lock()

        # Statistics
        self.total_requests = 0
        self.rejected_requests = 0
        self.total_wait_time = 0.0

        logger.info(
            "rate_limiter_initialized",
            name=name,
            rate=rate,
            capacity=capacity,
        )

    async def acquire(
        self,
        tokens: int = 1,
        blocking: bool = True,
    ) -> bool:
        """
        Acquire tokens from bucket.

        Args:
            tokens: Number of tokens to acquire
            blocking: If True, wait for tokens; if False, return immediately

        Returns:
            True if tokens acquired, False if not blocking and unavailable

        Raises:
            RateLimitExceededError: If non-blocking and tokens unavailable
        """
        async with self.lock:
            # Refill bucket based on time elapsed
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate,
            )
            self.last_update = now

            self.total_requests += 1

            # Check if tokens available
            if self.tokens >= tokens:
                # Success: Consume tokens
                self.tokens -= tokens
                logger.debug(
                    "rate_limiter_acquired",
                    name=self.name,
                    tokens=tokens,
                    remaining=self.tokens,
                )
                return True

            # Not enough tokens
            if not blocking:
                # Non-blocking: Reject immediately
                self.rejected_requests += 1
                logger.warning(
                    "rate_limiter_rejected",
                    name=self.name,
                    tokens_requested=tokens,
                    tokens_available=self.tokens,
                )
                raise RateLimitExceededError(
                    operation=self.name,
                    rate=self.rate,
                )

            # Blocking: Wait for tokens to refill
            wait_time = (tokens - self.tokens) / self.rate
            self.total_wait_time += wait_time

            logger.debug(
                "rate_limiter_waiting",
                name=self.name,
                tokens=tokens,
                wait_time_seconds=wait_time,
            )

        # Release lock while waiting (allow other operations)
        await asyncio.sleep(wait_time)

        # Re-acquire lock and consume tokens
        async with self.lock:
            # Update tokens after wait
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate,
            )
            self.last_update = now

            # Consume tokens
            self.tokens = max(0, self.tokens - tokens)

            logger.debug(
                "rate_limiter_acquired_after_wait",
                name=self.name,
                tokens=tokens,
                remaining=self.tokens,
            )

            return True

    def get_stats(self) -> Dict[str, any]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with:
            - name: Rate limiter name
            - rate: Tokens per second
            - capacity: Bucket capacity
            - current_tokens: Current tokens available
            - total_requests: Total requests processed
            - rejected_requests: Requests rejected (non-blocking mode)
            - total_wait_time: Total time spent waiting
            - rejection_rate: Percentage of rejected requests
        """
        rejection_rate = (
            (self.rejected_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
        )

        return {
            "name": self.name,
            "rate": self.rate,
            "capacity": self.capacity,
            "current_tokens": round(self.tokens, 2),
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "total_wait_time_seconds": round(self.total_wait_time, 2),
            "rejection_rate_percent": round(rejection_rate, 2),
        }

    def reset(self) -> None:
        """Reset rate limiter to initial state."""
        self.tokens = float(self.capacity)
        self.last_update = time.time()
        self.total_requests = 0
        self.rejected_requests = 0
        self.total_wait_time = 0.0
        logger.info("rate_limiter_reset", name=self.name)


# Global rate limiters for common operations
RATE_LIMITERS: Dict[str, RateLimiter] = {
    "ingest": RateLimiter(
        rate=10.0,  # 10 ingests per second
        capacity=20,  # Burst up to 20
        name="ingest",
    ),
    "batch_ingest": RateLimiter(
        rate=2.0,  # 2 batches per second
        capacity=5,  # Burst up to 5
        name="batch_ingest",
    ),
    "compression": RateLimiter(
        rate=5.0,  # 5 compressions per second
        capacity=10,  # Burst up to 10
        name="compression",
    ),
}


def get_rate_limiter(operation: str) -> RateLimiter:
    """
    Get rate limiter for operation.

    Args:
        operation: Operation name (e.g., "ingest", "batch_ingest")

    Returns:
        RateLimiter instance

    Raises:
        KeyError: If operation not configured
    """
    if operation not in RATE_LIMITERS:
        logger.warning(
            "rate_limiter_not_configured",
            operation=operation,
            available=list(RATE_LIMITERS.keys()),
        )
        raise KeyError(f"Rate limiter not configured for operation: {operation}")

    return RATE_LIMITERS[operation]


def configure_rate_limiter(
    operation: str,
    rate: float,
    capacity: int,
) -> None:
    """
    Configure rate limiter for operation.

    Args:
        operation: Operation name
        rate: Tokens per second
        capacity: Bucket capacity

    Example:
        configure_rate_limiter("ingest", rate=20.0, capacity=40)
    """
    RATE_LIMITERS[operation] = RateLimiter(
        rate=rate,
        capacity=capacity,
        name=operation,
    )

    logger.info(
        "rate_limiter_configured",
        operation=operation,
        rate=rate,
        capacity=capacity,
    )
