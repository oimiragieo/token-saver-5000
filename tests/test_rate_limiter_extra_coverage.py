"""rate limiter — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_basic(self):
        from src.rate_limiter import RateLimiter

        rl = RateLimiter(rate=100.0, capacity=10, name="test")
        await rl.acquire()  # should succeed

    @pytest.mark.asyncio
    async def test_acquire_non_blocking_insufficient(self):
        from src.rate_limiter import RateLimiter
        from src.error_types import RateLimitExceededError

        rl = RateLimiter(rate=0.001, capacity=1, name="test")
        await rl.acquire()  # drain the single token

        with pytest.raises(RateLimitExceededError):
            await rl.acquire(blocking=False)
