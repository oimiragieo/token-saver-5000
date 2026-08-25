"""ace handlers rate limit — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock
import numpy as np
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
        "ace_framework": MagicMock(),
        "focus_manager": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_semantic_node(text="test", importance=0.5, embedding=None):
    node = MagicMock()
    node.text = text
    node.importance = importance
    node.embedding = embedding if embedding is not None else np.random.rand(384).astype(np.float32)
    node.metadata = {"tokens": 10, "position": 0, "entities": []}
    return node


def _make_code_chunk(
    name="func", chunk_type="function", code="def f(): pass", docstring="", start_line=1, end_line=5
):
    chunk = MagicMock()
    chunk.name = name
    chunk.chunk_type = chunk_type
    chunk.code = code
    chunk.docstring = docstring
    chunk.start_line = start_line
    chunk.end_line = end_line
    return chunk


class TestACEHandlersRateLimit:
    """Cover rate limit paths in ACE handlers."""

    @pytest.mark.asyncio
    async def test_ace_execute_rate_limit(self):
        """Cover lines 239-240."""
        from src.handlers.ace_handlers import handle_ace_generate, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_generate(ctx, {"task": "test"})

    @pytest.mark.asyncio
    async def test_ace_reflect_rate_limit(self):
        """Cover lines 305-306."""
        from src.handlers.ace_handlers import handle_ace_reflect, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_reflect(ctx, {"trajectory": [], "outcome": "test", "success": True})

    @pytest.mark.asyncio
    async def test_ace_update_context_rate_limit(self):
        """Cover lines 364-365."""
        from src.handlers.ace_handlers import handle_ace_curate, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_curate(ctx, {"insights": []})

    @pytest.mark.asyncio
    async def test_ace_add_bullets_rate_limit(self):
        """Cover lines 424-425."""
        from src.handlers.ace_handlers import handle_ace_grow_context, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_grow_context(ctx, {"bullets": []})

    @pytest.mark.asyncio
    async def test_ace_update_confidence_rate_limit(self):
        """Cover lines 479-480."""
        from src.handlers.ace_handlers import handle_ace_refine_context, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_refine_context(ctx, {"bullet_ids": [], "success": True})

    @pytest.mark.asyncio
    async def test_ace_get_context_rate_limit(self):
        """Cover lines 543-544."""
        from src.handlers.ace_handlers import handle_ace_get_playbook, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_get_playbook(ctx, {})

    @pytest.mark.asyncio
    async def test_ace_full_cycle_rate_limit(self):
        """Cover lines 627-628."""
        from src.handlers.ace_handlers import handle_ace_execute_cycle, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_ace_execute_cycle(
                ctx, {"task": "test", "outcome": "done", "success": True}
            )
