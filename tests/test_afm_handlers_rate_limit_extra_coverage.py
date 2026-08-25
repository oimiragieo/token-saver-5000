"""afm handlers rate limit — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

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


class TestAFMHandlersRateLimit:
    """Cover rate limit paths in AFM handlers."""

    @pytest.mark.asyncio
    async def test_afm_add_message_rate_limit(self):
        """Cover lines 49-50."""
        from src.handlers.afm_handlers import handle_afm_add_message, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_add_message(ctx, {"role": "user", "content": "hi"})

    @pytest.mark.asyncio
    async def test_afm_get_context_rate_limit(self):
        """Cover lines 106-107."""
        from src.handlers.afm_handlers import handle_afm_build_context, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_build_context(ctx, {"current_query": "q", "budget_tokens": 100})

    @pytest.mark.asyncio
    async def test_afm_get_stats_rate_limit(self):
        """Cover lines 183-184."""
        from src.handlers.afm_handlers import handle_afm_get_stats, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_get_stats(ctx, {})

    @pytest.mark.asyncio
    async def test_afm_clear_history_rate_limit(self):
        """Cover lines 223-224."""
        from src.handlers.afm_handlers import handle_afm_clear_history, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_clear_history(ctx, {})

    @pytest.mark.asyncio
    async def test_afm_export_history_rate_limit(self):
        """Cover lines 270-271."""
        from src.handlers.afm_handlers import handle_afm_export_history, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_export_history(ctx, {})

    @pytest.mark.asyncio
    async def test_afm_import_history_rate_limit(self):
        """Cover lines 346-347."""
        from src.handlers.afm_handlers import handle_afm_import_history, RATE_LIMITERS
        from src.rate_limiter import RateLimitExceededError

        ctx = _make_mock_context()
        RATE_LIMITERS["compression"].acquire = AsyncMock(
            side_effect=RateLimitExceededError("limit", rate=1.0)
        )
        with pytest.raises(ValueError, match="Rate limit"):
            await handle_afm_import_history(ctx, {})
