"""
Regression tests for the offline/degraded embedding infinite-recursion bug.

BUG (diagnosed during PR #207):
    When BOTH neural tiers are unavailable (sentence-transformers absent AND
    the ONNX model not cached / HF unreachable), ``EmbeddingManager.encode``
    on the STANDARD tier recursed indefinitely:

        encode(STANDARD)
          → _encode_standard
            → get_text_embedder() returns _EmbeddingManagerAdapter
              → adapter.encode → manager.encode(ONNX)
                → _encode_onnx FAILS
                  → _encode_with_fallback(ONNX)
                    → tier != STANDARD, so re-enters _encode_standard
                      → adapter again → encode(ONNX) → ... LOOP

    ~330 frames deep, building an O(n^2) ~66KB nested error string (~14s CPU)
    before finally raising. ``asyncio.wait_for`` at the call site could bound
    the AWAIT but never cancel the worker thread.

FIX:
    1. ``_encode_with_fallback`` skips the STANDARD branch when
       ``SentenceTransformer is None`` (STANDARD can never load offline; the
       branch only re-enters the adapter).
    2. ``_EmbeddingManagerAdapter.encode`` raises ONE clean ``RuntimeError``
       immediately when ``ONNX_AVAILABLE`` is False, instead of re-entering
       ``manager.encode`` and bouncing through the fallback machinery.

These tests force both neural tiers unavailable and assert ``encode`` raises a
clean ``RuntimeError`` at BOUNDED recursion depth in well under one second.
Each test fails (recursion / timeout) against the pre-fix code.
"""

import sys
import time

import pytest

import src.embeddings as emb
from src.embeddings import EmbeddingManager, EmbeddingTier


@pytest.fixture(autouse=True)
def _isolated_singleton():
    """Give every test a fresh EmbeddingManager singleton.

    The manager is a process-wide singleton whose tier is locked at first
    construction; reset before AND after so neither this test nor any
    neighbour is poisoned by a leaked instance.
    """
    EmbeddingManager.reset_for_testing()
    yield
    EmbeddingManager.reset_for_testing()


def _force_no_sentence_transformers(monkeypatch):
    """Simulate sentence-transformers being absent (ONNX-only deployment).

    ``SentenceTransformer is None`` is exactly the runtime state of the
    production ONNX-only Docker image, and the trigger for
    ``get_text_embedder`` to hand back the ``_EmbeddingManagerAdapter``.
    """
    monkeypatch.setattr(emb, "SentenceTransformer", None)


def test_offline_no_onnx_raises_clean_runtime_error_without_recursion(monkeypatch):
    """Both neural tiers gone (no ST, no ONNX runtime) -> one clean RuntimeError.

    This is the "ONNX entirely unavailable" half of the bug (missing
    onnxruntime/optimum). The adapter guard must short-circuit immediately.
    """
    _force_no_sentence_transformers(monkeypatch)
    # ONNX runtime not importable at all.
    monkeypatch.setattr(emb, "ONNX_AVAILABLE", False)

    manager = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)

    # Tight recursion budget: the pre-fix loop blew through ~330 frames. A
    # correct one-shot raise stays within a handful of frames, so even a very
    # low limit must not be hit. Set a generous-but-bounded ceiling so the
    # recursion bug trips RecursionError (a clear failure) rather than hanging.
    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    start = time.perf_counter()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            manager.encode(["hello world"], tier=EmbeddingTier.STANDARD)
    finally:
        sys.setrecursionlimit(original_limit)
    elapsed = time.perf_counter() - start

    # Bounded time: the bug spent ~14s of CPU building the nested error string.
    assert elapsed < 1.0, f"encode took {elapsed:.2f}s — recursion/timeout not bounded"

    # A RecursionError (a RuntimeError subclass) IS the bug manifesting: the
    # offline path must raise a CLEAN domain RuntimeError, never blow the stack.
    assert not isinstance(exc_info.value, RecursionError), (
        "got RecursionError — the offline path is still recursing instead of "
        "raising one clean RuntimeError"
    )

    # The error must be a clean, finite message — NOT the megabyte-scale
    # nested 'Original error: ... Original error: ...' chain the bug produced.
    message = str(exc_info.value)
    assert len(message) < 4000, (
        f"error message is {len(message)} chars — looks like the O(n^2) "
        "nested-recursion error string, not a single clean raise"
    )
    assert message.count("Original error:") <= 1, (
        "error message chains multiple 'Original error:' fragments — "
        "the recursion is still happening"
    )


def test_offline_onnx_model_unloadable_does_not_recurse(monkeypatch):
    """ONNX runtime present but the model can't load (HF unreachable / corrupt).

    This is the "ONNX model not cached/unreachable" half of the bug. ONNX_AVAILABLE
    is True, so the adapter still delegates to encode(ONNX); _encode_onnx fails at
    MODEL-LOAD time. The _encode_with_fallback STANDARD-branch guard must prevent
    re-entry into the adapter.
    """
    _force_no_sentence_transformers(monkeypatch)
    # ONNX runtime "available" so the adapter delegates to encode(ONNX)...
    monkeypatch.setattr(emb, "ONNX_AVAILABLE", True)

    # ...but the ONNX model itself can't be constructed (unreachable HF / corrupt
    # cache). _encode_onnx raises at the point it builds the ONNXEmbeddingManager.
    def _boom(*_args, **_kwargs):
        raise RuntimeError("ONNX model unreachable (simulated offline / corrupt cache)")

    monkeypatch.setattr(EmbeddingManager, "_encode_onnx", _boom)

    manager = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    start = time.perf_counter()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            manager.encode(["hello world"], tier=EmbeddingTier.STANDARD)
    finally:
        sys.setrecursionlimit(original_limit)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"encode took {elapsed:.2f}s — recursion/timeout not bounded"

    message = str(exc_info.value)
    assert (
        len(message) < 4000
    ), f"error message is {len(message)} chars — recursion error string, not a clean raise"
    assert (
        message.count("Original error:") <= 1
    ), "error message chains multiple 'Original error:' fragments — recursion still happening"


def test_offline_no_onnx_request_onnx_tier_also_bounded(monkeypatch):
    """Directly requesting ONNX with no usable backend raises cleanly too.

    Covers the path where a caller (e.g. the API plan-gating layer) requests
    EmbeddingTier.ONNX directly while both backends are unavailable. Must not
    fall into the STANDARD<->ONNX bounce.
    """
    _force_no_sentence_transformers(monkeypatch)
    monkeypatch.setattr(emb, "ONNX_AVAILABLE", False)

    manager = EmbeddingManager(tier=EmbeddingTier.ONNX, enable_cache=False)

    original_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(200)
    start = time.perf_counter()
    try:
        with pytest.raises(RuntimeError):
            manager.encode(["hello world"], tier=EmbeddingTier.ONNX)
    finally:
        sys.setrecursionlimit(original_limit)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"encode took {elapsed:.2f}s — recursion/timeout not bounded"


def test_adapter_raises_immediately_when_onnx_unavailable(monkeypatch):
    """Unit-level: the adapter itself raises rather than re-entering encode.

    Locks the defense-in-depth guard directly so a future refactor of the
    fallback chain can't silently reopen the recursion at the adapter.
    """
    _force_no_sentence_transformers(monkeypatch)
    monkeypatch.setattr(emb, "ONNX_AVAILABLE", False)

    manager = EmbeddingManager(tier=EmbeddingTier.STANDARD, enable_cache=False)
    adapter = emb._EmbeddingManagerAdapter(manager)

    with pytest.raises(RuntimeError) as exc_info:
        adapter.encode(["hello world"])

    # Must mention there is no usable backend; must NOT recurse into manager.encode.
    assert "no usable embedding backend" in str(exc_info.value).lower()
