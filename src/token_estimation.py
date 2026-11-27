"""
Token estimation utilities for Semantic Modulator.

Provides dual-mode estimation: accurate (tiktoken) and fast (byte-based).
Falls back gracefully when tiktoken is not installed.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import sys
import threading
from typing import Any
from weakref import WeakKeyDictionary

_TOKEN_CACHE_MAX_CHARS = 65_536
_TOKEN_CACHE_MAX_SIZE = 256

_SHARED_ENCODER: Any = None
_SHARED_ENCODER_LOADED = False
_SHARED_ENCODER_TIKTOKEN_MOD: Any = None
_SHARED_ENCODER_LOCK = threading.Lock()
_ENCODER_CACHE_LOCK = threading.Lock()
_ENCODER_COUNT_CACHES: WeakKeyDictionary[Any, OrderedDict[tuple[bytes, int], int]] = (
    WeakKeyDictionary()
)


def _load_tiktoken_encoder():
    """Try to load tiktoken cl100k_base encoder. Returns encoder or None."""
    try:
        import tiktoken  # type: ignore

        if tiktoken is None:
            return None
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


_DEFAULT_LOAD_TIKTOKEN_ENCODER = _load_tiktoken_encoder


def _get_shared_tiktoken_encoder():
    """Return cached default tiktoken encoder while honoring test monkeypatches."""
    global _SHARED_ENCODER, _SHARED_ENCODER_LOADED, _SHARED_ENCODER_TIKTOKEN_MOD

    if _load_tiktoken_encoder is not _DEFAULT_LOAD_TIKTOKEN_ENCODER:
        return _load_tiktoken_encoder()

    if "tiktoken" in sys.modules and sys.modules["tiktoken"] is None:
        return None

    with _SHARED_ENCODER_LOCK:
        current_mod = sys.modules.get("tiktoken")
        if _SHARED_ENCODER_LOADED and current_mod is _SHARED_ENCODER_TIKTOKEN_MOD:
            return _SHARED_ENCODER
        enc = _load_tiktoken_encoder()
        _SHARED_ENCODER = enc
        _SHARED_ENCODER_LOADED = True
        _SHARED_ENCODER_TIKTOKEN_MOD = sys.modules.get("tiktoken")
        return enc


def _text_cache_key(text: str) -> tuple[bytes, int]:
    digest = hashlib.blake2b(text.encode("utf-8", errors="surrogatepass"), digest_size=32).digest()
    return (digest, len(text))


def _count_tokens_with_encoder(
    enc: Any,
    text: str,
    fallback_cache: OrderedDict[tuple[bytes, int], int] | None = None,
) -> int:
    if len(text) > _TOKEN_CACHE_MAX_CHARS:
        return len(enc.encode(text))

    key = _text_cache_key(text)
    with _ENCODER_CACHE_LOCK:
        try:
            cache = _ENCODER_COUNT_CACHES.get(enc)
            if cache is None:
                cache = OrderedDict()
                _ENCODER_COUNT_CACHES[enc] = cache
        except TypeError:
            cache = fallback_cache
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached

    count = len(enc.encode(text))
    with _ENCODER_CACHE_LOCK:
        try:
            cache = _ENCODER_COUNT_CACHES.get(enc)
            if cache is None:
                cache = OrderedDict()
                _ENCODER_COUNT_CACHES[enc] = cache
        except TypeError:
            cache = fallback_cache
        if cache is not None:
            if key in cache:
                cache.move_to_end(key)
            else:
                cache[key] = count
                while len(cache) > _TOKEN_CACHE_MAX_SIZE:
                    cache.popitem(last=False)
    return count


class TokenEstimator:
    """Estimates token counts for text using tiktoken or fast byte-based fallbacks.

    Args:
        None. The tiktoken encoder is loaded lazily on first instantiation.
    """

    def __init__(self) -> None:
        self._encoder = _get_shared_tiktoken_encoder()
        self._nonweakref_cache: OrderedDict[tuple[bytes, int], int] = OrderedDict()
        self._nonweakref_encoder_ref: Any = self._encoder

    def count_tokens(self, text: str) -> int:
        """Accurate token count using tiktoken cl100k_base, falling back to len//4.

        Args:
            text: Input text to count tokens for.

        Returns:
            Estimated or exact token count.
        """
        if not text:
            return 0
        enc = self._encoder
        if enc is not None:
            try:
                if enc is not self._nonweakref_encoder_ref:
                    with _ENCODER_CACHE_LOCK:
                        self._nonweakref_cache.clear()
                        self._nonweakref_encoder_ref = enc
                return _count_tokens_with_encoder(enc, text, fallback_cache=self._nonweakref_cache)
            except Exception:
                pass
        return len(text) // 4

    def estimate_fast(self, text: str) -> int:
        """Fast estimate: len(text) // 4.

        Args:
            text: Input text.

        Returns:
            Integer token estimate.
        """
        return len(text) // 4

    def estimate_json(self, text: str) -> int:
        """JSON-density estimate: len(text) // 2 (JSON is token-dense).

        Args:
            text: Input text (typically JSON serialized).

        Returns:
            Integer token estimate.
        """
        return len(text) // 2

    def count_bytes(self, text: str) -> int:
        """Exact UTF-8 byte count.

        Args:
            text: Input text.

        Returns:
            Number of bytes in the UTF-8 encoding.
        """
        return len(text.encode("utf-8"))

    def estimate_gemini(self, text: str) -> int:
        """Gemini-compatible estimate: 0.25 tok/ASCII char, 1.3 tok/non-ASCII char.

        Falls back to len(text)//4 for text > 100,000 chars (matching Gemini CLI behavior).

        Args:
            text: Input text to estimate.

        Returns:
            Integer token estimate compatible with Gemini CLI token counting.
        """
        if not text:
            return 0
        if len(text) > 100_000:
            return len(text) // 4
        if text.isascii():
            return len(text) // 4
        n_ascii = len(text.encode("ascii", "ignore"))
        n_non_ascii = len(text) - n_ascii
        est = n_ascii * 0.25 + n_non_ascii * 1.3
        # For N <= 100_000, max sequential float64 rounding drift is < 1.5e-6;
        # only values within 1e-5 of an integer can cross floor() differently.
        if abs(est - round(est)) > 1e-5:
            return int(est)
        tokens = 0.0
        for ch in text:
            tokens += 0.25 if ord(ch) <= 127 else 1.3
        return int(tokens)

    def estimate_all(self, text: str) -> dict:
        """Compute all estimation strategies in one call.

        Args:
            text: Input text to estimate.

        Returns:
            Dict with keys: accurate, estimated, json_estimated, bytes, gemini_estimated.
        """
        return {
            "accurate": self.count_tokens(text),
            "estimated": self.estimate_fast(text),
            "json_estimated": self.estimate_json(text),
            "bytes": self.count_bytes(text),
            "gemini_estimated": self.estimate_gemini(text),
        }
