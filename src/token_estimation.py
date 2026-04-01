"""
Token estimation utilities for Semantic Modulator.

Provides dual-mode estimation: accurate (tiktoken) and fast (byte-based).
Falls back gracefully when tiktoken is not installed.
"""

from __future__ import annotations


def _load_tiktoken_encoder():
    """Try to load tiktoken cl100k_base encoder. Returns encoder or None."""
    try:
        import tiktoken  # type: ignore

        if tiktoken is None:
            return None
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


class TokenEstimator:
    """Estimates token counts for text using tiktoken or fast byte-based fallbacks.

    Args:
        None. The tiktoken encoder is loaded lazily on first instantiation.
    """

    def __init__(self) -> None:
        self._encoder = _load_tiktoken_encoder()

    def count_tokens(self, text: str) -> int:
        """Accurate token count using tiktoken cl100k_base, falling back to len//4.

        Args:
            text: Input text to count tokens for.

        Returns:
            Estimated or exact token count.
        """
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
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
