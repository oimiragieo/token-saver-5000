"""Applies Token Saver compression pipeline to MCP tool responses."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..token_refiner import TokenRefiner
from ..meta_tokens import MetaTokenCompressor

# Minimum text length (characters) before compression is attempted.
# Texts shorter than this pass through unchanged to avoid skeleton overhead
# expanding small payloads.
_MIN_COMPRESS_CHARS = 100

# Minimum description length before description compression is applied.
_MIN_DESC_COMPRESS_CHARS = 200


@dataclass
class InterceptionStats:
    """Statistics produced by a single interception pass."""

    original_chars: int
    compressed_chars: int
    tokens_saved_estimate: int
    pipeline_stages: list[str] = field(default_factory=list)


class ResponseInterceptor:
    """Compresses MCP tool responses through the Token Saver pipeline.

    Pipeline applied in order:

    1. :class:`~src.token_refiner.TokenRefiner` — removes low-value filler tokens.
    2. :class:`~src.meta_tokens.MetaTokenCompressor` — lossless n-gram deduplication.

    Args:
        refiner_ratio: Fraction of tokens to keep (0.0–1.0).  Passed to
            :meth:`~src.token_refiner.TokenRefiner.refine` as *target_ratio*.
            Default ``0.7`` (keep 70%, remove up to 30% filler).
        enable_meta_tokens: Whether to apply the MetaToken stage.  Default ``True``.
    """

    def __init__(
        self,
        refiner_ratio: float = 0.7,
        enable_meta_tokens: bool = True,
    ) -> None:
        self._refiner = TokenRefiner()
        self._meta: MetaTokenCompressor | None = (
            MetaTokenCompressor() if enable_meta_tokens else None
        )
        self._refiner_ratio = refiner_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intercept_text(self, text: str) -> tuple[str, InterceptionStats]:
        """Compress *text* through the pipeline.

        Texts shorter than ``_MIN_COMPRESS_CHARS`` pass through unchanged.

        Args:
            text: The tool-response text to compress.

        Returns:
            Tuple of ``(compressed_text, stats)``.  When the text is too short
            to benefit from compression ``compressed_text == text``.
        """
        if not text:
            return text, InterceptionStats(0, 0, 0, [])

        original_len = len(text)

        if original_len < _MIN_COMPRESS_CHARS:
            return text, InterceptionStats(original_len, original_len, 0, [])

        current = text
        stages: list[str] = []

        # Stage 1: Token refinement
        refined_result = self._refiner.refine(current, target_ratio=self._refiner_ratio)
        if refined_result.removed_count > 0:
            current = refined_result.refined_text
            stages.append("token_refiner")

        # Stage 2: Meta-token compression
        if self._meta is not None:
            meta_result = self._meta.compress(current)
            if meta_result.savings_tokens > 0:
                current = meta_result.compressed_text
                stages.append("meta_tokens")

        compressed_len = len(current)
        # Rough estimate: 1 token ≈ 4 characters
        tokens_saved = max(0, (original_len - compressed_len) // 4)

        return current, InterceptionStats(
            original_chars=original_len,
            compressed_chars=compressed_len,
            tokens_saved_estimate=tokens_saved,
            pipeline_stages=stages,
        )

    def intercept_tool_descriptions(self, tools: list[dict]) -> list[dict]:
        """Compress tool description strings (not schemas, just descriptions).

        Tool names and ``inputSchema`` fields are never modified.  Only the
        ``description`` field of tools longer than ``_MIN_DESC_COMPRESS_CHARS``
        characters is compressed.

        Args:
            tools: List of tool dicts from an upstream ``tools/list`` response.

        Returns:
            New list of tool dicts with compressed descriptions.
        """
        result = []
        for tool in tools:
            new_tool = dict(tool)
            desc = tool.get("description", "")
            if desc and len(desc) > _MIN_DESC_COMPRESS_CHARS:
                refined = self._refiner.refine(desc, target_ratio=0.6)
                new_tool["description"] = refined.refined_text
            result.append(new_tool)
        return result
