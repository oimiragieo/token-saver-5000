"""Applies Token Saver compression pipeline to MCP tool responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..token_refiner import TokenRefiner
from ..meta_tokens import MetaTokenCompressor
from ..cli_output_optimizer import CLIOutputOptimizer
from ..identifier_preservation import apply_identifier_guard, extract_critical_identifiers
from .schema_levels import DEFAULT_BYTE_EXACT_READ_TOOLS, normalize_tool_name

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

    @property
    def savings_pct(self) -> float:
        """Percentage of characters saved (0.0–100.0)."""
        if self.original_chars == 0:
            return 0.0
        return round((1 - self.compressed_chars / self.original_chars) * 100, 1)

    @property
    def original_tokens_estimate(self) -> int:
        """Rough token count for the original text (1 token ≈ 4 chars)."""
        return self.original_chars // 4

    @property
    def compressed_tokens_estimate(self) -> int:
        """Rough token count for the compressed text (1 token ≈ 4 chars)."""
        return self.compressed_chars // 4


class ResponseInterceptor:
    """Compresses MCP tool responses through the Token Saver pipeline.

    Pipeline applied in order:

    0. :class:`~src.cli_output_optimizer.CLIOutputOptimizer` — CLI output filtering
       (optional, enabled via *enable_cli_filters*).
    1. :class:`~src.token_refiner.TokenRefiner` — removes low-value filler tokens.
    2. :class:`~src.meta_tokens.MetaTokenCompressor` — lossless n-gram deduplication.

    Args:
        refiner_ratio: Fraction of tokens to keep (0.0–1.0).  Passed to
            :meth:`~src.token_refiner.TokenRefiner.refine` as *target_ratio*.
            Default ``0.7`` (keep 70%, remove up to 30% filler).
        enable_meta_tokens: Whether to apply the MetaToken stage.  Default ``True``.
        enable_cli_filters: Whether to apply Stage 0 CLI output filtering before
            the token refinement stage.  Default ``False`` to preserve backward
            compatibility.
    """

    def __init__(
        self,
        refiner_ratio: float = 0.7,
        enable_meta_tokens: bool = True,
        enable_cli_filters: bool = False,
        preserve_identifiers: bool = False,
        byte_exact_read_tools: Iterable[str] | None = None,
    ) -> None:
        self._refiner = TokenRefiner()
        self._meta: MetaTokenCompressor | None = (
            MetaTokenCompressor() if enable_meta_tokens else None
        )
        self._refiner_ratio = refiner_ratio
        self._cli_optimizer: CLIOutputOptimizer | None = (
            CLIOutputOptimizer() if enable_cli_filters else None
        )
        # Opt-in "amnesia tax" guard: reinject execution-critical identifiers
        # (file paths, error codes, symbols, URLs, stack frames, env vars, UUIDs)
        # that compression dropped, so an agent reading a compressed tool result
        # never loses the token it needs. Off by default for backward compat.
        self._preserve_identifiers = preserve_identifiers
        if byte_exact_read_tools is None:
            self._byte_exact_read_tools = DEFAULT_BYTE_EXACT_READ_TOOLS
        else:
            self._byte_exact_read_tools = frozenset(
                str(t).strip().lower() for t in byte_exact_read_tools if str(t).strip()
            )

    def is_byte_exact_read_tool(self, tool_name: str | None) -> bool:
        """Return ``True`` if *tool_name* is protected by the byte-exact read guard."""
        if not tool_name:
            return False
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in self._byte_exact_read_tools or norm in self._byte_exact_read_tools

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def intercept_text(
        self,
        text: str,
        tool_name: str | None = None,
        *,
        is_error: bool = False,
        structured_content: Any = None,
        force_compress: bool = False,
    ) -> tuple[str, InterceptionStats]:
        """Compress *text* through the pipeline unless guarded.

        Never compresses errors (``is_error=True``), ``structuredContent``
        responses, or byte-exact file-read tools unless ``force_compress=True``.
        Texts shorter than ``_MIN_COMPRESS_CHARS`` also pass through unchanged.
        """
        if not text:
            return text, InterceptionStats(0, 0, 0, [])

        original_len = len(text)

        if is_error:
            return text, InterceptionStats(
                original_len, original_len, 0, ["passthrough:error_guard"]
            )

        if structured_content is not None:
            return text, InterceptionStats(
                original_len, original_len, 0, ["passthrough:structured_guard"]
            )

        if not force_compress and self.is_byte_exact_read_tool(tool_name):
            return text, InterceptionStats(
                original_len, original_len, 0, ["passthrough:read_guard"]
            )

        if original_len < _MIN_COMPRESS_CHARS:
            return text, InterceptionStats(original_len, original_len, 0, [])

        current = text
        stages: list[str] = []

        # Stage 0: CLI output filtering (optional)
        if self._cli_optimizer is not None:
            cli_result = self._cli_optimizer.filter(current, command_hint=tool_name)
            if cli_result.strategy_applied != "passthrough":
                current = cli_result.filtered_text
                stages.append(f"cli_filter:{cli_result.strategy_applied}")

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

        # Stage 3: identifier preservation (opt-in). Extract execution-critical
        # identifiers from the ORIGINAL text and reinject any that compression
        # dropped, so an agent never has to re-run a tool to recover a file
        # path / error code / symbol the compressor elided (the "amnesia tax").
        # Same guarantee the REST /v1/compress/tool-output endpoint provides.
        if self._preserve_identifiers:
            critical = extract_critical_identifiers(text)
            current, reinjected = apply_identifier_guard(current, critical)
            if reinjected:
                stages.append(f"identifier_guard:{len(reinjected)}")

        compressed_len = len(current)
        # Rough estimate: 1 token ≈ 4 characters
        tokens_saved = max(0, (original_len - compressed_len) // 4)

        return current, InterceptionStats(
            original_chars=original_len,
            compressed_chars=compressed_len,
            tokens_saved_estimate=tokens_saved,
            pipeline_stages=stages,
        )

    def intercept_mcp_result(
        self,
        result: dict[str, Any],
        tool_name: str | None = None,
        *,
        force_compress: bool = False,
    ) -> tuple[dict[str, Any], InterceptionStats]:
        """Intercept a structured MCP ``CallToolResult`` dict.

        Never compresses ``isError=True``, ``structuredContent``, non-text items,
        or byte-exact read tools unless ``force_compress=True``.
        """
        if result.get("isError") is True:
            return dict(result), InterceptionStats(0, 0, 0, ["passthrough:error_guard"])
        if result.get("structuredContent") is not None:
            return dict(result), InterceptionStats(0, 0, 0, ["passthrough:structured_guard"])
        if not force_compress and self.is_byte_exact_read_tool(tool_name):
            return dict(result), InterceptionStats(0, 0, 0, ["passthrough:read_guard"])

        items = result.get("content")
        if not isinstance(items, list):
            return dict(result), InterceptionStats(0, 0, 0, [])

        new_items: list[Any] = []
        total_orig = 0
        total_comp = 0
        stages: list[str] = []
        for item in items:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                comp_text, item_stats = self.intercept_text(
                    item["text"], tool_name=tool_name, force_compress=force_compress
                )
                total_orig += item_stats.original_chars
                total_comp += item_stats.compressed_chars
                for st in item_stats.pipeline_stages:
                    if st not in stages:
                        stages.append(st)
                new_item = dict(item)
                new_item["text"] = comp_text
                new_items.append(new_item)
            else:
                if "passthrough:non_text_guard" not in stages:
                    stages.append("passthrough:non_text_guard")
                new_items.append(item)

        new_result = dict(result)
        new_result["content"] = new_items
        tokens_saved = max(0, (total_orig - total_comp) // 4)
        return new_result, InterceptionStats(
            original_chars=total_orig,
            compressed_chars=total_comp,
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
