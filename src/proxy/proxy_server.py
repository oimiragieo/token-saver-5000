"""MCP proxy server: wraps an upstream MCP server with transparent compression.

The :class:`ProxyServer` class contains all proxy logic and is designed to be
independently testable without starting any real MCP server.  The async run
loop (stdio I/O, upstream subprocess management) lives in the entry-point
script ``scripts/token_saver_proxy.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .response_interceptor import ResponseInterceptor
from .schema_compressor import SchemaCompressor


@dataclass
class ProxyConfig:
    """Configuration for the proxy server.

    Args:
        upstream_command: Executable to launch as the upstream MCP server.
        upstream_args: Arguments passed to *upstream_command*.
        upstream_env: Optional extra environment variables for the subprocess.
        upstream_cwd: Optional working directory for the subprocess.
        provider: AI provider hint (e.g. ``"anthropic"``, ``"google"``).
            Used for future provider-specific optimisations.
        enable_schema_compression: When ``True``, expose 3 meta-tools instead of
            the full upstream tool list.
        refiner_ratio: Fraction of tokens to keep per tool-result (0.0–1.0).
        enable_meta_tokens: Whether to run the MetaToken compression stage.
    """

    upstream_command: str
    upstream_args: list[str] = field(default_factory=list)
    upstream_env: dict[str, str] | None = None
    upstream_cwd: str | None = None
    provider: str = "unknown"
    enable_schema_compression: bool = False
    refiner_ratio: float = 0.7
    enable_meta_tokens: bool = True


@dataclass
class ProxySessionMetrics:
    """Cumulative metrics for a proxy session."""

    total_calls: int = 0
    total_original_chars: int = 0
    total_compressed_chars: int = 0
    total_tokens_saved: int = 0
    by_tool: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def savings_pct(self) -> float:
        if self.total_original_chars == 0:
            return 0.0
        return round((1 - self.total_compressed_chars / self.total_original_chars) * 100, 1)

    def record(self, tool_name: str, stats: dict[str, Any]) -> None:
        """Accumulate stats from a single tool call."""
        self.total_calls += 1
        self.total_original_chars += stats.get("original_chars", 0)
        self.total_compressed_chars += stats.get("compressed_chars", 0)
        self.total_tokens_saved += stats.get("tokens_saved_estimate", 0)

        entry = self.by_tool.setdefault(tool_name, {"calls": 0, "tokens_saved": 0})
        entry["calls"] += 1
        entry["tokens_saved"] += stats.get("tokens_saved_estimate", 0)

    def summary_line(self) -> str:
        """One-line summary suitable for stderr."""
        return (
            f"{self.total_calls} calls | "
            f"{self.total_tokens_saved:,} tokens saved | "
            f"{self.savings_pct:.0f}% avg compression"
        )


class ProxyServer:
    """Transparent MCP proxy that compresses upstream responses.

    This class is **not** an MCP server itself — it encapsulates the proxy
    logic (tool-list processing, result compression, meta-tool dispatch) so
    it can be unit-tested without any network or subprocess involvement.

    The MCP server run loop that connects to the upstream server and serves
    stdio is in ``scripts/token_saver_proxy.py``.

    Args:
        config: :class:`ProxyConfig` instance describing the upstream server
            and compression settings.
    """

    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self._interceptor = ResponseInterceptor(
            refiner_ratio=config.refiner_ratio,
            enable_meta_tokens=config.enable_meta_tokens,
        )
        self._schema_compressor: SchemaCompressor | None = None
        self.metrics = ProxySessionMetrics()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_schema_compression(self, upstream_tools: list[dict[str, Any]]) -> None:
        """Initialise schema compression from the upstream tool list.

        This is a no-op when :attr:`ProxyConfig.enable_schema_compression` is
        ``False``.

        Args:
            upstream_tools: Raw tool dicts from the upstream ``tools/list`` call.
        """
        if self.config.enable_schema_compression:
            self._schema_compressor = SchemaCompressor(upstream_tools)

    # ------------------------------------------------------------------
    # Tool-list processing
    # ------------------------------------------------------------------

    def get_tools(self, upstream_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process the upstream ``tools/list`` response.

        When schema compression is enabled returns the 3 meta-tool schemas.
        Otherwise returns the upstream tools with descriptions compressed via
        :meth:`ResponseInterceptor.intercept_tool_descriptions`.

        Args:
            upstream_tools: Raw tool dicts from the upstream server.

        Returns:
            Processed tool dicts to return to the downstream MCP client.
        """
        if self._schema_compressor is not None:
            return self._schema_compressor.meta_tool_schemas()
        return self._interceptor.intercept_tool_descriptions(upstream_tools)

    # ------------------------------------------------------------------
    # Result compression
    # ------------------------------------------------------------------

    def process_tool_result(self, tool_name: str, result_text: str) -> tuple[str, dict[str, Any]]:
        """Compress an upstream tool result text.

        Args:
            tool_name: Name of the tool that produced this result (used for
                logging / future per-tool tuning).
            result_text: Text content returned by the upstream tool.

        Returns:
            Tuple of ``(compressed_text, stats_dict)``.  *stats_dict* has the
            keys ``original_chars``, ``compressed_chars``,
            ``tokens_saved_estimate``, ``savings_pct``, and ``pipeline_stages``.
        """
        compressed, stats = self._interceptor.intercept_text(result_text)
        stats_dict = {
            "original_chars": stats.original_chars,
            "compressed_chars": stats.compressed_chars,
            "tokens_saved_estimate": stats.tokens_saved_estimate,
            "savings_pct": stats.savings_pct,
            "pipeline_stages": stats.pipeline_stages,
        }
        self.metrics.record(tool_name, stats_dict)
        return compressed, stats_dict

    # ------------------------------------------------------------------
    # Meta-tool dispatch
    # ------------------------------------------------------------------

    def handle_meta_tool_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Handle a meta-tool call if schema compression is active.

        Returns ``None`` when schema compression is not enabled or *name* is
        not one of the three meta-tool names.

        For ``invoke_tool`` the returned JSON contains
        ``"_invoke_upstream": true``, which the caller must interpret as a
        forwarding directive (call the named tool on the upstream server).

        Args:
            name: Tool name as received from the downstream client.
            arguments: Tool arguments dict.

        Returns:
            JSON string result, or ``None`` if this is not a meta-tool call.
        """
        if self._schema_compressor is None:
            return None
        if name not in SchemaCompressor._META_NAMES:
            return None
        return self._schema_compressor.handle_meta_tool(name, arguments)
