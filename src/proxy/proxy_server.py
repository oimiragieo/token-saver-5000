"""MCP proxy server: wraps an upstream MCP server with transparent compression.

The :class:`ProxyServer` class contains all proxy logic and is designed to be
independently testable without starting any real MCP server.  The async run
loop (stdio I/O, upstream subprocess management) lives in the entry-point
script ``scripts/token_saver_proxy.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .cache_router import INTERNAL_ROUTER_TOOL_NAME, CacheRouter
from .response_interceptor import ResponseInterceptor
from .schema_compressor import SchemaCompressor, validate_tool_arguments
from .schema_levels import (
    DEFAULT_BYTE_EXACT_READ_TOOLS,
    DEFAULT_CORE_PINNED_TOOLS,
    RouterPolicy,
    SchemaCompressionLevel,
)


@dataclass
class ProxyConfig:
    """Configuration for the proxy server."""

    upstream_command: str
    upstream_args: list[str] = field(default_factory=list)
    upstream_env: dict[str, str] | None = None
    upstream_cwd: str | None = None
    provider: str = "unknown"
    enable_schema_compression: bool = False
    refiner_ratio: float = 0.7
    enable_meta_tokens: bool = True
    preserve_identifiers: bool = False
    schema_level: SchemaCompressionLevel | None = None
    max_schema_tokens: int = 4000
    pin_tools: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)
    whitelist: list[str] = field(default_factory=list)
    validate_arguments: bool = True
    byte_exact_read_tools: frozenset[str] = field(
        default_factory=lambda: DEFAULT_BYTE_EXACT_READ_TOOLS
    )

    def effective_schema_level(self) -> SchemaCompressionLevel | None:
        """Return the active :class:`SchemaCompressionLevel`, or ``None`` for legacy mode."""
        if self.schema_level is not None:
            return self.schema_level
        if self.enable_schema_compression:
            return SchemaCompressionLevel.L2_LAZY
        return None


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
    """Transparent MCP proxy that compresses upstream responses and schemas."""

    def __init__(self, config: ProxyConfig) -> None:
        self.config = config
        self._policy = RouterPolicy(
            blacklist=frozenset(config.blacklist),
            whitelist=frozenset(config.whitelist),
            core_pinned_tools=DEFAULT_CORE_PINNED_TOOLS | set(config.pin_tools),
            max_schema_tokens=config.max_schema_tokens,
            byte_exact_read_tools=config.byte_exact_read_tools,
        )
        self._interceptor = ResponseInterceptor(
            refiner_ratio=config.refiner_ratio,
            enable_meta_tokens=config.enable_meta_tokens,
            preserve_identifiers=config.preserve_identifiers,
            byte_exact_read_tools=self._policy.byte_exact_read_tools,
        )
        self._schema_compressor: SchemaCompressor | None = None
        self._cache_router: CacheRouter | None = None
        self._upstream_tools_by_name: dict[str, dict[str, Any]] = {}
        self.metrics = ProxySessionMetrics()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_schema_compression(self, upstream_tools: list[dict[str, Any]]) -> None:
        """Initialise schema compression / routing from the upstream tool list."""
        self._upstream_tools_by_name = {
            t["name"]: t for t in upstream_tools if isinstance(t, dict) and t.get("name")
        }
        level = self.config.effective_schema_level()
        if level is None:
            return
        self._schema_compressor = SchemaCompressor(
            upstream_tools,
            level=level,
            policy=self._policy,
        )
        if level == SchemaCompressionLevel.DUAL_PATH_ROUTER:
            self._cache_router = CacheRouter(upstream_tools, policy=self._policy)

    # ------------------------------------------------------------------
    # Tool-list processing
    # ------------------------------------------------------------------

    def get_tools(self, upstream_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process the upstream ``tools/list`` response."""
        level = self.config.effective_schema_level()
        if level == SchemaCompressionLevel.DUAL_PATH_ROUTER and self._cache_router is not None:
            return self._cache_router.get_resident_tools()
        if level is not None and self._schema_compressor is not None:
            return self._schema_compressor.get_tools(level)
        return self._interceptor.intercept_tool_descriptions(upstream_tools)

    # ------------------------------------------------------------------
    # Pre-binding validation & Result compression
    # ------------------------------------------------------------------

    def _active_meta_names(self) -> frozenset[str]:
        """Return the set of meta-tool names active for the current schema level."""
        level = self.config.effective_schema_level()
        if level == SchemaCompressionLevel.DUAL_PATH_ROUTER and self._cache_router is not None:
            return frozenset({INTERNAL_ROUTER_TOOL_NAME})
        if level in (SchemaCompressionLevel.L2_LAZY, SchemaCompressionLevel.L3_ADAPTIVE_PINNED):
            if self._schema_compressor is not None:
                return SchemaCompressor._META_NAMES
        return frozenset()

    def validate_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        via_internal_router: bool = False,
    ) -> dict[str, Any]:
        """Validate tool existence, state preconditions, and argument schema."""
        if self._cache_router is not None:
            if tool_name == INTERNAL_ROUTER_TOOL_NAME:
                return {"allowed": True, "reason": "ok", "errors": [], "error_message": ""}
            if (
                not via_internal_router
                and tool_name in self._cache_router._by_name
                and not self._cache_router.is_resident_tool(tool_name)
            ):
                msg = (
                    f"Tool '{tool_name}' is not resident in DUAL_PATH_ROUTER mode; "
                    f"discover and invoke it via '{INTERNAL_ROUTER_TOOL_NAME}'."
                )
                return {
                    "allowed": False,
                    "reason": "non_resident_tool",
                    "errors": [msg],
                    "error_message": msg,
                }
            return self._cache_router.validate_call(
                tool_name,
                arguments,
                validate_arguments=self.config.validate_arguments,
            )

        active_meta = self._active_meta_names()
        if tool_name in active_meta:
            return {"allowed": True, "reason": "ok", "errors": [], "error_message": ""}

        all_meta = SchemaCompressor._META_NAMES | {INTERNAL_ROUTER_TOOL_NAME}
        if tool_name in all_meta and tool_name not in self._upstream_tools_by_name:
            msg = f"Unknown tool '{tool_name}'"
            return {
                "allowed": False,
                "reason": "unknown_tool",
                "errors": [msg],
                "error_message": msg,
            }

        if self._upstream_tools_by_name and tool_name not in self._upstream_tools_by_name:
            msg = f"Unknown tool '{tool_name}'"
            return {
                "allowed": False,
                "reason": "unknown_tool",
                "errors": [msg],
                "error_message": msg,
            }

        if self._policy.is_blacklisted(tool_name):
            msg = f"Tool '{tool_name}' is blacklisted"
            return {
                "allowed": False,
                "reason": "blacklisted",
                "errors": [msg],
                "error_message": msg,
            }
        if self._policy.whitelist and not self._policy.is_whitelisted(tool_name):
            msg = f"Tool '{tool_name}' is not whitelisted"
            return {
                "allowed": False,
                "reason": "not_whitelisted",
                "errors": [msg],
                "error_message": msg,
            }

        tool = self._upstream_tools_by_name.get(tool_name)
        if tool is not None and arguments is not None and self.config.validate_arguments:
            val = validate_tool_arguments(tool_name, arguments, tool.get("inputSchema", {}))
            if not val["valid"]:
                return {
                    "allowed": False,
                    "reason": "invalid_arguments",
                    "errors": val["errors"],
                    "error_message": val["error_message"],
                }

        return {"allowed": True, "reason": "ok", "errors": [], "error_message": ""}

    def process_tool_result(
        self,
        tool_name: str,
        result_text: str,
        *,
        is_error: bool = False,
        structured_content: Any = None,
        force_compress: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Compress an upstream tool result text while respecting read/error guards."""
        compressed, stats = self._interceptor.intercept_text(
            result_text,
            tool_name=tool_name,
            is_error=is_error,
            structured_content=structured_content,
            force_compress=force_compress,
        )
        stats_dict = {
            "original_chars": stats.original_chars,
            "compressed_chars": stats.compressed_chars,
            "tokens_saved_estimate": stats.tokens_saved_estimate,
            "savings_pct": stats.savings_pct,
            "pipeline_stages": stats.pipeline_stages,
        }
        self.metrics.record(tool_name, stats_dict)
        if self._schema_compressor is not None:
            self._schema_compressor.record_tool_call(tool_name)
        self.commit_tool_effects(tool_name, succeeded=not is_error)
        return compressed, stats_dict

    def commit_tool_effects(self, tool_name: str, succeeded: bool) -> None:
        """Commit deferred state effects in CacheRouter when execution succeeds."""
        if self._cache_router is not None:
            self._cache_router.commit_tool_effects(tool_name, succeeded=succeeded)

    # ------------------------------------------------------------------
    # Meta-tool dispatch
    # ------------------------------------------------------------------

    def handle_meta_tool_call(self, name: str, arguments: dict[str, Any]) -> str | None:
        """Handle a meta-tool or ``internal_router`` call when active."""
        if name == INTERNAL_ROUTER_TOOL_NAME and self._cache_router is not None:
            if not isinstance(arguments, dict):
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "error_message": (
                            f"internal_router arguments must be an object, "
                            f"got {type(arguments).__name__}"
                        ),
                    }
                )
            return json.dumps(
                self._cache_router.route_internal_call(
                    arguments,
                    validate_arguments=self.config.validate_arguments,
                )
            )

        if self._schema_compressor is None:
            return None
        if name not in self._active_meta_names():
            return None
        if not isinstance(arguments, dict):
            return json.dumps(
                {
                    "status": "error",
                    "reason": "invalid_arguments",
                    "error_message": (
                        f"{name} arguments must be an object, got {type(arguments).__name__}"
                    ),
                }
            )

        if name == "invoke_tool":
            raw_target = arguments.get("tool_name", "")
            if not isinstance(raw_target, str) or not raw_target.strip():
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "error_message": "tool_name must be a non-empty string",
                    }
                )
            target = raw_target
            raw_args = arguments.get("arguments", {})
            if raw_args is None:
                raw_args = {}
            if not isinstance(raw_args, dict):
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "error_message": (
                            f"arguments must be an object, got {type(raw_args).__name__}"
                        ),
                    }
                )
            gate = self.validate_call(
                target,
                raw_args if self.config.validate_arguments else None,
            )
            if not gate["allowed"]:
                return json.dumps({"status": "error", **gate})
            self._schema_compressor.record_tool_call(target)

        return self._schema_compressor.handle_meta_tool(name, arguments)
