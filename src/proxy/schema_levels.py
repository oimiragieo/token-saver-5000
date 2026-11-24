"""Schema compression levels and routing policy contracts for the MCP proxy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

DEFAULT_CORE_PINNED_TOOLS: frozenset[str] = frozenset(
    {
        "shell",
        "bash",
        "read",
        "read_file",
        "edit",
        "edit_file",
        "apply_patch",
        "write",
        "write_file",
        "view",
        "cat",
        "grep",
        "glob",
    }
)

DEFAULT_BYTE_EXACT_READ_TOOLS: frozenset[str] = frozenset(
    {
        "read",
        "read_file",
        "view",
        "open",
        "cat",
        "show",
        "contents",
    }
)


class SchemaCompressionLevel(str, Enum):
    """Supported MCP tool schema compression modes."""

    L0_LOSSLESS = "L0_LOSSLESS"
    L1_SUMMARY = "L1_SUMMARY"
    L2_LAZY = "L2_LAZY"
    L3_ADAPTIVE_PINNED = "L3_ADAPTIVE_PINNED"
    DUAL_PATH_ROUTER = "DUAL_PATH_ROUTER"


_LEVEL_ALIASES: dict[str, SchemaCompressionLevel] = {
    "l0": SchemaCompressionLevel.L0_LOSSLESS,
    "l0_lossless": SchemaCompressionLevel.L0_LOSSLESS,
    "lossless": SchemaCompressionLevel.L0_LOSSLESS,
    "l1": SchemaCompressionLevel.L1_SUMMARY,
    "l1_summary": SchemaCompressionLevel.L1_SUMMARY,
    "summary": SchemaCompressionLevel.L1_SUMMARY,
    "l2": SchemaCompressionLevel.L2_LAZY,
    "l2_lazy": SchemaCompressionLevel.L2_LAZY,
    "lazy": SchemaCompressionLevel.L2_LAZY,
    "l3": SchemaCompressionLevel.L3_ADAPTIVE_PINNED,
    "l3_adaptive_pinned": SchemaCompressionLevel.L3_ADAPTIVE_PINNED,
    "adaptive": SchemaCompressionLevel.L3_ADAPTIVE_PINNED,
    "dual": SchemaCompressionLevel.DUAL_PATH_ROUTER,
    "dual_path": SchemaCompressionLevel.DUAL_PATH_ROUTER,
    "dual_path_router": SchemaCompressionLevel.DUAL_PATH_ROUTER,
}


def parse_schema_level(value: str | SchemaCompressionLevel) -> SchemaCompressionLevel:
    """Parse a string or enum into :class:`SchemaCompressionLevel`."""
    if isinstance(value, SchemaCompressionLevel):
        return value
    key = str(value).strip().lower().replace("-", "_")
    if key in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[key]
    valid = ", ".join(level.value for level in SchemaCompressionLevel)
    raise ValueError(f"Unknown schema compression level {value!r}. Expected one of: {valid}")


def normalize_tool_name(tool_name: str) -> str:
    """Normalize a potentially namespaced tool name for policy matching."""
    stripped = tool_name.strip().lower()
    for sep in (":", ".", "/"):
        if sep in stripped:
            stripped = stripped.rsplit(sep, 1)[-1]
    return stripped


@dataclass
class RouterPolicy:
    """Routing and stubbing policy for schema compression and CacheRouter.

    Core execution tools (``shell``, ``bash``, ``read``, ``read_file``, ``edit``,
    ``apply_patch``, etc.) are always protected from being stubbed or blacklisted.
    """

    blacklist: frozenset[str] | Iterable[str] = field(default_factory=frozenset)
    whitelist: frozenset[str] | Iterable[str] = field(default_factory=frozenset)
    core_pinned_tools: frozenset[str] | Iterable[str] = field(
        default_factory=lambda: DEFAULT_CORE_PINNED_TOOLS
    )
    max_schema_tokens: int = 4000
    byte_exact_read_tools: frozenset[str] | Iterable[str] = field(
        default_factory=lambda: DEFAULT_BYTE_EXACT_READ_TOOLS
    )

    def __post_init__(self) -> None:
        if self.max_schema_tokens <= 0:
            raise ValueError(f"max_schema_tokens must be positive, got {self.max_schema_tokens}")

        pinned = frozenset(str(t).strip() for t in self.core_pinned_tools if str(t).strip())
        # Guarantee core execution tools are never stubbed or omitted.
        self.core_pinned_tools = pinned | DEFAULT_CORE_PINNED_TOOLS
        self._pinned_keys: frozenset[str] = frozenset(
            k for t in self.core_pinned_tools for k in (t.lower(), normalize_tool_name(t)) if k
        )

        raw_blacklist = frozenset(str(t).strip() for t in self.blacklist if str(t).strip())
        self.blacklist = frozenset(
            t
            for t in raw_blacklist
            if t.lower() not in self._pinned_keys
            and normalize_tool_name(t) not in self._pinned_keys
        )
        self._blacklist_keys: frozenset[str] = frozenset(
            k for t in self.blacklist for k in (t.lower(), normalize_tool_name(t)) if k
        )

        self.whitelist = frozenset(str(t).strip() for t in self.whitelist if str(t).strip())
        self._whitelist_keys: frozenset[str] = frozenset(
            k for t in self.whitelist for k in (t.lower(), normalize_tool_name(t)) if k
        )
        self.byte_exact_read_tools = frozenset(
            str(t).strip().lower() for t in self.byte_exact_read_tools if str(t).strip()
        )

    def is_core_tool(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is a built-in resident core execution tool."""
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in DEFAULT_CORE_PINNED_TOOLS or norm in DEFAULT_CORE_PINNED_TOOLS

    def is_core_pinned(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is a protected core or pinned execution tool."""
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in self._pinned_keys or norm in self._pinned_keys

    def is_blacklisted(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is blacklisted (never true for core tools)."""
        if self.is_core_pinned(tool_name):
            return False
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in self._blacklist_keys or norm in self._blacklist_keys

    def is_whitelisted(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is allowed by the whitelist (or whitelist is empty)."""
        if not self.whitelist:
            return True
        if self.is_core_pinned(tool_name):
            return True
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in self._whitelist_keys or norm in self._whitelist_keys

    def is_byte_exact_read(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is a byte-exact file-read tool."""
        lower = tool_name.strip().lower()
        norm = normalize_tool_name(tool_name)
        return lower in self.byte_exact_read_tools or norm in self.byte_exact_read_tools
