"""Meta-tool pattern: replaces N upstream tools with 3 search/inspect/invoke meta-tools.

Instead of exposing every upstream tool directly (which consumes many tokens in the
``tools/list`` response), ``SchemaCompressor`` creates a tiny 3-tool surface:

- ``search_tools``   — keyword search across all upstream tool names and descriptions.
- ``get_tool_schema``— retrieve the full input schema for a specific tool.
- ``invoke_tool``    — call an upstream tool by name (proxy handles the forwarding).

The AI can discover and use all upstream tools through these three meta-tools, saving
the bulk of the schema tokens upfront.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Number of search results returned by default
_DEFAULT_TOP_K = 5


@dataclass
class ToolEntry:
    """A single upstream tool entry."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolIndex:
    """Simple keyword search index over tool names and descriptions.

    Scoring is bag-of-words: each query term that appears in the combined
    ``name + description`` text adds 1 to the score.  Ties are broken by
    insertion order (stable sort).

    Args:
        tools: List of :class:`ToolEntry` objects to index.
    """

    def __init__(self, tools: list[ToolEntry]) -> None:
        self._tools: dict[str, ToolEntry] = {t.name: t for t in tools}
        self._entries: list[ToolEntry] = list(tools)

    @classmethod
    def from_tool_dicts(cls, tools: list[dict[str, Any]]) -> "ToolIndex":
        """Build a :class:`ToolIndex` from raw tool dicts.

        Each dict is expected to have the keys ``name``, ``description``, and
        optionally ``inputSchema``.
        """
        entries = [
            ToolEntry(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in tools
        ]
        return cls(entries)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant tools for *query*.

        When *query* is empty or blank all tools are returned (up to *top_k*).

        Args:
            query: Space-separated search keywords.
            top_k: Maximum number of results.

        Returns:
            List of dicts with keys ``name``, ``description`` (first 200 chars),
            and ``relevance`` (integer score).
        """
        query_lower = query.lower().strip()
        if not query_lower:
            # No query → return top_k tools sorted by name
            return [
                {"name": e.name, "description": e.description[:200], "relevance": 0}
                for e in self._entries[:top_k]
            ]

        query_terms = query_lower.split()
        scored: list[tuple[ToolEntry, int]] = []
        for entry in self._entries:
            text = f"{entry.name} {entry.description}".lower()
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {"name": e.name, "description": e.description[:200], "relevance": s}
            for e, s in scored[:top_k]
        ]

    def get_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Return the full tool dict for *tool_name*, or ``None`` if not found."""
        entry = self._tools.get(tool_name)
        if entry is None:
            return None
        return {
            "name": entry.name,
            "description": entry.description,
            "inputSchema": entry.input_schema,
        }

    def list_all(self) -> list[dict[str, Any]]:
        """Return a compact list of all tools (name + first 100 chars of description)."""
        return [{"name": e.name, "description": e.description[:100]} for e in self._entries]

    @property
    def tool_count(self) -> int:
        """Number of tools in the index."""
        return len(self._entries)


class SchemaCompressor:
    """Generates 3 meta-tools that replace a full upstream tool list.

    The meta-tools allow an AI to discover and invoke any upstream tool without
    receiving the full schema listing upfront, saving significant context tokens.

    Args:
        upstream_tools: Raw tool dicts from the upstream ``tools/list`` response.
    """

    _META_NAMES: frozenset[str] = frozenset({"search_tools", "get_tool_schema", "invoke_tool"})

    def __init__(self, upstream_tools: list[dict[str, Any]]) -> None:
        self._index = ToolIndex.from_tool_dicts(upstream_tools)
        self._original_count = len(upstream_tools)

    # ------------------------------------------------------------------
    # Meta-tool definitions
    # ------------------------------------------------------------------

    def meta_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the 3 meta-tool schema dicts.

        These replace the full upstream tool listing when schema compression is
        enabled.  The descriptions reference the total upstream tool count so
        the AI knows what it's searching.

        Returns:
            List of 3 tool dicts (``name``, ``description``, ``inputSchema``).
        """
        count = self._original_count
        return [
            {
                "name": "search_tools",
                "description": (
                    f"Search {count} available upstream tools by keyword. "
                    "Returns top matches with names and descriptions. "
                    "Use this to discover which tool to call."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keywords",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": _DEFAULT_TOP_K,
                            "description": "Number of results to return",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_tool_schema",
                "description": (
                    "Get the full input schema for a specific upstream tool by exact name. "
                    "Use search_tools first to find the right name."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Exact tool name",
                        },
                    },
                    "required": ["tool_name"],
                },
            },
            {
                "name": "invoke_tool",
                "description": (
                    "Invoke an upstream tool by name with arguments. "
                    "Use search_tools to find the right tool, then get_tool_schema to "
                    "see its parameters, then call this to execute it."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool to invoke",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Tool arguments",
                            "default": {},
                        },
                    },
                    "required": ["tool_name"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Meta-tool dispatch
    # ------------------------------------------------------------------

    def handle_meta_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Handle a meta-tool call synchronously.

        Args:
            name: Meta-tool name (must be one of the 3 meta-tools).
            arguments: Tool arguments dict.

        Returns:
            JSON string result.  For ``invoke_tool``, returns a marker dict
            with ``"_invoke_upstream": true`` so the caller knows to forward
            the call to the upstream server.
        """
        if name == "search_tools":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", _DEFAULT_TOP_K))
            results = self._index.search(query, top_k=top_k)
            return json.dumps(
                {
                    "status": "success",
                    "results": results,
                    "total_tools": self._original_count,
                }
            )

        if name == "get_tool_schema":
            tool_name = arguments.get("tool_name", "")
            schema = self._index.get_schema(tool_name)
            if schema is not None:
                return json.dumps({"status": "success", "tool": schema})
            available = [e["name"] for e in self._index.list_all()]
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Tool '{tool_name}' not found",
                    "available_tools": available[:20],
                }
            )

        if name == "invoke_tool":
            # Return a forwarding marker — the proxy server handles the actual call.
            tool_name = arguments.get("tool_name", "")
            tool_args = arguments.get("arguments", {})
            return json.dumps(
                {
                    "_invoke_upstream": True,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                }
            )

        return json.dumps({"status": "error", "message": f"Unknown meta-tool: {name}"})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def original_tool_count(self) -> int:
        """Number of tools in the upstream index."""
        return self._original_count

    @property
    def index(self) -> ToolIndex:
        """Direct access to the underlying :class:`ToolIndex`."""
        return self._index
