"""Schema compression (L0-L3) and <0.5ms pre-binding argument validation.

Supports:
- ``L0_LOSSLESS``: strip ``$schema``, ``$comment``, redundant ``title``, empty
  descriptions, and normalize description whitespace.
- ``L1_SUMMARY``: verbatim first sentence + up to 2 constraint sentences.
- ``L2_LAZY``: 3 meta-tools (``search_tools``, ``get_tool_schema``, ``invoke_tool``).
- ``L3_ADAPTIVE_PINNED``: freeze hot + core pinned tools in a prefix whose SHA-256
  never shifts, appending newly unlocked tools strictly at the tail.
- ``validate_tool_arguments``: fast structural JSON Schema validation without regex
  evaluation (no ReDoS).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schema_levels import RouterPolicy, SchemaCompressionLevel

# Number of search results returned by default
_DEFAULT_TOP_K = 5
_DEFAULT_PINNED_TOP_K = 5
_MAX_VALIDATION_DEPTH = 16
_MAX_VALIDATION_NODES = 1024
_MAX_VALIDATION_ERRORS = 32

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONSTRAINT_CUES: tuple[str, ...] = (
    "must",
    "required",
    "only",
    "never",
    "always",
    "cannot",
    "can't",
    "do not",
    "don't",
    "max",
    "min",
    "limit",
    "at least",
    "at most",
    "between",
    "valid",
    "format",
    "note:",
    "warning:",
    "important:",
)


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _is_redundant_title(title: Any, prop_name: str | None) -> bool:
    if not isinstance(title, str):
        return False
    if prop_name is None:
        return True
    norm_title = title.strip().lower().replace("_", " ").replace("-", " ")
    norm_prop = prop_name.strip().lower().replace("_", " ").replace("-", " ")
    return norm_title == norm_prop


_LITERAL_DATA_KEYS: frozenset[str] = frozenset({"enum", "const", "default", "examples"})
_SCHEMA_MAP_KEYS: frozenset[str] = frozenset(
    {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
)


def _clean_schema_node(node: Any, prop_name: str | None = None) -> Any:
    """Recursively apply L0 lossless cleanup to a JSON Schema dict."""
    if isinstance(node, list):
        return [_clean_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _LITERAL_DATA_KEYS:
            cleaned[key] = copy.deepcopy(value)
            continue
        if key in ("$schema", "$comment"):
            continue
        if key == "title" and _is_redundant_title(value, prop_name):
            continue
        if key == "description":
            if isinstance(value, str):
                norm = _normalize_ws(value)
                if not norm:
                    continue
                cleaned[key] = norm
            continue
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            cleaned[key] = {
                p_name: _clean_schema_node(p_schema, prop_name=p_name)
                for p_name, p_schema in value.items()
            }
            continue
        cleaned[key] = _clean_schema_node(value)
    return cleaned


def compact_schema_l0(tool: dict[str, Any]) -> dict[str, Any]:
    """Apply L0 lossless compaction to a single tool dict."""
    compacted: dict[str, Any] = {"name": tool.get("name", "")}
    desc = tool.get("description")
    if isinstance(desc, str):
        norm_desc = _normalize_ws(desc)
        if norm_desc:
            compacted["description"] = norm_desc
    if "inputSchema" in tool:
        compacted["inputSchema"] = _clean_schema_node(tool["inputSchema"])
    for extra_key, extra_val in tool.items():
        if extra_key not in ("name", "description", "inputSchema"):
            compacted[extra_key] = extra_val
    return compacted


def summarize_description_l1(description: str) -> str:
    """Extract verbatim first sentence plus up to 2 constraint sentences."""
    norm = _normalize_ws(description)
    if not norm:
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(norm) if s.strip()]
    if not sentences:
        return norm

    selected: list[str] = [sentences[0]]
    constraints_added = 0
    for sentence in sentences[1:]:
        lower = sentence.lower()
        if any(cue in lower for cue in _CONSTRAINT_CUES):
            selected.append(sentence)
            constraints_added += 1
            if constraints_added >= 2:
                break
    return " ".join(selected)


def compact_schema_l1(tool: dict[str, Any]) -> dict[str, Any]:
    """Apply L1 verbatim-summary compaction to a single tool dict."""
    compacted = compact_schema_l0(tool)
    desc = compacted.get("description", "")
    if desc:
        compacted["description"] = summarize_description_l1(desc)
    return compacted


def _estimate_tool_tokens(tool: dict[str, Any]) -> int:
    return max(1, len(json.dumps(tool, separators=(",", ":"))) // 4)


def _matches_json_type(value: Any, expected_type: str) -> bool:
    """Check JSON Schema primitive type with strict bool vs int distinction."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return True


def _enum_contains(enum_values: list[Any], value: Any) -> bool:
    """Exact enum membership check preserving bool vs int distinction."""
    val_is_bool = isinstance(value, bool)
    for candidate in enum_values:
        if val_is_bool != isinstance(candidate, bool):
            continue
        if value == candidate:
            return True
    return False


@dataclass
class _ValidationState:
    errors: list[str]
    node_count: int = 0
    max_depth: int = _MAX_VALIDATION_DEPTH
    max_nodes: int = _MAX_VALIDATION_NODES
    max_errors: int = _MAX_VALIDATION_ERRORS
    budget_exceeded: bool = False

    def add_error(self, message: str) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(message)


def _validate_node(
    value: Any,
    schema: dict[str, Any],
    path: str,
    state: _ValidationState,
    depth: int,
) -> None:
    if len(state.errors) >= state.max_errors or state.budget_exceeded:
        return

    if depth > state.max_depth:
        state.add_error(f"{path}: maximum validation depth ({state.max_depth}) exceeded")
        return

    state.node_count += 1
    if state.node_count > state.max_nodes:
        state.budget_exceeded = True
        state.add_error(f"{path}: validation node budget exceeded ({state.max_nodes})")
        return

    if not isinstance(schema, dict):
        return

    expected_type = schema.get("type")
    if expected_type is not None:
        allowed_types = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(_matches_json_type(value, t) for t in allowed_types):
            state.add_error(f"{path}: expected type {expected_type!r}, got {type(value).__name__}")
            return

    if "enum" in schema and isinstance(schema["enum"], list):
        if not _enum_contains(schema["enum"], value):
            state.add_error(f"{path}: value {value!r} not in enum {schema['enum']!r}")

    if isinstance(value, str):
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if min_len is not None and len(value) < min_len:
            state.add_error(f"{path}: string length {len(value)} < minLength {min_len}")
        if max_len is not None and len(value) > max_len:
            state.add_error(f"{path}: string length {len(value)} > maxLength {max_len}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exc_min = schema.get("exclusiveMinimum")
        exc_max = schema.get("exclusiveMaximum")
        if minimum is not None and value < minimum:
            state.add_error(f"{path}: {value} < minimum {minimum}")
        if maximum is not None and value > maximum:
            state.add_error(f"{path}: {value} > maximum {maximum}")
        if exc_min is not None and value <= exc_min:
            state.add_error(f"{path}: {value} <= exclusiveMinimum {exc_min}")
        if exc_max is not None and value >= exc_max:
            state.add_error(f"{path}: {value} >= exclusiveMaximum {exc_max}")

    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            state.add_error(f"{path}: array length {len(value)} < minItems {min_items}")
        if max_items is not None and len(value) > max_items:
            state.add_error(f"{path}: array length {len(value)} > maxItems {max_items}")
            return
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                if len(state.errors) >= state.max_errors or state.budget_exceeded:
                    break
                _validate_node(item, items_schema, f"{path}[{idx}]", state, depth + 1)

    if isinstance(value, dict):
        min_props = schema.get("minProperties")
        max_props = schema.get("maxProperties")
        if min_props is not None and len(value) < min_props:
            state.add_error(f"{path}: object properties {len(value)} < minProperties {min_props}")
        if max_props is not None and len(value) > max_props:
            state.add_error(f"{path}: object properties {len(value)} > maxProperties {max_props}")
            return

        required = schema.get("required")
        if isinstance(required, list):
            for req_key in required:
                if len(state.errors) >= state.max_errors:
                    break
                if req_key not in value:
                    state.add_error(f"{path}: missing required field '{req_key}'")

        props = schema.get("properties")
        if isinstance(props, dict):
            if schema.get("additionalProperties") is False:
                for key in value:
                    if len(state.errors) >= state.max_errors:
                        break
                    if key not in props:
                        state.add_error(f"{path}: unexpected additional property '{key}'")
            for key, prop_val in value.items():
                if len(state.errors) >= state.max_errors or state.budget_exceeded:
                    break
                if key in props and isinstance(props[key], dict):
                    _validate_node(prop_val, props[key], f"{path}.{key}", state, depth + 1)


def validate_tool_arguments(
    tool_name: str,
    arguments: Any,
    schema: dict[str, Any],
    *,
    max_depth: int = _MAX_VALIDATION_DEPTH,
    max_nodes: int = _MAX_VALIDATION_NODES,
    max_errors: int = _MAX_VALIDATION_ERRORS,
) -> dict[str, Any]:
    """Perform fast (<0.5ms) structural pre-binding validation on tool arguments.

    Checks required fields, JSON types (strictly distinguishing ``bool`` from
    ``int``), enums, numeric/length bounds, and ``additionalProperties: false``.
    Regex ``pattern`` matching is intentionally skipped to prevent ReDoS.

    Returns:
        Dict with keys ``valid`` (bool), ``errors`` (list[str]), and
        ``error_message`` (str).
    """
    errors: list[str] = []
    if not isinstance(arguments, dict):
        errors.append(f"{tool_name}: arguments must be an object, got {type(arguments).__name__}")
        return {"valid": False, "errors": errors, "error_message": "; ".join(errors)}

    state = _ValidationState(
        errors=errors,
        max_depth=max_depth,
        max_nodes=max_nodes,
        max_errors=max_errors,
    )
    _validate_node(arguments, schema or {}, tool_name, state, depth=0)
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "error_message": "; ".join(errors),
    }


@dataclass
class ToolEntry:
    """A single upstream tool entry."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolIndex:
    """Simple keyword search index over tool names and descriptions."""

    def __init__(self, tools: list[ToolEntry]) -> None:
        self._tools: dict[str, ToolEntry] = {t.name: t for t in tools}
        self._entries: list[ToolEntry] = list(tools)

    @classmethod
    def from_tool_dicts(cls, tools: list[dict[str, Any]]) -> "ToolIndex":
        """Build a :class:`ToolIndex` from raw tool dicts."""
        entries = [
            ToolEntry(
                name=t.get("name", ""),
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in tools
        ]
        return cls(entries)

    def search(self, query: str, top_k: int = _DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Return the *top_k* most relevant tools for *query*."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            return []
        query_lower = query.lower().strip()
        if not query_lower:
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
    """Compacts upstream MCP tool schemas according to ``SchemaCompressionLevel``.

    Defaults to ``L2_LAZY`` (the 3-meta-tool surface) for backward compatibility.
    """

    _META_NAMES: frozenset[str] = frozenset({"search_tools", "get_tool_schema", "invoke_tool"})

    def __init__(
        self,
        upstream_tools: list[dict[str, Any]],
        level: SchemaCompressionLevel = SchemaCompressionLevel.L2_LAZY,
        policy: RouterPolicy | None = None,
        top_k_pinned: int = _DEFAULT_PINNED_TOP_K,
    ) -> None:
        self._raw_tools: list[dict[str, Any]] = list(upstream_tools)
        self._raw_by_name: dict[str, dict[str, Any]] = {
            t.get("name", ""): t for t in self._raw_tools if t.get("name")
        }
        self._original_count = len(upstream_tools)
        self._level = level
        self._policy = policy or RouterPolicy()
        self._top_k_pinned = top_k_pinned
        allowed_tools = [t for t in self._raw_tools if self._is_tool_allowed(t.get("name", ""))]
        self._index = ToolIndex.from_tool_dicts(allowed_tools)

        # L3 adaptive session state
        self._call_counts: Counter[str] = Counter()
        self._frozen_prefix: list[dict[str, Any]] | None = None
        self._frozen_prefix_names: set[str] = set()
        self._prefix_sha256: str = ""
        self._tail_tools: list[dict[str, Any]] = []
        self._tail_names: set[str] = set()

    def _is_tool_allowed(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is permitted by blacklist and whitelist."""
        if not tool_name:
            return False
        if self._policy.is_blacklisted(tool_name):
            return False
        if self._policy.whitelist and not self._policy.is_whitelisted(tool_name):
            return False
        return True

    # ------------------------------------------------------------------
    # Meta-tool definitions (L2 / L3 tail discovery)
    # ------------------------------------------------------------------

    def meta_tool_schemas(self) -> list[dict[str, Any]]:
        """Return the 3 meta-tool schema dicts."""
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
    # L0-L3 compaction & adaptive prefix pinning
    # ------------------------------------------------------------------

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool call for L3 adaptive frequency tracking."""
        if tool_name and tool_name not in self._META_NAMES:
            self._call_counts[tool_name] += 1

    def unlock_tool(self, tool_name: str) -> bool:
        """Append a newly unlocked tool strictly at the tail of L3 tools.

        Returns ``True`` if the tool was appended, ``False`` if it was already
        present in the frozen prefix or tail, or unknown.
        """
        if tool_name in self._frozen_prefix_names or tool_name in self._tail_names:
            return False
        raw = self._raw_by_name.get(tool_name)
        if raw is None or not self._is_tool_allowed(tool_name):
            return False
        self._tail_tools.append(compact_schema_l0(raw))
        self._tail_names.add(tool_name)
        return True

    def _build_l3_prefix(self) -> list[dict[str, Any]]:
        """Freeze the turn-0 L3 prefix (core/whitelisted + top-K hot tools + meta-tools)."""
        pinned: list[dict[str, Any]] = []
        pinned_names: set[str] = set()
        tokens_used = 0
        budget = self._policy.max_schema_tokens

        # 1. Core execution tools & whitelisted tools first (core tools never stubbed)
        for tool in self._raw_tools:
            name = tool.get("name", "")
            if not self._is_tool_allowed(name):
                continue
            if self._policy.is_core_pinned(name) or (
                bool(self._policy.whitelist) and self._policy.is_whitelisted(name)
            ):
                compacted = compact_schema_l0(tool)
                pinned.append(compacted)
                pinned_names.add(name)
                tokens_used += _estimate_tool_tokens(compacted)

        # 2. Top-K hot tools by recorded frequency (within token budget)
        hot_candidates = sorted(
            (t for t in self._raw_tools if t.get("name", "") not in pinned_names),
            key=lambda t: self._call_counts.get(t.get("name", ""), 0),
            reverse=True,
        )
        added_hot = 0
        for tool in hot_candidates:
            if added_hot >= self._top_k_pinned:
                break
            name = tool.get("name", "")
            if not self._is_tool_allowed(name):
                continue
            compacted = compact_schema_l0(tool)
            est = _estimate_tool_tokens(compacted)
            if tokens_used + est > budget and pinned:
                continue
            pinned.append(compacted)
            pinned_names.add(name)
            tokens_used += est
            added_hot += 1

        prefix = pinned + self.meta_tool_schemas()
        self._frozen_prefix = prefix
        self._frozen_prefix_names = pinned_names
        canonical = json.dumps(prefix, sort_keys=True, separators=(",", ":"))
        self._prefix_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return prefix

    @property
    def prefix_sha256(self) -> str:
        """SHA-256 hex digest of the frozen L3 prefix (empty until built)."""
        return self._prefix_sha256

    def get_tools(self, level: SchemaCompressionLevel | None = None) -> list[dict[str, Any]]:
        """Return compacted tool schemas for the requested compression level."""
        active = level or self._level
        if active == SchemaCompressionLevel.L0_LOSSLESS:
            return [
                compact_schema_l0(t)
                for t in self._raw_tools
                if self._is_tool_allowed(t.get("name", ""))
            ]
        if active == SchemaCompressionLevel.L1_SUMMARY:
            return [
                compact_schema_l1(t)
                for t in self._raw_tools
                if self._is_tool_allowed(t.get("name", ""))
            ]
        if active == SchemaCompressionLevel.L2_LAZY:
            return self.meta_tool_schemas()
        if active == SchemaCompressionLevel.L3_ADAPTIVE_PINNED:
            if self._frozen_prefix is None:
                self._build_l3_prefix()
            assert self._frozen_prefix is not None
            return list(self._frozen_prefix) + list(self._tail_tools)
        return self.meta_tool_schemas()

    # ------------------------------------------------------------------
    # Meta-tool dispatch
    # ------------------------------------------------------------------

    def handle_meta_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Handle a meta-tool call synchronously."""
        if not isinstance(arguments, dict):
            return json.dumps(
                {
                    "status": "error",
                    "reason": "invalid_arguments",
                    "message": f"{name} arguments must be an object, got {type(arguments).__name__}",
                    "error_message": f"{name} arguments must be an object, got {type(arguments).__name__}",
                }
            )

        if name == "search_tools":
            raw_query = arguments.get("query", "")
            query = (
                raw_query
                if isinstance(raw_query, str)
                else ("" if raw_query is None else str(raw_query))
            )
            raw_top_k = arguments.get("top_k", _DEFAULT_TOP_K)
            if isinstance(raw_top_k, bool):
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_top_k",
                        "message": "top_k must be an integer, not bool",
                        "error_message": "top_k must be an integer, not bool",
                    }
                )
            if isinstance(raw_top_k, int):
                top_k = raw_top_k
            elif isinstance(raw_top_k, str):
                try:
                    top_k = int(raw_top_k.strip())
                except ValueError:
                    return json.dumps(
                        {
                            "status": "error",
                            "reason": "invalid_top_k",
                            "message": f"Invalid top_k value: {raw_top_k!r}",
                            "error_message": f"Invalid top_k value: {raw_top_k!r}",
                        }
                    )
            else:
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_top_k",
                        "message": f"top_k must be an integer, got {type(raw_top_k).__name__}",
                        "error_message": f"top_k must be an integer, got {type(raw_top_k).__name__}",
                    }
                )
            if top_k <= 0:
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_top_k",
                        "message": f"top_k must be >= 1, got {top_k}",
                        "error_message": f"top_k must be >= 1, got {top_k}",
                    }
                )
            top_k = min(20, top_k)
            results = self._index.search(query, top_k=top_k)
            return json.dumps(
                {
                    "status": "success",
                    "results": results,
                    "total_tools": self._original_count,
                }
            )

        if name == "get_tool_schema":
            raw_tool_name = arguments.get("tool_name", "")
            if not isinstance(raw_tool_name, str) or not raw_tool_name.strip():
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "message": "tool_name must be a non-empty string",
                        "error_message": "tool_name must be a non-empty string",
                    }
                )
            tool_name = raw_tool_name
            if tool_name in self._raw_by_name and not self._is_tool_allowed(tool_name):
                reason = (
                    "blacklisted" if self._policy.is_blacklisted(tool_name) else "not_whitelisted"
                )
                msg = f"Tool '{tool_name}' is {reason.replace('_', ' ')}"
                return json.dumps(
                    {
                        "status": "error",
                        "reason": reason,
                        "message": msg,
                        "error_message": msg,
                    }
                )
            schema = self._index.get_schema(tool_name)
            if schema is not None:
                if self._level == SchemaCompressionLevel.L3_ADAPTIVE_PINNED:
                    self.unlock_tool(tool_name)
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
            raw_tool_name = arguments.get("tool_name", "")
            if not isinstance(raw_tool_name, str) or not raw_tool_name.strip():
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "message": "tool_name must be a non-empty string",
                        "error_message": "tool_name must be a non-empty string",
                    }
                )
            tool_name = raw_tool_name
            if not self._is_tool_allowed(tool_name):
                reason = (
                    "blacklisted" if self._policy.is_blacklisted(tool_name) else "not_whitelisted"
                )
                msg = f"Tool '{tool_name}' is {reason.replace('_', ' ')}"
                return json.dumps(
                    {
                        "status": "error",
                        "reason": reason,
                        "message": msg,
                        "error_message": msg,
                    }
                )
            tool_args = arguments.get("arguments", {})
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                return json.dumps(
                    {
                        "status": "error",
                        "reason": "invalid_arguments",
                        "message": f"arguments must be an object, got {type(tool_args).__name__}",
                        "error_message": f"arguments must be an object, got {type(tool_args).__name__}",
                    }
                )
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
