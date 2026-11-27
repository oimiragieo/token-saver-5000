"""Dual-Path CacheRouter with 4-stage filtering, ISO/BM25 relevance, and state gating.

Implements:
- 4-stage filter ``F_Y(F_K(F_W(F_B(T))))``:
  Blacklist -> Whitelist -> Top-K BM25 relevance -> Yellowlist soft-avoidance
  with top-1 fallback.
- State precondition gating (``state_t |= effects``) and hallucination rejection
  for unknown tools, unmet preconditions, and invalid arguments.
- ``INTERNAL_ROUTER`` meta-tool schema and isolated routing channel so the main
  model's resident ``tools[]`` block remains byte-identical across turns.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import re
from collections import Counter
from typing import Any, Iterable

from .schema_compressor import compact_schema_l0, validate_tool_arguments
from .schema_levels import RouterPolicy

INTERNAL_ROUTER_TOOL_NAME = "internal_router"

INTERNAL_ROUTER_SCHEMA: dict[str, Any] = {
    "name": INTERNAL_ROUTER_TOOL_NAME,
    "description": (
        "Discover, inspect, or route long-tail MCP tools without mutating the "
        "resident tool prefix. Actions: 'discover' (query), 'inspect' (tool_name), "
        "'route' (tool_name + arguments)."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["discover", "inspect", "route"],
                "default": "discover",
                "description": "Routing action to perform",
            },
            "query": {
                "type": "string",
                "description": "Search query for action='discover'",
            },
            "top_k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 25,
                "default": 5,
                "description": "Maximum candidates to return",
            },
            "tool_name": {
                "type": "string",
                "description": "Target upstream tool name for inspect/route",
            },
            "arguments": {
                "type": "object",
                "default": {},
                "description": "Arguments for action='route'",
            },
        },
        "required": ["action"],
    },
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("_", " ").replace("-", " "))


def _tool_document_tokens(tool: dict[str, Any]) -> list[str]:
    parts = [tool.get("name", ""), tool.get("description", "")]
    schema = tool.get("inputSchema")
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            parts.extend(props.keys())
    return _tokenize(" ".join(str(p) for p in parts if p))


class CacheRouter:
    """Dual-path tool router keeping the resident prefix cache-stable."""

    def __init__(
        self,
        upstream_tools: list[dict[str, Any]],
        policy: RouterPolicy | None = None,
        initial_state: Iterable[str] = (),
        yellowlist: Iterable[str] | None = None,
    ) -> None:
        self._policy = policy or RouterPolicy()
        self._tools: list[dict[str, Any]] = [compact_schema_l0(t) for t in upstream_tools]
        self._by_name: dict[str, dict[str, Any]] = {
            t["name"]: t for t in self._tools if t.get("name")
        }
        self._state: set[str] = set(initial_state)
        self._preconditions: dict[str, set[str]] = {}
        self._effects: dict[str, set[str]] = {}

        for raw in upstream_tools:
            name = raw.get("name", "")
            if not name:
                continue
            input_schema = (
                raw.get("inputSchema") if isinstance(raw.get("inputSchema"), dict) else {}
            )
            pre = raw.get("x-preconditions")
            if pre is None:
                pre = input_schema.get("x-preconditions")
            eff = raw.get("x-effects")
            if eff is None:
                eff = input_schema.get("x-effects")
            if isinstance(pre, (list, set, tuple, frozenset)):
                self._preconditions[name] = {str(x) for x in pre}
            if isinstance(eff, (list, set, tuple, frozenset)):
                self._effects[name] = {str(x) for x in eff}

        # Precompute BM25 corpus statistics once at init
        self._doc_tokens: dict[str, list[str]] = {
            t["name"]: _tool_document_tokens(t) for t in self._tools if t.get("name")
        }
        self._doc_tf: dict[str, Counter[str]] = {
            name: Counter(toks) for name, toks in self._doc_tokens.items()
        }
        n_docs = max(1, len(self._doc_tokens))
        self._avg_dl = sum(len(toks) for toks in self._doc_tokens.values()) / n_docs
        df: Counter[str] = Counter()
        for toks in self._doc_tokens.values():
            df.update(set(toks))
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        self._doc_dl: dict[str, int] = {name: sum(tf.values()) for name, tf in self._doc_tf.items()}
        k1 = 1.5
        b = 0.75
        avg_dl = self._avg_dl or 1.0
        postings: dict[str, list[tuple[int, float]]] = {}
        for tool_idx, tool in enumerate(self._tools):
            name = tool.get("name")
            if not name:
                continue
            tf = self._doc_tf.get(name)
            if not tf:
                continue
            dl = self._doc_dl[name]
            dl_norm = k1 * (1.0 - b + b * (dl / avg_dl))
            for term, freq in tf.items():
                if freq == 0:
                    continue
                idf = self._idf.get(term, 0.0)
                weight = idf * (freq * (k1 + 1.0)) / (freq + dl_norm)
                postings.setdefault(term, []).append((tool_idx, weight))
        self._postings: dict[str, list[tuple[int, float]]] = postings

        self._policy_cache: tuple[tuple[Any, Any, Any] | None, list[tuple[int, dict[str, Any]]]] = (
            None,
            [],
        )
        self._rebuild_policy_eligible()

        # Resident prefix: core pinned tools present in catalog + INTERNAL_ROUTER
        resident = [
            t
            for t in self._tools
            if not self._policy.is_blacklisted(t["name"])
            and (
                self._policy.is_core_pinned(t["name"])
                or (bool(self._policy.whitelist) and self._policy.is_whitelisted(t["name"]))
            )
        ]
        resident.append(INTERNAL_ROUTER_SCHEMA)
        self._resident_tools: list[dict[str, Any]] = resident
        self._resident_names: frozenset[str] = frozenset(
            t["name"] for t in resident if t.get("name")
        )
        canonical = json.dumps(resident, sort_keys=True, separators=(",", ":"))
        self._resident_sha256: str = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        if yellowlist is None:
            self._default_yellowlist: frozenset[str] = frozenset(
                t["name"] for t in resident if t["name"] != INTERNAL_ROUTER_TOOL_NAME
            )
        else:
            self._default_yellowlist = frozenset(yellowlist)

    # ------------------------------------------------------------------
    # Resident block (byte-identical across turns)
    # ------------------------------------------------------------------

    def get_resident_tools(self) -> list[dict[str, Any]]:
        """Return the frozen resident tool list (core pinned + internal_router)."""
        return list(self._resident_tools)

    @property
    def resident_sha256(self) -> str:
        """SHA-256 digest of the frozen resident ``tools[]`` block."""
        return self._resident_sha256

    # ------------------------------------------------------------------
    # State precondition gating
    # ------------------------------------------------------------------

    @property
    def state(self) -> frozenset[str]:
        """Current session state flags (``state_t``)."""
        return frozenset(self._state)

    def register_preconditions(
        self,
        tool_name: str,
        requires: Iterable[str] = (),
        effects: Iterable[str] = (),
    ) -> None:
        """Declare state preconditions and post-execution effects for *tool_name*."""
        self._preconditions[tool_name] = set(requires)
        self._effects[tool_name] = set(effects)

    def is_precondition_satisfied(self, tool_name: str) -> bool:
        """Return ``True`` if all preconditions of *tool_name* are in ``state_t``."""
        req = self._preconditions.get(tool_name, set())
        return req.issubset(self._state)

    def advance_state(self, tool_name: str) -> frozenset[str]:
        """Apply post-execution state effects (``state_t |= effects``)."""
        eff = self._effects.get(tool_name)
        if eff:
            self._state |= eff
        return self.state

    def commit_tool_effects(self, tool_name: str, succeeded: bool) -> frozenset[str]:
        """Apply post-execution state effects only when upstream execution succeeded."""
        if succeeded:
            return self.advance_state(tool_name)
        return self.state

    def is_resident_tool(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* is in the resident tool prefix."""
        return tool_name in self._resident_names

    # ------------------------------------------------------------------
    # 4-stage filter: F_Y(F_K(F_W(F_B(T))))
    # ------------------------------------------------------------------

    def _policy_cache_key(self, policy: RouterPolicy | None = None) -> tuple[Any, Any, Any]:
        pol = self._policy if policy is None else policy
        return (
            pol.blacklist,
            pol.whitelist,
            pol.core_pinned_tools,
        )

    def _rebuild_policy_eligible(
        self, policy: RouterPolicy | None = None
    ) -> list[tuple[int, dict[str, Any]]]:
        pol = self._policy if policy is None else policy
        key = self._policy_cache_key(pol)
        has_whitelist = bool(pol.whitelist)
        eligible: list[tuple[int, dict[str, Any]]] = []
        for idx, tool in enumerate(self._tools):
            name = tool["name"]
            if pol.is_blacklisted(name):
                continue
            if has_whitelist and not pol.is_whitelisted(name):
                continue
            eligible.append((idx, tool))
        self._policy_cache = (key, eligible)
        return eligible

    def _bm25_score(self, tool_name: str, query_terms: list[str]) -> float:
        if not query_terms:
            return 0.0
        tf = self._doc_tf.get(tool_name)
        if not tf:
            return 0.0
        dl = self._doc_dl.get(tool_name, 0)
        k1 = 1.5
        b = 0.75
        avg_dl = self._avg_dl or 1.0
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = self._idf.get(term, 0.0)
            denom = freq + k1 * (1.0 - b + b * (dl / avg_dl))
            score += idf * (freq * (k1 + 1.0)) / denom
        return score

    def filter_tools(
        self,
        query: str = "",
        top_k: int = 5,
        yellowlist: Iterable[str] | None = None,
        respect_preconditions: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute 4-stage filter ``F_Y(F_K(F_W(F_B(T))))`` over the tool catalog."""
        # Stage 1 & 2: F_B (Blacklist) + F_W (Whitelist) cached until RouterPolicy changes
        policy = self._policy
        cached_key, policy_eligible = self._policy_cache
        if self._policy_cache_key(policy) != cached_key:
            policy_eligible = self._rebuild_policy_eligible(policy)

        # Precondition ISO gating
        if respect_preconditions:
            eligible = [
                (tool_idx, t)
                for tool_idx, t in policy_eligible
                if self.is_precondition_satisfied(t["name"])
            ]
        else:
            eligible = policy_eligible

        if not eligible:
            return []

        # Stage 3: F_K — Top-K BM25 relevance via inverted posting index
        k = max(1, top_k)
        q_terms = _tokenize(query)
        if not q_terms:
            stage_k = [t for _, t in eligible[:k]]
        else:
            eligible_idx_set = {tool_idx for tool_idx, _ in eligible}
            scores: dict[int, float] = {}
            postings = self._postings
            for term in q_terms:
                hits = postings.get(term)
                if not hits:
                    continue
                for tool_idx, weight in hits:
                    if tool_idx in eligible_idx_set:
                        scores[tool_idx] = scores.get(tool_idx, 0.0) + weight
            if not scores:
                stage_k = [t for _, t in eligible[:k]]
            else:
                scored = [
                    (-scores.get(tool_idx, 0.0), pos, t)
                    for pos, (tool_idx, t) in enumerate(eligible)
                ]
                top_items = heapq.nsmallest(k, scored) if k < len(scored) else sorted(scored)
                stage_k = [item[2] for item in top_items]

        # Stage 4: F_Y — Yellowlist soft-avoidance with top-1 fallback
        yl = self._default_yellowlist if yellowlist is None else frozenset(yellowlist)
        if not yl:
            return stage_k
        stage_y = [t for t in stage_k if t["name"] not in yl]
        if not stage_y and stage_k:
            return [stage_k[0]]
        return stage_y

    # ------------------------------------------------------------------
    # Hallucination rejection gate & INTERNAL_ROUTER dispatch
    # ------------------------------------------------------------------

    def validate_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        validate_arguments: bool = True,
    ) -> dict[str, Any]:
        """Reject unknown tools, blacklisted/non-whitelisted tools, unmet preconditions, or bad args."""
        if tool_name not in self._by_name:
            return {
                "allowed": False,
                "reason": "unknown_tool",
                "errors": [f"Unknown tool '{tool_name}'"],
                "error_message": f"Unknown tool '{tool_name}'",
            }
        if self._policy.is_blacklisted(tool_name):
            return {
                "allowed": False,
                "reason": "blacklisted",
                "errors": [f"Tool '{tool_name}' is blacklisted"],
                "error_message": f"Tool '{tool_name}' is blacklisted",
            }
        if self._policy.whitelist and not self._policy.is_whitelisted(tool_name):
            return {
                "allowed": False,
                "reason": "not_whitelisted",
                "errors": [f"Tool '{tool_name}' is not whitelisted"],
                "error_message": f"Tool '{tool_name}' is not whitelisted",
            }
        missing = sorted(self._preconditions.get(tool_name, set()) - self._state)
        if missing:
            msg = f"Tool '{tool_name}' missing state preconditions: {', '.join(missing)}"
            return {
                "allowed": False,
                "reason": "precondition_unmet",
                "missing_preconditions": missing,
                "errors": [msg],
                "error_message": msg,
            }
        if arguments is not None and validate_arguments:
            schema = self._by_name[tool_name].get("inputSchema", {})
            val = validate_tool_arguments(tool_name, arguments, schema)
            if not val["valid"]:
                return {
                    "allowed": False,
                    "reason": "invalid_arguments",
                    "errors": val["errors"],
                    "error_message": val["error_message"],
                }
        return {"allowed": True, "reason": "ok", "errors": [], "error_message": ""}

    def route_internal_call(
        self,
        arguments: dict[str, Any],
        *,
        validate_arguments: bool = True,
    ) -> dict[str, Any]:
        """Handle an ``internal_router`` meta-tool call in the isolated channel."""
        if not isinstance(arguments, dict):
            return {
                "status": "error",
                "reason": "invalid_arguments",
                "error_message": (
                    f"internal_router arguments must be an object, got {type(arguments).__name__}"
                ),
            }

        raw_action = arguments.get("action", "discover")
        if not isinstance(raw_action, str):
            return {
                "status": "error",
                "reason": "unknown_action",
                "error_message": f"action must be a string, got {type(raw_action).__name__}",
            }
        action = raw_action.lower()

        if action == "discover":
            raw_query = arguments.get("query", "")
            if raw_query is None:
                query = ""
            elif isinstance(raw_query, str):
                query = raw_query
            else:
                return {
                    "status": "error",
                    "action": "discover",
                    "reason": "invalid_arguments",
                    "error_message": f"query must be a string, got {type(raw_query).__name__}",
                }
            raw_top_k = arguments.get("top_k", 5)
            if isinstance(raw_top_k, bool):
                return {
                    "status": "error",
                    "action": "discover",
                    "reason": "invalid_top_k",
                    "error_message": "top_k must be an integer, not bool",
                }
            if isinstance(raw_top_k, int):
                top_k = raw_top_k
            elif isinstance(raw_top_k, str):
                try:
                    top_k = int(raw_top_k.strip())
                except ValueError:
                    return {
                        "status": "error",
                        "action": "discover",
                        "reason": "invalid_top_k",
                        "error_message": f"Invalid top_k value: {raw_top_k!r}",
                    }
            else:
                return {
                    "status": "error",
                    "action": "discover",
                    "reason": "invalid_top_k",
                    "error_message": f"top_k must be an integer, got {type(raw_top_k).__name__}",
                }
            top_k = max(1, min(20, top_k))
            candidates = self.filter_tools(query=query, top_k=top_k)
            return {
                "status": "success",
                "action": "discover",
                "results": [
                    {
                        "name": t["name"],
                        "description": t.get("description", "")[:200],
                    }
                    for t in candidates
                ],
                "total_tools": len(self._tools),
            }

        if action == "inspect":
            raw_tool_name = arguments.get("tool_name")
            if not isinstance(raw_tool_name, str) or not raw_tool_name.strip():
                return {
                    "status": "error",
                    "action": "inspect",
                    "reason": "invalid_arguments",
                    "error_message": "tool_name must be a non-empty string",
                }
            tool_name = raw_tool_name
            gate = self.validate_call(tool_name, validate_arguments=validate_arguments)
            if not gate["allowed"]:
                return {"status": "error", "action": "inspect", **gate}
            return {
                "status": "success",
                "action": "inspect",
                "tool": self._by_name[tool_name],
            }

        if action == "route":
            raw_tool_name = arguments.get("tool_name")
            if not isinstance(raw_tool_name, str) or not raw_tool_name.strip():
                return {
                    "status": "error",
                    "action": "route",
                    "reason": "invalid_arguments",
                    "error_message": "tool_name must be a non-empty string",
                }
            tool_name = raw_tool_name
            if self._policy.is_core_tool(tool_name):
                msg = (
                    f"Tool '{tool_name}' is a resident core tool and cannot be tunneled "
                    f"through '{INTERNAL_ROUTER_TOOL_NAME}'; invoke '{tool_name}' directly."
                )
                return {
                    "status": "error",
                    "action": "route",
                    "allowed": False,
                    "reason": "resident_tool_direct_only",
                    "errors": [msg],
                    "error_message": msg,
                }
            tool_args = arguments.get("arguments", {})
            if tool_args is None:
                tool_args = {}
            if not isinstance(tool_args, dict):
                return {
                    "status": "error",
                    "action": "route",
                    "reason": "invalid_arguments",
                    "error_message": f"arguments must be an object, got {type(tool_args).__name__}",
                }
            gate = self.validate_call(tool_name, tool_args, validate_arguments=validate_arguments)
            if not gate["allowed"]:
                return {"status": "error", "action": "route", **gate}
            return {
                "_invoke_upstream": True,
                "tool_name": tool_name,
                "arguments": tool_args,
            }

        return {
            "status": "error",
            "reason": "unknown_action",
            "error_message": f"Unknown internal_router action: {action!r}",
        }
