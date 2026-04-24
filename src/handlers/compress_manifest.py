"""MCP tool manifest compression handler (v1.8.0 A2b).

Compresses the ``description`` fields of an MCP ``tools/list`` response so
downstream agents see shorter tool descriptions without losing ``inputSchema``
semantics.  Neutralises Atlassian's ``mcp-compressor`` in sales conversations
by demonstrating that gotcontext already covers schema-level compression.

Contract:
- ``inputSchema`` is preserved byte-for-byte (JSON-equivalent).
- ``name`` is preserved verbatim.
- ``description`` is compressed via the existing token-estimation approach
  (length-based, no heavy model dependencies) so the handler works in the
  SaaS gateway without torch/sentence-transformers.
- ``stats`` in the response carries ``input_tokens``, ``output_tokens``,
  ``savings_pct`` so callers can report savings without a second round-trip.
"""

from __future__ import annotations

import copy
import re
from typing import Any

# ---------------------------------------------------------------------------
# Token estimation (lightweight — no model dependency)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _estimate_tokens(text: str) -> int:
    """Estimate token count without tiktoken dependency.

    Uses the same chars/4 heuristic relied on throughout the codebase for
    contexts where tiktoken is not guaranteed to be present (e.g. SaaS
    gateway running without the full torch stack).
    """
    return max(0, len(text) // 4)


# ---------------------------------------------------------------------------
# Description compression
# ---------------------------------------------------------------------------

# Maximum number of words to keep per description after compression.
# Chosen empirically: enough to convey the tool's purpose to an orchestrating
# agent while cutting 40-70% of verbose descriptions.
_MAX_WORDS = 40


def _compress_description(description: str) -> str:
    """Compress a single tool description by extracting the leading sentence(s).

    Strategy:
    1. If the description is already short (<= _MAX_WORDS words), return as-is.
    2. Otherwise keep the first sentence (ends at `. ` or `.\n`) which almost
       always contains the tool's core purpose.
    3. If the first sentence alone exceeds _MAX_WORDS, truncate at the word
       boundary and append an ellipsis.

    This is intentionally conservative — ``inputSchema`` carries the full
    parameter contracts; agents only need the description to route tool
    selection, not to understand every nuance.
    """
    if not description or not description.strip():
        return description

    words = _WORD_RE.findall(description)
    if len(words) <= _MAX_WORDS:
        return description

    # Try to keep first sentence.
    first_sentence_match = re.search(r"[.!?](?:\s|$)", description)
    if first_sentence_match:
        candidate = description[: first_sentence_match.end()].strip()
        candidate_words = _WORD_RE.findall(candidate)
        if len(candidate_words) <= _MAX_WORDS:
            return candidate

    # Fall back to word-boundary truncation.
    return " ".join(words[:_MAX_WORDS]) + "…"


# ---------------------------------------------------------------------------
# Public handler
# ---------------------------------------------------------------------------


def handle_compress_manifest(params: dict[str, Any]) -> dict[str, Any]:
    """Compress an MCP tools/list manifest.

    Input:
        {
            "manifest": {
                "tools": [
                    {"name": str, "description": str, "inputSchema": dict},
                    ...
                ]
            }
        }

    Output:
        {
            "manifest": {
                "tools": [...compressed descriptions, schemas preserved...]
            },
            "stats": {
                "input_tokens": int,
                "output_tokens": int,
                "savings_pct": float   # 0.0–100.0
            }
        }

    The ``inputSchema`` for every tool is preserved byte-for-byte
    (JSON-equivalent).  Only ``description`` fields are modified.
    ``name`` is never touched.
    """
    manifest = params.get("manifest", {})
    tools: list[dict[str, Any]] = manifest.get("tools", [])

    if not tools:
        return {
            "manifest": {"tools": []},
            "stats": {"input_tokens": 0, "output_tokens": 0, "savings_pct": 0.0},
        }

    total_input_tokens = 0
    total_output_tokens = 0
    compressed_tools: list[dict[str, Any]] = []

    for tool in tools:
        # Deep-copy so the original manifest is never mutated.
        out_tool = copy.deepcopy(tool)

        description = tool.get("description", "") or ""
        compressed = _compress_description(description)

        in_toks = _estimate_tokens(description)
        out_toks = _estimate_tokens(compressed)
        total_input_tokens += in_toks
        total_output_tokens += out_toks

        out_tool["description"] = compressed
        # inputSchema is already deep-copied — no further action needed.
        compressed_tools.append(out_tool)

    if total_input_tokens > 0:
        savings_pct = round(
            max(0.0, (total_input_tokens - total_output_tokens) / total_input_tokens * 100.0),
            2,
        )
    else:
        savings_pct = 0.0

    return {
        "manifest": {"tools": compressed_tools},
        "stats": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "savings_pct": savings_pct,
        },
    }
