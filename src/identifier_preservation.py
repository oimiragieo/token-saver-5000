"""Execution-critical identifier preservation for compressed tool output.

Ported from the API tool-output endpoint so the compression ENGINE (and the
``gotcontext wrap`` proxy's :class:`~src.proxy.response_interceptor.ResponseInterceptor`)
can guarantee that file paths, error codes, symbols, URLs, stack frames, env var
names, and UUIDs survive compression — the "amnesia tax" fix (openai/codex
#18318): naive compression drops the very tokens an agent needs to avoid
re-running a tool.

This is the canonical home. ``api/app/routers/v1/compress_tool_output.py`` is
being migrated to import ``extract_critical_identifiers`` / ``apply_identifier_guard``
from here (DRY) so the REST endpoint and the proxy share one implementation.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns that mark execution-critical tokens which MUST survive compression.
# ---------------------------------------------------------------------------
_IDENTIFIER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # File paths with optional line/col: src/api/main.py:98 or /app/main.py:12:3
    ("file_path_loc", re.compile(r"(?:[\w./\-]+/)?[\w\-]+\.\w{1,8}(?::\d+(?::\d+)?)")),
    # Stack frames: at FunctionName (file.ts:12)  /  at file.ts:12
    ("stack_frame", re.compile(r"\bat\s+\S+\s*\(\S+:\d+(?::\d+)?\)")),
    # Error codes: ECONNREFUSED, TS2724, E501, ERR_*, etc.
    ("error_code", re.compile(r"\b[A-Z][A-Z_0-9]{2,}(?:Error|Exception|Warning)?\b")),
    # HTTP status codes in context: 404, 500, 422, etc.
    ("http_status", re.compile(r"\b[2345]\d{2}\b")),
    # URLs
    ("url", re.compile(r"https?://\S+")),
    # UUIDs
    ("uuid", re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")),
    # Env var names: UPPERCASE_WITH_UNDERSCORES
    ("env_var", re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")),
    # Python/TS/Go identifiers in error contexts (camelCase or snake_case symbols)
    ("symbol", re.compile(r"\b(?:[a-z][a-zA-Z0-9_]*|[A-Z][a-zA-Z0-9_]*)\b")),
]

# Minimum token length to preserve: we don't force-keep single chars.
_MIN_PRESERVE_TOKEN_LEN = 3

# Maximum identifiers to re-inject (safety bound to avoid turning a 90% saving
# into a 0% saving on identifier-dense output).
_MAX_REINJECT = 200


def extract_critical_identifiers(text: str) -> list[str]:
    """Extract all execution-critical identifier tokens from *text*.

    Returns a deduplicated list of tokens in first-seen order. These tokens
    MUST survive compression.
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for _pname, pat in _IDENTIFIER_PATTERNS:
        for m in pat.finditer(text):
            tok = m.group(0).strip()
            if len(tok) >= _MIN_PRESERVE_TOKEN_LEN and tok not in seen:
                seen.add(tok)
                tokens.append(tok)
    return tokens


def apply_identifier_guard(
    compressed: str,
    critical_tokens: list[str],
) -> tuple[str, list[str]]:
    """Ensure all *critical_tokens* appear in *compressed*.

    For each missing token, appends a compact preservation footer.
    Returns ``(final_text, list_of_reinjected_tokens)``.
    """
    missing = [t for t in critical_tokens if t not in compressed]
    if not missing:
        return compressed, []

    # Limit reinject batch to avoid turning a 90% saving into a 0% saving.
    to_reinject = missing[:_MAX_REINJECT]
    footer = "\n[preserved identifiers: " + ", ".join(to_reinject) + "]"
    return compressed + footer, to_reinject
