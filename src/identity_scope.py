"""Helpers for tenant-aware document identity and filtering."""

from __future__ import annotations

from typing import Dict, Optional
from urllib.parse import quote, unquote

_SCOPE_PREFIX = "scope__"
_FIELD_TOKENS = (
    ("workspace_id", "w"),
    ("user_id", "u"),
    ("agent_id", "a"),
    ("session_id", "s"),
)


def _encode(value: str) -> str:
    # quote() intentionally leaves "_" unescaped, but our "__" delimiter uses
    # underscores, so encode them explicitly to keep parsing unambiguous.
    return quote(value, safe="").replace("_", "%5F")


def _decode(value: str) -> str:
    return unquote(value)


def _scope_values(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    return {
        "workspace_id": workspace_id,
        "user_id": user_id,
        "agent_id": agent_id,
        "session_id": session_id,
    }


def compose_scoped_file_id(
    file_id: str,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    """Compose a canonical internal file id for a tenant-scoped document."""
    if not any((workspace_id, user_id, agent_id, session_id)):
        return file_id

    scope = _scope_values(workspace_id, user_id, agent_id, session_id)
    parts = [_SCOPE_PREFIX.rstrip("_")]
    for field, token in _FIELD_TOKENS:
        value = scope[field]
        if value is not None:
            parts.append(f"{token}={_encode(value)}")
    parts.append(f"f={_encode(file_id)}")
    return "__".join(parts)


def parse_scoped_file_id(scoped_or_raw_file_id: str) -> Dict[str, Optional[str]]:
    """Parse a scoped file id into visible file id plus optional scope fields."""
    if not scoped_or_raw_file_id.startswith(_SCOPE_PREFIX):
        return {
            "file_id": scoped_or_raw_file_id,
            "workspace_id": None,
            "user_id": None,
            "agent_id": None,
            "session_id": None,
        }

    parsed: Dict[str, Optional[str]] = {
        "file_id": None,
        "workspace_id": None,
        "user_id": None,
        "agent_id": None,
        "session_id": None,
    }
    token_to_field = {token: field for field, token in _FIELD_TOKENS}
    for part in scoped_or_raw_file_id.split("__")[1:]:
        if "=" not in part:
            continue
        token, raw_value = part.split("=", 1)
        if token == "f":
            parsed["file_id"] = _decode(raw_value)
        elif token in token_to_field:
            parsed[token_to_field[token]] = _decode(raw_value)

    parsed["file_id"] = parsed["file_id"] or scoped_or_raw_file_id
    return parsed


def display_file_id(scoped_or_raw_file_id: str) -> str:
    """Return the external file id for a stored internal identity."""
    return str(parse_scoped_file_id(scoped_or_raw_file_id)["file_id"])


def has_scope(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    return any((workspace_id, user_id, agent_id, session_id))


def scope_matches(
    scoped_or_raw_file_id: str,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> bool:
    """
    Return True when a document belongs to the requested scope.

    With no requested scope, only unscoped documents match. This preserves
    backward-compatible single-tenant behavior while preventing scoped data
    from leaking into legacy calls.
    """
    parsed = parse_scoped_file_id(scoped_or_raw_file_id)
    requested = _scope_values(workspace_id, user_id, agent_id, session_id)

    if not has_scope(**requested):
        return not has_scope(
            parsed["workspace_id"],
            parsed["user_id"],
            parsed["agent_id"],
            parsed["session_id"],
        )

    for field, requested_value in requested.items():
        if requested_value is not None and parsed[field] != requested_value:
            return False
    return True
