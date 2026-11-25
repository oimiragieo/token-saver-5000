"""Shared allowlist validation for caller-supplied storage identifiers.

Used to reject ``file_id``/``session_id``-style values that could escape an
intended storage directory when joined onto a base ``Path`` (CWE-22). A
denylist alone (rejecting only ``".."``) is exactly how this class of bug
happened: an absolute path or a Windows drive-letter/UNC path bypasses a
``".."``-only check while still escaping the base directory, because
``Path.__truediv__`` silently REPLACES the whole path when the right-hand
operand is itself absolute -- it does not raise, and it does not confine
the result to the left-hand base.

Prefer a strict allowlist charset over trying to enumerate every unsafe
shape; a denylist has to be complete on every call, an allowlist only has
to be complete once.
"""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath

# file_id-style identifiers may legitimately contain forward-slash path
# segments (e.g. "docs/audits/foo/bar.md" -- see persistence.py's
# _atomic_write_json docstring, which documents this as an intentional
# customer-facing pattern). session_id/workspace_id/user_id/agent_id style
# identifiers never need directory segments.
_SAFE_ID_WITH_SUBDIRS_RE = re.compile(r"^[a-zA-Z0-9_\-./]+$")
_SAFE_ID_NO_SUBDIRS_RE = re.compile(r"^[a-zA-Z0-9_\-.]+$")


def validate_safe_storage_id(
    value: str, field_name: str = "id", *, allow_subdirs: bool = False
) -> str | None:
    """Return an error message if *value* is unsafe to join onto a base
    storage ``Path``, else ``None``.

    Checks (in order): non-allowlisted characters, ``".."`` parent-directory
    references, backslashes, and any value that resolves to an absolute
    path on POSIX or Windows (drive letter, UNC, or POSIX-rooted). The
    absolute-path check is explicit and platform-dual rather than relying
    on a single ``Path(value).is_absolute()`` call, because
    ``PureWindowsPath`` and the POSIX ``Path`` disagree on values like
    ``"/etc/passwd"`` (no drive letter) depending on which flavor evaluates
    them, and this validator must refuse it regardless of which OS it runs
    on.
    """
    if not value:
        return None

    charset_re = _SAFE_ID_WITH_SUBDIRS_RE if allow_subdirs else _SAFE_ID_NO_SUBDIRS_RE
    if not charset_re.match(value):
        allowed = (
            "alphanumeric characters, underscores, hyphens, dots, or forward slashes"
            if allow_subdirs
            else "alphanumeric characters, underscores, hyphens, or dots"
        )
        return f"{field_name} must contain only {allowed}"

    if ".." in value:
        return f"{field_name} must not contain '..' (parent-directory reference)"

    if "\\" in value:
        return f"{field_name} must not contain backslashes"

    if value.startswith("/"):
        return f"{field_name} must not be an absolute path"

    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return f"{field_name} must not be an absolute path"

    return None
