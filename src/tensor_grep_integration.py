"""Optional tensor-grep integration for AST-aware code compression and search.

All public functions gracefully return fallback values (with ``available=False``)
when the ``tg`` CLI is not installed, so callers never need to guard against
ImportError or FileNotFoundError.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RepoMapResult:
    """Result from ``get_repo_map()``."""

    path: str = ""
    files: list[str] = field(default_factory=list)
    symbols: list[dict[str, Any]] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)
    available: bool = True


@dataclass
class CodeSearchResult:
    """Result from ``code_search()``."""

    pattern: str = ""
    matches: list[dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    available: bool = True


@dataclass
class ASTSearchResult:
    """Result from ``ast_search()``."""

    pattern: str = ""
    matches: list[dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    available: bool = True


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def is_available() -> bool:
    """Return True if the ``tg`` binary is present on PATH."""
    return shutil.which("tg") is not None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_repo_map(directory: str | Path, timeout: float = 30.0) -> RepoMapResult:
    """
    Invoke ``tg map --json <directory>`` and parse the result.

    Returns a ``RepoMapResult`` with ``available=False`` if tensor-grep is not
    installed, or a result with empty lists on any subprocess/parse failure.
    """
    if not is_available():
        return RepoMapResult(path=str(directory), available=False)

    try:
        result = subprocess.run(
            ["tg", "map", "--json", str(directory)],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return RepoMapResult(path=str(directory), available=True)

        data = json.loads(result.stdout)
        return RepoMapResult(
            path=str(directory),
            files=data.get("files", []),
            symbols=data.get("symbols", []),
            raw_json=data,
            available=True,
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return RepoMapResult(path=str(directory), available=True)


def code_search(
    pattern: str,
    directory: str | Path,
    use_index: bool = True,
    timeout: float = 30.0,
) -> CodeSearchResult:
    """
    Invoke ``tg <pattern> <directory> --json [--index]`` and parse matches.

    Returns a ``CodeSearchResult`` with ``available=False`` if tensor-grep is
    not installed, or a result with empty matches on any subprocess/parse failure.
    """
    if not is_available():
        return CodeSearchResult(pattern=pattern, available=False)

    try:
        cmd = ["tg", pattern, str(directory), "--json"]
        if use_index:
            cmd.append("--index")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return CodeSearchResult(pattern=pattern, available=True)

        data = json.loads(result.stdout)
        matches = data.get("matches", [])
        return CodeSearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=data.get("total_matches", len(matches)),
            available=True,
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return CodeSearchResult(pattern=pattern, available=True)


def ast_search(
    pattern: str,
    directory: str | Path,
    lang: str | None = None,
    timeout: float = 30.0,
) -> ASTSearchResult:
    """
    Invoke ``tg run <pattern> <directory> --json [--lang <lang>]`` and parse matches.

    Returns an ``ASTSearchResult`` with ``available=False`` if tensor-grep is
    not installed, or a result with empty matches on any subprocess/parse failure.
    """
    if not is_available():
        return ASTSearchResult(pattern=pattern, available=False)

    try:
        cmd = ["tg", "run", pattern, str(directory), "--json"]
        if lang:
            cmd.extend(["--lang", lang])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return ASTSearchResult(pattern=pattern, available=True)

        data = json.loads(result.stdout)
        matches = data.get("matches", [])
        return ASTSearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=data.get("total_matches", len(matches)),
            available=True,
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return ASTSearchResult(pattern=pattern, available=True)
