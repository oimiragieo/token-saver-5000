"""Optional tensor-grep integration for AST-aware code compression and search.

All public functions gracefully return fallback values (with ``available=False``)
when the ``tg`` CLI is not installed, so callers never need to guard against
ImportError or FileNotFoundError.

tensor-grep is our own sibling project (``C:/dev/projects/tensor-grep``).
The integration layer here wraps the four CLI surfaces that gotcontext uses:

  * ``tg map``              → :func:`get_repo_map`
  * ``tg <pattern>``        → :func:`code_search`   (streaming NDJSON to avoid
                              the pipe-buffer deadlock that ``--json`` causes on
                              large repos — see B2 in the 2026-04-19 stress-test
                              report; fixed here via ``--ndjson``)
  * ``tg run``              → :func:`ast_search`
  * ``tg context-render``   → :func:`get_context_render`   (added v1.8.0)
  * ``tg scan``             → :func:`scan_ruleset`          (added v1.8.0)

Minimum tensor-grep version: **>=1.8.0**.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TG_ARG_MAX_LEN = 512
_MAX_STREAM_BYTES = 10 * 1024 * 1024  # 10MB limit on subprocess output stream
_MAX_STREAM_MATCHES = 5000  # Cap maximum match objects accumulated


class TgInputError(ValueError):
    """Caller-side validation failure for tensor-grep arguments."""


def sanitize_tg_positional(value: str | None, *, field_name: str = "argument") -> str:
    """Validate a user-controlled positional before it reaches the tg argv.

    Prevents argument-injection (CWE-88) where leading dashes can be parsed as CLI flags.
    """
    if value is None:
        raise TgInputError(f"{field_name}: required")
    v = value.strip()
    if not v:
        raise TgInputError(f"{field_name}: must not be empty")
    if len(v) > _TG_ARG_MAX_LEN:
        raise TgInputError(f"{field_name}: too long ({len(v)} > {_TG_ARG_MAX_LEN} chars)")
    if "\x00" in v:
        raise TgInputError(f"{field_name}: null byte not allowed")
    if any(ord(ch) < 0x20 for ch in v):
        raise TgInputError(f"{field_name}: control characters not allowed")
    if v.startswith("-"):
        raise TgInputError(
            f"{field_name}: must not start with '-' — rejected to prevent argument injection"
        )
    return v


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
    overflow: bool = False


@dataclass
class ASTSearchResult:
    """Result from ``ast_search()``."""

    pattern: str = ""
    matches: list[dict[str, Any]] = field(default_factory=list)
    total_matches: int = 0
    available: bool = True
    overflow: bool = False


@dataclass
class ContextRenderResult:
    """Result from ``get_context_render()``.

    ``ranked_files`` is the list of ``{path, score, snippets}`` dicts returned
    by ``tg context-render --json``. ``render`` is the prompt-ready text bundle
    (populated when the render profile includes a text body). ``raw_json`` holds
    the unprocessed response for callers that need it.
    """

    query: str = ""
    ranked_files: list[dict[str, Any]] = field(default_factory=list)
    render: str = ""
    raw_json: dict = field(default_factory=dict)
    available: bool = True


@dataclass
class ScanFinding:
    """Single finding from ``tg scan``."""

    rule_id: str = ""
    severity: str = ""
    path: str = ""
    line: int = 0
    message: str = ""
    fingerprint: str = ""
    evidence: str = ""


@dataclass
class ScanResult:
    """Result from ``scan_ruleset()``."""

    path: str = ""
    ruleset: str = ""
    findings: list[ScanFinding] = field(default_factory=list)
    total_findings: int = 0
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
    max_bytes: int = _MAX_STREAM_BYTES,
    max_matches: int = _MAX_STREAM_MATCHES,
) -> CodeSearchResult:
    """
    Search for *pattern* under *directory* using tensor-grep's streaming NDJSON
    output (``--ndjson``), accumulating match objects up to bounded memory limits.

    Returns a ``CodeSearchResult`` with ``available=False`` if tensor-grep is
    not installed, or a result with empty matches on any subprocess/parse failure.
    """
    if not is_available():
        return CodeSearchResult(pattern=pattern, available=False)

    try:
        sanitized_pattern = sanitize_tg_positional(pattern, field_name="pattern")
        sanitized_dir = sanitize_tg_positional(str(directory), field_name="directory")

        terms = sanitized_pattern.split()
        if len(terms) > 1:
            tg_pattern = "|".join(re.escape(t) for t in terms)
        else:
            tg_pattern = sanitized_pattern

        cmd = ["tg", tg_pattern, sanitized_dir, "--ndjson"]
        if use_index:
            cmd.append("--index")

        matches: list[dict[str, Any]] = []
        total_bytes_read = 0
        overflow = False

        # Stream NDJSON line-by-line so the pipe never fills.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            assert proc.stdout is not None  # always true when stdout=PIPE
            for line in proc.stdout:
                line_bytes = len(line.encode("utf-8", errors="replace"))
                total_bytes_read += line_bytes
                if total_bytes_read > max_bytes or len(matches) >= max_matches:
                    overflow = True
                    proc.kill()
                    break

                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                except json.JSONDecodeError:
                    continue

                if isinstance(obj, dict):
                    matches.append(obj)
                elif isinstance(obj, list):
                    matches.extend(obj)

            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return CodeSearchResult(pattern=pattern, available=True)
        finally:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        if proc.returncode != 0 and not overflow:
            return CodeSearchResult(pattern=pattern, available=True)

        return CodeSearchResult(
            pattern=pattern,
            matches=matches,
            total_matches=len(matches),
            available=True,
            overflow=overflow,
        )
    except (OSError, TgInputError):
        return CodeSearchResult(pattern=pattern, available=True)


def ast_search(
    pattern: str,
    directory: str | Path,
    lang: str | None = None,
    timeout: float = 30.0,
    max_bytes: int = _MAX_STREAM_BYTES,
) -> ASTSearchResult:
    """
    Invoke ``tg run <pattern> <directory> --json [--lang <lang>]`` and parse matches.

    Returns an ``ASTSearchResult`` with ``available=False`` if tensor-grep is
    not installed, or a result with empty matches on any subprocess/parse failure.
    """
    if not is_available():
        return ASTSearchResult(pattern=pattern, available=False)

    try:
        sanitized_pattern = sanitize_tg_positional(pattern, field_name="pattern")
        sanitized_dir = sanitize_tg_positional(str(directory), field_name="directory")

        cmd = ["tg", "run", sanitized_pattern, sanitized_dir, "--json"]
        if lang:
            sanitized_lang = sanitize_tg_positional(lang, field_name="lang")
            cmd.extend(["--lang", sanitized_lang])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if len(result.stdout.encode("utf-8", errors="replace")) > max_bytes:
            return ASTSearchResult(pattern=pattern, available=True, overflow=True)

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
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError, TgInputError):
        return ASTSearchResult(pattern=pattern, available=True)


def get_context_render(
    directory: str | Path,
    query: str,
    *,
    max_files: int = 20,
    max_sources: int | None = None,
    max_symbols_per_file: int | None = None,
    max_render_chars: int | None = None,
    max_tokens: int | None = None,
    render_profile: str = "llm",
    optimize_context: bool = True,
    timeout: float = 60.0,
) -> ContextRenderResult:
    """
    Invoke ``tg context-render --json --query <query> <directory>`` and parse the
    ranked-file bundle into a :class:`ContextRenderResult`.

    This surface is the preferred way to produce prompt-ready repo context for
    AI features. The ranking fuses BM25 signal with the blast-radius graph
    distance so the most relevant files appear first.

    Parameters
    ----------
    directory:
        Root of the repository (or sub-tree) to analyse.
    query:
        Free-form search query driving the ranking.
    max_files:
        Maximum number of files to include in the ranked context.
    max_sources:
        Cap on the number of source snippets per file (passed as
        ``--max-sources``). ``None`` lets tg use its default.
    max_symbols_per_file:
        Cap on symbols rendered per file (``--max-symbols-per-file``).
    max_render_chars:
        Hard character cap on the rendered text body.
    max_tokens:
        Token budget passed to tg's built-in token estimator.
    render_profile:
        One of ``full``, ``compact``, or ``llm`` (default: ``"llm"``).
    optimize_context:
        Whether tg should run its context-optimization pass. Defaults
        to ``True``; set ``False`` for deterministic output.
    timeout:
        Subprocess wall-clock budget in seconds.

    Returns
    -------
    ContextRenderResult
        ``available=False`` when tg is not installed. On subprocess error or
        parse failure returns an empty result with ``available=True`` (degraded).
    """
    if not is_available():
        return ContextRenderResult(query=query, available=False)

    cmd = [
        "tg",
        "context-render",
        "--json",
        "--query",
        query,
        "--max-files",
        str(max_files),
        "--render-profile",
        render_profile,
        str(directory),
    ]
    if max_sources is not None:
        cmd.extend(["--max-sources", str(max_sources)])
    if max_symbols_per_file is not None:
        cmd.extend(["--max-symbols-per-file", str(max_symbols_per_file)])
    if max_render_chars is not None:
        cmd.extend(["--max-render-chars", str(max_render_chars)])
    if max_tokens is not None:
        cmd.extend(["--max-tokens", str(max_tokens)])
    if not optimize_context:
        cmd.append("--no-optimize-context")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            return ContextRenderResult(query=query, available=True)

        data = json.loads(result.stdout)
        return ContextRenderResult(
            query=query,
            ranked_files=data.get("ranked_files", []),
            render=data.get("render", ""),
            raw_json=data,
            available=True,
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return ContextRenderResult(query=query, available=True)


def scan_ruleset(
    path: str | Path,
    ruleset: str,
    *,
    language: str | None = None,
    include_evidence: bool = False,
    baseline: str | Path | None = None,
    timeout: float = 60.0,
) -> ScanResult:
    """
    Invoke ``tg scan --json --ruleset <ruleset> <path>`` and parse findings.

    Parameters
    ----------
    path:
        File or directory to scan.
    ruleset:
        Built-in ruleset name (e.g. ``"secrets"``, ``"security"``) or path to a
        custom ``.toml`` ruleset file.
    language:
        Restrict scanning to a single language (``--language``).
    include_evidence:
        Emit the surrounding code snippet for each finding
        (``--include-evidence-snippets``).
    baseline:
        Path to a previously written baseline file; findings that appear in
        the baseline are suppressed from the result.
    timeout:
        Subprocess wall-clock budget in seconds.

    Returns
    -------
    ScanResult
        ``available=False`` when tg is not installed. On subprocess error or
        parse failure returns an empty result with ``available=True`` (degraded).
    """
    if not is_available():
        return ScanResult(path=str(path), ruleset=ruleset, available=False)

    cmd = [
        "tg",
        "scan",
        "--json",
        "--ruleset",
        ruleset,
        "--path",
        str(path),
    ]
    if language:
        cmd.extend(["--language", language])
    if include_evidence:
        cmd.append("--include-evidence-snippets")
    if baseline is not None:
        cmd.extend(["--baseline", str(baseline)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        # tg scan exits 1 when findings are present (non-zero = "found issues").
        # We treat rc=1 as a successful scan with findings, not a subprocess error.
        if result.returncode not in (0, 1):
            return ScanResult(path=str(path), ruleset=ruleset, available=True)

        data = json.loads(result.stdout)
        raw_findings = data.get("findings", [])
        findings = [
            ScanFinding(
                rule_id=f.get("rule_id", ""),
                severity=f.get("severity", ""),
                path=f.get("path", ""),
                line=int(f.get("line", 0)),
                message=f.get("message", ""),
                fingerprint=f.get("fingerprint", ""),
                evidence=f.get("evidence", ""),
            )
            for f in raw_findings
        ]
        return ScanResult(
            path=str(path),
            ruleset=ruleset,
            findings=findings,
            total_findings=data.get("total_findings", len(findings)),
            available=True,
        )
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return ScanResult(path=str(path), ruleset=ruleset, available=True)
