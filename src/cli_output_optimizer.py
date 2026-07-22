"""
CLIOutputOptimizer — RTK-inspired CLI output filtering for Token Saver 5000.

Auto-detects command type from raw CLI output and applies the appropriate
filtering strategy to reduce token usage before the output enters the MCP
context window.

Supported strategies:
    git_diff        → summary stats (files, insertions, deletions)
    git_status      → grouped by modification type
    test_output     → failures only + summary line
    install_output  → drops progress, keeps summary + warnings
    lint_output     → groups by rule code with occurrence counts
    json_output     → top-level key/type map with array previews
    ansi_output     → strips ANSI escape codes and carriage returns
    progress_output → removes %, spinner, and ellipsis lines
    tree_output     → collapses common path prefixes
    log_output      → deduplicates consecutive identical lines
    generic         → conservative fallback for unclassified verbose output
                      (CR-resolve, progress-strip, consecutive exact/masked
                      dedup, stack-frame elision, blank-run + timestamp-only
                      line collapse); only applied when detection lands
                      "unknown" AND the caller opts in via
                      ``filter(..., fallback_strategy="generic")``
    unknown         → passthrough (no change) unless ``fallback_strategy`` opts in

RTK fallback: if ``shutil.which("rtk")`` finds a binary and the text exceeds
500 characters, the optimizer tries ``rtk --json`` via subprocess first.
If RTK fails or is absent, the pure-Python implementation is used.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class FilterResult:
    """Result of a single :meth:`CLIOutputOptimizer.filter` call.

    Attributes:
        original_text:   Unmodified input text.
        filtered_text:   Text after strategy application.
        original_lines:  Line count of *original_text*.
        filtered_lines:  Line count of *filtered_text*.
        command_detected: Detected (or hinted) command type string.
        strategy_applied: Name of the strategy method that was applied,
                          or ``"passthrough"`` when no strategy was needed.
        compression_pct: ``(1 - filtered_lines / original_lines) * 100``.
                         0.0 when *original_lines* == 0.
    """

    original_text: str
    filtered_text: str
    original_lines: int
    filtered_lines: int
    command_detected: str
    strategy_applied: str
    compression_pct: float = field(init=False)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.original_lines > 0:
            self.compression_pct = (1.0 - self.filtered_lines / self.original_lines) * 100.0
        else:
            self.compression_pct = 0.0


# ---------------------------------------------------------------------------
# Minimum text length before any strategy is attempted
# ---------------------------------------------------------------------------
_MIN_FILTER_CHARS = 50

# RTK: try the external binary when text is longer than this.
_RTK_MIN_CHARS = 500

# Strategy dispatch map: command-type → strategy method name
STRATEGY_MAP: dict[str, str] = {
    "git_diff": "_git_diff_stats",
    "git_status": "_git_status_compact",
    "test_output": "_test_failure_focus",
    "install_output": "_install_summary",
    "lint_output": "_lint_group",
    "json_output": "_json_structure",
    "ansi_output": "_strip_ansi",
    "progress_output": "_progress_strip",
    "tree_output": "_tree_compress",
    "log_output": "_log_dedup",
    "docker_output": "_docker_compact",
    "generic": "_generic_conservative",
}

# ANSI escape-code regex (includes all terminating letters)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Spinner characters used by many CLI progress bars
_SPINNER_CHARS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

# ---------------------------------------------------------------------------
# #137 — generic conservative fallback (unclassified "unknown" output).
#
# Every rule below is CONSECUTIVE-RUN-SCOPED and always annotates what it
# removed. This is deliberately more cautious than the type-specific
# strategies above: no global/non-adjacent dedup (that eats distinct data
# rows — measured 99.1% loss on unique prose during the design audit), no
# silent timestamp-prefix stripping on message-bearing lines, no frame
# elision on short (<=8) blocks, no blank-line collapse to zero.
# ---------------------------------------------------------------------------

# Only lines with this many stripped chars (and at least one alnum char) are
# eligible for dedup/collapse — protects short structural syntax lines (a
# JSON "}," or a YAML "- name:") from being treated as repeated noise even
# when they repeat exactly or near-identically.
_MIN_COLLAPSIBLE_CONTENT_CHARS = 8

# Masked near-duplicate runs need at least this many CONSECUTIVE lines
# sharing a mask before they are eligible for first+last+elision collapse.
_MASKED_RUN_MIN_LEN = 5

# Stack-frame block elision never fires on a block this size or smaller.
_STACK_FRAME_BLOCK_MIN_LEN = 8
_STACK_FRAME_KEEP_TOP = 5
_STACK_FRAME_KEEP_BOTTOM = 2

# Timestamp-shape regex reused by both the near-dup masker and the
# timestamp-only-line dropper: ISO-8601-ish ("2026-07-21T10:00:00.123Z" /
# "...+02:00") or a bare "HH:MM:SS(.ms)" clock reading.
_TIMESTAMP_SHAPE_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
)
# Full UUIDs — a distinctive NON-pure-digit shape (8-4-4-4-12 hex groups
# joined by hyphens; hyphens can never appear in a plain integer/hash).
_MASK_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
# codex adversarial-gate finding 1, round 2 (2026-07-21): plain integers/
# decimals are DELIBERATELY NOT masked — see `_mask_line` docstring. The
# old `_MASK_NUMBER_RE = re.compile(r"\b\d+\b")` masked EVERY standalone
# number, which silently collapsed distinct numeric data (`processed batch
# 1..50`, dollar amounts, ports, ids) into first+last — worse than 0%
# savings.
#
# codex adversarial-gate finding 1, round 3 (2026-07-22): hex-masking was
# ALSO removed entirely (the prior `_MASK_HEX_RE = re.compile(r"\b0x[0-9a-
# fA-F]+\b|\b[0-9a-f]{8,}\b")`). Digits 0-9 are ALL valid hex characters, so
# an 8+ digit run like a batch id ("batch 12345678") matched that pattern
# just as readily as a real hex hash — `batch 12345678` and `batch
# 12345679` masked IDENTICALLY and could collapse as if they were the same
# noisy line, even though they are genuinely distinct numeric data. A hash
# that VARIES per line is exactly this kind of distinct data and must
# never collapse; a hash that REPEATS IDENTICALLY is already handled by
# exact-dup collapse (no masking needed there at all). So hex-masking only
# added collision risk with zero compensating value — dropped. Only
# TIMESTAMP-shaped and UUID-shaped tokens are masked now: both have
# distinctive NON-pure-digit shapes (a timestamp carries "-"/":"/"T"/"Z";
# a UUID is 8-4-4-4-12 hex groups joined by hyphens) that a plain integer
# can never coincidentally match.

# "Distinct identifier" detectors used ONLY to EXEMPT a masked near-dup run
# from collapse — a run containing one of these looks like distinct DATA,
# not repeated noisy log lines, even if the surrounding text lines up.
_DISTINCT_URL_RE = re.compile(r"https?://\S+")
_DISTINCT_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
# Mirrors identifier_preservation.py's ReDoS-hardened file_path_loc shape
# (bounded {1,20}/{1,8} quantifiers — no unbounded nested repetition).
_DISTINCT_FILE_PATH_RE = re.compile(r"(?:[\w.\-]+/){1,20}[\w\-]+\.\w{1,8}(?::\d+(?::\d+)?)?")
# ALL-CAPS runs of 3+ chars (letters/digits/underscore after the first
# letter) — matches genuine error codes (ECONNREFUSED, TS2724, E501) as well
# as common log-level tags (WARN, ERROR, INFO); the latter are excluded via
# _LOG_LEVEL_WORDS below so a repetitive log storm's level tag never forces
# a false exemption.
_DISTINCT_ERROR_CODE_RE = re.compile(r"\b[A-Z][A-Z_0-9]{2,}(?:Error|Exception|Warning)?\b")
_LOG_LEVEL_WORDS = frozenset(
    {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL", "TRACE", "FATAL", "NOTICE"}
)

# Stack-frame header shapes: Python's "File "x.py", line N, in func" and the
# JS/Java/Node "at name (file:line:col)" / "at file:line:col" single-line form.
_PY_TRACEBACK_FRAME_RE = re.compile(r'^\s*File\s+"[^"]+",\s+line\s+\d+')
_JS_JAVA_FRAME_RE = re.compile(r"^\s*at\s+\S.*:\d+(?::\d+)?\)?\s*$")
# These always start a NEW block — never merged with frames on either side.
_BLOCK_BOUNDARY_RE = re.compile(r"^\s*(?:Caused by:|During handling of the above exception)")

# A full line that is ENTIRELY a timestamp + log level with no message —
# never matches a message-bearing line (the trailing "$" requires nothing
# else on the line), so a timestamp PREFIX on a real message is never
# stripped, only whole timestamp+level-only lines are dropped.
_TIMESTAMP_LEVEL_ONLY_RE = re.compile(
    r"^\s*(?:" + _TIMESTAMP_SHAPE_RE.pattern + r")"
    r"\s*[\[\(]?\s*(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|TRACE|FATAL|NOTICE)\s*[\]\)]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# codex adversarial-gate finding 2 (2026-07-21, narrowed 2026-07-22):
# structured-content bypass.
#
# The >=8-char collapsibility guard does NOT protect JSON/YAML record lines
# (a JSON object or YAML list item is easily >=8 chars). Collapsing repeated
# or near-duplicate structured lines corrupts syntax (mismatched brackets)
# AND silently deletes array/list cardinality — a real 50-element JSONL
# block would misreport as fewer records. A structured line BYPASSES
# generic dedup/collapse entirely (verbatim), regardless of length.
#
# codex adversarial-gate finding 2, round 3 (2026-07-22): the original
# `json.loads` fallback was capped at `_STRUCTURED_LINE_MAX_CHARS` (2000)
# for performance/ReDoS-adjacent caution — but that meant a long JSONL
# record or array element (a big single-line object with many fields, or
# a large embedded string) LOST its structured-content protection past
# that length. Fixed with an O(1) SHAPE check (`_looks_like_json_value_shape`
# — just the first/last character after stripping an optional trailing
# comma, never a full parse) that fires regardless of length, so a
# multi-kilobyte JSON object/array/string line is still recognized and
# protected without paying (or risking) a full `json.loads` on it.
# ---------------------------------------------------------------------------
_JSON_STRUCTURAL_ONLY_RE = re.compile(r"^[\{\}\[\],]+,?$")
_JSON_KEY_VALUE_RE = re.compile(r'^"[^"]*"\s*:\s*.+?,?$')
_YAML_LIST_ITEM_RE = re.compile(r"^-\s+\S")
# Only used to bound the (optional, short-value-only) full `json.loads`
# fallback below — NOT used to gate the shape check, which is length-free.
_STRUCTURED_LINE_MAX_CHARS = 2000


def _looks_like_json_value_shape(stripped: str) -> bool:
    """Cheap O(1) shape check (first/last character only, no full JSON
    parse) for a JSON object/array/string value spanning the WHOLE line —
    works regardless of length, so a long JSONL record or array element
    stays protected without needing (or risking) `json.loads` on it."""
    body = stripped[:-1] if stripped.endswith(",") else stripped
    if len(body) < 2:
        return False
    first, last = body[0], body[-1]
    if first == "{" and last == "}":
        return True
    if first == "[" and last == "]":
        return True
    if first == '"' and last == '"':
        return True
    return False


def _is_structured_line(line: str) -> bool:
    """True when *line* is a JSON object/array-element/structural-bracket
    line, a JSONL record, or a YAML list item — these BYPASS generic
    dedup/collapse entirely (verbatim), never treated as noise even when
    they repeat exactly or near-identically, and REGARDLESS OF LENGTH."""
    stripped = line.strip()
    if not stripped:
        return False
    if _JSON_STRUCTURAL_ONLY_RE.match(stripped):
        return True
    if _YAML_LIST_ITEM_RE.match(stripped):
        return True
    if _JSON_KEY_VALUE_RE.match(stripped):
        return True
    if _looks_like_json_value_shape(stripped):
        return True
    if len(stripped) > _STRUCTURED_LINE_MAX_CHARS:
        return False
    # Short-value fallback (bare number/true/false/null array elements,
    # or anything the shape check didn't recognize) — a full JSON value on
    # its own line. Strip one trailing comma before attempting to parse
    # (multi-line JSON/JSONL commonly trails a comma per element).
    candidate = stripped[:-1] if stripped.endswith(",") else stripped
    try:
        json.loads(candidate)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _is_collapsible_content_line(line: str) -> bool:
    """Only lines with >= `_MIN_COLLAPSIBLE_CONTENT_CHARS` stripped chars,
    at least one alnum char, AND that are NOT structured JSON/YAML content
    are eligible for dedup/collapse."""
    stripped = line.strip()
    if len(stripped) < _MIN_COLLAPSIBLE_CONTENT_CHARS:
        return False
    if _is_structured_line(line):
        return False
    return any(c.isalnum() for c in stripped)


# ---------------------------------------------------------------------------
# codex adversarial-gate finding 3 (2026-07-21, narrowed 2026-07-22):
# CR-resolution DATA LOSS.
#
# The old `_cr_resolve_lines` fired on EVERY lone "\r", so an old-Mac-style
# record (`\r` as the ONLY line terminator) or a `\r`-delimited data payload
# lost every field except the last. Round 2 narrowed this to require a
# percentage/spinner/dots SIGNAL somewhere among the "\r"-delimited chunks
# — but round 3 found that check STILL too broad: `re.search(r"\d+%", ...)`
# fires on any segment that merely CONTAINS a "%" ANYWHERE, so
# "discount 95%\rprice=10" (a single "\r", real substantive data on BOTH
# sides) got wrongly truncated to just "price=10", losing "discount 95%".
#
# Genuine overwrite shape now requires EITHER:
#   (a) MULTIPLE "\r"s (>=3 segments total) — a repeated-overwrite
#       sequence is itself strong evidence of a real progress bar, even
#       when individual frames aren't perfectly "pure" on their own (e.g.
#       "Downloading 45%\rDownloading 46%\rDone" resolves to "Done" even
#       though "Downloading 45%" carries a label prefix, not just a bare
#       percentage) — a genuine `\r`-delimited flat-data format with 3+
#       fields per line is presumed rare enough in CLI/tool-output content
#       that this is an accepted, EXPLICIT trade-off, not a silent one; OR
#   (b) for a SINGLE "\r" (exactly 2 segments), the ONE pre-final segment
#       must be ENTIRELY progress noise on its own (see
#       `_is_pure_progress_segment`) — real data that merely CONTAINS a
#       "%" or starts with a spinner-like glyph is never enough by itself.
# ---------------------------------------------------------------------------


def _is_pure_progress_segment(stripped: str) -> bool:
    """True when *stripped* segment text is ENTIRELY progress noise (bare
    percentage / spinner-only / spinner-led with only a bare-percentage or
    dots remainder / dots-only) — no other substantive tokens. Used ONLY
    to judge whether a "\\r"-delimited pre-final segment is genuine
    overwrite noise (see `_looks_like_progress_overwrite`) — kept
    independent of `_strip_progress_noise_lines` (finding 4, confirmed
    fixed) so this narrower check can never regress that one."""
    if not stripped:
        return True
    if _BARE_PERCENTAGE_RE.match(stripped):
        return True
    if _DOTS_ONLY_RE.match(stripped):
        return True
    if all(c in _SPINNER_CHARS or c.isspace() for c in stripped):
        return True
    if stripped[0] in _SPINNER_CHARS:
        remainder = stripped.lstrip("".join(_SPINNER_CHARS)).strip()
        if not remainder or _BARE_PERCENTAGE_RE.match(remainder) or _DOTS_ONLY_RE.match(remainder):
            return True
    return False


def _looks_like_progress_overwrite(raw_line: str) -> bool:
    """True when a "\\r"-containing line shows genuine terminal
    overwrite/progress-bar shape (see the finding-3-round-3 module comment
    above for the exact (a)/(b) rule) — rather than being a plain-data
    line that happens to use "\\r" as a field or record separator
    (old-Mac line endings, "\\r"-separated payload fields). Those must
    stay intact, never truncated to the last segment.
    """
    segments = raw_line.split("\r")
    if len(segments) > 2:
        return True
    return _is_pure_progress_segment(segments[0].strip())


def _cr_resolve_lines(text: str) -> list[str]:
    """Resolve carriage-return overwrites to the terminal's final rendered
    state per physical line, then split on newlines — but ONLY when the
    line actually shows progress/overwrite shape (see
    `_looks_like_progress_overwrite`).

    CRLF ("\\r\\n") is normalized to a plain newline first so it is never
    mistaken for an in-place progress overwrite. A remaining lone "\\r"
    that looks like a genuine terminal overwrite (progress bars, spinners)
    is resolved to the segment AFTER the last "\\r" (the final rendered
    content). A "\\r" with no progress/overwrite signal is left FULLY
    INTACT — it may be an old-Mac-style record separator or a
    "\\r"-delimited data payload, and truncating it would be silent data
    loss, not compression.
    """
    normalized = text.replace("\r\n", "\n")
    resolved: list[str] = []
    for raw_line in normalized.split("\n"):
        if "\r" in raw_line and _looks_like_progress_overwrite(raw_line):
            raw_line = raw_line.rsplit("\r", 1)[-1]
        resolved.append(raw_line)
    return resolved


# ---------------------------------------------------------------------------
# codex adversarial-gate finding 4 (2026-07-21): progress-noise OVER-DROP.
#
# The old rule dropped ANY line starting with a spinner glyph, even when
# real content followed ("⠋ candidate accepted" lost its whole message).
# Only drop a line that is ENTIRELY progress (spinner-only / dots-only /
# bare-percentage with NO other substantive tokens); a spinner/percentage
# followed by real content has the glyph stripped but the message kept.
# ---------------------------------------------------------------------------
_BARE_PERCENTAGE_RE = re.compile(r"^\d+%$")
_DOTS_ONLY_RE = re.compile(r"^[.·]+$")


def _strip_progress_noise_lines(lines: list[str]) -> list[str]:
    """Drop lines that are ENTIRELY progress noise (bare "N%", spinner-only,
    dots/ellipsis-only). A spinner-LED line has its leading glyph(s)
    stripped and the remainder re-evaluated: if the remainder is empty (or
    is itself a bare percentage / dots-only remnant) the whole line is
    still pure progress and is dropped; otherwise the remainder — the real
    message — is KEPT (e.g. "⠋ candidate accepted" -> "candidate accepted").
    """
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        # Bare percentage / dots-only line (nothing else on the line).
        if _BARE_PERCENTAGE_RE.match(stripped) or _DOTS_ONLY_RE.match(stripped):
            continue

        if stripped[0] in _SPINNER_CHARS:
            remainder = stripped.lstrip("".join(_SPINNER_CHARS)).strip()
            if not remainder:
                continue  # spinner-only — nothing else on the line
            if _BARE_PERCENTAGE_RE.match(remainder) or _DOTS_ONLY_RE.match(remainder):
                continue  # spinner + bare percentage/dots — still pure progress
            kept.append(remainder)  # real content after the glyph — keep it
            continue

        kept.append(line)
    return kept


def _flush_dup_run(line: str, run: int) -> list[str]:
    """Render one exact-duplicate run: collapse to "<line> (repeated N
    times)" only when the line is eligible for collapse (see
    `_is_collapsible_content_line`) — short structural lines (JSON "},",
    YAML "- name:") are kept verbatim, every occurrence, even when they
    repeat exactly."""
    if run <= 1:
        return [line]
    if _is_collapsible_content_line(line):
        return [f"{line} (repeated {run} times)"]
    return [line] * run


def _collapse_exact_dup_runs(lines: list[str]) -> list[str]:
    """Collapse CONSECUTIVE exact-duplicate line runs — mirrors
    :meth:`CLIOutputOptimizer._log_dedup`'s algorithm, scoped by the
    collapsibility guard above."""
    if not lines:
        return lines
    output: list[str] = []
    current = lines[0]
    run = 1
    for line in lines[1:]:
        if line == current:
            run += 1
            continue
        output.extend(_flush_dup_run(current, run))
        current = line
        run = 1
    output.extend(_flush_dup_run(current, run))
    return output


def _mask_line(line: str) -> str:
    """Normalize ONLY provably-volatile fields — timestamps and UUIDs — to
    shared placeholders so structurally-identical log lines that differ
    ONLY in those fields hash to the same mask.

    codex adversarial-gate finding 1, round 2 (2026-07-21): plain integers/
    decimals are DELIBERATELY left untouched. An arbitrary number is not
    provably noise — it might be a batch counter, a dollar amount, a port,
    or any other MEANINGFUL distinct value, and masking it away would
    silently collapse genuinely DISTINCT data rows to first+last (a filter
    that silently drops signal is worse than 0% savings).

    codex adversarial-gate finding 1, round 3 (2026-07-22): hex-masking is
    ALSO removed — digits are valid hex characters, so an 8+ digit run
    (a plain numeric id/counter) matched the old hex-blob pattern just as
    readily as a real hash, causing the SAME collision. Only timestamp and
    UUID shapes survive masking: both carry non-digit punctuation
    (`-`/`:`/`T`/`Z`) that a bare integer can never coincidentally produce.
    A run that still differs after this masking — by a plain number, a
    hex-looking id, or any other text — is DISTINCT DATA and is never
    grouped for collapse. Never mutates the actual output — used purely as
    a grouping signal."""
    masked = _TIMESTAMP_SHAPE_RE.sub("\x00TS\x00", line)
    masked = _MASK_UUID_RE.sub("\x00UUID\x00", masked)
    return masked


def _has_distinct_identifier(line: str) -> bool:
    """True when *line* contains a URL, UUID, file path, or a genuine
    error-code-shaped token — marking it as likely-distinct DATA rather than
    repeated noise. Common log-level tags (WARN/ERROR/...) are excluded so a
    repetitive log storm's level tag never forces a false exemption."""
    if _DISTINCT_URL_RE.search(line) or _DISTINCT_UUID_RE.search(line):
        return True
    if _DISTINCT_FILE_PATH_RE.search(line):
        return True
    for match in _DISTINCT_ERROR_CODE_RE.finditer(line):
        if match.group(0) not in _LOG_LEVEL_WORDS:
            return True
    return False


def _flush_masked_run(run_lines: list[str]) -> list[str]:
    """Render one masked-near-dup run: collapse to first+elision+last only
    when the run is long enough, every line is collapse-eligible, and no
    line carries a distinct identifier (URL/UUID/path/error-code)."""
    if len(run_lines) < _MASKED_RUN_MIN_LEN:
        return run_lines
    if not all(_is_collapsible_content_line(ln) for ln in run_lines):
        return run_lines
    if any(_has_distinct_identifier(ln) for ln in run_lines):
        return run_lines
    elided = len(run_lines) - 2
    return [run_lines[0], f"... ({elided} similar lines elided) ...", run_lines[-1]]


def _collapse_masked_near_dup_runs(lines: list[str]) -> list[str]:
    """Collapse CONSECUTIVE runs of masked near-duplicate lines (see
    `_mask_line` / `_flush_masked_run`)."""
    if not lines:
        return lines
    output: list[str] = []
    run: list[str] = [lines[0]]
    run_mask = _mask_line(lines[0])
    for line in lines[1:]:
        mask = _mask_line(line)
        if mask == run_mask:
            run.append(line)
            continue
        output.extend(_flush_masked_run(run))
        run = [line]
        run_mask = mask
    output.extend(_flush_masked_run(run))
    return output


def _classify_line_kind(line: str) -> str:
    """Classify a line for stack-frame block grouping: "boundary" (Caused
    by / During handling — always starts a new block), "header" (a Python
    File-line or JS/Java "at ..." frame), or "other"."""
    if _BLOCK_BOUNDARY_RE.match(line):
        return "boundary"
    if _PY_TRACEBACK_FRAME_RE.match(line) or _JS_JAVA_FRAME_RE.match(line):
        return "header"
    return "other"


def _group_frame_units(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group raw lines into (kind, unit_lines) tuples, where a "frame" unit
    is a header line plus at most one following non-header/non-boundary
    context line (Python's "    do_something()" source-context line under a
    "File ..." header) — so a header+context pair is elided or kept
    together as one frame."""
    units: list[tuple[str, list[str]]] = []
    i = 0
    n = len(lines)
    while i < n:
        kind = _classify_line_kind(lines[i])
        if kind == "header":
            unit_lines = [lines[i]]
            i += 1
            if i < n and _classify_line_kind(lines[i]) == "other" and lines[i].strip() != "":
                unit_lines.append(lines[i])
                i += 1
            units.append(("frame", unit_lines))
        else:
            units.append((kind, [lines[i]]))
            i += 1
    return units


def _elide_stack_frame_blocks(lines: list[str]) -> list[str]:
    """Elide the middle of long contiguous stack-frame blocks, keeping the
    top `_STACK_FRAME_KEEP_TOP` + bottom `_STACK_FRAME_KEEP_BOTTOM` frames.
    NEVER fires on a block of `_STACK_FRAME_BLOCK_MIN_LEN` or fewer frames.
    A "Caused by:" / "During handling of the above exception" line is never
    itself a frame, so it naturally starts a new block boundary — two frame
    groups separated by one of these lines are never merged."""
    if not lines:
        return lines
    units = _group_frame_units(lines)
    output: list[str] = []
    block: list[list[str]] = []

    def _flush_block() -> None:
        if len(block) > _STACK_FRAME_BLOCK_MIN_LEN:
            elided = len(block) - _STACK_FRAME_KEEP_TOP - _STACK_FRAME_KEEP_BOTTOM
            for unit_lines in block[:_STACK_FRAME_KEEP_TOP]:
                output.extend(unit_lines)
            output.append(f"... ({elided} frames elided) ...")
            for unit_lines in block[-_STACK_FRAME_KEEP_BOTTOM:]:
                output.extend(unit_lines)
        else:
            for unit_lines in block:
                output.extend(unit_lines)
        block.clear()

    for kind, unit_lines in units:
        if kind == "frame":
            block.append(unit_lines)
            continue
        if block:
            _flush_block()
        output.extend(unit_lines)
    if block:
        _flush_block()
    return output


def _collapse_blank_runs(lines: list[str]) -> list[str]:
    """Collapse runs of 3+ consecutive blank lines down to exactly 1 blank
    line — NEVER 0 (a single blank line is preserved as a paragraph break)."""
    if not lines:
        return lines
    output: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip() == "":
            blank_run += 1
            continue
        if blank_run:
            output.extend([""] * (1 if blank_run >= 3 else blank_run))
            blank_run = 0
        output.append(line)
    if blank_run:
        output.extend([""] * (1 if blank_run >= 3 else blank_run))
    return output


def _drop_timestamp_only_lines(lines: list[str]) -> list[str]:
    """Drop lines that are ENTIRELY a timestamp + log level with no message
    content. Never strips a timestamp PREFIX off a message-bearing line —
    only whole lines matching (nothing else on the line) are dropped."""
    return [ln for ln in lines if not _TIMESTAMP_LEVEL_ONLY_RE.match(ln)]


class CLIOutputOptimizer:
    """Auto-detecting CLI output optimizer.

    Usage::

        optimizer = CLIOutputOptimizer()
        result = optimizer.filter(raw_output)
        print(result.filtered_text)
        print(f"{result.compression_pct:.1f}% compression")
    """

    def __init__(self, tee_store: Optional[object] = None):
        self._tee_store = tee_store
        self._rule_engine = self._load_rule_engine()

    @staticmethod
    def _load_rule_engine() -> Optional[object]:
        """Load user-defined filter rules from .gotcontext.toml (if present)."""
        try:
            from .filter_rules import FilterRuleEngine

            engine = FilterRuleEngine()
            engine.load_rules()
            return engine if engine.rules else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        text: str,
        command_hint: Optional[str] = None,
        fallback_strategy: Optional[str] = None,
    ) -> FilterResult:
        """Filter *text* by auto-detecting command type and applying a strategy.

        Args:
            text:         Raw CLI output to filter.
            command_hint: Optional override for the detected command type.
                          Must be a key in :data:`STRATEGY_MAP` (or ``"unknown"``).
            fallback_strategy: Optional :data:`STRATEGY_MAP` key applied ONLY
                          when the resolved command type is ``"unknown"``
                          (whether from auto-detection or an unrecognized
                          hint). ``command_detected`` stays ``"unknown"``
                          (truthful — detection genuinely found no pattern
                          match); ``strategy_applied`` reflects whichever
                          fallback strategy ran. :meth:`filter` is shared by
                          the ``filter_cli_output`` MCP tool, ``/v1/filter-cli``,
                          and the ``gotcontext wrap`` proxy — leave ``None``
                          (unchanged passthrough-on-unknown default) unless
                          the caller has explicitly opted in (e.g. the
                          tool-output endpoint passing ``"generic"``).

        Returns:
            A :class:`FilterResult` with the filtered text and metadata.
        """
        if not text:
            return FilterResult(
                original_text=text,
                filtered_text=text,
                original_lines=0,
                filtered_lines=0,
                command_detected="unknown",
                strategy_applied="passthrough",
            )

        original_lines = len(text.splitlines())

        # Short texts: passthrough — unless a command_hint forces a specific strategy
        if len(text) < _MIN_FILTER_CHARS and not command_hint:
            return FilterResult(
                original_text=text,
                filtered_text=text,
                original_lines=original_lines,
                filtered_lines=original_lines,
                command_detected="unknown",
                strategy_applied="passthrough",
            )

        # --- RTK fallback (optional external binary) ---
        rtk_result = self._try_rtk(text)
        if rtk_result is not None:
            filtered_lines = len(rtk_result.splitlines())
            return FilterResult(
                original_text=text,
                filtered_text=rtk_result,
                original_lines=original_lines,
                filtered_lines=filtered_lines,
                command_detected=command_hint or "rtk",
                strategy_applied="rtk",
            )

        # --- Detect or accept hint ---
        if command_hint and command_hint in STRATEGY_MAP:
            command_type = command_hint
        else:
            command_type = self.detect_command(text)

        # --- Always strip ANSI first when codes are present ---
        working_text = text
        if _ANSI_RE.search(working_text):
            working_text = self._strip_ansi(working_text)
            # If ANSI was the *only* concern, we're done
            if command_type == "ansi_output":
                filtered_lines = len(working_text.splitlines())
                return FilterResult(
                    original_text=text,
                    filtered_text=working_text,
                    original_lines=original_lines,
                    filtered_lines=filtered_lines,
                    command_detected=command_type,
                    strategy_applied="_strip_ansi",
                )
            # Re-detect on clean text if we hadn't set a hint
            if command_hint is None:
                command_type = self.detect_command(working_text)

        # --- Dispatch to content strategy ---
        strategy_name = STRATEGY_MAP.get(command_type)
        # #137: unclassified output gets a conservative fallback strategy
        # ONLY when the caller opts in — command_detected stays "unknown"
        # (truthful; detection still found no real pattern match).
        if command_type == "unknown" and fallback_strategy and fallback_strategy in STRATEGY_MAP:
            strategy_name = STRATEGY_MAP[fallback_strategy]
        if strategy_name and strategy_name != "_strip_ansi":
            strategy_fn = getattr(self, strategy_name)
            filtered_text = strategy_fn(working_text)
            applied = strategy_name
        else:
            filtered_text = working_text
            applied = "passthrough"

        # --- Apply user-defined TOML filter rules (post-filter) ---
        if self._rule_engine and command_type:
            rule_result = self._rule_engine.apply(filtered_text, command_type)
            if rule_result is not None and rule_result != filtered_text:
                filtered_text = rule_result
                applied = f"{applied}+user_rules"

        filtered_lines = len(filtered_text.splitlines())
        result = FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            original_lines=original_lines,
            filtered_lines=filtered_lines,
            command_detected=command_type,
            strategy_applied=applied,
        )

        # Tee original if compression is significant
        if self._tee_store and result.compression_pct > 0:
            tee_id = self._tee_store.store(
                original_text=text,
                compressed_text=filtered_text,
                compression_pct=result.compression_pct,
                source="cli_optimizer",
                command_hint=command_type,
            )
            if tee_id:
                result.metadata["tee_id"] = tee_id

        return result

    def detect_command(self, text: str) -> str:
        """Detect the type of CLI output from its content.

        Returns a command-type string that maps to a strategy in
        :data:`STRATEGY_MAP`, or ``"unknown"`` when no pattern matches.
        """
        if "diff --git" in text:
            return "git_diff"
        if any(p in text for p in ["modified:", "new file:", "deleted:", "Untracked files:"]):
            return "git_status"
        # Docker output: ps, images, logs patterns
        if re.search(r"CONTAINER\s+ID|IMAGE\s+.*COMMAND|REPOSITORY\s+TAG", text):
            return "docker_output"
        if re.search(r"\b(PASSED|FAILED)\b.*\b(PASSED|FAILED)\b", text):
            return "test_output"
        # Summary line: "N passed, M failed" or "N passed" + "M failed" in a summary
        if re.search(r"\d+\s+(?:passed|failed)", text, re.IGNORECASE) and re.search(
            r"(?:passed|failed).*(?:passed|failed)", text, re.IGNORECASE
        ):
            return "test_output"
        if any(p in text for p in ["added", "packages in", "Successfully installed", "up to date"]):
            return "install_output"
        # Lint check before tree/json so that "file:line:col: CODE" lines are handled correctly.
        # Match both "file:line:col: error/warning message" and "file:line:col: E501 message" (ruff/flake8).
        if re.search(
            r"^\S+:\d+:\d+:(?:.*(?:error|warning)|[ \t]+[A-Z]\d+)",
            text,
            re.MULTILINE | re.IGNORECASE,
        ):
            return "lint_output"
        # Log dedup: check before json_output because repeated lines may start with [
        lines = text.splitlines()
        if len(lines) > 10 and len(set(lines)) < len(lines) * 0.7:
            return "log_output"
        if text.lstrip().startswith(("{", "[")):
            return "json_output"
        if "\x1b[" in text:
            return "ansi_output"
        if re.search(r"\d+%|\r|⠋|⠙|⠹", text):
            return "progress_output"
        if any(c in text for c in ["├──", "└──"]) or (
            text.count("/") > 3 and text.count("/") > text.count("\n") * 0.8
        ):
            return "tree_output"
        return "unknown"

    # ------------------------------------------------------------------
    # Strategy implementations (private)
    # ------------------------------------------------------------------

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes and carriage returns from *text*."""
        text = _ANSI_RE.sub("", text)
        text = text.replace("\r", "")
        return text

    def _git_diff_stats(self, text: str) -> str:
        """Summarize a ``git diff`` into per-file change previews + summary."""
        files: list[dict] = []
        current_file: dict | None = None
        insertions = 0
        deletions = 0

        for line in text.splitlines():
            if line.startswith("diff --git "):
                parts = line.split(" b/", 1)
                current_file = {
                    "name": parts[1].strip() if len(parts) == 2 else "?",
                    "adds": 0,
                    "dels": 0,
                    "lines": [],
                }
                files.append(current_file)
            elif line.startswith("rename from ") and current_file:
                current_file["rename_from"] = line[len("rename from ") :]
            elif line.startswith("+") and not line.startswith("+++"):
                insertions += 1
                if current_file:
                    current_file["adds"] += 1
                    if len(current_file["lines"]) < 3:
                        current_file["lines"].append(line)
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
                if current_file:
                    current_file["dels"] += 1
                    if len(current_file["lines"]) < 3:
                        current_file["lines"].append(line)

        file_count = len(files)
        noun = "file" if file_count == 1 else "files"
        output_parts: list[str] = []

        for f in files:
            rename = f" (renamed from {f['rename_from']})" if "rename_from" in f else ""
            header = f"[{f['name']}] (+{f['adds']} -{f['dels']}){rename}"
            output_parts.append(header)
            for ln in f["lines"]:
                output_parts.append(f"  {ln}")
            extra = (f["adds"] + f["dels"]) - len(f["lines"])
            if extra > 0:
                output_parts.append(f"  ... +{extra} more changes")

        output_parts.append(
            f"{file_count} {noun} changed, +{insertions} insertions, -{deletions} deletions"
        )
        return "\n".join(output_parts)

    def _git_status_compact(self, text: str) -> str:
        """Group ``git status`` output by modification type."""
        groups: dict[str, list[str]] = {
            "Modified": [],
            "New": [],
            "Deleted": [],
            "Renamed": [],
        }

        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("modified:"):
                path = stripped[len("modified:") :].strip()
                groups["Modified"].append(path)
            elif stripped.startswith("new file:"):
                path = stripped[len("new file:") :].strip()
                groups["New"].append(path)
            elif stripped.startswith("deleted:"):
                path = stripped[len("deleted:") :].strip()
                groups["Deleted"].append(path)
            elif stripped.startswith("renamed:"):
                path = stripped[len("renamed:") :].strip()
                groups["Renamed"].append(path)

        output_lines: list[str] = []
        for label, paths in groups.items():
            if paths:
                names = ", ".join(paths)
                output_lines.append(f"{label} ({len(paths)}): {names}")
        return "\n".join(output_lines) if output_lines else text

    def _test_failure_focus(self, text: str) -> str:
        """Extract structured failure info and summary from test output.

        For pytest: parses the ``=== FAILURES ===`` section, extracts test names
        from ``___`` delimiters, and keeps assertion lines (``E`` / ``>``).
        Caps at 5 failures with a ``+N more`` note.
        """
        lines = text.splitlines()

        # --- Try structured pytest failure extraction ---
        in_failures = False
        failures: list[dict] = []
        current_failure: dict | None = None
        summary_line: Optional[str] = None

        for line in lines:
            stripped = line.strip()
            # Detect failures section header
            if re.match(r"^=+\s*FAILURES\s*=+$", stripped):
                in_failures = True
                continue
            # Detect end of failures
            if in_failures and re.match(r"^=+\s*(?:short test summary|warnings)", stripped):
                in_failures = False
                continue
            # Summary line
            if re.search(r"={5,}.*(?:passed|failed)", stripped, re.IGNORECASE):
                summary_line = line
                continue
            if re.search(r"\d+\s+(?:passed|failed)", stripped, re.IGNORECASE):
                summary_line = line
                continue

            if in_failures:
                # Test name delimiter: ___ test_name ___
                m = re.match(r"^_+\s+(.+?)\s+_+$", stripped)
                if m:
                    current_failure = {"name": m.group(1), "assertions": []}
                    failures.append(current_failure)
                    continue
                # Assertion lines (> or E prefix)
                if current_failure is not None:
                    if stripped.startswith("E ") or stripped.startswith("> "):
                        if len(current_failure["assertions"]) < 5:
                            current_failure["assertions"].append(stripped)

        if failures:
            result_parts: list[str] = []
            shown = failures[:5]
            for f in shown:
                result_parts.append(f"FAILED {f['name']}")
                for a in f["assertions"]:
                    result_parts.append(f"  {a}")
            if len(failures) > 5:
                result_parts.append(f"... +{len(failures) - 5} more failures")
            if summary_line:
                result_parts.append(summary_line)
            return "\n".join(result_parts)

        # --- Fallback: simple line-matching for jest/cargo/generic ---
        failure_lines: list[str] = []
        is_pytest = any("PASSED" in ln or "FAILED" in ln for ln in lines)
        is_jest = any("✓" in ln or "✗" in ln for ln in lines)
        is_cargo = any(" ok" in ln or "FAILED" in ln for ln in lines)

        for line in lines:
            if is_pytest and "FAILED" in line:
                failure_lines.append(line)
            elif is_jest and "✗" in line:
                failure_lines.append(line)
            elif is_cargo and "FAILED" in line:
                failure_lines.append(line)

        result_parts = failure_lines[:5]
        if len(failure_lines) > 5:
            result_parts.append(f"... +{len(failure_lines) - 5} more failures")
        if summary_line:
            result_parts.append(summary_line)

        if not result_parts:
            return text

        return "\n".join(result_parts)

    def _install_summary(self, text: str) -> str:
        """Keep warnings and the final summary line from package-install output.

        Strips progress/download/spinner lines; keeps warnings and the
        "added N packages" / "Successfully installed X" / "up to date" line.
        """
        lines = text.splitlines()
        kept: list[str] = []
        summary_patterns = [
            r"\badded\b.*\bpackages?\b",
            r"Successfully installed",
            r"up to date",
            r"already satisfied",
            r"found \d+ vulnerabilit",
        ]
        warning_patterns = [r"\bwarn\b", r"\bwarning\b", r"\bdeprecated\b"]
        skip_patterns = [
            r"⸨",
            r"^\s*\u280b",  # spinner
            r"timing\s",
            r"^npm timing",
            r"idealTree",
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(re.search(p, stripped, re.IGNORECASE) for p in skip_patterns):
                continue
            if any(re.search(p, stripped, re.IGNORECASE) for p in summary_patterns):
                kept.append(line)
                continue
            if any(re.search(p, stripped, re.IGNORECASE) for p in warning_patterns):
                kept.append(line)
                continue

        return "\n".join(kept) if kept else text

    def _lint_group(self, text: str) -> str:
        """Group lint output by rule code with counts and file breakdown.

        Tries structured JSON first (ruff/eslint ``--format json``), then falls
        back to regex parsing of ``file:line:col: CODE message`` lines.
        """
        # --- Tier 1: Try structured JSON input (ruff --output-format json) ---
        try:
            data = json.loads(text)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                code_counts: Counter = Counter()
                file_counts: Counter = Counter()
                code_desc: dict[str, str] = {}
                fixable_count = 0
                for item in data:
                    code = item.get("code") or item.get("ruleId") or "unknown"
                    msg = item.get("message", "")
                    filename = item.get("filename") or item.get("filePath") or "?"
                    code_counts[code] += 1
                    file_counts[filename] += 1
                    if code not in code_desc:
                        code_desc[code] = msg[:80]
                    if item.get("fix") or item.get("fixable"):
                        fixable_count += 1

                output_lines: list[str] = []
                output_lines.append(f"Lint: {len(data)} issues in {len(file_counts)} files")
                output_lines.append("")
                output_lines.append("By rule:")
                for code, count in code_counts.most_common():
                    output_lines.append(f"  {code} ({count}x): {code_desc.get(code, '')}")
                output_lines.append("")
                output_lines.append("By file:")
                for filename, count in file_counts.most_common(10):
                    output_lines.append(f"  {filename}: {count} issues")
                if len(file_counts) > 10:
                    output_lines.append(f"  ... +{len(file_counts) - 10} more files")
                if fixable_count:
                    output_lines.append(f"\n{fixable_count} auto-fixable (use --fix)")
                return "\n".join(output_lines)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # --- Tier 2: Regex parsing of file:line:col: CODE message ---
        lint_re = re.compile(r"^(\S+):(\d+):(\d+):\s+([A-Z]\d+)\s+(.*)", re.MULTILINE)
        matches = lint_re.findall(text)

        if not matches:
            return text

        code_counts = Counter()
        file_counts: Counter = Counter()
        code_desc = {}
        for filename, _line, _col, code, desc in matches:
            code_counts[code] += 1
            file_counts[filename] += 1
            if code not in code_desc:
                code_desc[code] = desc.strip()[:80]

        output_lines = []
        output_lines.append(f"Lint: {len(matches)} issues in {len(file_counts)} files")
        output_lines.append("")
        output_lines.append("By rule:")
        for code, count in code_counts.most_common():
            output_lines.append(f"  {code} ({count}x): {code_desc[code]}")
        output_lines.append("")
        output_lines.append("By file:")
        for filename, count in file_counts.most_common(10):
            output_lines.append(f"  {filename}: {count} issues")
        if len(file_counts) > 10:
            output_lines.append(f"  ... +{len(file_counts) - 10} more files")

        # Keep non-lint summary lines
        for line in text.splitlines():
            if not lint_re.match(line) and line.strip():
                if not any(line.strip().startswith(p) for p in ["Found ", "All checks"]):
                    continue
                output_lines.append(line)

        return "\n".join(output_lines)

    def _log_dedup(self, text: str) -> str:
        """Collapse consecutive or near-consecutive identical lines."""
        lines = text.splitlines()
        if not lines:
            return text

        output: list[str] = []
        current_line = lines[0]
        run_count = 1

        for line in lines[1:]:
            if line == current_line:
                run_count += 1
            else:
                if run_count > 1:
                    output.append(f"{current_line} (repeated {run_count} times)")
                else:
                    output.append(current_line)
                current_line = line
                run_count = 1

        # Flush last group
        if run_count > 1:
            output.append(f"{current_line} (repeated {run_count} times)")
        else:
            output.append(current_line)

        return "\n".join(output)

    def _docker_compact(self, text: str) -> str:
        """Compact docker ps/images/logs output.

        For ``docker ps``: shows name, image, status, ports in compact form.
        For ``docker images``: shows repo:tag + size.
        For logs: deduplicates repeated lines with ``(×N)`` counts.
        """
        lines = text.splitlines()
        if not lines:
            return text

        header = lines[0]

        # docker ps: CONTAINER ID  IMAGE  COMMAND  CREATED  STATUS  PORTS  NAMES
        if "CONTAINER ID" in header and "IMAGE" in header:
            output_parts: list[str] = [f"Docker containers ({len(lines) - 1}):"]
            for row in lines[1:]:
                cols = row.split()
                if len(cols) >= 7:
                    # Last col is usually the name
                    name = cols[-1]
                    image = cols[1]
                    # Status is typically cols[4] + cols[5]
                    status_start = row.find("Up") if "Up" in row else row.find("Exited")
                    if status_start >= 0:
                        status_end = row.find("  ", status_start + 1)
                        status = row[status_start : status_end if status_end > 0 else None].strip()
                    else:
                        status = "?"
                    output_parts.append(f"  {name}: {image} [{status}]")
                elif row.strip():
                    output_parts.append(f"  {row.strip()}")
            return "\n".join(output_parts)

        # docker images: REPOSITORY  TAG  IMAGE ID  CREATED  SIZE
        if "REPOSITORY" in header and "TAG" in header:
            output_parts = [f"Docker images ({len(lines) - 1}):"]
            for row in lines[1:]:
                cols = row.split()
                if len(cols) >= 5:
                    repo = cols[0]
                    tag = cols[1]
                    size = cols[-1]
                    output_parts.append(f"  {repo}:{tag} ({size})")
                elif row.strip():
                    output_parts.append(f"  {row.strip()}")
            return "\n".join(output_parts)

        # docker logs: deduplicate repeated lines
        return self._log_dedup(text)

    def _json_structure(self, text: str) -> str:
        """Extract schema-like structure from JSON output.

        Shows types instead of values, truncates long strings, and previews
        arrays with first item + count.
        """
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

        def _schema(obj: object, depth: int = 0, max_depth: int = 3) -> str:
            indent = "  " * depth
            if depth >= max_depth:
                return f"{indent}..."

            if isinstance(obj, list):
                total = len(obj)
                if total == 0:
                    return f"{indent}[] (empty)"
                first_schema = _schema(obj[0], depth + 1, max_depth)
                if total == 1:
                    return f"{indent}[1 item]:\n{first_schema}"
                return f"{indent}[{total} items, showing first]:\n{first_schema}\n{indent}... +{total - 1} more"

            if isinstance(obj, dict):
                if not obj:
                    return f"{indent}{{}} (empty)"
                lines: list[str] = []
                for key, value in obj.items():
                    if isinstance(value, str):
                        val_preview = value[:80] + ("..." if len(value) > 80 else "")
                        lines.append(f'{indent}{key}: "{val_preview}"')
                    elif isinstance(value, (int, float)):
                        lines.append(f"{indent}{key}: {value}")
                    elif isinstance(value, bool):
                        lines.append(f"{indent}{key}: {str(value).lower()}")
                    elif value is None:
                        lines.append(f"{indent}{key}: null")
                    elif isinstance(value, list):
                        if len(value) > 1:
                            lines.append(
                                f"{indent}{key}: [{len(value)} items, +{len(value) - 1} more]"
                            )
                        else:
                            lines.append(f"{indent}{key}: [{len(value)} items]")
                        if value and depth < max_depth - 1:
                            lines.append(_schema(value[0], depth + 2, max_depth))
                    elif isinstance(value, dict):
                        lines.append(f"{indent}{key}: {{{len(value)} keys}}")
                    else:
                        lines.append(f"{indent}{key}: <{type(value).__name__}>")
                return "\n".join(lines)

            if isinstance(obj, str):
                preview = obj[:80] + ("..." if len(obj) > 80 else "")
                return f'{indent}"{preview}"'
            return f"{indent}{obj}"

        return _schema(data)

    def _tree_compress(self, text: str) -> str:
        """Collapse common path prefixes in tree-like output."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        # Remove tree-drawing characters
        clean_paths: list[str] = []
        for line in lines:
            path = line.lstrip("├─└│ \t")
            if path:
                clean_paths.append(path)

        if not clean_paths:
            return text

        # Group by directory prefix
        dir_files: dict[str, list[str]] = {}
        for path in clean_paths:
            if "/" in path:
                parts = path.rsplit("/", 1)
                directory = parts[0] + "/"
                filename = parts[1]
            else:
                directory = "./"
                filename = path
            dir_files.setdefault(directory, []).append(filename)

        output_lines: list[str] = []
        for directory, files in sorted(dir_files.items()):
            if len(files) == 1:
                output_lines.append(f"{directory}{files[0]}")
            else:
                output_lines.append(f"{directory} ({len(files)} files): {', '.join(files)}")

        return "\n".join(output_lines)

    def _progress_strip(self, text: str) -> str:
        """Remove progress-indicator lines (%, spinners, ellipsis)."""
        result_lines: list[str] = []
        for line in text.splitlines():
            # Skip lines with carriage return (in-place progress)
            if "\r" in line:
                continue
            stripped = line.strip()
            # Skip bare percentage lines
            if re.match(r"^\s*\d+%\s*$", line):
                continue
            # Skip spinner-only lines
            if stripped and all(c in _SPINNER_CHARS or c.isspace() for c in stripped):
                continue
            # Skip lines that are only spinner + short label
            if stripped and stripped[0] in _SPINNER_CHARS:
                continue
            # Skip lines that are only dots / ellipsis
            if re.match(r"^\s*[.·]+\s*$", line):
                continue
            result_lines.append(line)

        return "\n".join(result_lines)

    def _generic_conservative(self, text: str) -> str:
        """Conservative fallback strategy for unclassified verbose tool
        output (#137). Applied ONLY when auto-detection lands ``"unknown"``
        AND the caller opts in via ``filter(..., fallback_strategy="generic")``.

        Deliberately more cautious than the type-specific strategies above:
        every collapse rule is scoped to CONSECUTIVE runs (never a global or
        non-adjacent dedup, which would silently eat distinct data rows) and
        always annotates what was removed. Pipeline (order matters):

            1. CR-resolve: each line collapses to its final rendered segment
               (the text after the last ``\\r``), preserving in-place
               progress-bar output instead of dropping the whole line.
            2. Drop spinner/percentage/dots-only progress noise.
            3. Collapse consecutive EXACT-duplicate line runs.
            4. Collapse consecutive MASKED near-duplicate runs (>=5 lines
               sharing a timestamp/hex/number-normalized shape) to
               first+last+an elision annotation, unless the run contains a
               distinct file path / URL / UUID / error code.
            5. Elide the middle of long (>8) contiguous stack-frame blocks,
               keeping the top 5 + bottom 2 frames per block.
            6. Collapse blank-line runs of 3+ down to exactly 1.
            7. Drop lines that are ENTIRELY a timestamp + log level with no
               message content (never strips a timestamp prefix off a
               message-bearing line).
        """
        lines = _cr_resolve_lines(text)
        if not lines:
            return text
        lines = _strip_progress_noise_lines(lines)
        lines = _collapse_exact_dup_runs(lines)
        lines = _collapse_masked_near_dup_runs(lines)
        lines = _elide_stack_frame_blocks(lines)
        lines = _collapse_blank_runs(lines)
        lines = _drop_timestamp_only_lines(lines)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # RTK fallback (optional external binary)
    # ------------------------------------------------------------------

    def _try_rtk(self, text: str) -> Optional[str]:
        """Try the ``rtk`` binary if available and text > 500 chars.

        Returns the filtered text string on success, ``None`` otherwise.
        """
        if len(text) <= _RTK_MIN_CHARS:
            return None
        if not shutil.which("rtk"):
            return None
        try:
            proc = subprocess.run(
                ["rtk", "--json"],
                input=text,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if proc.returncode != 0:
                return None
            payload = json.loads(proc.stdout)
            return payload.get("filtered") or payload.get("output") or None
        except Exception:
            return None
