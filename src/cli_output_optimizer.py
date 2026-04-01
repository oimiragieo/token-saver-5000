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
    unknown         → passthrough (no change)

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
}

# ANSI escape-code regex (includes all terminating letters)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

# Spinner characters used by many CLI progress bars
_SPINNER_CHARS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


class CLIOutputOptimizer:
    """Auto-detecting CLI output optimizer.

    Usage::

        optimizer = CLIOutputOptimizer()
        result = optimizer.filter(raw_output)
        print(result.filtered_text)
        print(f"{result.compression_pct:.1f}% compression")
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, text: str, command_hint: Optional[str] = None) -> FilterResult:
        """Filter *text* by auto-detecting command type and applying a strategy.

        Args:
            text:         Raw CLI output to filter.
            command_hint: Optional override for the detected command type.
                          Must be a key in :data:`STRATEGY_MAP` (or ``"unknown"``).

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
        if strategy_name and strategy_name != "_strip_ansi":
            strategy_fn = getattr(self, strategy_name)
            filtered_text = strategy_fn(working_text)
            applied = strategy_name
        else:
            filtered_text = working_text
            applied = "passthrough"

        filtered_lines = len(filtered_text.splitlines())
        return FilterResult(
            original_text=text,
            filtered_text=filtered_text,
            original_lines=original_lines,
            filtered_lines=filtered_lines,
            command_detected=command_type,
            strategy_applied=applied,
        )

    def detect_command(self, text: str) -> str:
        """Detect the type of CLI output from its content.

        Returns a command-type string that maps to a strategy in
        :data:`STRATEGY_MAP`, or ``"unknown"`` when no pattern matches.
        """
        if "diff --git" in text:
            return "git_diff"
        if any(p in text for p in ["modified:", "new file:", "deleted:", "Untracked files:"]):
            return "git_status"
        if re.search(r"\b(PASSED|FAILED|passed|failed)\b.*\b(PASSED|FAILED|passed|failed)\b", text):
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
            text.count("/") > 0 and text.count("/") > text.count("\n") * 0.5
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
        """Summarize a ``git diff`` into file list + insertion/deletion counts."""
        files: list[str] = []
        insertions = 0
        deletions = 0

        for line in text.splitlines():
            if line.startswith("diff --git "):
                # "diff --git a/path b/path" → extract b/path
                parts = line.split(" b/", 1)
                if len(parts) == 2:
                    files.append(parts[1].strip())
            elif line.startswith("+") and not line.startswith("+++"):
                insertions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1

        file_count = len(files)
        noun = "file" if file_count == 1 else "files"
        summary = f"{file_count} {noun} changed, +{insertions} insertions, -{deletions} deletions"
        file_list = "\n".join(f"  {f}" for f in files)
        return f"{file_list}\n{summary}"

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
        """Keep only failure lines and the summary line from test output."""
        lines = text.splitlines()
        failure_lines: list[str] = []
        summary_line: Optional[str] = None

        # Detect framework
        is_pytest = any("PASSED" in ln or "FAILED" in ln for ln in lines)
        is_jest = any("✓" in ln or "✗" in ln for ln in lines)
        is_cargo = any(" ok" in ln or "FAILED" in ln for ln in lines)

        for line in lines:
            stripped = line.strip()
            # Summary line heuristic: lots of = signs or "passed/failed" counts
            if re.search(r"={5,}.*(?:passed|failed)", stripped, re.IGNORECASE):
                summary_line = line
                continue
            if re.search(r"\d+\s+(?:passed|failed)", stripped, re.IGNORECASE):
                summary_line = line
                continue

            if is_pytest and "FAILED" in line:
                failure_lines.append(line)
            elif is_jest and "✗" in line:
                failure_lines.append(line)
            elif is_cargo and "FAILED" in line:
                failure_lines.append(line)

        result_parts = failure_lines[:]
        if summary_line:
            result_parts.append(summary_line)

        if not result_parts:
            # Nothing specific found — return the whole thing
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
        """Group lint output by rule code with occurrence counts."""
        # Pattern: file:line:col: CODE message
        lint_re = re.compile(r"^\S+:\d+:\d+:\s+([A-Z]\d+)\s+(.*)", re.MULTILINE)
        matches = lint_re.findall(text)

        if not matches:
            return text

        # Count per code + capture first description
        code_counts: Counter = Counter()
        code_desc: dict[str, str] = {}
        for code, desc in matches:
            code_counts[code] += 1
            if code not in code_desc:
                code_desc[code] = desc.strip()

        output_lines: list[str] = []
        for code, count in code_counts.most_common():
            output_lines.append(f"{code} ({code_desc[code]}): {count} occurrences")

        # Keep summary lines (lines that don't match the lint pattern)
        for line in text.splitlines():
            if not lint_re.match(line) and line.strip():
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

    def _json_structure(self, text: str) -> str:
        """Extract top-level keys and value types from JSON output."""
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

        if isinstance(data, list):
            total = len(data)
            if total == 0:
                return "[] (empty array)"
            first = data[0]
            preview = json.dumps(first, ensure_ascii=False)
            if len(preview) > 80:
                preview = preview[:77] + "..."
            if total > 1:
                return f"[array of {total} items]\nFirst item: {preview}\n... and {total - 1} more items"
            return f"[array of {total} item]\nFirst item: {preview}"

        if isinstance(data, dict):
            lines: list[str] = []
            for key, value in data.items():
                vtype = type(value).__name__
                if isinstance(value, list):
                    lines.append(
                        f"{key}: array ({len(value)} items)"
                        + (f", first: {json.dumps(value[0])}" if value else "")
                    )
                elif isinstance(value, dict):
                    lines.append(f"{key}: object ({len(value)} keys)")
                else:
                    val_str = str(value)
                    if len(val_str) > 60:
                        val_str = val_str[:57] + "..."
                    lines.append(f"{key}: {vtype} = {val_str}")
            return "\n".join(lines)

        # Scalar JSON value
        return text

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
