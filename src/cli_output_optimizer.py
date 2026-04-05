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

    def __init__(self, tee_store: Optional[object] = None):
        self._tee_store = tee_store

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
