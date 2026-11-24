"""Session-scoped diff-on-reread tracker for avoiding redundant full-file reloads."""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import OrderedDict
from typing import Any

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_NEWLINE_MARKER = "\\ No newline at end of file"


def reconstruct_from_diff(base_content: str, diff_text: str) -> str:
    """Reconstruct target content from ``base_content`` and a unified ``diff_text``."""
    if not diff_text:
        return base_content

    base_lines = base_content.splitlines(keepends=True)
    diff_lines = diff_text.splitlines(keepends=True)
    out: list[str] = []
    base_idx = 0
    i = 0

    while i < len(diff_lines):
        line = diff_lines[i]
        if line.startswith(("--- ", "+++ ")):
            i += 1
            continue

        match = _HUNK_RE.match(line)
        if not match:
            i += 1
            continue

        old_start = int(match.group(1))
        # Unified diff uses 1-based line numbers (0 when old file is empty).
        target_base_idx = max(0, old_start - 1) if old_start > 0 else 0
        while base_idx < target_base_idx and base_idx < len(base_lines):
            out.append(base_lines[base_idx])
            base_idx += 1

        i += 1
        while i < len(diff_lines):
            hline = diff_lines[i]
            if hline.startswith("@@ "):
                break
            if hline.startswith(_NO_NEWLINE_MARKER):
                i += 1
                continue
            prefix = hline[:1]
            payload = hline[1:]
            next_is_no_newline = i + 1 < len(diff_lines) and diff_lines[i + 1].startswith(
                _NO_NEWLINE_MARKER
            )
            if next_is_no_newline and payload.endswith("\n"):
                payload = payload[:-1]

            if prefix == " ":
                out.append(base_lines[base_idx] if base_idx < len(base_lines) else payload)
                base_idx += 1
            elif prefix == "-":
                base_idx += 1
            elif prefix == "+":
                out.append(payload)
            else:
                break
            i += 1

    while base_idx < len(base_lines):
        out.append(base_lines[base_idx])
        base_idx += 1

    return "".join(out)


def _format_unified_diff(diff_raw: list[str]) -> str:
    """Format unified diff lines while preserving exact line endings and no-newline markers."""
    formatted: list[str] = []
    for entry in diff_raw:
        if entry.startswith(("--- ", "+++ ", "@@ ")):
            formatted.append(entry if entry.endswith("\n") else entry + "\n")
        elif entry.endswith("\n"):
            formatted.append(entry)
        else:
            formatted.append(entry + "\n" + _NO_NEWLINE_MARKER + "\n")
    return "".join(formatted)


class SessionDiffTracker:
    """Tracks ``(session_id, file_path) -> (sha256, lines)`` with bounded LRU eviction."""

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries <= 0:
            raise ValueError("'max_entries' must be > 0")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, str], tuple[str, list[str]]] = OrderedDict()

    def _put(self, key: tuple[str, str], sha256: str, lines: list[str]) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
        self._entries[key] = (sha256, lines)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def read(
        self,
        session_id: str,
        file_path: str,
        content: str,
        force_full: bool = False,
    ) -> dict[str, Any]:
        """Return compact marker, unified diff, or full content for ``(session_id, file_path)``."""
        key = (session_id, file_path)
        content_bytes = content.encode("utf-8")
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        curr_lines = content.splitlines(keepends=True)
        total_lines = len(curr_lines)

        prev = self._entries.get(key)
        if force_full or prev is None:
            self._put(key, sha256, curr_lines)
            return {
                "mode": "full",
                "content": content,
                "sha256": sha256,
                "total_lines": total_lines,
                "diff_lines": 0,
            }

        prev_sha, prev_lines = prev
        if prev_sha == sha256:
            self._entries.move_to_end(key)
            return {
                "mode": "unchanged",
                "content": f"UNCHANGED (sha256={sha256[:12]}, {total_lines} lines)",
                "sha256": sha256,
                "total_lines": total_lines,
                "diff_lines": 0,
            }

        diff_raw = list(
            difflib.unified_diff(
                prev_lines,
                curr_lines,
                fromfile=file_path,
                tofile=file_path,
                lineterm="\n",
            )
        )
        prev_content = "".join(prev_lines)
        self._put(key, sha256, curr_lines)

        diff_text = _format_unified_diff(diff_raw)
        diff_bytes = len(diff_text.encode("utf-8"))
        within_line_budget = len(diff_raw) < 0.6 * max(1, total_lines)
        within_byte_budget = diff_bytes < 0.6 * max(1, len(content_bytes))

        if (
            diff_raw
            and within_line_budget
            and within_byte_budget
            and reconstruct_from_diff(prev_content, diff_text) == content
        ):
            return {
                "mode": "diff",
                "content": diff_text,
                "sha256": sha256,
                "total_lines": total_lines,
                "diff_lines": len(diff_raw),
            }

        return {
            "mode": "full",
            "content": content,
            "sha256": sha256,
            "total_lines": total_lines,
            "diff_lines": len(diff_raw),
        }

    def __len__(self) -> int:
        return len(self._entries)
