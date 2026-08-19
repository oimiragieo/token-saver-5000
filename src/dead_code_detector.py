"""Dead code detection via import graph analysis.

Identifies files that are never imported by other files in a directory.
Uses regex-based import parsing (no AST required) for speed.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENTRY_PATTERNS = [
    "main.py",
    "app.py",
    "server.py",
    "run.py",
    "manage.py",
    "wsgi.py",
    "asgi.py",
    "cli.py",
    "setup.py",
    "conftest.py",
    "__init__.py",
    "__main__.py",
]

DEFAULT_NEVER_DEAD_PATTERNS = [
    "test_",
    "tests/",
    "conftest",
    "__init__",
    "__main__",
    "setup.py",
    "manage.py",
    "migrations/",
]


@dataclass
class DeadCodeReport:
    total_files: int = 0
    dead_files: list[str] = field(default_factory=list)
    dead_file_count: int = 0
    live_files: list[str] = field(default_factory=list)
    live_file_count: int = 0
    tokens_saved: int = 0
    entry_points: list[str] = field(default_factory=list)


def _extract_imports(code: str) -> list[str]:
    """Extract imported module names from Python code using regex."""
    imports = []
    # from X import Y
    for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.MULTILINE):
        imports.append(m.group(1))
    # import X
    for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.MULTILINE):
        imports.append(m.group(1))
    return imports


def _module_name_to_possible_files(module_name: str, directory: str) -> list[str]:
    """Convert a Python module path to possible file paths."""
    parts = module_name.split(".")
    possibilities = []
    # Direct file match
    possibilities.append(os.path.join(directory, *parts) + ".py")
    # Package match
    possibilities.append(os.path.join(directory, *parts, "__init__.py"))
    # First component only (for relative-style)
    if len(parts) > 0:
        possibilities.append(os.path.join(directory, parts[0] + ".py"))
    return possibilities


def _is_entry_point(file_path: str, entry_patterns: list[str]) -> bool:
    """Check if a file matches entry point patterns."""
    name = os.path.basename(file_path)
    for pattern in entry_patterns:
        if name == pattern or pattern in file_path:
            return True
    return False


def _is_never_dead(file_path: str) -> bool:
    """Check if a file should never be marked as dead."""
    name = os.path.basename(file_path)
    path_lower = file_path.replace("\\", "/").lower()
    for pattern in DEFAULT_NEVER_DEAD_PATTERNS:
        if pattern in name or pattern in path_lower:
            return True
    return False


def detect_dead_files(
    directory: str,
    files: list[str] | None = None,
    entry_patterns: list[str] | None = None,
) -> DeadCodeReport:
    """Detect files that are never imported by other files.

    Algorithm:
    1. Collect all Python files in directory
    2. Parse imports from each file (regex, fast)
    3. Build import graph: A imports B -> edge A->B
    4. Identify entry points + never-dead files
    5. BFS from entry points to find all reachable files
    6. Unreachable = dead candidates
    """
    if entry_patterns is None:
        entry_patterns = DEFAULT_ENTRY_PATTERNS

    directory = str(Path(directory).resolve())

    # Collect files
    if files is None:
        import glob

        files = sorted(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True))

    if not files:
        return DeadCodeReport()

    report = DeadCodeReport(total_files=len(files))

    # Build import graph
    imported_by: dict[str, set[str]] = defaultdict(set)  # file -> set of files that import it
    file_contents: dict[str, str] = {}

    for fpath in files:
        try:
            content = Path(fpath).read_text(encoding="utf-8", errors="replace")
            file_contents[fpath] = content
        except OSError:
            continue

    # Resolve every candidate file ONCE, into a lookup keyed by its real path.
    #
    # The previous form scanned `files` linearly for each import and called
    # `.resolve()` on BOTH sides of the comparison inside that innermost loop:
    #
    #     for target in files:
    #         if Path(target).resolve() == Path(resolved).resolve():
    #
    # so the cost was O(files x imports x files) realpath SYSCALLS. On this
    # repo's own `src/` that is millions of them: `detect_dead_files("src")`
    # exceeded a 300s local timeout, and the same call took the CI "Full
    # Validation" job down via pytest-timeout inside `_joinrealpath`.
    #
    # Resolving each target once turns the inner scan into a dict hit: O(files)
    # syscalls total, then O(1) per import. Same comparison, same first-match
    # semantics (one target per resolved path), no behaviour change.
    resolved_targets: dict[str, str] = {}
    for target in files:
        try:
            resolved_targets.setdefault(str(Path(target).resolve()), target)
        except OSError:
            # A broken symlink or a vanished file must not abort the scan; fall
            # back to the literal path so the entry is still reachable.
            resolved_targets.setdefault(target, target)

    for fpath in files:
        content = file_contents.get(fpath)
        if content is None:
            continue

        imports = _extract_imports(content)
        for module in imports:
            possible_files = _module_name_to_possible_files(module, directory)
            for possible in possible_files:
                try:
                    key = str(Path(possible).resolve()) if os.path.exists(possible) else possible
                except OSError:
                    key = possible
                target = resolved_targets.get(key)
                if target is not None:
                    imported_by[target].add(fpath)

    # Identify entry points and never-dead files
    always_live: set[str] = set()
    for fpath in files:
        if _is_entry_point(fpath, entry_patterns) or _is_never_dead(fpath):
            always_live.add(fpath)
            report.entry_points.append(fpath)

    # Files that are imported by at least one other file
    has_importers = {f for f, importers in imported_by.items() if importers}

    # Live = entry points + imported files
    live = always_live | has_importers

    # Dead = everything else
    for fpath in files:
        if fpath in live:
            report.live_files.append(fpath)
        else:
            report.dead_files.append(fpath)
            # Estimate token savings
            content = file_contents.get(fpath, "")
            report.tokens_saved += len(content) // 4

    report.dead_file_count = len(report.dead_files)
    report.live_file_count = len(report.live_files)

    return report
