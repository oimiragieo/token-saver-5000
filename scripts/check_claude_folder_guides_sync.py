#!/usr/bin/env python3
"""
Regenerate per-folder claude.md guides and fail if git working tree differs.

Use in CI and pre-commit so guides cannot drift from the generator. Only
`claude.md` files under src/, tests/, and scripts/ are compared (other local
edits do not affect the exit code). The generator writes UTF-8 with LF newlines
only; `.gitattributes` enforces `eol=lf` for `**/claude.md`.

Usage (from repo root):
  python scripts/check_claude_folder_guides_sync.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = REPO_ROOT / "scripts" / "generate_claude_folder_guides.py"


def iter_guide_paths() -> list[Path]:
    out: list[Path] = []
    for sub in ("src", "tests", "scripts"):
        base = REPO_ROOT / sub
        if not base.is_dir():
            continue
        for p in base.rglob("claude.md"):
            if ".egg-info" in p.parts:
                continue
            out.append(p)
    return sorted(out)


def main() -> int:
    gen = subprocess.run([sys.executable, str(GENERATOR)], cwd=REPO_ROOT)
    if gen.returncode != 0:
        return gen.returncode

    paths = [p.relative_to(REPO_ROOT).as_posix() for p in iter_guide_paths()]
    if not paths:
        print("check_claude_folder_guides_sync: no claude.md guides found", file=sys.stderr)
        return 1

    diff = subprocess.run(
        ["git", "diff", "--exit-code", "--", *paths],
        cwd=REPO_ROOT,
    )
    if diff.returncode != 0:
        print(
            "\nPer-folder claude.md guides are out of date.\n"
            "Run: python scripts/generate_claude_folder_guides.py\n"
            "Then commit the updated claude.md files.\n",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
