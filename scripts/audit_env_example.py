#!/usr/bin/env python3
"""Audit that every os.getenv/os.environ key in src/ is documented in .env.example.

VAL-DOCKER-002 gate. Usage (from repo root):
  python scripts/audit_env_example.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
ENV_EXAMPLE = REPO_ROOT / ".env.example"

ENV_PATTERN = re.compile(
    r"os\.(?:getenv|environ\.get)\(\s*['\"]([A-Z][A-Z0-9_]+)['\"]"
    r"|os\.environ\[['\"]([A-Z][A-Z0-9_]+)['\"]\]"
)


def collect_src_env_vars() -> set[str]:
    found: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in ENV_PATTERN.finditer(text):
            found.add(match.group(1) or match.group(2))
    return found


def collect_documented_env_vars() -> set[str]:
    text = ENV_EXAMPLE.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r"^([A-Z][A-Z0-9_]*)=", text, re.M))


def main() -> int:
    if not ENV_EXAMPLE.is_file():
        print(f"Missing {ENV_EXAMPLE}", file=sys.stderr)
        return 1

    src_vars = collect_src_env_vars()
    documented = collect_documented_env_vars()
    missing = sorted(src_vars - documented)

    if missing:
        print("Undocumented env vars (add to .env.example):", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"OK: all {len(src_vars)} src/ env vars are listed in .env.example")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
