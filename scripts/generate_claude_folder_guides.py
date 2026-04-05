#!/usr/bin/env python3
"""
Regenerate per-folder claude.md guides and print a manifest for CLAUDE.md.

Output is deterministic across OSes: UTF-8, LF-only line endings on write
(`Path.write_text` with `newline="\\n"`), sorted directory listings, and sorted
`os.walk` branches so Linux CI matches Windows devs.

Usage (from repo root):
  python scripts/generate_claude_folder_guides.py
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

SKIP_DIR_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".semantic_modulator_data",
    "node_modules",
    ".egg-info",
    "dist",
    "build",
    ".venv",
    "venv",
}

TARGET_TOP_LEVELS = ("src", "tests", "scripts")


def iter_doc_dirs(base: Path) -> Iterable[Path]:
    yield base
    for root, dirs, _files in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIR_PARTS and not d.startswith("."))
        for d in dirs:
            yield Path(root) / d


def skip_path(path: Path) -> bool:
    if any(p in SKIP_DIR_PARTS for p in path.parts):
        return True
    return any(part.endswith(".egg-info") for part in path.parts)


def module_summary(tree: ast.AST) -> str:
    doc = ast.get_docstring(tree)
    if not doc:
        return ""
    paras = [p.strip() for p in doc.strip().split("\n\n") if p.strip()]
    if not paras:
        return ""
    first_line = paras[0].splitlines()[0].strip()
    return first_line[:500]


def extract_symbols(tree: ast.AST) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append(("class", node.name))
        elif isinstance(node, ast.FunctionDef):
            out.append(("def", node.name))
        elif isinstance(node, ast.AsyncFunctionDef):
            out.append(("async def", node.name))
    return out


def analyze_py(path: Path) -> tuple[str, list[tuple[str, str]]]:
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, OSError):
        return ("(unparsed — syntax error or unreadable)", [])
    summ = module_summary(tree)
    if not summ:
        summ = "(no module docstring — see symbols below)"
    return (summ, extract_symbols(tree))


def list_non_py(path: Path) -> list[str]:
    exts = {".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".sh", ".cjs", ".js"}
    names = []
    for p in sorted(path.iterdir(), key=lambda x: x.name.casefold()):
        if p.name == "claude.md":
            continue
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."):
            names.append(p.name)
    return names


def rel_posix(p: Path) -> str:
    return p.relative_to(REPO_ROOT).as_posix()


def build_folder_markdown(folder: Path) -> str:
    rel = rel_posix(folder)
    depth = len(folder.relative_to(REPO_ROOT).parts)
    root_claude = f"{'../' * depth}CLAUDE.md"
    py_files = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix == ".py"),
        key=lambda p: p.name.casefold(),
    )
    other = list_non_py(folder)
    lines: list[str] = [
        f"# Folder guide: `{rel}/`",
        "",
        f"Breadcrumb for AI navigation. Master index: [`CLAUDE.md`]({root_claude}).",
        "",
        "## Contents",
        "",
    ]
    if not py_files and not other:
        lines.append("_No tracked artifacts in this folder (or only non-documented file types)._")
        lines.append("")
        return "\n".join(lines)

    if py_files:
        lines.append("### Python modules")
        lines.append("")
        for pf in py_files:
            summ, syms = analyze_py(pf)
            lines.append(f"#### `{pf.name}`")
            lines.append("")
            lines.append(summ)
            lines.append("")
            if syms:
                lines.append("| Kind | Name |")
                lines.append("|------|------|")
                for kind, name in syms:
                    lines.append(f"| `{kind}` | `{name}` |")
                lines.append("")
            else:
                lines.append("_No top-level classes or functions (may re-export only)._")
                lines.append("")

    if other:
        lines.append("### Other files")
        lines.append("")
        for name in other:
            lines.append(f"- `{name}`")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Symbols are **top-level only** (nested methods and inner functions are not listed). "
        "Regenerate: `python scripts/generate_claude_folder_guides.py`."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    written: list[str] = []
    for top in TARGET_TOP_LEVELS:
        base = REPO_ROOT / top
        if not base.is_dir():
            continue
        for folder in iter_doc_dirs(base):
            if skip_path(folder):
                continue
            has_py = any(p.suffix == ".py" for p in folder.iterdir() if p.is_file())
            has_other = bool(list_non_py(folder))
            if not has_py and not has_other:
                continue
            md = build_folder_markdown(folder)
            out = folder / "claude.md"
            out.write_text(md, encoding="utf-8", newline="\n")
            written.append(rel_posix(out))

    print(f"Wrote {len(written)} claude.md files:")
    for w in sorted(written):
        print(f"  {w}")


if __name__ == "__main__":
    main()
