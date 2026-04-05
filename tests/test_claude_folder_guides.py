"""Contracts for per-folder claude.md guides (determinism, line endings)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iter_claude_guides() -> list[Path]:
    paths: list[Path] = []
    for sub in ("src", "tests", "scripts"):
        base = ROOT / sub
        if not base.is_dir():
            continue
        for p in base.rglob("claude.md"):
            if ".egg-info" in p.parts:
                continue
            paths.append(p)
    return sorted(paths, key=lambda p: p.as_posix().casefold())


def test_claude_folder_guides_have_no_crlf():
    """Guides must be LF-only so Windows devs and Linux CI agree (see generate_claude_folder_guides)."""
    for path in _iter_claude_guides():
        data = path.read_bytes()
        assert b"\r\n" not in data, (
            f"{path.relative_to(ROOT)} contains CRLF; run "
            f"`python scripts/generate_claude_folder_guides.py` and commit."
        )


def test_gitattributes_sets_lf_for_claude_guides():
    ga = ROOT / ".gitattributes"
    assert ga.is_file()
    text = ga.read_text(encoding="utf-8")
    assert "claude.md" in text
    assert "eol=lf" in text
