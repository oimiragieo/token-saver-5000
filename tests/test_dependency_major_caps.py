"""Dependencies whose API we CALL must carry a major-version cap.

THREE INSTANCES OF THIS CLASS IN THE GOTCONTEXT REPOS IN ONE DAY (2026-08-19):

    svix>=1.40.0   -> 2.0.0   every Clerk webhook broken in production
    ruff>=0.1.0    -> newer   Quality Gate red for months, hiding Full Validation
    mcp>=0.9.0     -> 2.0.0   Server.list_tools removed; 74 tests failed

Each was an unbounded floor on a package whose API the code calls directly.
A `>=` with no ceiling is a promise that the maintainers will never rename
anything -- and CI, which resolves fresh every run, is where that promise gets
tested. Local venvs pin whatever they installed months ago and stay quiet: on
the mcp break, local had 1.28.0 and CI had 2.0.0.

This guard is a RATCHET, not a blanket rule. Capping every dependency would be
noise; the registry below names the ones whose SYMBOLS this codebase touches,
where a major bump is an API migration rather than a bugfix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

# package -> (expected cap, why this one matters)
#
# Add a package here when the code calls its API directly and a major bump
# would be a migration. The REASON is required: a bare entry is a rule nobody
# can evaluate later.
MUST_BE_CAPPED: dict[str, tuple[str, str]] = {
    "mcp": (
        "<2",
        "router_binding.py binds Server.list_tools / call_tool; mcp 2.0.0 "
        "removed list_tools and failed 74 tests",
    ),
}


def _requirement_lines(path: Path) -> list[str]:
    """Non-comment, non-blank lines, with inline comments stripped."""
    out = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if line:
            out.append(line)
    return out


@pytest.mark.parametrize("package", sorted(MUST_BE_CAPPED))
def test_the_pin_carries_a_major_cap(package: str) -> None:
    cap, why = MUST_BE_CAPPED[package]

    found = []
    for path in (REQUIREMENTS, PYPROJECT):
        for line in _requirement_lines(path):
            if re.match(rf"^{re.escape(package)}\s*[><=~!]", line):
                found.append((path.name, line))

    assert found, (
        f"{package} is not declared in requirements.txt or pyproject.toml. "
        f"If it was removed, drop it from MUST_BE_CAPPED too - a registry entry "
        f"for a package nobody depends on is a guard that can never fire."
    )

    for filename, line in found:
        assert cap in line, (
            f"{filename}: `{line}` has no `{cap}` cap.\n"
            f"WHY THIS ONE: {why}.\n"
            f"An unbounded floor means CI resolves whatever PyPI published this "
            f"morning, while your venv keeps the version you installed months "
            f"ago - so the break appears only in CI and only for whoever pushes "
            f"next."
        )


def test_both_manifests_agree_on_every_capped_package() -> None:
    """A cap in one manifest and not the other is a cap that does not hold.

    requirements.txt drives CI; pyproject.toml drives `pip install -e .` and
    anything consuming this as a package. Capping only one leaves a real
    install path unbounded.
    """
    for package, (cap, _why) in MUST_BE_CAPPED.items():
        in_req = [ln for ln in _requirement_lines(REQUIREMENTS) if ln.startswith(package)]
        in_toml = [ln for ln in _requirement_lines(PYPROJECT) if ln.startswith(package)]
        if in_req and in_toml:
            assert (cap in in_req[0]) == (cap in in_toml[0]), (
                f"{package} cap disagrees between manifests:\n"
                f"  requirements.txt: {in_req[0]}\n"
                f"  pyproject.toml:   {in_toml[0]}"
            )


def test_the_guard_can_actually_fail() -> None:
    """NON-VACUITY: a parser that matches nothing would pass every arm above.

    Pinned against a synthetic unbounded line, so a change to `_requirement_lines`
    that silently stops matching real declarations is caught here rather than by
    the next major bump.
    """
    lines = _requirement_lines(REQUIREMENTS)
    assert lines, "requirement parsing returned nothing - the guard is inert"

    assert any(
        re.match(r"^mcp\s*[><=~!]", ln) for ln in lines
    ), "the parser no longer finds the mcp declaration it is meant to police"

    # A comment-only line must not be read as a requirement, or a note ABOUT a
    # package would satisfy the cap check for it.
    fake = "# mcp>=0.9.0 was unbounded and broke 74 tests"
    assert not [x for x in _requirement_lines_from_text(fake) if x.startswith("mcp")]


def _requirement_lines_from_text(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if line:
            out.append(line)
    return out
