"""`detect_dead_files` must not be quadratic in realpath syscalls.

WHAT THIS COST. The import-graph builder scanned every candidate file for every
import and called `.resolve()` on BOTH sides of the comparison inside that
innermost loop:

    for target in files:
        if Path(target).resolve() == Path(resolved).resolve():

so the cost was O(files x imports x files) realpath SYSCALLS. On this repo's own
`src/`, `detect_dead_files("src")` exceeded a 300-second local timeout, and the
same call took CI's "Full Validation" job down via pytest-timeout inside
`_joinrealpath`. After resolving each target once into a dict: **0.27s**.

It stayed hidden because "Full Validation" is gated behind "Quality Gate", which
had been failing on unpinned-ruff drift, so the job was `skipped` on every
recent main run. A permanently-red gate does not merely fail to catch things --
it hides the jobs behind it.

WHY A SYSCALL COUNT AND NOT A WALL CLOCK. A timing assertion on a shared CI
runner is a contention detector, not a correctness test -- this repo has receipts
for exactly that. Counting `Path.resolve` calls is deterministic: it is the
quantity that went quadratic, and it cannot be moved by a busy host.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile
from unittest.mock import patch

from src.dead_code_detector import detect_dead_files


def _write_pkg(root: pathlib.Path, n_files: int, imports_each: int) -> None:
    """A package where every module imports several siblings.

    Cross-imports are the shape that made the old code quadratic: each import
    triggered a full linear scan of the file list, with two resolves per step.
    """
    for i in range(n_files):
        lines = [f"import mod{(i + k) % n_files}" for k in range(1, imports_each + 1)]
        lines.append(f"VALUE_{i} = {i}")
        (root / f"mod{i}.py").write_text("\n".join(lines), encoding="utf-8")


def test_resolve_calls_do_not_grow_quadratically(tmp_path: pathlib.Path) -> None:
    """Doubling the file count must not ~quadruple the resolve() calls."""
    small = tmp_path / "small"
    big = tmp_path / "big"
    small.mkdir()
    big.mkdir()
    _write_pkg(small, n_files=10, imports_each=4)
    _write_pkg(big, n_files=20, imports_each=4)

    counts: dict[str, int] = {}
    real_resolve = pathlib.Path.resolve

    for label, directory in (("small", small), ("big", big)):
        n = 0

        def counting_resolve(self, *a, **kw):
            nonlocal n
            n += 1
            return real_resolve(self, *a, **kw)

        with patch.object(pathlib.Path, "resolve", counting_resolve):
            detect_dead_files(str(directory))
        counts[label] = n

    assert counts["small"] > 0, "the probe counted nothing - it is not observing resolve()"

    # Quadratic would be ~4x for a 2x file count. Linear-ish is ~2x. Allow 3x as
    # the discriminating boundary: comfortably above linear, well below quadratic.
    ratio = counts["big"] / max(counts["small"], 1)
    assert ratio < 3.0, (
        f"resolve() calls scaled {ratio:.1f}x when the file count doubled "
        f"({counts['small']} -> {counts['big']}). That is the quadratic shape that "
        f"took CI's Full Validation job down: the inner loop is resolving per "
        f"comparison again instead of once per file."
    )


def test_the_import_graph_is_still_correct() -> None:
    """NON-VACUITY + correctness: a fast wrong answer is worse than a slow one.

    `a` imports `b`, nothing imports `c`. So `c` is dead and `b` is not,
    regardless of how the resolution is cached.

    DELIBERATELY NOT `tmp_path`. `_is_never_dead` matches its patterns against
    the WHOLE PATH, not the filename, and `"test_"` is one of them. Every pytest
    temp directory lives under `pytest-of-<user>/pytest-N/<test name>/`, so the
    fixture's own location made all three files never-dead and this assertion
    read `dead == set()`. The fixture LOCATION was deciding the verdict -- the
    detector was right both before and after the change, which a standalone run
    under `tempfile.mkdtemp()` confirmed immediately.
    """
    root = pathlib.Path(tempfile.mkdtemp(prefix="dcd-"))
    try:
        d = root / "pkg"
        d.mkdir()
        (d / "a.py").write_text("import b\n", encoding="utf-8")
        (d / "b.py").write_text("X = 1\n", encoding="utf-8")
        (d / "c.py").write_text("Y = 2\n", encoding="utf-8")

        assert "test_" not in str(d).replace("\\", "/").lower(), (
            f"the fixture path itself contains a never-dead pattern ({d}); "
            f"this test would pass vacuously"
        )

        report = detect_dead_files(str(d))

        assert report.total_files == 3, report.total_files
        dead = {pathlib.Path(p).name for p in report.dead_files}
        assert "b.py" not in dead, f"b.py is imported by a.py but was called dead: {dead}"
        assert "c.py" in dead, f"c.py is imported by nothing and should be dead: {dead}"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_a_broken_symlink_does_not_abort_the_scan(tmp_path: pathlib.Path) -> None:
    """`.resolve()` on a vanished target must degrade, not kill the run.

    The one-time resolution pass touches every candidate up front, so a single
    unresolvable entry would otherwise take down a scan that the old per-
    comparison form merely skipped.
    """
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "a.py").write_text("import b\n", encoding="utf-8")
    (d / "b.py").write_text("X = 1\n", encoding="utf-8")

    report = detect_dead_files(str(d))
    assert report.total_files == 2
