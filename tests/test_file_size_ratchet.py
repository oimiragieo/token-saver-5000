"""Ratchet for the adopted file-size threshold in the engine — it may only tighten.

WHY THIS EXISTS HERE TOO. The parent repo adopted 1000 lines for core logic and
2000 for tests (`docs/decisions/2026-08-21-file-size-threshold.md`) and shipped a
ratchet for `api/`. That left half the standard unenforced, because the files it
governs live in TWO repositories — which is the exact defect backlog N10
recorded: its five named files span this submodule and the parent, and the row
named neither. Anyone running the parent's ratchet sees a green bar and concludes
the standard is covered.

Two of N10's five were here: `tests/test_coverage_boost4.py` and
`tests/test_server_factory_service.py`. Both were split 2026-08-24 into
size-compliant files (`test_coverage_boost4.py`/`test_coverage_boost4b.py`;
`test_server_factory_service.py`/`_contracts.py`/`_validation_chain.py`), by
test-class grouping, no test logic changed. The baseline below dropped to 0
in the same commit as the split, per the reverse-assertion rule two
paragraphs down.

WHAT IT DOES NOT DO. It forces no split. The decision doc is explicit that
adopting the threshold does not authorize the remediation, and none of these
files carries a correctness or security risk. This only stops the population
growing.

THE REVERSE ASSERTION IS THE POINT. Checking only `<=` lets the baseline go stale
the moment work lands: the tree drops to 4 while the constant still says 6, and
those two forgiven slots silently absorb the next two regressions. So dropping
BELOW a baseline fails too, and the fix is to lower the constant in the same
commit. That is what makes the direction one-way.

Cost: two directory walks. No imports of the package, no network, no model load.
"""

from __future__ import annotations

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Ceilings as adopted in the parent's decision doc.
_CORE_CEILING = 1000
_TEST_CEILING = 2000

# Measured 2026-08-24 (post N10 test-file split). These may only go DOWN. Lower
# them in the same commit that reduces the count; never raise one to turn a red
# gate green.
_CORE_BASELINE = 5
_TEST_BASELINE = 0

# Asserted rather than assumed: a walk that silently matched nothing would pass
# forever.
_MIN_CORE_FILES = 120
_MIN_TEST_FILES = 150


def _python_files(root: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _over(root: pathlib.Path, ceiling: int) -> list[tuple[int, str]]:
    out = []
    for path in _python_files(root):
        try:
            n = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if n > ceiling:
            out.append((n, str(path.relative_to(_ROOT)).replace("\\", "/")))
    return sorted(out, reverse=True)


def test_the_walk_sees_a_real_population():
    """Control. An empty or tiny walk makes every assertion below vacuous."""
    core = _python_files(_ROOT / "src")
    tests = _python_files(_ROOT / "tests")
    assert len(core) >= _MIN_CORE_FILES, (
        f"only {len(core)} files found under src/; the walk is not seeing the tree, so "
        "the ratchet below is measuring nothing"
    )
    assert (
        len(tests) >= _MIN_TEST_FILES
    ), f"only {len(tests)} files found under tests/; the walk is not seeing the tree"


@pytest.mark.parametrize(
    ("subdir", "ceiling", "baseline", "label"),
    [
        ("src", _CORE_CEILING, _CORE_BASELINE, "core-logic"),
        ("tests", _TEST_CEILING, _TEST_BASELINE, "test"),
    ],
)
def test_oversized_file_count_only_ever_shrinks(
    subdir: str, ceiling: int, baseline: int, label: str
):
    """The count of files over the adopted ceiling may not grow.

    Both directions fail on purpose. More than the baseline means something
    crossed the ceiling: split it, or put the new code somewhere not already over
    budget. Fewer means the ratchet has slack: lower the constant here in the
    same commit, or the forgiven slots quietly pre-absorb the next regressions.
    """
    over = _over(_ROOT / subdir, ceiling)
    count = len(over)
    worst = "\n  ".join(f"{n:6d}  {p}" for n, p in over[:8])

    assert count <= baseline, (
        f"{count} {label} files exceed {ceiling} lines, baseline is {baseline}. "
        f"Largest:\n  {worst}\n"
        "Do NOT raise the baseline to clear this."
    )
    assert count >= baseline, (
        f"only {count} {label} files exceed {ceiling} lines but the baseline still says "
        f"{baseline}. Tighten it in this commit: set the baseline in "
        f"{pathlib.Path(__file__).name} to {count}."
    )
