"""Cross-process output determinism regression gate.

Locks the 2026-06-19 fix for non-deterministic skeleton output. The bug:
``semantic_compressor.py`` returned ``list(set(entities))[:max_entities]`` for the
"Key entities:" line. ``set`` iteration order is ``PYTHONHASHSEED``-dependent, so
truncating the scrambled set kept DIFFERENT entities across processes — the skeleton
text varied run-to-run. Fix: ``list(dict.fromkeys(entities))`` (first-occurrence,
deterministic).

This module is the empirical completeness gate the design review demanded: rather than
asserting "only one line was non-deterministic" from a static grep, it runs full
``read_skeleton`` in two subprocesses with DIFFERENT ``PYTHONHASHSEED`` values and asserts
byte-identical output — catching any future hash-order leak anywhere in the output path.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from src.semantic_compressor import SemanticCompressor

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_extract_key_entities_first_occurrence_deterministic():
    """The fix returns first-occurrence order, independent of hash seed.

    With the old ``list(set(...))[:N]`` this could return any N of the 6 entities in
    any order; with ``dict.fromkeys`` it is always the first N in appearance order.
    """
    compressor = SemanticCompressor()
    text = "the Alpha then Beta then Gamma then Delta then Epsilon then Zeta then Alpha here"
    result = compressor._extract_key_entities(text, max_entities=4)
    assert result == ["Alpha", "Beta", "Gamma", "Delta"]


def test_extract_key_entities_dedup_preserves_order():
    """Dedup keeps the first occurrence and drops later repeats, order intact."""
    compressor = SemanticCompressor()
    text = "x Apple then Banana then Apple then Cherry then Banana end"
    result = compressor._extract_key_entities(text, max_entities=10)
    assert result == ["Apple", "Banana", "Cherry"]


def test_skeleton_deterministic_same_process():
    """Compressing the same input twice in one process yields identical skeletons."""
    text = (
        "The Roman Empire was a powerful state founded by Augustus. Trajan expanded it "
        "to its peak. Constantine later moved the capital to Byzantium, renaming it "
        "Constantinople. Odoacer deposed Romulus Augustulus in the West. "
    ) * 3
    c1 = SemanticCompressor()
    c1.ingest_file(text, file_id="same_proc")
    first = c1.read_skeleton("same_proc")
    c2 = SemanticCompressor()
    c2.ingest_file(text, file_id="same_proc")
    second = c2.read_skeleton("same_proc")
    assert first == second


# Worker compresses a few structurally diverse fixtures and prints the concatenated
# skeletons. Run in subprocesses with differing PYTHONHASHSEED to prove the output
# does not depend on hash-randomized set/dict iteration anywhere in the pipeline.
_SKELETON_WORKER = r"""
import sys
from src.semantic_compressor import SemanticCompressor

FIXTURES = {
    "rome": (
        "The Roman Empire was among the most powerful forces of the ancient world. "
        "Augustus founded it. Trajan expanded it to its peak. Constantine moved the "
        "capital to Byzantium. Odoacer deposed Romulus Augustulus in the West. "
    ) * 3,
    "bio": (
        "Photosynthesis converts light into chemical energy inside chloroplasts using "
        "Chlorophyll. The Calvin cycle runs in the stroma while light reactions run in "
        "the Thylakoid membranes producing Glucose and Oxygen. "
    ) * 3,
    "code": (
        "class Cache:\n    def get(self, key):\n        return self.store.get(key)\n"
        "    def put(self, key, value):\n        self.store[key] = value\n"
    ) * 3,
}

c = SemanticCompressor()
parts = []
for fid in sorted(FIXTURES):
    c.ingest_file(FIXTURES[fid], file_id=fid)
    parts.append(c.read_skeleton(fid))
sys.stdout.write("\n===\n".join(parts))
"""


def _run_worker(hashseed: str) -> str:
    env = {"PYTHONHASHSEED": hashseed, "PYTHONPATH": str(_REPO_ROOT)}
    import os

    full_env = {**os.environ, **env}
    proc = subprocess.run(
        [sys.executable, "-c", _SKELETON_WORKER],
        cwd=str(_REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"worker failed (seed={hashseed}):\n{proc.stderr[-2000:]}"
    return proc.stdout


@pytest.mark.slow
def test_skeleton_deterministic_across_hashseed():
    """Full skeleton output is byte-identical under different PYTHONHASHSEED values.

    This is the empirical completeness gate: it would fail if ANY set/dict/frozenset
    iteration in the output path leaked hash-order non-determinism, not just the known
    entity line.
    """
    out_a = _run_worker("0")
    out_b = _run_worker("12345")
    assert out_a == out_b, "skeleton output differs across PYTHONHASHSEED -> non-determinism leak"
