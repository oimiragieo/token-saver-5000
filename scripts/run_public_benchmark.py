#!/usr/bin/env python3
"""Public, reproducible-by-anyone compression benchmark for token-saver-5000.

This is the one-command entrypoint referenced by ``BENCHMARK.md``. It measures
compression ratio and token savings on the committed ``benchmarks/corpus/``
files (small / medium / large prose docs + a Python code sample) using the
same word-count methodology as ``tests/test_gtm_benchmarks.py`` (the GTM claim
regression-lock suite), so the numbers in ``BENCHMARK.md`` can be reproduced
byte-for-byte by anyone who clones the repo.

Usage:
    PYTHONPATH=. python scripts/run_public_benchmark.py
    PYTHONPATH=. python scripts/run_public_benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _measure(text: str, file_id: str) -> dict:
    """Measure compression ratio using a fresh compressor instance.

    Mirrors the exact methodology in tests/test_gtm_benchmarks.py so these
    numbers reproduce the locked GTM regression thresholds.
    """
    from src.semantic_compressor import SemanticCompressor

    compressor = SemanticCompressor()
    compressor.ingest_file(text, file_id)
    skeleton = compressor.read_skeleton(file_id)
    original_tokens = len(text.split())
    skeleton_tokens = len(skeleton.split()) if skeleton else original_tokens
    ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
    savings_pct = (1 - skeleton_tokens / original_tokens) * 100 if original_tokens > 0 else 0.0
    return {
        "original_tokens": original_tokens,
        "skeleton_tokens": skeleton_tokens,
        "compression_ratio": round(ratio, 2),
        "token_savings_pct": round(savings_pct, 2),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the public reproducible compression benchmark.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=PROJECT_ROOT / "benchmarks" / "corpus",
        help="Directory containing small.txt / medium.txt / large.txt / code/.",
    )
    parser.add_argument(
        "--code-files",
        type=int,
        default=5,
        help="Number of code/*.py files to concatenate for the code case (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write results as JSON.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    corpus_dir: Path = args.corpus_dir

    cases: list[tuple[str, str]] = []
    for name in ("small", "medium", "large"):
        path = corpus_dir / f"{name}.txt"
        if path.exists():
            cases.append((name, path.read_text(encoding="utf-8")))
        else:
            print(f"skip: {path} not found", file=sys.stderr)

    code_dir = corpus_dir / "code"
    if code_dir.exists():
        py_files = sorted(code_dir.glob("*.py"))[: args.code_files]
        if py_files:
            all_code = "\n\n".join(f.read_text(encoding="utf-8") for f in py_files)
            cases.append(("code", all_code))

    if not cases:
        print(f"No corpus files found under {corpus_dir}", file=sys.stderr)
        return 2

    results = {}
    print(f"{'Case':<10} {'Orig tokens':>12} {'Skeleton tokens':>16} {'Ratio':>8} {'Savings':>9}")
    print("-" * 60)
    for name, text in cases:
        result = _measure(text, f"pub_{name}")
        results[name] = result
        print(
            f"{name:<10} {result['original_tokens']:>12} {result['skeleton_tokens']:>16} "
            f"{result['compression_ratio']:>7.2f}x {result['token_savings_pct']:>8.2f}%"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
