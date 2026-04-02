#!/usr/bin/env python
"""Token Saver 5000 Benchmark: Compare token usage with and without compression.

Runs identical tasks through Claude Code and/or Gemini CLI, measuring real API
token consumption. Compares raw context vs Token Saver-compressed context.

Usage:
    python scripts/benchmark_token_savings.py --dry-run
    python scripts/benchmark_token_savings.py --mode skill --sizes small --providers claude
    python scripts/benchmark_token_savings.py --mode both --output results.json --verbose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cli_benchmark.runner import run_benchmark, run_search_compress_benchmark


def _print_search_compress_table(results: list) -> str:
    """Render search-then-compress results as an ASCII table.

    Args:
        results: List of SearchCompressResult instances.

    Returns:
        Formatted table string.
    """
    if not results:
        return "No search-compress results to display."

    header = (
        f"{'Query':<35} {'Scanned':>8} {'Matched':>8} {'Method':<14} "
        f"{'Naive':>10} {'Cmp-All':>10} {'Srch+Cmp':>10} "
        f"{'vs Naive':>10} {'vs CmpAll':>10}"
    )
    sep = "-" * len(header)
    rows = [sep, header, sep]
    for r in results:
        q = r.query[:33] + ".." if len(r.query) > 35 else r.query
        rows.append(
            f"{q:<35} {r.files_scanned:>8} {r.files_matched:>8} {r.search_method:<14} "
            f"{r.naive_all_tokens:>10,} {r.naive_compressed_tokens:>10,} "
            f"{r.search_compress_tokens:>10,} "
            f"{r.search_vs_naive_savings_pct:>9.1f}% "
            f"{r.search_vs_compress_all_savings_pct:>9.1f}%"
        )
    rows.append(sep)
    rows.append("")
    rows.append(
        "Methodology: 'vs Naive' = search+compress vs uncompressed all files. "
        "'vs CmpAll' = vs compress-all baseline."
    )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Token Saver 5000 Benchmark: measure real token savings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["skill", "mcp", "both", "search_compress"],
        default="skill",
        help=(
            "Benchmark mode: skill (pre-compress), mcp (live MCP), both, "
            "or search_compress (search-then-compress pipeline on code corpus)"
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=["small", "medium", "large", "all"],
        default=["all"],
        help="Corpus sizes to test (default: all)",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["claude", "gemini", "codex", "opencode", "all"],
        default=["all"],
        help="CLI providers to test (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON results to file",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip actual CLI calls")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat each config N times")
    parser.add_argument("--model-claude", type=str, default=None, help="Override Claude model")
    parser.add_argument("--model-gemini", type=str, default=None, help="Override Gemini model")
    parser.add_argument("--model-codex", type=str, default=None, help="Override Codex model")
    parser.add_argument("--model-opencode", type=str, default=None, help="Override OpenCode model")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    args = parser.parse_args()

    sizes = None if "all" in args.sizes else args.sizes
    providers = None if "all" in args.providers else args.providers

    if args.verbose:
        print("Token Saver 5000 Benchmark")
        print(f"  Mode: {args.mode}")
        if args.mode != "search_compress":
            print(f"  Sizes: {sizes or 'all'}")
            print(f"  Providers: {providers or 'all'}")
            print(f"  Dry run: {args.dry_run}")
        print()

    # search_compress mode: purely local measurement, no CLI providers needed
    if args.mode == "search_compress":
        corpus_dir = Path(__file__).resolve().parent.parent / "benchmarks"
        sc_results = run_search_compress_benchmark(
            corpus_dir=corpus_dir,
            verbose=args.verbose,
        )
        print()
        print("Search-Then-Compress Pipeline Benchmark")
        print(_print_search_compress_table(sc_results))
        return 0

    report = run_benchmark(
        mode=args.mode,
        sizes=sizes,
        providers=providers,
        dry_run=args.dry_run,
        repeats=args.repeats,
        model_claude=args.model_claude,
        model_gemini=args.model_gemini,
        model_codex=args.model_codex,
        model_opencode=args.model_opencode,
        verbose=args.verbose,
    )

    # Print table
    print()
    print(report.to_table())

    # Save JSON if requested
    if args.output:
        report.to_json(args.output)
        if args.verbose:
            print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
