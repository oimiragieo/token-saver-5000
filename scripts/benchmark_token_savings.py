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

from src.cli_benchmark.runner import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Token Saver 5000 Benchmark: measure real token savings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["skill", "mcp", "both"],
        default="skill",
        help="Benchmark mode: skill (pre-compress), mcp (live MCP), or both (default: skill)",
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
        print(f"  Sizes: {sizes or 'all'}")
        print(f"  Providers: {providers or 'all'}")
        print(f"  Dry run: {args.dry_run}")
        print()

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
