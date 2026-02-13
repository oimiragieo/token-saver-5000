#!/usr/bin/env python3
"""Profile token counts for raw vs compressed context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.semantic_compressor import SemanticCompressor  # noqa: E402


def _read_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return sys.stdin.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile token usage before/after compression.")
    parser.add_argument("--text", type=str, default="", help="Inline text input.")
    parser.add_argument("--file", type=Path, help="Path to UTF-8 text file.")
    parser.add_argument(
        "--file-id", type=str, default="skill_profile_doc", help="Document identifier."
    )
    parser.add_argument(
        "--skeleton-ratio",
        type=float,
        default=0.2,
        help="Skeleton ratio used by compressor.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    text = _read_text(args)
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        return 2

    compressor = SemanticCompressor(skeleton_ratio=args.skeleton_ratio)
    result = compressor.ingest_file(text, args.file_id)
    payload = {
        "file_id": args.file_id,
        "original_tokens": result.total_tokens,
        "skeleton_tokens": result.skeleton_tokens,
        "compression_ratio": round(result.compression_ratio, 3),
        "token_savings_pct": round(
            (1 - result.skeleton_tokens / max(result.total_tokens, 1)) * 100, 2
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
