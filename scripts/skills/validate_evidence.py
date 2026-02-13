#!/usr/bin/env python3
"""Validate evidence sufficiency for a query against compressed context."""

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
    parser = argparse.ArgumentParser(description="Check evidence sufficiency for a query.")
    parser.add_argument("--text", type=str, default="", help="Inline text input.")
    parser.add_argument("--file", type=Path, help="Path to UTF-8 text file.")
    parser.add_argument(
        "--file-id", type=str, default="skill_evidence_doc", help="Document identifier."
    )
    parser.add_argument("--query", type=str, required=True, help="Query to validate.")
    parser.add_argument("--top-k", type=int, default=5, help="Evidence node count.")
    parser.add_argument("--min-similarity", type=float, default=0.35, help="Sufficiency threshold.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    text = _read_text(args)
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        return 2

    compressor = SemanticCompressor()
    compressor.ingest_file(text, args.file_id)
    evidence = compressor.retrieve_evidence(
        query=args.query,
        file_id=args.file_id,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
    )
    payload = {
        "file_id": args.file_id,
        "query": args.query,
        "sufficient": evidence.sufficient,
        "best_score": round(evidence.best_score, 3),
        "threshold": evidence.threshold,
        "used_expanded_search": evidence.used_expanded_search,
        "message": evidence.message,
        "node_ids": evidence.node_ids,
    }
    print(json.dumps(payload, indent=2))
    return 0 if evidence.sufficient else 1


if __name__ == "__main__":
    raise SystemExit(main())
