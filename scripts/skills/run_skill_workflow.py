#!/usr/bin/env python3
"""Run end-to-end token-saver skill workflow in one command."""

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
    parser = argparse.ArgumentParser(
        description="Run profile + compression + evidence validation workflow."
    )
    parser.add_argument("--text", type=str, default="", help="Inline text input.")
    parser.add_argument("--file", type=Path, help="Path to UTF-8 text file.")
    parser.add_argument(
        "--file-id", type=str, default="skill_workflow_doc", help="Document identifier."
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "query_guided", "evidence_aware"],
        default="baseline",
        help="Compression selection mode.",
    )
    parser.add_argument(
        "--query", type=str, default="", help="Query for guided/evidence-aware modes."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Evidence node count.")
    parser.add_argument("--min-similarity", type=float, default=0.35, help="Evidence threshold.")
    parser.add_argument("--skeleton-ratio", type=float, default=0.2, help="Skeleton ratio.")
    parser.add_argument(
        "--fail-on-insufficient-evidence",
        action="store_true",
        help="Exit non-zero if evidence validation is insufficient.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    text = _read_text(args)
    if not text.strip():
        print("No input text provided.", file=sys.stderr)
        return 2
    if args.mode != "baseline" and not args.query.strip():
        print("--query is required for query_guided and evidence_aware modes.", file=sys.stderr)
        return 2

    compressor = SemanticCompressor(skeleton_ratio=args.skeleton_ratio)
    ingest = compressor.ingest_file(text, args.file_id)

    profile = {
        "file_id": args.file_id,
        "original_tokens": ingest.total_tokens,
        "skeleton_tokens": ingest.skeleton_tokens,
        "compression_ratio": round(ingest.compression_ratio, 3),
        "token_savings_pct": round(
            (1 - ingest.skeleton_tokens / max(ingest.total_tokens, 1)) * 100, 2
        ),
    }

    evidence = None
    if args.mode == "query_guided":
        skeleton = compressor._generate_skeleton(args.file_id, query=args.query)
    elif args.mode == "evidence_aware":
        evidence = compressor.retrieve_evidence(
            query=args.query,
            file_id=args.file_id,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
        )
        skeleton = compressor._generate_skeleton(
            args.file_id,
            query=args.query,
            anchor_node_ids=set(evidence.node_ids),
        )
    else:
        skeleton = compressor._generate_skeleton(args.file_id)

    compressed = {
        "file_id": args.file_id,
        "mode": args.mode,
        "query": args.query or None,
        "skeleton_tokens": skeleton.skeleton_tokens,
        "compression_ratio": round(skeleton.compression_ratio, 3),
        "skeleton_text": skeleton.skeleton_text,
    }

    evidence_validation = None
    if args.query.strip():
        if evidence is None:
            evidence = compressor.retrieve_evidence(
                query=args.query,
                file_id=args.file_id,
                top_k=args.top_k,
                min_similarity=args.min_similarity,
            )
        evidence_validation = {
            "sufficient": evidence.sufficient,
            "best_score": round(evidence.best_score, 3),
            "threshold": evidence.threshold,
            "used_expanded_search": evidence.used_expanded_search,
            "message": evidence.message,
            "node_ids": evidence.node_ids,
        }

    payload = {
        "profile": profile,
        "compressed": compressed,
        "evidence_validation": evidence_validation,
    }
    print(json.dumps(payload, indent=2))

    if (
        args.fail_on_insufficient_evidence
        and evidence_validation
        and not evidence_validation["sufficient"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
