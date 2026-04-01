"""Wraps Token Saver skill scripts for pre-compression."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..token_refiner import TokenRefiner

SKILLS_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "token-saver-context-compression" / "scripts"
)

_refiner = TokenRefiner()


@dataclass
class CompressResult:
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    refined: bool = False


def compress_file(file_path: Path, mode: str = "baseline", refine: bool = True) -> CompressResult:
    """Compress a file using the Token Saver skill script.

    Args:
        file_path: Path to the file to compress.
        mode: Compression mode (baseline, query_guided, evidence_aware).
        refine: If True, apply token-level refinement after compression.
    """
    script = SKILLS_DIR / "compress_context.py"
    if not script.exists():
        raise FileNotFoundError(f"Skill script not found: {script}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--file",
            str(file_path),
            "--mode",
            mode,
            "--output-format",
            "json",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(SKILLS_DIR),
        timeout=120,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"compress_context.py failed (exit {result.returncode}): {result.stderr[:500]}"
        )
    data = json.loads(result.stdout)
    compressed_text = data.get("compressed_text", "")
    compressed_tokens = data.get("compressed_tokens", 0)

    # Apply token-level refinement for additional compression
    if refine and compressed_text:
        refined = _refiner.refine(compressed_text, target_ratio=0.7)
        compressed_text = refined.refined_text
        compressed_tokens = refined.refined_tokens

    original_tokens = data.get("original_tokens", 0)
    return CompressResult(
        compressed_text=compressed_text,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=round(original_tokens / max(1, compressed_tokens), 1),
        refined=refine,
    )


def compress_text(text: str, mode: str = "baseline", refine: bool = True) -> CompressResult:
    """Compress text content by writing to a temp file first."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        tmp_path = Path(f.name)
    try:
        return compress_file(tmp_path, mode=mode, refine=refine)
    finally:
        tmp_path.unlink(missing_ok=True)
