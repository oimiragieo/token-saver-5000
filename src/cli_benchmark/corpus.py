"""Corpus loader for benchmark harness."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus"


@dataclass
class CorpusEntry:
    name: str
    file_path: Path
    description: str
    content: str
    line_count: int


def load_manifest(corpus_dir: Path = CORPUS_DIR) -> dict:
    """Load and parse manifest.json."""
    manifest_path = corpus_dir / "manifest.json"
    with open(manifest_path) as f:
        return json.load(f)


def load_corpus(name: str, corpus_dir: Path = CORPUS_DIR) -> CorpusEntry:
    """Load a single corpus file by name (small/medium/large)."""
    manifest = load_manifest(corpus_dir)
    entry = next((f for f in manifest["files"] if f["name"] == name), None)
    if entry is None:
        valid = [f["name"] for f in manifest["files"]]
        raise ValueError(f"Unknown corpus {name!r}. Available: {valid}")
    file_path = corpus_dir / entry["file"]
    content = file_path.read_text(encoding="utf-8")
    return CorpusEntry(
        name=entry["name"],
        file_path=file_path,
        description=entry["description"],
        content=content,
        line_count=len(content.splitlines()),
    )


def load_all_corpus(corpus_dir: Path = CORPUS_DIR) -> list[CorpusEntry]:
    """Load all corpus files."""
    manifest = load_manifest(corpus_dir)
    return [load_corpus(f["name"], corpus_dir) for f in manifest["files"]]


def build_prompt(context: str, corpus_dir: Path = CORPUS_DIR) -> str:
    """Build the benchmark prompt with context inserted."""
    manifest = load_manifest(corpus_dir)
    template = manifest["prompt_template"]
    return template.replace("{context}", context)
