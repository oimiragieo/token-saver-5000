"""Search-then-compress pipeline: tensor-grep search -> semantic compression.

Implements the research-backed pattern (cAST, SmartChunk, SeleCom):
1. Search for query-relevant code files
2. Compress only matched files
3. Compare against naive (everything) approach
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

from .tensor_grep_integration import code_search
from .tensor_grep_integration import is_available as tg_available
from .cli_benchmark.compressor import compress_text


@dataclass
class SearchCompressResult:
    """Result from a single search-then-compress pipeline run."""

    query: str
    # Search phase
    files_scanned: int = 0
    files_matched: int = 0
    search_method: str = "glob_fallback"
    matched_files: list[str] = field(default_factory=list)
    # Compress phase (search-first)
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    compression_ratio: float = 0.0
    document_savings_pct: float = 0.0
    # Naive comparison (compress everything)
    naive_all_tokens: int = 0
    naive_compressed_tokens: int = 0
    # Search-first vs naive
    search_compress_tokens: int = 0
    search_vs_naive_savings_pct: float = 0.0
    search_vs_compress_all_savings_pct: float = 0.0
    # Trace
    stages: list[str] = field(default_factory=list)


def _glob_search(
    query: str,
    directory: str,
    extensions: list[str] | None = None,
) -> list[str]:
    """Fallback search using glob + simple content grep.

    Args:
        query: Search query string; individual words are matched independently.
        directory: Root directory to search recursively.
        extensions: Glob patterns to include (default: Python and common code files).

    Returns:
        List of file paths whose content contains at least one query term.
    """
    if extensions is None:
        extensions = ["*.py", "*.js", "*.ts", "*.go", "*.rs"]

    all_files: list[str] = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))

    query_terms = query.lower().split()
    matched: list[str] = []
    for fpath in all_files:
        try:
            content = Path(fpath).read_text(encoding="utf-8", errors="replace").lower()
            if any(term in content for term in query_terms):
                matched.append(fpath)
        except OSError:
            continue
    return matched


def _count_tokens(text: str) -> int:
    """Fast token count estimate: one token per four characters.

    Args:
        text: Input text to estimate.

    Returns:
        Integer token estimate.
    """
    return len(text) // 4


def search_then_compress(
    directory: str,
    query: str,
    max_files: int = 20,
    refine: bool = True,
) -> SearchCompressResult:
    """Execute the search-then-compress pipeline.

    Phases:
    1. Collect ALL .py files (for naive baseline measurement).
    2. Search directory for files relevant to query (tensor-grep or glob fallback).
    3. Read all files to establish naive token count.
    4. Compress ALL files (compress-all baseline).
    5. Read and compress ONLY matched files (search-first path).
    6. Calculate and attach comparison metrics.

    Args:
        directory: Path to directory containing code files.
        query: Natural-language or keyword search query.
        max_files: Maximum number of matched files to compress.
        refine: Whether to apply token-level refinement after compression.

    Returns:
        Populated SearchCompressResult with all metrics filled in.
    """
    result = SearchCompressResult(query=query)
    directory = str(Path(directory).resolve())

    # Phase 1: Collect ALL files (for naive baseline)
    all_py_files = sorted(glob.glob(os.path.join(directory, "**", "*.py"), recursive=True))
    result.files_scanned = len(all_py_files)
    result.stages.append("scan")

    if not all_py_files:
        return result

    # Phase 2: Search for query-relevant files
    if tg_available():
        search_result = code_search(query, directory, use_index=False)
        if search_result.available and search_result.matches:
            matched_paths = list(
                {m.get("file", "") for m in search_result.matches if m.get("file")}
            )
            # Resolve to absolute paths
            matched_paths = [
                os.path.join(directory, p) if not os.path.isabs(p) else p for p in matched_paths
            ]
            result.matched_files = sorted(matched_paths[:max_files])
            result.search_method = "tensor_grep"
            result.stages.append("tensor_grep_search")
        else:
            result.matched_files = sorted(_glob_search(query, directory)[:max_files])
            result.search_method = "glob_fallback"
            result.stages.append("glob_search")
    else:
        result.matched_files = sorted(_glob_search(query, directory)[:max_files])
        result.search_method = "glob_fallback"
        result.stages.append("glob_search")

    result.files_matched = len(result.matched_files)

    # Phase 3: Read all files (for naive token count)
    all_content = ""
    for fpath in all_py_files:
        try:
            all_content += Path(fpath).read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
    result.naive_all_tokens = _count_tokens(all_content)
    result.stages.append("read_all")

    # Phase 4: Compress ALL files (compress-all baseline)
    try:
        all_compressed = compress_text(all_content, refine=refine)
        result.naive_compressed_tokens = all_compressed.compressed_tokens
    except Exception:
        result.naive_compressed_tokens = result.naive_all_tokens  # fallback
    result.stages.append("compress_all")

    # Phase 5: Read and compress ONLY matched files
    if result.matched_files:
        matched_content = ""
        for fpath in result.matched_files:
            try:
                matched_content += Path(fpath).read_text(encoding="utf-8", errors="replace") + "\n"
            except OSError:
                continue

        result.total_original_tokens = _count_tokens(matched_content)

        try:
            matched_compressed = compress_text(matched_content, refine=refine)
            result.total_compressed_tokens = matched_compressed.compressed_tokens
            result.search_compress_tokens = matched_compressed.compressed_tokens
        except Exception:
            result.total_compressed_tokens = result.total_original_tokens
            result.search_compress_tokens = result.total_original_tokens

        result.stages.append("compress_matched")

    # Phase 6: Calculate metrics
    if result.total_original_tokens > 0:
        result.compression_ratio = round(
            result.total_original_tokens / max(1, result.total_compressed_tokens), 1
        )
        result.document_savings_pct = round(
            (result.total_original_tokens - result.total_compressed_tokens)
            / result.total_original_tokens
            * 100,
            1,
        )

    if result.naive_all_tokens > 0:
        result.search_vs_naive_savings_pct = round(
            (result.naive_all_tokens - result.search_compress_tokens)
            / result.naive_all_tokens
            * 100,
            1,
        )

    if result.naive_compressed_tokens > 0:
        result.search_vs_compress_all_savings_pct = round(
            (result.naive_compressed_tokens - result.search_compress_tokens)
            / result.naive_compressed_tokens
            * 100,
            1,
        )

    result.stages.append("report")
    return result
