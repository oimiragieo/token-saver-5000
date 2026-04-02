"""Tests for the search-then-compress pipeline.

Uses the real code corpus at benchmarks/corpus/code/ — no mocking needed
for local file operations. Tensor-grep is optional; tests pass with glob fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.search_compress_pipeline import SearchCompressResult, _glob_search, search_then_compress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_CORPUS = _REPO_ROOT / "benchmarks" / "corpus" / "code"
CORPUS_DIR = _REPO_ROOT / "benchmarks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_exists() -> bool:
    """Return True if the code corpus directory is non-empty."""
    return CODE_CORPUS.exists() and any(CODE_CORPUS.glob("*.py"))


# ---------------------------------------------------------------------------
# Corpus existence guard — skip all corpus tests if corpus missing
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    not _corpus_exists(),
    reason="Code corpus not found at benchmarks/corpus/code/",
)


# ---------------------------------------------------------------------------
# Search-level tests
# ---------------------------------------------------------------------------


def test_search_finds_auth_files():
    """Query 'authentication' should match auth.py from the corpus."""
    matched = _glob_search("authentication", str(CODE_CORPUS))
    basenames = [os.path.basename(p) for p in matched]
    assert "auth.py" in basenames, f"Expected auth.py in {basenames}"


def test_search_finds_database_files():
    """Query 'database connection' should match database.py."""
    matched = _glob_search("database connection", str(CODE_CORPUS))
    basenames = [os.path.basename(p) for p in matched]
    assert "database.py" in basenames, f"Expected database.py in {basenames}"


def test_search_finds_api_files():
    """Query 'rate limiting' should match middleware.py or api.py."""
    matched = _glob_search("rate limiting", str(CODE_CORPUS))
    basenames = {os.path.basename(p) for p in matched}
    assert basenames & {
        "middleware.py",
        "api.py",
    }, f"Expected middleware.py or api.py in {basenames}"


# ---------------------------------------------------------------------------
# Pipeline result field tests
# ---------------------------------------------------------------------------


def test_search_compress_result_fields():
    """All required SearchCompressResult fields should be populated after a run."""
    result = search_then_compress(str(CODE_CORPUS), "authentication")

    assert isinstance(result.query, str)
    assert isinstance(result.files_scanned, int)
    assert isinstance(result.files_matched, int)
    assert isinstance(result.search_method, str)
    assert isinstance(result.matched_files, list)
    assert isinstance(result.total_original_tokens, int)
    assert isinstance(result.total_compressed_tokens, int)
    assert isinstance(result.compression_ratio, float)
    assert isinstance(result.document_savings_pct, float)
    assert isinstance(result.naive_all_tokens, int)
    assert isinstance(result.naive_compressed_tokens, int)
    assert isinstance(result.search_compress_tokens, int)
    assert isinstance(result.search_vs_naive_savings_pct, float)
    assert isinstance(result.search_vs_compress_all_savings_pct, float)
    assert isinstance(result.stages, list)


def test_search_compress_fewer_than_naive():
    """search_compress_tokens must be strictly less than naive_all_tokens."""
    result = search_then_compress(str(CODE_CORPUS), "authentication token JWT")
    # Search+compress touches fewer files -> fewer tokens than raw all-files
    assert result.search_compress_tokens < result.naive_all_tokens, (
        f"search_compress_tokens={result.search_compress_tokens} "
        f"should be < naive_all_tokens={result.naive_all_tokens}"
    )


def test_search_compress_fewer_than_compress_all():
    """search_compress_tokens should be <= naive_compressed_tokens (compress-all baseline)."""
    result = search_then_compress(str(CODE_CORPUS), "database connection pool")
    # Searching fewer files -> result should not exceed compress-all output
    assert result.search_compress_tokens <= result.naive_compressed_tokens, (
        f"search_compress_tokens={result.search_compress_tokens} "
        f"should be <= naive_compressed_tokens={result.naive_compressed_tokens}"
    )


def test_search_vs_naive_savings_positive():
    """search_vs_naive_savings_pct should be positive (we always save vs raw all-files)."""
    result = search_then_compress(str(CODE_CORPUS), "authentication token JWT")
    assert (
        result.search_vs_naive_savings_pct > 0
    ), f"Expected positive savings vs naive, got {result.search_vs_naive_savings_pct}%"


def test_search_vs_compress_all_savings_nonneg():
    """search_vs_compress_all_savings_pct must be >= 0."""
    result = search_then_compress(str(CODE_CORPUS), "database connection pool")
    assert result.search_vs_compress_all_savings_pct >= 0, (
        f"Expected non-negative savings vs compress-all, "
        f"got {result.search_vs_compress_all_savings_pct}%"
    )


def test_max_files_limit():
    """matched_files must not exceed max_files."""
    max_files = 3
    result = search_then_compress(str(CODE_CORPUS), "class", max_files=max_files)
    assert (
        len(result.matched_files) <= max_files
    ), f"Expected <= {max_files} matched files, got {len(result.matched_files)}"


def test_stages_populated():
    """stages list must be non-empty after a successful run."""
    result = search_then_compress(str(CODE_CORPUS), "authentication")
    assert result.stages, "stages list should not be empty"
    assert "scan" in result.stages


def test_search_method_is_string():
    """search_method must be either 'tensor_grep' or 'glob_fallback'."""
    result = search_then_compress(str(CODE_CORPUS), "authentication")
    valid_methods = {"tensor_grep", "glob_fallback"}
    assert (
        result.search_method in valid_methods
    ), f"search_method={result.search_method!r} not in {valid_methods}"


# ---------------------------------------------------------------------------
# Edge-case / robustness tests
# ---------------------------------------------------------------------------


def test_empty_directory(tmp_path: Path):
    """search_then_compress on an empty directory should return gracefully."""
    result = search_then_compress(str(tmp_path), "authentication")
    assert result.files_scanned == 0
    assert result.files_matched == 0
    assert result.naive_all_tokens == 0
    assert result.search_compress_tokens == 0
    assert "scan" in result.stages


def test_nonexistent_directory():
    """search_then_compress on a non-existent path should return gracefully."""
    result = search_then_compress("/nonexistent/path/that/does/not/exist", "query")
    assert result.files_scanned == 0
    assert result.search_compress_tokens == 0


def test_glob_fallback_works():
    """With tensor-grep patched as unavailable, glob fallback should still find files."""
    with patch("src.search_compress_pipeline.tg_available", return_value=False):
        result = search_then_compress(str(CODE_CORPUS), "authentication token JWT")

    assert result.search_method == "glob_fallback"
    assert result.files_matched > 0
    assert "auth.py" in [os.path.basename(p) for p in result.matched_files]


def test_compression_ratio_reasonable():
    """compression_ratio for matched files should be between 1.0 and 50.0."""
    result = search_then_compress(str(CODE_CORPUS), "authentication token JWT")
    if result.total_original_tokens > 0:
        assert (
            1.0 <= result.compression_ratio <= 50.0
        ), f"compression_ratio={result.compression_ratio} out of expected range"


def test_files_scanned_matches_directory():
    """files_scanned should equal the number of .py files found recursively."""
    import glob as glob_mod

    expected = len(sorted(glob_mod.glob(str(CODE_CORPUS / "**" / "*.py"), recursive=True)))
    result = search_then_compress(str(CODE_CORPUS), "authentication")
    assert (
        result.files_scanned == expected
    ), f"files_scanned={result.files_scanned} != expected={expected}"


# ---------------------------------------------------------------------------
# run_search_compress_benchmark integration test
# ---------------------------------------------------------------------------


def test_run_benchmark_function():
    """run_search_compress_benchmark should return a non-empty list of results."""
    from src.cli_benchmark.runner import run_search_compress_benchmark

    results = run_search_compress_benchmark(corpus_dir=CORPUS_DIR, verbose=False)
    assert isinstance(results, list)
    assert len(results) > 0, "Expected at least one result"
    for r in results:
        assert isinstance(r, SearchCompressResult)
        assert r.query  # each result should have a query
