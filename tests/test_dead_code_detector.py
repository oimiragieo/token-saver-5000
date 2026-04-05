"""Tests for dead_code_detector.py -- import-graph-based dead file detection.

Uses the real benchmark corpus at benchmarks/corpus/code/ as the primary
test fixture (10 Python files + tests/ sub-directory).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.dead_code_detector import (
    DeadCodeReport,
    detect_dead_files,
    _extract_imports,
    _is_entry_point,
    _is_never_dead,
    DEFAULT_ENTRY_PATTERNS,
)

CORPUS_DIR = Path(__file__).parent.parent / "benchmarks" / "corpus" / "code"

# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def test_extract_imports_from_statement():
    code = "import os\nimport sys\n"
    result = _extract_imports(code)
    assert "os" in result
    assert "sys" in result


def test_extract_from_imports():
    code = "from pathlib import Path\nfrom typing import Any\n"
    result = _extract_imports(code)
    assert "pathlib" in result
    assert "typing" in result


def test_is_entry_point_server():
    assert _is_entry_point("/project/server.py", DEFAULT_ENTRY_PATTERNS)


def test_is_entry_point_init():
    assert _is_entry_point("/pkg/__init__.py", DEFAULT_ENTRY_PATTERNS)


def test_is_never_dead_test_prefix():
    assert _is_never_dead("/project/test_auth.py")


def test_is_never_dead_conftest():
    assert _is_never_dead("/project/conftest.py")


# ---------------------------------------------------------------------------
# Corpus-based detection
# ---------------------------------------------------------------------------


def test_detect_report_fields():
    """All DeadCodeReport fields must be populated after a run."""
    report = detect_dead_files(str(CORPUS_DIR))
    assert isinstance(report, DeadCodeReport)
    assert report.total_files > 0
    assert isinstance(report.dead_files, list)
    assert isinstance(report.live_files, list)
    assert isinstance(report.entry_points, list)
    assert report.dead_file_count == len(report.dead_files)
    assert report.live_file_count == len(report.live_files)
    assert report.dead_file_count + report.live_file_count == report.total_files


def test_detect_test_files_live():
    """Files matching test_ prefix should never be marked dead."""
    report = detect_dead_files(str(CORPUS_DIR))
    dead_basenames = {os.path.basename(f) for f in report.dead_files}
    for name in dead_basenames:
        assert not name.startswith("test_"), f"test file marked dead: {name}"


def test_detect_entry_points_live():
    """__init__.py and conftest.py should not appear in dead_files."""
    report = detect_dead_files(str(CORPUS_DIR))
    dead_basenames = {os.path.basename(f) for f in report.dead_files}
    assert "__init__.py" not in dead_basenames
    assert "conftest.py" not in dead_basenames


def test_detect_tokens_saved_non_negative():
    """tokens_saved must be >= 0 (zero when no dead files exist)."""
    report = detect_dead_files(str(CORPUS_DIR))
    assert report.tokens_saved >= 0


def test_detect_tokens_saved_positive_when_dead():
    """tokens_saved should be > 0 whenever dead files are found."""
    report = detect_dead_files(str(CORPUS_DIR))
    if report.dead_file_count > 0:
        assert report.tokens_saved > 0


def test_detect_empty_directory():
    """Running on an empty directory returns a zeroed report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        report = detect_dead_files(tmpdir)
    assert report.total_files == 0
    assert report.dead_file_count == 0
    assert report.live_file_count == 0
    assert report.tokens_saved == 0


def test_detect_single_file_is_live():
    """A single standalone file with no imports is classified as live (entry-point heuristic)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "app.py"
        p.write_text("x = 1\n", encoding="utf-8")
        report = detect_dead_files(tmpdir)
    # app.py matches "app.py" in DEFAULT_ENTRY_PATTERNS → live
    assert report.total_files == 1
    assert report.dead_file_count == 0


def test_detect_imported_files_live():
    """A file explicitly imported by another should be in live_files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        lib = Path(tmpdir) / "lib.py"
        lib.write_text("VALUE = 42\n", encoding="utf-8")
        main = Path(tmpdir) / "main.py"
        main.write_text("import lib\nprint(lib.VALUE)\n", encoding="utf-8")
        report = detect_dead_files(tmpdir)
    live_basenames = {os.path.basename(f) for f in report.live_files}
    assert "lib.py" in live_basenames
    assert "main.py" in live_basenames


def test_detect_unimported_file_is_dead():
    """A file not imported by any other and not an entry point should be dead."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orphan = Path(tmpdir) / "orphan.py"
        orphan.write_text("# this module is never used\ndef unused(): pass\n", encoding="utf-8")
        main = Path(tmpdir) / "main.py"
        main.write_text("print('hello')\n", encoding="utf-8")
        report = detect_dead_files(tmpdir)
    dead_basenames = {os.path.basename(f) for f in report.dead_files}
    assert "orphan.py" in dead_basenames


def test_detect_handles_missing_files():
    """Passing a files list with non-existent paths should not raise."""
    report = detect_dead_files(str(CORPUS_DIR), files=["/nonexistent/path.py"])
    # File not readable -> graceful skip, total_files = 1, no crash
    assert report.total_files == 1


def test_detect_all_entry_patterns_live():
    """When all files match entry patterns they are all classified as live."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["main.py", "app.py", "server.py"]:
            (Path(tmpdir) / name).write_text("x = 1\n", encoding="utf-8")
        report = detect_dead_files(tmpdir)
    assert report.dead_file_count == 0
    assert report.live_file_count == 3
