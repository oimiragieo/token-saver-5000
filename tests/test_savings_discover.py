"""Tests for the savings discovery module."""

from __future__ import annotations

import os
import tempfile

from src.savings_discover import (
    COMPRESSION_ESTIMATES,
    DEFAULT_COMPRESSION_RATIO,
    MIN_COMPRESSIBLE_TOKENS,
    RECOMMENDED_THRESHOLD,
    ContextAnalyzer,
    DiscoveryReport,
    SavingsOpportunity,
    format_report,
)

# ---------------------------------------------------------------------------
# SavingsOpportunity
# ---------------------------------------------------------------------------


class TestSavingsOpportunity:
    def test_tokens_saved(self):
        opp = SavingsOpportunity(
            path="test.py",
            token_count=1000,
            estimated_compressed=150,
            estimated_savings_pct=85.0,
            recommendation="compress",
        )
        assert opp.tokens_saved == 850

    def test_to_dict_has_required_keys(self):
        opp = SavingsOpportunity(
            path="test.py",
            token_count=1000,
            estimated_compressed=150,
            estimated_savings_pct=85.0,
            recommendation="compress",
        )
        d = opp.to_dict()
        assert d["path"] == "test.py"
        assert d["token_count"] == 1000
        assert d["estimated_compressed"] == 150
        assert d["tokens_saved"] == 850
        assert d["estimated_savings_pct"] == 85.0
        assert d["recommendation"] == "compress"


# ---------------------------------------------------------------------------
# DiscoveryReport
# ---------------------------------------------------------------------------


class TestDiscoveryReport:
    def test_empty_report_to_dict(self):
        report = DiscoveryReport()
        d = report.to_dict()
        assert d["summary"]["files_scanned"] == 0
        assert d["summary"]["potential_savings_pct"] == 0.0
        assert d["opportunities"] == []

    def test_report_with_opportunities(self):
        opp = SavingsOpportunity("f.py", 1000, 150, 85.0, "compress")
        report = DiscoveryReport(
            opportunities=[opp],
            total_tokens_scanned=1000,
            total_potential_savings=850,
            files_scanned=1,
            files_compressible=1,
        )
        d = report.to_dict()
        assert d["summary"]["files_compressible"] == 1
        assert d["summary"]["total_potential_savings"] == 850
        assert len(d["opportunities"]) == 1


# ---------------------------------------------------------------------------
# ContextAnalyzer - text analysis
# ---------------------------------------------------------------------------


class TestAnalyzeText:
    def test_small_text_skipped(self):
        analyzer = ContextAnalyzer()
        # ~10 tokens
        opp = analyzer.analyze_text("hello world", label="tiny")
        assert opp.recommendation == "skip"
        assert opp.estimated_savings_pct == 0.0

    def test_medium_text_direct_read(self):
        analyzer = ContextAnalyzer()
        # 100-500 tokens: about 400-2000 chars
        text = "word " * 120  # ~120 tokens
        opp = analyzer.analyze_text(text, label="medium")
        assert opp.recommendation == "direct_read"

    def test_large_text_compress(self):
        analyzer = ContextAnalyzer()
        # >500 tokens: about 2000+ chars
        text = "word " * 600  # ~600 tokens
        opp = analyzer.analyze_text(text, label="large")
        assert opp.recommendation == "compress"
        assert opp.estimated_savings_pct > 0
        assert opp.tokens_saved > 0

    def test_file_ext_affects_ratio(self):
        analyzer = ContextAnalyzer()
        text = "x " * 600
        py_opp = analyzer.analyze_text(text, label="f.py", file_ext=".py")
        json_opp = analyzer.analyze_text(text, label="f.json", file_ext=".json")
        # JSON should have higher compression ratio
        assert json_opp.estimated_savings_pct > py_opp.estimated_savings_pct

    def test_unknown_ext_uses_default(self):
        analyzer = ContextAnalyzer()
        text = "x " * 600
        opp = analyzer.analyze_text(text, label="f.xyz", file_ext=".xyz")
        assert opp.recommendation == "compress"
        # Should use DEFAULT_COMPRESSION_RATIO
        assert opp.estimated_savings_pct > 0


# ---------------------------------------------------------------------------
# ContextAnalyzer - file analysis
# ---------------------------------------------------------------------------


class TestAnalyzeFile:
    def test_analyze_existing_file(self):
        analyzer = ContextAnalyzer()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write("x = 1\n" * 200)  # ~200 tokens
            f.flush()
            path = f.name

        try:
            opp = analyzer.analyze_file(path)
            assert opp is not None
            assert opp.token_count > 0
        finally:
            os.unlink(path)

    def test_analyze_nonexistent_file_returns_none(self):
        analyzer = ContextAnalyzer()
        opp = analyzer.analyze_file("/nonexistent/file.py")
        assert opp is None

    def test_analyze_binary_file_skipped(self):
        analyzer = ContextAnalyzer()
        with tempfile.NamedTemporaryFile(suffix=".pyc", delete=False) as f:
            f.write(b"\x00\x01\x02")
            path = f.name
        try:
            opp = analyzer.analyze_file(path)
            assert opp is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ContextAnalyzer - directory scan
# ---------------------------------------------------------------------------


class TestScanDirectory:
    def test_scan_empty_dir(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            report = analyzer.scan_directory(tmpdir)
            assert report.files_scanned == 0
            assert report.files_compressible == 0

    def test_scan_dir_with_large_files(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large .py file
            path = os.path.join(tmpdir, "big.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("def foo():\n    pass\n" * 200)

            report = analyzer.scan_directory(tmpdir)
            assert report.files_scanned >= 1
            assert report.files_compressible >= 1
            assert report.total_potential_savings > 0

    def test_scan_dir_with_small_files(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tiny.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("x = 1\n")

            report = analyzer.scan_directory(tmpdir)
            assert report.files_scanned >= 1
            assert report.files_compressible == 0

    def test_scan_skips_pycache(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "__pycache__")
            os.makedirs(cache_dir)
            path = os.path.join(cache_dir, "module.cpython-312.pyc")
            with open(path, "wb") as f:
                f.write(b"\x00" * 1000)

            report = analyzer.scan_directory(tmpdir)
            assert report.files_scanned == 0

    def test_scan_max_files_limit(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(10):
                path = os.path.join(tmpdir, f"file_{i}.py")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("x = 1\n" * 200)

            report = analyzer.scan_directory(tmpdir, max_files=3)
            assert report.files_scanned <= 3

    def test_scan_sorted_by_savings(self):
        analyzer = ContextAnalyzer()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small file
            with open(os.path.join(tmpdir, "small.py"), "w", encoding="utf-8") as f:
                f.write("x = 1\n" * 200)
            # Large file
            with open(os.path.join(tmpdir, "large.py"), "w", encoding="utf-8") as f:
                f.write("def func():\n    return True\n" * 500)

            report = analyzer.scan_directory(tmpdir)
            if len(report.opportunities) >= 2:
                assert report.opportunities[0].tokens_saved >= report.opportunities[1].tokens_saved

    def test_scan_nonexistent_dir(self):
        analyzer = ContextAnalyzer()
        report = analyzer.scan_directory("/nonexistent/dir")
        assert report.files_scanned == 0


# ---------------------------------------------------------------------------
# ContextAnalyzer - analyze_items
# ---------------------------------------------------------------------------


class TestAnalyzeItems:
    def test_analyze_items_list(self):
        analyzer = ContextAnalyzer()
        items = [
            {"text": "x " * 600, "label": "big.py", "file_ext": ".py"},
            {"text": "y " * 10, "label": "tiny.txt", "file_ext": ".txt"},
        ]
        report = analyzer.analyze_items(items)
        assert report.files_scanned == 2
        assert report.files_compressible == 1

    def test_analyze_items_empty(self):
        analyzer = ContextAnalyzer()
        report = analyzer.analyze_items([])
        assert report.files_scanned == 0


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_format_empty_report(self):
        report = DiscoveryReport()
        text = format_report(report)
        assert "Files scanned: 0" in text
        assert "Potential savings: 0 tokens" in text

    def test_format_report_with_opportunities(self):
        opp = SavingsOpportunity("big.py", 1000, 150, 85.0, "compress")
        report = DiscoveryReport(
            opportunities=[opp],
            total_tokens_scanned=1000,
            total_potential_savings=850,
            files_scanned=1,
            files_compressible=1,
        )
        text = format_report(report)
        assert "big.py" in text
        assert "85%" in text
        assert "850" in text

    def test_format_report_respects_top_n(self):
        opps = [SavingsOpportunity(f"file_{i}.py", 1000, 150, 85.0, "compress") for i in range(10)]
        report = DiscoveryReport(
            opportunities=opps,
            total_tokens_scanned=10000,
            total_potential_savings=8500,
            files_scanned=10,
            files_compressible=10,
        )
        text = format_report(report, top_n=3)
        assert "Top 3" in text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_compression_estimates_has_common_extensions(self):
        for ext in [".py", ".js", ".json", ".md", ".yaml"]:
            assert ext in COMPRESSION_ESTIMATES

    def test_thresholds_are_ordered(self):
        assert MIN_COMPRESSIBLE_TOKENS < RECOMMENDED_THRESHOLD

    def test_default_ratio_is_reasonable(self):
        assert 0.5 < DEFAULT_COMPRESSION_RATIO < 1.0
