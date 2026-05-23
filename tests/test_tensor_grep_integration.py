"""
Tests for tensor_grep_integration.py

All subprocess calls are mocked so that tests run without tensor-grep installed.
"""

from __future__ import annotations

import io
import json
import subprocess
from unittest.mock import MagicMock, patch

from src.tensor_grep_integration import (
    ASTSearchResult,
    CodeSearchResult,
    ContextRenderResult,
    RepoMapResult,
    ScanFinding,
    ScanResult,
    ast_search,
    code_search,
    get_context_render,
    get_repo_map,
    is_available,
    scan_ruleset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed_process(stdout: str, returncode: int = 0) -> MagicMock:
    cp = MagicMock()
    cp.stdout = stdout
    cp.returncode = returncode
    return cp


def _make_popen(lines: list[str], returncode: int = 0) -> MagicMock:
    """Return a mock Popen whose stdout is a line-iterable StringIO."""
    proc = MagicMock()
    proc.stdout = io.StringIO("\n".join(lines) + ("\n" if lines else ""))
    proc.returncode = returncode
    proc.wait = MagicMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_is_available_with_tg_installed(self) -> None:
        with patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"):
            assert is_available() is True

    def test_is_available_without_tg(self) -> None:
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            assert is_available() is False


# ---------------------------------------------------------------------------
# get_repo_map
# ---------------------------------------------------------------------------


class TestGetRepoMap:
    def test_repo_map_not_available(self) -> None:
        """Returns RepoMapResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = get_repo_map("/some/dir")

        assert isinstance(result, RepoMapResult)
        assert result.available is False
        assert result.files == []
        assert result.symbols == []

    def test_repo_map_parses_json(self) -> None:
        """Parses JSON output into files and symbols lists."""
        sample = json.dumps(
            {
                "files": ["src/main.py", "src/utils.py"],
                "symbols": [{"name": "main", "file": "src/main.py"}],
            }
        )
        cp = _make_completed_process(sample)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = get_repo_map("/repo")

        assert result.available is True
        assert result.files == ["src/main.py", "src/utils.py"]
        assert len(result.symbols) == 1
        assert result.symbols[0]["name"] == "main"

    def test_repo_map_timeout_returns_graceful_result(self) -> None:
        """TimeoutExpired should yield a non-crashing result with available=True."""
        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch(
                "src.tensor_grep_integration.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tg", timeout=30),
            ),
        ):
            result = get_repo_map("/repo")

        assert result.available is True
        assert result.files == []


# ---------------------------------------------------------------------------
# code_search  (B2 fix: now uses subprocess.Popen + --ndjson)
# ---------------------------------------------------------------------------


class TestCodeSearch:
    def test_code_search_not_available(self) -> None:
        """Returns CodeSearchResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = code_search("def main", "/repo")

        assert isinstance(result, CodeSearchResult)
        assert result.available is False
        assert result.matches == []

    def test_code_search_uses_ndjson_flag(self) -> None:
        """B2 fix: code_search must use --ndjson (not --json)."""
        proc = _make_popen([json.dumps({"file": "src/main.py", "line": 10, "text": "def main():"})])

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc) as mock_popen,
        ):
            code_search("def main", "/repo")

        cmd = mock_popen.call_args[0][0]
        assert "--ndjson" in cmd
        assert "--json" not in cmd

    def test_code_search_parses_ndjson_matches(self) -> None:
        """Parses NDJSON stream of match objects."""
        match_obj = {"file": "src/main.py", "line": 10, "text": "def main():"}
        proc = _make_popen([json.dumps(match_obj)])

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc),
        ):
            result = code_search("def main", "/repo")

        assert result.available is True
        assert len(result.matches) == 1
        assert result.matches[0]["file"] == "src/main.py"

    def test_code_search_parses_multiple_ndjson_lines(self) -> None:
        """Each NDJSON line is a separate match object; all are collected."""
        lines = [
            json.dumps({"file": "a.py", "line": 1}),
            json.dumps({"file": "b.py", "line": 2}),
            json.dumps({"file": "c.py", "line": 3}),
        ]
        proc = _make_popen(lines)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc),
        ):
            result = code_search("pattern", "/repo")

        assert result.available is True
        assert len(result.matches) == 3

    def test_code_search_with_index_flag(self) -> None:
        """use_index=True should add --index to the subprocess command."""
        proc = _make_popen([])

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc) as mock_popen,
        ):
            code_search("foo", "/repo", use_index=True)

        cmd = mock_popen.call_args[0][0]
        assert "--index" in cmd

    def test_code_search_without_index_flag(self) -> None:
        """use_index=False should NOT add --index to the subprocess command."""
        proc = _make_popen([])

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc) as mock_popen,
        ):
            code_search("foo", "/repo", use_index=False)

        cmd = mock_popen.call_args[0][0]
        assert "--index" not in cmd

    def test_code_search_returns_pattern(self) -> None:
        """The pattern field of the result should match the input pattern."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = code_search("my_pattern", "/repo")

        assert result.pattern == "my_pattern"

    def test_code_search_skips_invalid_json_lines(self) -> None:
        """Malformed NDJSON lines are silently skipped."""
        lines = [
            "not-json-at-all",
            json.dumps({"file": "ok.py", "line": 5}),
            "{broken",
        ]
        proc = _make_popen(lines)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc),
        ):
            result = code_search("pattern", "/repo")

        assert result.available is True
        assert len(result.matches) == 1
        assert result.matches[0]["file"] == "ok.py"

    def test_code_search_timeout_returns_graceful_result(self) -> None:
        """TimeoutExpired yields available=True with empty matches.

        proc.wait is called twice:
          1. proc.wait(timeout=...)  — raises TimeoutExpired
          2. proc.wait()             — cleanup call after proc.kill(); returns 0
        """
        proc = _make_popen([])
        proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="tg", timeout=30), 0]

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.Popen", return_value=proc),
        ):
            result = code_search("pattern", "/repo")

        assert result.available is True
        assert result.matches == []
        proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# ast_search
# ---------------------------------------------------------------------------


class TestASTSearch:
    def test_ast_search_not_available(self) -> None:
        """Returns ASTSearchResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = ast_search("class Foo", "/repo")

        assert isinstance(result, ASTSearchResult)
        assert result.available is False

    def test_ast_search_parses_matches(self) -> None:
        """Parses matches list and total_matches from JSON output."""
        sample = json.dumps(
            {
                "matches": [
                    {"file": "src/models.py", "node": "class Foo"},
                ],
                "total_matches": 1,
            }
        )
        cp = _make_completed_process(sample)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = ast_search("class Foo", "/repo")

        assert result.available is True
        assert result.total_matches == 1
        assert result.matches[0]["node"] == "class Foo"

    def test_ast_search_with_lang(self) -> None:
        """Providing lang should add --lang <lang> to the subprocess command."""
        cp = _make_completed_process(json.dumps({"matches": [], "total_matches": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            ast_search("class Foo", "/repo", lang="python")

        cmd = mock_run.call_args[0][0]
        assert "--lang" in cmd
        assert "python" in cmd

    def test_ast_search_without_lang(self) -> None:
        """Omitting lang should not add --lang to the command."""
        cp = _make_completed_process(json.dumps({"matches": [], "total_matches": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            ast_search("class Foo", "/repo")

        cmd = mock_run.call_args[0][0]
        assert "--lang" not in cmd


# ---------------------------------------------------------------------------
# get_context_render
# ---------------------------------------------------------------------------


class TestGetContextRender:
    def test_context_render_not_available(self) -> None:
        """Returns ContextRenderResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = get_context_render("/repo", "how does auth work")

        assert isinstance(result, ContextRenderResult)
        assert result.available is False
        assert result.ranked_files == []
        assert result.render == ""

    def test_context_render_parses_json(self) -> None:
        """Parses ranked_files and render from JSON output."""
        payload = {
            "ranked_files": ["src/auth.py", "src/middleware.py"],
            "render": "## Auth\n\ndef verify_token(...):\n    ...",
            "query": "how does auth work",
        }
        cp = _make_completed_process(json.dumps(payload))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = get_context_render("/repo", "how does auth work")

        assert result.available is True
        assert result.ranked_files == ["src/auth.py", "src/middleware.py"]
        assert "verify_token" in result.render
        assert result.query == "how does auth work"

    def test_context_render_includes_query_in_command(self) -> None:
        """The query string is passed to tg context-render."""
        cp = _make_completed_process(json.dumps({}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            get_context_render("/repo", "find the auth logic")

        cmd = mock_run.call_args[0][0]
        assert "context-render" in cmd
        assert "--query" in cmd
        assert "find the auth logic" in cmd

    def test_context_render_passes_render_profile(self) -> None:
        """render_profile parameter is passed via --render-profile."""
        cp = _make_completed_process(json.dumps({}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            get_context_render("/repo", "query", render_profile="compact")

        cmd = mock_run.call_args[0][0]
        assert "--render-profile" in cmd
        assert "compact" in cmd

    def test_context_render_passes_max_files(self) -> None:
        """max_files parameter is passed via --max-files."""
        cp = _make_completed_process(json.dumps({}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            get_context_render("/repo", "query", max_files=5)

        cmd = mock_run.call_args[0][0]
        assert "--max-files" in cmd
        assert "5" in cmd

    def test_context_render_no_optimize_context_flag(self) -> None:
        """optimize_context=False adds --no-optimize-context to the command."""
        cp = _make_completed_process(json.dumps({}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            get_context_render("/repo", "query", optimize_context=False)

        cmd = mock_run.call_args[0][0]
        assert "--no-optimize-context" in cmd

    def test_context_render_timeout_returns_graceful_result(self) -> None:
        """TimeoutExpired yields available=True with empty result."""
        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch(
                "src.tensor_grep_integration.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tg", timeout=60),
            ),
        ):
            result = get_context_render("/repo", "query")

        assert result.available is True
        assert result.ranked_files == []
        assert result.render == ""

    def test_context_render_bad_json_returns_graceful_result(self) -> None:
        """Malformed JSON from tg returns available=True with empty fields."""
        cp = _make_completed_process("this is not json")

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = get_context_render("/repo", "query")

        assert result.available is True
        assert result.ranked_files == []


# ---------------------------------------------------------------------------
# scan_ruleset
# ---------------------------------------------------------------------------


class TestScanRuleset:
    def test_scan_not_available(self) -> None:
        """Returns ScanResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = scan_ruleset("/repo", "secrets")

        assert isinstance(result, ScanResult)
        assert result.available is False
        assert result.findings == []

    def test_scan_rc0_clean(self) -> None:
        """rc=0 means scan completed with no findings."""
        payload = {"findings": [], "total_findings": 0}
        cp = _make_completed_process(json.dumps(payload), returncode=0)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = scan_ruleset("/repo", "secrets")

        assert result.available is True
        assert result.findings == []
        assert result.total_findings == 0

    def test_scan_rc1_findings_present(self) -> None:
        """rc=1 means findings exist — NOT a subprocess error."""
        finding = {
            "rule_id": "no-hardcoded-secrets",
            "severity": "high",
            "path": "config.py",
            "line": 12,
            "message": "Hardcoded API key detected",
            "fingerprint": "abc123",
            "evidence": None,
        }
        payload = {"findings": [finding], "total_findings": 1}
        cp = _make_completed_process(json.dumps(payload), returncode=1)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = scan_ruleset("/repo", "secrets")

        assert result.available is True
        assert result.total_findings == 1
        assert len(result.findings) == 1
        f = result.findings[0]
        assert isinstance(f, ScanFinding)
        assert f.rule_id == "no-hardcoded-secrets"
        assert f.severity == "high"
        assert f.path == "config.py"
        assert f.line == 12
        assert f.message == "Hardcoded API key detected"

    def test_scan_rc2_is_error(self) -> None:
        """rc>1 is a real subprocess error; available=True but findings empty."""
        cp = _make_completed_process("tg: ruleset not found", returncode=2)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = scan_ruleset("/repo", "nonexistent-ruleset")

        assert result.available is True
        assert result.findings == []
        assert result.total_findings == 0

    def test_scan_passes_ruleset_flag(self) -> None:
        """The --ruleset flag is passed to tg scan."""
        cp = _make_completed_process(json.dumps({"findings": [], "total_findings": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            scan_ruleset("/repo", "owasp-top-10")

        cmd = mock_run.call_args[0][0]
        assert "scan" in cmd
        assert "--ruleset" in cmd
        assert "owasp-top-10" in cmd

    def test_scan_passes_language_flag(self) -> None:
        """language parameter adds --language <lang> to the command."""
        cp = _make_completed_process(json.dumps({"findings": [], "total_findings": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            scan_ruleset("/repo", "secrets", language="python")

        cmd = mock_run.call_args[0][0]
        assert "--language" in cmd
        assert "python" in cmd

    def test_scan_without_language_flag(self) -> None:
        """Omitting language should not add --language to the command."""
        cp = _make_completed_process(json.dumps({"findings": [], "total_findings": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            scan_ruleset("/repo", "secrets")

        cmd = mock_run.call_args[0][0]
        assert "--language" not in cmd

    def test_scan_include_evidence_flag(self) -> None:
        """include_evidence=True adds --include-evidence-snippets to the command."""
        cp = _make_completed_process(json.dumps({"findings": [], "total_findings": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            scan_ruleset("/repo", "secrets", include_evidence=True)

        cmd = mock_run.call_args[0][0]
        assert "--include-evidence-snippets" in cmd

    def test_scan_timeout_returns_graceful_result(self) -> None:
        """TimeoutExpired yields available=True with empty findings."""
        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch(
                "src.tensor_grep_integration.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="tg", timeout=60),
            ),
        ):
            result = scan_ruleset("/repo", "secrets")

        assert result.available is True
        assert result.findings == []
