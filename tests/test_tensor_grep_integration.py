"""
Tests for tensor_grep_integration.py

All subprocess calls are mocked so that tests run without tensor-grep installed.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

from src.tensor_grep_integration import (
    ASTSearchResult,
    CodeSearchResult,
    RepoMapResult,
    ast_search,
    code_search,
    get_repo_map,
    is_available,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completed_process(stdout: str, returncode: int = 0) -> MagicMock:
    cp = MagicMock()
    cp.stdout = stdout
    cp.returncode = returncode
    return cp


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
# code_search
# ---------------------------------------------------------------------------


class TestCodeSearch:
    def test_code_search_not_available(self) -> None:
        """Returns CodeSearchResult with available=False when tg is absent."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = code_search("def main", "/repo")

        assert isinstance(result, CodeSearchResult)
        assert result.available is False
        assert result.matches == []

    def test_code_search_parses_matches(self) -> None:
        """Parses matches list and total_matches from JSON output."""
        sample = json.dumps(
            {
                "matches": [
                    {"file": "src/main.py", "line": 10, "text": "def main():"},
                ],
                "total_matches": 1,
            }
        )
        cp = _make_completed_process(sample)

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp),
        ):
            result = code_search("def main", "/repo")

        assert result.available is True
        assert result.total_matches == 1
        assert result.matches[0]["file"] == "src/main.py"

    def test_code_search_with_index_flag(self) -> None:
        """use_index=True should add --index to the subprocess command."""
        cp = _make_completed_process(json.dumps({"matches": [], "total_matches": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            code_search("foo", "/repo", use_index=True)

        cmd = mock_run.call_args[0][0]
        assert "--index" in cmd

    def test_code_search_without_index_flag(self) -> None:
        """use_index=False should NOT add --index to the subprocess command."""
        cp = _make_completed_process(json.dumps({"matches": [], "total_matches": 0}))

        with (
            patch("src.tensor_grep_integration.shutil.which", return_value="/usr/bin/tg"),
            patch("src.tensor_grep_integration.subprocess.run", return_value=cp) as mock_run,
        ):
            code_search("foo", "/repo", use_index=False)

        cmd = mock_run.call_args[0][0]
        assert "--index" not in cmd

    def test_code_search_returns_pattern(self) -> None:
        """The pattern field of the result should match the input pattern."""
        with patch("src.tensor_grep_integration.shutil.which", return_value=None):
            result = code_search("my_pattern", "/repo")

        assert result.pattern == "my_pattern"


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
