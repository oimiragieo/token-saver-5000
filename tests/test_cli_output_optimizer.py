"""
Tests for CLIOutputOptimizer (TDD — written before implementation).

Covers:
- Command detection heuristics for all supported output types
- All 10 strategy implementations
- Integration: chaining, passthrough, compression_pct, command_hint override
- Edge cases: empty input, short input, unknown type
"""

from __future__ import annotations

import json

import pytest

from src.cli_output_optimizer import CLIOutputOptimizer, FilterResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GIT_DIFF_FIXTURE = """diff --git a/src/main.py b/src/main.py
index abc1234..def5678 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,8 @@ def main():
     print("hello")
+    print("world")
+    print("!")
diff --git a/src/utils.py b/src/utils.py
index 111..222 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,3 +1,2 @@
 def helper():
-    pass
"""

GIT_STATUS_FIXTURE = """On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)

        modified:   src/main.py
        modified:   src/utils.py
        modified:   tests/test_main.py

Untracked files:
  (use "git add <file>..." to include in what will be committed)

        new_feature.py
        scratch.txt
"""

PYTEST_FIXTURE = """tests/test_auth.py::test_login PASSED
tests/test_auth.py::test_logout PASSED
tests/test_auth.py::test_invalid_token FAILED
tests/test_api.py::test_get_users PASSED
tests/test_api.py::test_create_user PASSED
tests/test_api.py::test_delete_user FAILED
tests/test_api.py::test_update_user PASSED

========================= 2 failed, 5 passed =========================
"""

NPM_INSTALL_FIXTURE = """npm warn deprecated inflight@1.0.6
npm warn deprecated glob@7.2.3
\u2838\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2839 \u280f idealTree: timing idealTree
added 247 packages, and audited 248 packages in 12s

38 packages are looking for funding
  run `npm fund` for details

found 0 vulnerabilities
"""

ANSI_FIXTURE = (
    "\x1b[32m\u2713\x1b[0m test passed\n"
    "\x1b[31m\u2717\x1b[0m test failed\n"
    "\x1b[33mwarning\x1b[0m: something"
)

LINT_FIXTURE = """src/main.py:10:1: E501 line too long (120 > 100 characters)
src/main.py:15:1: E501 line too long (115 > 100 characters)
src/main.py:20:5: W293 whitespace before ':'
src/utils.py:5:1: E501 line too long (105 > 100 characters)
src/utils.py:12:1: F401 'os' imported but unused

Found 5 errors in 2 files
"""

JSON_FIXTURE = json.dumps(
    {
        "status": "ok",
        "items": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}, {"id": 3}],
        "count": 3,
        "metadata": {"version": "1.0"},
    }
)

PROGRESS_FIXTURE = (
    "Downloading model weights\n"
    "  10%\n"
    "  50%\n"
    "  \u280b loading\n"
    "  100%\n"
    "Download complete\n"
)

TREE_FIXTURE = (
    "src/handlers/ace_handlers.py\n"
    "src/handlers/afm_handlers.py\n"
    "src/handlers/compression_handlers.py\n"
    "src/utils/helpers.py\n"
    "src/utils/validators.py\n"
)

LOG_REPEATED_FIXTURE = "\n".join(["[INFO] heartbeat ok"] * 15 + ["[ERROR] connection lost"])


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestDetectCommand:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_detect_git_diff(self):
        assert self.opt.detect_command(GIT_DIFF_FIXTURE) == "git_diff"

    def test_detect_git_status(self):
        assert self.opt.detect_command(GIT_STATUS_FIXTURE) == "git_status"

    def test_detect_pytest_output(self):
        assert self.opt.detect_command(PYTEST_FIXTURE) == "test_output"

    def test_detect_npm_install(self):
        assert self.opt.detect_command(NPM_INSTALL_FIXTURE) == "install_output"

    def test_detect_ansi_output(self):
        assert self.opt.detect_command(ANSI_FIXTURE) == "ansi_output"

    def test_detect_lint_output(self):
        assert self.opt.detect_command(LINT_FIXTURE) == "lint_output"

    def test_detect_json_output(self):
        assert self.opt.detect_command(JSON_FIXTURE) == "json_output"

    def test_detect_log_output_repeated_lines(self):
        assert self.opt.detect_command(LOG_REPEATED_FIXTURE) == "log_output"

    def test_detect_unknown_plain_text(self):
        plain = "Hello world.\nThis is a simple sentence.\nNothing special here.\n"
        assert self.opt.detect_command(plain) == "unknown"


# ---------------------------------------------------------------------------
# Strategy: ANSI stripping
# ---------------------------------------------------------------------------


class TestAnsiStrip:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_ansi_strip_removes_color_codes(self):
        result = self.opt.filter(ANSI_FIXTURE)
        assert "\x1b[" not in result.filtered_text
        assert "\x1b[32m" not in result.filtered_text
        assert "\x1b[0m" not in result.filtered_text

    def test_ansi_strip_preserves_text(self):
        result = self.opt.filter(ANSI_FIXTURE)
        assert "test passed" in result.filtered_text
        assert "test failed" in result.filtered_text
        assert "warning" in result.filtered_text

    def test_ansi_strip_removes_carriage_returns(self):
        text_with_cr = "line one\r\nline two\r\n"
        result = self.opt._strip_ansi(text_with_cr)
        assert "\r" not in result


# ---------------------------------------------------------------------------
# Strategy: Git diff stats
# ---------------------------------------------------------------------------


class TestGitDiffStats:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_git_diff_stats_extracts_summary(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        text = result.filtered_text
        # Must mention file count and insertion/deletion counts
        assert "files changed" in text or "file changed" in text

    def test_git_diff_stats_lists_files(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        text = result.filtered_text
        assert "main.py" in text
        assert "utils.py" in text

    def test_git_diff_stats_insertion_count(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        text = result.filtered_text
        # Two + lines in main.py hunk
        assert "+2" in text or "2 insertion" in text or "+2 insertion" in text

    def test_git_diff_stats_deletion_count(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        text = result.filtered_text
        # One - line in utils.py hunk
        assert "-1" in text or "1 deletion" in text or "-1 deletion" in text


# ---------------------------------------------------------------------------
# Strategy: Git status compact
# ---------------------------------------------------------------------------


class TestGitStatusCompact:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_git_status_groups_by_type(self):
        result = self.opt.filter(GIT_STATUS_FIXTURE)
        text = result.filtered_text
        # Modified group should appear
        assert "Modified" in text or "modified" in text

    def test_git_status_lists_modified_files(self):
        result = self.opt.filter(GIT_STATUS_FIXTURE)
        text = result.filtered_text
        assert "main.py" in text

    def test_git_status_shorter_than_original(self):
        result = self.opt.filter(GIT_STATUS_FIXTURE)
        assert len(result.filtered_text) < len(GIT_STATUS_FIXTURE)


# ---------------------------------------------------------------------------
# Strategy: Test failure focus
# ---------------------------------------------------------------------------


class TestFailureFocus:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_failure_focus_keeps_failures_only(self):
        result = self.opt.filter(PYTEST_FIXTURE)
        text = result.filtered_text
        assert "test_invalid_token" in text
        assert "test_delete_user" in text

    def test_failure_focus_preserves_summary_line(self):
        result = self.opt.filter(PYTEST_FIXTURE)
        text = result.filtered_text
        # The "2 failed, 5 passed" summary must be in the output
        assert "failed" in text
        assert "passed" in text

    def test_failure_focus_all_pass_returns_summary(self):
        all_pass = (
            "tests/test_foo.py::test_a PASSED\n"
            "tests/test_foo.py::test_b PASSED\n\n"
            "========================= 2 passed =========================\n"
        )
        result = self.opt.filter(all_pass, command_hint="test_output")
        text = result.filtered_text
        # No failures, just the summary
        assert "passed" in text
        # PASSED lines may be omitted when there are no failures
        assert "FAILED" not in text

    def test_failure_focus_omits_passing_tests(self):
        result = self.opt.filter(PYTEST_FIXTURE)
        text = result.filtered_text
        # Passing-test lines should be stripped
        assert "test_login PASSED" not in text
        assert "test_logout PASSED" not in text


# ---------------------------------------------------------------------------
# Strategy: Install summary
# ---------------------------------------------------------------------------


class TestInstallSummary:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_install_summary_strips_progress(self):
        result = self.opt.filter(NPM_INSTALL_FIXTURE)
        text = result.filtered_text
        # Progress / idealTree lines should be gone
        assert "idealTree" not in text

    def test_install_summary_keeps_summary_line(self):
        result = self.opt.filter(NPM_INSTALL_FIXTURE)
        text = result.filtered_text
        assert "247" in text  # package count

    def test_install_summary_keeps_warnings(self):
        result = self.opt.filter(NPM_INSTALL_FIXTURE)
        text = result.filtered_text
        assert "deprecated" in text or "warn" in text


# ---------------------------------------------------------------------------
# Strategy: Lint grouping
# ---------------------------------------------------------------------------


class TestLintGroup:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_lint_group_counts_by_rule(self):
        result = self.opt.filter(LINT_FIXTURE)
        text = result.filtered_text
        # E501 appears 3 times
        assert "E501" in text
        assert "3" in text

    def test_lint_group_preserves_summary(self):
        result = self.opt.filter(LINT_FIXTURE)
        text = result.filtered_text
        assert "5" in text or "errors" in text

    def test_lint_group_includes_all_rules(self):
        result = self.opt.filter(LINT_FIXTURE)
        text = result.filtered_text
        assert "W293" in text
        assert "F401" in text


# ---------------------------------------------------------------------------
# Strategy: JSON structure
# ---------------------------------------------------------------------------


class TestJsonStructure:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_json_structure_extracts_keys(self):
        result = self.opt.filter(JSON_FIXTURE)
        text = result.filtered_text
        assert "status" in text
        assert "count" in text
        assert "metadata" in text

    def test_json_structure_array_shows_first_item(self):
        result = self.opt.filter(JSON_FIXTURE)
        text = result.filtered_text
        # Array should be summarized, not fully expanded
        assert "items" in text
        # Either shows "... and N more" or similar truncation
        assert "more" in text or "array" in text.lower() or "list" in text.lower()

    def test_json_structure_invalid_json_passes_through(self):
        bad = "{not valid json"
        result = self.opt.filter(bad, command_hint="json_output")
        # Should not crash; text passes through or is minimally processed
        assert result.filtered_text is not None

    def test_json_structure_array_root(self):
        arr = json.dumps([{"a": 1}, {"b": 2}, {"c": 3}])
        result = self.opt.filter(arr, command_hint="json_output")
        text = result.filtered_text
        assert text is not None


# ---------------------------------------------------------------------------
# Strategy: Log dedup
# ---------------------------------------------------------------------------


class TestLogDedup:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_log_dedup_collapses_repeated(self):
        result = self.opt.filter(LOG_REPEATED_FIXTURE)
        text = result.filtered_text
        # The 15 identical lines should be collapsed
        assert "repeated" in text or "15" in text or "×" in text or "x15" in text.lower()

    def test_log_dedup_preserves_unique(self):
        result = self.opt.filter(LOG_REPEATED_FIXTURE)
        text = result.filtered_text
        assert "connection lost" in text

    def test_log_dedup_shorter_than_original(self):
        result = self.opt.filter(LOG_REPEATED_FIXTURE)
        assert len(result.filtered_text) < len(LOG_REPEATED_FIXTURE)


# ---------------------------------------------------------------------------
# Strategy: Progress strip
# ---------------------------------------------------------------------------


class TestProgressStrip:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_progress_strip_removes_percentage_lines(self):
        result = self.opt.filter(PROGRESS_FIXTURE, command_hint="progress_output")
        text = result.filtered_text
        assert "10%" not in text
        assert "50%" not in text
        assert "100%" not in text

    def test_progress_strip_removes_spinner(self):
        spinner_text = "\u280b loading\n\u2819 loading\nDone\n"
        result = self.opt.filter(spinner_text, command_hint="progress_output")
        text = result.filtered_text
        assert "Done" in text
        assert "\u280b" not in text

    def test_progress_strip_keeps_non_progress_lines(self):
        result = self.opt.filter(PROGRESS_FIXTURE, command_hint="progress_output")
        text = result.filtered_text
        assert "Downloading model weights" in text
        assert "Download complete" in text


# ---------------------------------------------------------------------------
# Strategy: Tree compress
# ---------------------------------------------------------------------------


class TestTreeCompress:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_tree_compress_collapses_common_prefix(self):
        result = self.opt.filter(TREE_FIXTURE, command_hint="tree_output")
        text = result.filtered_text
        # Should collapse src/handlers/ files
        assert "src/handlers/" in text
        assert "2 files" in text or "3 files" in text

    def test_tree_compress_shorter_than_original(self):
        result = self.opt.filter(TREE_FIXTURE, command_hint="tree_output")
        assert len(result.filtered_text) <= len(TREE_FIXTURE)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestFilterIntegration:
    def setup_method(self):
        self.opt = CLIOutputOptimizer()

    def test_filter_empty_text(self):
        result = self.opt.filter("")
        assert result.filtered_text == ""
        assert result.original_lines == 0
        assert result.filtered_lines == 0
        assert result.compression_pct == 0.0

    def test_filter_short_text_passthrough(self):
        short = "hello world"
        result = self.opt.filter(short)
        assert result.filtered_text == short

    def test_filter_result_has_all_fields(self):
        result = self.opt.filter(PYTEST_FIXTURE)
        assert hasattr(result, "original_text")
        assert hasattr(result, "filtered_text")
        assert hasattr(result, "original_lines")
        assert hasattr(result, "filtered_lines")
        assert hasattr(result, "command_detected")
        assert hasattr(result, "strategy_applied")
        assert hasattr(result, "compression_pct")

    def test_filter_chained_ansi_then_strategy(self):
        # ANSI-wrapped pytest output: should strip ANSI first, then apply test strategy
        ansi_pytest = (
            "\x1b[32mtests/test_auth.py::test_login PASSED\x1b[0m\n"
            "\x1b[31mtests/test_auth.py::test_login FAILED\x1b[0m\n"
            "1 failed, 1 passed\n"
        )
        result = self.opt.filter(ansi_pytest)
        # ANSI codes must be gone
        assert "\x1b[" not in result.filtered_text

    def test_compression_pct_correct(self):
        result = self.opt.filter(PYTEST_FIXTURE)
        expected = (1.0 - result.filtered_lines / result.original_lines) * 100
        assert abs(result.compression_pct - expected) < 1.0

    def test_command_hint_overrides_detection(self):
        # NPM fixture detected as install_output normally
        # Force it through test_output strategy via hint
        result = self.opt.filter(NPM_INSTALL_FIXTURE, command_hint="test_output")
        assert result.strategy_applied == "_test_failure_focus"

    def test_unknown_returns_unchanged(self):
        plain = "This is plain text with no special structure at all.\n" * 3
        result = self.opt.filter(plain)
        assert result.strategy_applied == "passthrough" or result.filtered_text == plain

    def test_filter_result_is_filterresult_instance(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        assert isinstance(result, FilterResult)

    def test_original_text_preserved_in_result(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        assert result.original_text == GIT_DIFF_FIXTURE

    def test_line_counts_are_accurate(self):
        result = self.opt.filter(GIT_DIFF_FIXTURE)
        assert result.original_lines == len(GIT_DIFF_FIXTURE.splitlines())
        assert result.filtered_lines == len(result.filtered_text.splitlines())
