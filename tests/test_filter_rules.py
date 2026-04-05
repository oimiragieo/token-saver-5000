"""Tests for the custom filter rules DSL engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.filter_rules import (
    FilterRule,
    FilterRuleEngine,
    FilterRuleSet,
    InlineTest,
)

# ---------------------------------------------------------------------------
# FilterRule basics
# ---------------------------------------------------------------------------


class TestFilterRule:
    def test_matches_simple_command(self):
        rule = FilterRule(name="test", match_command="my-build")
        assert rule.matches("my-build") is True
        assert rule.matches("MY-BUILD") is True
        assert rule.matches("other-tool") is False

    def test_matches_regex_pattern(self):
        rule = FilterRule(name="test", match_command=r"npm\s+(run\s+)?build")
        assert rule.matches("npm run build") is True
        assert rule.matches("npm build") is True
        assert rule.matches("yarn build") is False

    def test_matches_empty_command_returns_false(self):
        rule = FilterRule(name="test", match_command="")
        assert rule.matches("anything") is False


# ---------------------------------------------------------------------------
# FilterRule pipeline stages
# ---------------------------------------------------------------------------


class TestFilterRulePipeline:
    def test_strip_ansi(self):
        rule = FilterRule(name="test", strip_ansi=True)
        text = "\x1b[32mOK\x1b[0m\n\x1b[31mERROR\x1b[0m"
        result = rule.apply(text)
        assert result == "OK\nERROR"

    def test_strip_lines_matching(self):
        rule = FilterRule(name="test", strip_lines_matching=[r"^Progress:", r"^\s*$"])
        text = "Progress: 50%\nERROR: build failed\n\nProgress: 100%"
        result = rule.apply(text)
        assert result == "ERROR: build failed"

    def test_keep_lines_matching(self):
        rule = FilterRule(name="test", keep_lines_matching=[r"^ERROR", r"^WARNING"])
        text = "INFO: starting\nERROR: bad\nDEBUG: details\nWARNING: slow"
        result = rule.apply(text)
        assert result == "ERROR: bad\nWARNING: slow"

    def test_truncate_lines_at(self):
        rule = FilterRule(name="test", truncate_lines_at=10)
        text = "short\nthis is a very long line that should be truncated"
        result = rule.apply(text)
        lines = result.splitlines()
        assert lines[0] == "short"
        assert lines[1] == "this is a "
        assert len(lines[1]) == 10

    def test_head_lines(self):
        rule = FilterRule(name="test", head_lines=2)
        text = "line1\nline2\nline3\nline4\nline5"
        result = rule.apply(text)
        assert result == "line1\nline2"

    def test_tail_lines(self):
        rule = FilterRule(name="test", tail_lines=2)
        text = "line1\nline2\nline3\nline4\nline5"
        result = rule.apply(text)
        assert result == "line4\nline5"

    def test_max_lines_with_gap(self):
        rule = FilterRule(name="test", max_lines=4)
        text = "\n".join(f"line{i}" for i in range(10))
        result = rule.apply(text)
        lines = result.splitlines()
        assert len(lines) == 5  # 2 head + 1 gap + 2 tail
        assert "omitted" in lines[2]
        assert lines[0] == "line0"
        assert lines[-1] == "line9"

    def test_on_empty_fallback(self):
        rule = FilterRule(
            name="test",
            keep_lines_matching=[r"^NEVER_MATCHES"],
            on_empty="[no relevant output]",
        )
        text = "some text\nmore text"
        result = rule.apply(text)
        assert result == "[no relevant output]"

    def test_on_empty_not_triggered_when_output_exists(self):
        rule = FilterRule(
            name="test",
            keep_lines_matching=[r"^ERROR"],
            on_empty="[no relevant output]",
        )
        text = "ERROR: something\nINFO: details"
        result = rule.apply(text)
        assert result == "ERROR: something"

    def test_full_pipeline(self):
        rule = FilterRule(
            name="build",
            strip_ansi=True,
            strip_lines_matching=[r"^Progress:"],
            keep_lines_matching=[r"^ERROR", r"^SUMMARY"],
            truncate_lines_at=50,
            head_lines=10,
            on_empty="[build] No relevant output",
        )
        text = (
            "\x1b[32mProgress: 50%\x1b[0m\n"
            "Progress: 75%\n"
            "ERROR: module not found\n"
            "DEBUG: stack trace here\n"
            "SUMMARY: 1 error, 0 warnings"
        )
        result = rule.apply(text)
        assert "ERROR: module not found" in result
        assert "SUMMARY:" in result
        assert "Progress" not in result


# ---------------------------------------------------------------------------
# FilterRuleSet
# ---------------------------------------------------------------------------


class TestFilterRuleSet:
    def test_find_matching_rule(self):
        rule1 = FilterRule(name="build", match_command="npm build")
        rule2 = FilterRule(name="test", match_command="pytest")
        rule_set = FilterRuleSet(rules={"build": rule1, "test": rule2})

        assert rule_set.find_matching_rule("npm build") == rule1
        assert rule_set.find_matching_rule("pytest tests/") == rule2
        assert rule_set.find_matching_rule("git diff") is None


# ---------------------------------------------------------------------------
# FilterRuleEngine - TOML loading
# ---------------------------------------------------------------------------


class TestFilterRuleEngineLoad:
    def test_load_project_rules_from_toml(self):
        toml_content = """
[filters.my_build]
description = "My custom build output"
match_command = "my-build-tool"
strip_ansi = true
strip_lines_matching = ["^Progress:"]
keep_lines_matching = ["^ERROR", "^WARNING"]
truncate_lines_at = 120
head_lines = 50
on_empty = "[build] No output"

[filters.my_test]
description = "My test runner"
match_command = "my-test"
tail_lines = 20
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / ".gotcontext.toml"
            toml_path.write_text(toml_content, encoding="utf-8")

            engine = FilterRuleEngine()
            rule_set = engine.load_project_rules(tmpdir)

            assert len(rule_set.rules) == 2
            assert "my_build" in rule_set.rules
            assert "my_test" in rule_set.rules

            build_rule = rule_set.rules["my_build"]
            assert build_rule.strip_ansi is True
            assert build_rule.truncate_lines_at == 120
            assert build_rule.head_lines == 50
            assert "^ERROR" in build_rule.keep_lines_matching
            assert build_rule.on_empty == "[build] No output"

    def test_load_nonexistent_project_returns_empty(self):
        engine = FilterRuleEngine()
        rule_set = engine.load_project_rules("/nonexistent/path")
        assert len(rule_set.rules) == 0

    def test_load_with_inline_tests(self):
        toml_content = """
[filters.my_build]
match_command = "my-build"
strip_lines_matching = ["^Progress:"]

[[tests.my_build]]
name = "Error case"
input = "Progress: 50%\\nERROR: build failed\\nProgress: 100%"
expected = "ERROR: build failed"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / ".gotcontext.toml"
            toml_path.write_text(toml_content, encoding="utf-8")

            engine = FilterRuleEngine()
            rule_set = engine.load_project_rules(tmpdir)

            assert len(rule_set.tests) == 1
            assert rule_set.tests[0].name == "Error case"
            assert rule_set.tests[0].rule_name == "my_build"

    def test_load_legacy_token_saver_toml(self):
        toml_content = """
[filters.legacy]
match_command = "legacy-tool"
head_lines = 5
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / ".token-saver.toml"
            toml_path.write_text(toml_content, encoding="utf-8")

            engine = FilterRuleEngine()
            rule_set = engine.load_project_rules(tmpdir)
            assert "legacy" in rule_set.rules


# ---------------------------------------------------------------------------
# FilterRuleEngine - matching and applying
# ---------------------------------------------------------------------------


class TestFilterRuleEngineApply:
    def test_apply_matching_rule(self):
        engine = FilterRuleEngine()
        toml_content = """
[filters.build]
match_command = "make"
strip_lines_matching = ["^make\\\\["]
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            toml_path = Path(tmpdir) / ".gotcontext.toml"
            toml_path.write_text(toml_content, encoding="utf-8")
            engine.load_project_rules(tmpdir)

            result = engine.apply("make[1]: Entering\ncc -o main main.c", "make all")
            assert result is not None
            assert "cc -o main main.c" in result

    def test_apply_no_matching_rule_returns_none(self):
        engine = FilterRuleEngine()
        engine._project_rules = FilterRuleSet()
        result = engine.apply("some text", "unknown-tool")
        assert result is None

    def test_project_rules_take_precedence(self):
        engine = FilterRuleEngine()
        project_rule = FilterRule(name="test", match_command="pytest", head_lines=1)
        user_rule = FilterRule(name="test", match_command="pytest", tail_lines=1)
        engine._project_rules = FilterRuleSet(rules={"test": project_rule})
        engine._user_rules = FilterRuleSet(rules={"test": user_rule})

        text = "line1\nline2\nline3"
        result = engine.apply(text, "pytest tests/")
        assert result == "line1"  # head_lines=1 from project rule


# ---------------------------------------------------------------------------
# FilterRuleEngine - inline test verification
# ---------------------------------------------------------------------------


class TestFilterRuleEngineVerify:
    def test_verify_passing_test(self):
        rule = FilterRule(
            name="build",
            match_command="build",
            strip_lines_matching=[r"^Progress:"],
        )
        test = InlineTest(
            name="strips progress",
            rule_name="build",
            input_text="Progress: 50%\nERROR: failed",
            expected="ERROR: failed",
        )
        rule_set = FilterRuleSet(rules={"build": rule}, tests=[test])

        engine = FilterRuleEngine()
        results = engine.verify_tests(rule_set)

        assert len(results) == 1
        assert results[0]["passed"] is True

    def test_verify_failing_test(self):
        rule = FilterRule(name="build", match_command="build")
        test = InlineTest(
            name="wrong expectation",
            rule_name="build",
            input_text="hello",
            expected="world",
        )
        rule_set = FilterRuleSet(rules={"build": rule}, tests=[test])

        engine = FilterRuleEngine()
        results = engine.verify_tests(rule_set)

        assert len(results) == 1
        assert results[0]["passed"] is False
        assert "expected" in results[0]
        assert "actual" in results[0]

    def test_verify_missing_rule(self):
        test = InlineTest(
            name="missing rule",
            rule_name="nonexistent",
            input_text="hello",
            expected="hello",
        )
        rule_set = FilterRuleSet(tests=[test])

        engine = FilterRuleEngine()
        results = engine.verify_tests(rule_set)

        assert len(results) == 1
        assert results[0]["passed"] is False
        assert "not found" in results[0]["error"]


# ---------------------------------------------------------------------------
# Config paths
# ---------------------------------------------------------------------------


class TestConfigPaths:
    def test_project_config_path(self):
        path = FilterRuleEngine.get_project_config_path("/my/project")
        assert path.endswith(".gotcontext.toml")
        assert "my" in path

    def test_user_config_path(self):
        path = FilterRuleEngine.get_user_config_path()
        assert "gotcontext" in path
        assert path.endswith("filters.toml")
