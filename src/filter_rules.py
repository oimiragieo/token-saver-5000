"""Custom filter rules engine with TOML DSL.

Allows users to define project-specific CLI output filtering rules
in `.gotcontext.toml` (project-local) or `~/.config/gotcontext/filters.toml`
(user-global). Rules are matched by command hint and applied as an
8-stage pipeline before built-in strategies.

Pipeline stages (applied in order):
1. strip_ansi — remove ANSI escape codes
2. strip_lines_matching — remove lines matching regex patterns
3. keep_lines_matching — keep ONLY lines matching regex patterns
4. truncate_lines_at — truncate each line at N characters
5. head_lines — keep only first N lines
6. tail_lines — keep only last N lines
7. max_lines — cap total output at N lines (head + tail with gap)
8. on_empty — replacement text when output is empty after filtering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# ANSI escape code pattern
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\r")


@dataclass
class FilterRule:
    """A single user-defined filter rule."""

    name: str
    description: str = ""
    match_command: str = ""

    # Pipeline stages (all optional)
    strip_ansi: bool = False
    strip_lines_matching: List[str] = field(default_factory=list)
    keep_lines_matching: List[str] = field(default_factory=list)
    truncate_lines_at: int = 0
    head_lines: int = 0
    tail_lines: int = 0
    max_lines: int = 0
    on_empty: str = ""

    def matches(self, command_hint: str) -> bool:
        """Check if this rule matches the given command hint."""
        if not self.match_command:
            return False
        try:
            return bool(re.search(self.match_command, command_hint, re.IGNORECASE))
        except re.error:
            return self.match_command.lower() in command_hint.lower()

    def apply(self, text: str) -> str:
        """Apply the 8-stage pipeline to text."""
        lines = text.splitlines()

        # Stage 1: strip ANSI
        if self.strip_ansi:
            lines = [_ANSI_RE.sub("", line) for line in lines]

        # Stage 2: strip lines matching patterns
        if self.strip_lines_matching:
            try:
                compiled = [re.compile(p) for p in self.strip_lines_matching]
                lines = [line for line in lines if not any(r.search(line) for r in compiled)]
            except re.error:
                pass  # skip stage on invalid regex

        # Stage 3: keep only lines matching patterns
        if self.keep_lines_matching:
            try:
                compiled = [re.compile(p) for p in self.keep_lines_matching]
                lines = [line for line in lines if any(r.search(line) for r in compiled)]
            except re.error:
                pass  # skip stage on invalid regex

        # Stage 4: truncate lines
        if self.truncate_lines_at > 0:
            lines = [line[: self.truncate_lines_at] for line in lines]

        # Stage 5: head lines
        if self.head_lines > 0 and len(lines) > self.head_lines:
            lines = lines[: self.head_lines]

        # Stage 6: tail lines
        if self.tail_lines > 0 and len(lines) > self.tail_lines:
            lines = lines[-self.tail_lines :]

        # Stage 7: max lines (head + tail with gap marker)
        if self.max_lines > 0 and len(lines) > self.max_lines:
            half = self.max_lines // 2
            head = lines[:half]
            tail = lines[-half:] if half > 0 else []
            gap = len(lines) - len(head) - len(tail)
            lines = head + [f"... ({gap} lines omitted) ..."] + tail

        result = "\n".join(lines)

        # Stage 8: on_empty fallback
        if not result.strip() and self.on_empty:
            return self.on_empty

        return result


@dataclass
class InlineTest:
    """An inline test case defined alongside a filter rule."""

    name: str
    rule_name: str
    input_text: str
    expected: str


@dataclass
class FilterRuleSet:
    """A collection of filter rules loaded from TOML config files."""

    rules: Dict[str, FilterRule] = field(default_factory=dict)
    tests: List[InlineTest] = field(default_factory=list)
    source_file: str = ""

    def find_matching_rule(self, command_hint: str) -> Optional[FilterRule]:
        """Find the first rule that matches the command hint."""
        for rule in self.rules.values():
            if rule.matches(command_hint):
                return rule
        return None


class FilterRuleEngine:
    """Loads and applies user-defined filter rules from TOML config files."""

    def __init__(self) -> None:
        self._project_rules: Optional[FilterRuleSet] = None
        self._user_rules: Optional[FilterRuleSet] = None

    def load_project_rules(self, project_dir: str) -> FilterRuleSet:
        """Load rules from project-local `.gotcontext.toml`."""
        path = Path(project_dir) / ".gotcontext.toml"
        if not path.exists():
            # Also check legacy name
            path = Path(project_dir) / ".token-saver.toml"
        self._project_rules = self._load_file(str(path)) if path.exists() else FilterRuleSet()
        return self._project_rules

    def load_user_rules(self) -> FilterRuleSet:
        """Load rules from user-global `~/.config/gotcontext/filters.toml`."""
        path = Path.home() / ".config" / "gotcontext" / "filters.toml"
        self._user_rules = self._load_file(str(path)) if path.exists() else FilterRuleSet()
        return self._user_rules

    def find_rule(self, command_hint: str) -> Optional[FilterRule]:
        """Find a matching rule, checking project rules first then user rules."""
        if self._project_rules:
            rule = self._project_rules.find_matching_rule(command_hint)
            if rule:
                return rule
        if self._user_rules:
            rule = self._user_rules.find_matching_rule(command_hint)
            if rule:
                return rule
        return None

    def apply(self, text: str, command_hint: str) -> Optional[str]:
        """Apply matching filter rule to text. Returns None if no rule matches."""
        rule = self.find_rule(command_hint)
        if rule is None:
            return None
        return rule.apply(text)

    def verify_tests(self, rule_set: Optional[FilterRuleSet] = None) -> List[Dict[str, Any]]:
        """Run inline tests for a rule set. Returns list of test results."""
        results = []
        sets = [rule_set] if rule_set else [self._project_rules, self._user_rules]

        for rs in sets:
            if rs is None:
                continue
            for test in rs.tests:
                rule = rs.rules.get(test.rule_name)
                if rule is None:
                    results.append(
                        {
                            "name": test.name,
                            "rule": test.rule_name,
                            "passed": False,
                            "error": f"Rule '{test.rule_name}' not found",
                        }
                    )
                    continue

                actual = rule.apply(test.input_text)
                passed = actual.strip() == test.expected.strip()
                result: Dict[str, Any] = {
                    "name": test.name,
                    "rule": test.rule_name,
                    "passed": passed,
                }
                if not passed:
                    result["expected"] = test.expected
                    result["actual"] = actual
                results.append(result)

        return results

    @staticmethod
    def _load_file(path: str) -> FilterRuleSet:
        """Parse a TOML config file into a FilterRuleSet."""
        if tomllib is None:
            return FilterRuleSet(source_file=path)

        file_path = Path(path)
        if not file_path.exists():
            return FilterRuleSet(source_file=path)

        try:
            data = tomllib.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            return FilterRuleSet(source_file=path)

        rule_set = FilterRuleSet(source_file=path)

        # Parse [filters.*] sections
        filters = data.get("filters", {})
        for name, cfg in filters.items():
            if not isinstance(cfg, dict):
                continue
            rule = FilterRule(
                name=name,
                description=cfg.get("description", ""),
                match_command=cfg.get("match_command", ""),
                strip_ansi=cfg.get("strip_ansi", False),
                strip_lines_matching=cfg.get("strip_lines_matching", []),
                keep_lines_matching=cfg.get("keep_lines_matching", []),
                truncate_lines_at=cfg.get("truncate_lines_at", 0),
                head_lines=cfg.get("head_lines", 0),
                tail_lines=cfg.get("tail_lines", 0),
                max_lines=cfg.get("max_lines", 0),
                on_empty=cfg.get("on_empty", ""),
            )
            rule_set.rules[name] = rule

        # Parse [[tests.*]] sections
        tests_section = data.get("tests", {})
        for rule_name, test_list in tests_section.items():
            if not isinstance(test_list, list):
                continue
            for test_cfg in test_list:
                if not isinstance(test_cfg, dict):
                    continue
                inline_test = InlineTest(
                    name=test_cfg.get("name", "unnamed"),
                    rule_name=rule_name,
                    input_text=test_cfg.get("input", ""),
                    expected=test_cfg.get("expected", ""),
                )
                rule_set.tests.append(inline_test)

        return rule_set

    @staticmethod
    def get_project_config_path(project_dir: str) -> str:
        """Return the expected path for project filter config."""
        return str(Path(project_dir) / ".gotcontext.toml")

    @staticmethod
    def get_user_config_path() -> str:
        """Return the expected path for user-global filter config."""
        return str(Path.home() / ".config" / "gotcontext" / "filters.toml")
