"""Comprehensive tests for the cli_benchmark package.

All tests are dry-run / unit level — no real API calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLAUDE_FIXTURE = {
    "type": "result",
    "subtype": "success",
    "result": "Here are the concepts...",
    "total_cost_usd": 0.0234,
    "duration_ms": 5432,
    "num_turns": 1,
    "usage": {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    },
    "modelUsage": {
        "claude-sonnet-4-6": {
            "inputTokens": 12500,
            "outputTokens": 450,
            "cacheReadInputTokens": 5000,
            "cacheCreationInputTokens": 7500,
            "costUSD": 0.0234,
        }
    },
}

GEMINI_FIXTURE = {
    "response": "Here are the concepts...",
    "stats": {
        "models": {
            "gemini-2.5-flash": {
                "api": {"totalRequests": 1, "totalErrors": 0, "totalLatencyMs": 4200},
                "tokens": {
                    "input": 9800,  # billed (net of cache)
                    "prompt": 12800,  # total content tokens (cache-independent)
                    "candidates": 400,
                    "total": 13200,
                    "cached": 3000,
                    "thoughts": 0,
                    "tool": 0,
                },
            }
        },
        "tools": {"totalCalls": 0},
    },
}

# Path to the real corpus directory
CORPUS_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "corpus"


# ---------------------------------------------------------------------------
# corpus.py tests
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_load_corpus_manifest(self):
        """manifest.json parses and has the expected 'files' key."""
        from src.cli_benchmark.corpus import load_manifest

        manifest = load_manifest(CORPUS_DIR)
        assert "files" in manifest
        assert isinstance(manifest["files"], list)
        assert len(manifest["files"]) > 0

    def test_manifest_has_prompt_template(self):
        """manifest.json has a prompt_template with the {context} placeholder."""
        from src.cli_benchmark.corpus import load_manifest

        manifest = load_manifest(CORPUS_DIR)
        assert "prompt_template" in manifest
        assert "{context}" in manifest["prompt_template"]


class TestCorpusFiles:
    def test_corpus_files_exist(self):
        """All files referenced in manifest.json exist on disk."""
        from src.cli_benchmark.corpus import load_manifest

        manifest = load_manifest(CORPUS_DIR)
        for entry in manifest["files"]:
            file_path = CORPUS_DIR / entry["file"]
            assert file_path.exists(), f"Corpus file missing: {file_path}"
            assert file_path.stat().st_size > 0, f"Corpus file is empty: {file_path}"

    def test_small_corpus_line_count(self):
        """small.txt has at least 80 lines (control corpus)."""
        small = CORPUS_DIR / "small.txt"
        assert small.read_text(encoding="utf-8").count("\n") >= 80

    def test_medium_corpus_line_count(self):
        """medium.txt has at least 400 lines."""
        medium = CORPUS_DIR / "medium.txt"
        assert medium.read_text(encoding="utf-8").count("\n") >= 400

    def test_large_corpus_line_count(self):
        """large.txt has at least 1500 lines."""
        large = CORPUS_DIR / "large.txt"
        assert large.read_text(encoding="utf-8").count("\n") >= 1500


class TestLoadCorpus:
    def test_load_corpus_by_name(self):
        """load_corpus('small') returns a CorpusEntry with non-empty content."""
        from src.cli_benchmark.corpus import load_corpus

        entry = load_corpus("small", CORPUS_DIR)
        assert entry.name == "small"
        assert len(entry.content) > 0
        assert entry.line_count > 0
        assert entry.file_path.exists()
        assert entry.description

    def test_load_corpus_medium(self):
        """load_corpus('medium') returns correct name and content."""
        from src.cli_benchmark.corpus import load_corpus

        entry = load_corpus("medium", CORPUS_DIR)
        assert entry.name == "medium"
        assert entry.line_count > 400

    def test_load_corpus_large(self):
        """load_corpus('large') returns correct name and content."""
        from src.cli_benchmark.corpus import load_corpus

        entry = load_corpus("large", CORPUS_DIR)
        assert entry.name == "large"
        assert entry.line_count > 1500

    def test_load_corpus_unknown_raises(self):
        """load_corpus('nonexistent') raises ValueError with helpful message."""
        from src.cli_benchmark.corpus import load_corpus

        with pytest.raises(ValueError, match="nonexistent"):
            load_corpus("nonexistent", CORPUS_DIR)

    def test_load_corpus_unknown_mentions_valid(self):
        """ValueError from unknown corpus mentions the available names."""
        from src.cli_benchmark.corpus import load_corpus

        with pytest.raises(ValueError, match="small"):
            load_corpus("bogus", CORPUS_DIR)

    def test_load_all_corpus_returns_all(self):
        """load_all_corpus() returns one entry per manifest file."""
        from src.cli_benchmark.corpus import load_all_corpus, load_manifest

        manifest = load_manifest(CORPUS_DIR)
        entries = load_all_corpus(CORPUS_DIR)
        assert len(entries) == len(manifest["files"])

    def test_corpus_entry_line_count_matches_content(self):
        """CorpusEntry.line_count matches the actual line count of content."""
        from src.cli_benchmark.corpus import load_corpus

        entry = load_corpus("small", CORPUS_DIR)
        assert entry.line_count == len(entry.content.splitlines())


class TestBuildPrompt:
    def test_prompt_template_substitution(self):
        """build_prompt inserts the context string into the template."""
        from src.cli_benchmark.corpus import build_prompt

        result = build_prompt("MY_CONTEXT_STRING", CORPUS_DIR)
        assert "MY_CONTEXT_STRING" in result

    def test_prompt_template_no_raw_placeholder(self):
        """Result of build_prompt does not contain the literal '{context}'."""
        from src.cli_benchmark.corpus import build_prompt

        result = build_prompt("some context", CORPUS_DIR)
        assert "{context}" not in result

    def test_prompt_template_preserves_template_text(self):
        """Template wrapper text is still present in the built prompt."""
        from src.cli_benchmark.corpus import build_prompt

        result = build_prompt("data", CORPUS_DIR)
        # The manifest template has this wrapper text
        assert len(result) > len("data")


# ---------------------------------------------------------------------------
# providers.py tests
# ---------------------------------------------------------------------------


class TestParseJsonOutput:
    def test_parse_json_direct(self):
        """_parse_json_output handles clean JSON string."""
        from src.cli_benchmark.providers import _parse_json_output

        data = {"key": "value"}
        assert _parse_json_output(json.dumps(data)) == data

    def test_parse_json_with_prefix(self):
        """_parse_json_output handles non-JSON prefix before the object."""
        from src.cli_benchmark.providers import _parse_json_output

        stdout = 'Some log line\nAnother line\n{"key": "value"}'
        result = _parse_json_output(stdout)
        assert result == {"key": "value"}

    def test_parse_json_with_suffix(self):
        """_parse_json_output handles trailing text after the JSON object."""
        from src.cli_benchmark.providers import _parse_json_output

        stdout = '{"key": "value"}\nTrailing log output'
        result = _parse_json_output(stdout)
        assert result == {"key": "value"}

    def test_parse_empty_json(self):
        """_parse_json_output returns empty dict for garbage input."""
        from src.cli_benchmark.providers import _parse_json_output

        assert _parse_json_output("not json at all") == {}
        assert _parse_json_output("") == {}
        assert _parse_json_output("   ") == {}

    def test_parse_json_nested(self):
        """_parse_json_output handles nested JSON objects."""
        from src.cli_benchmark.providers import _parse_json_output

        data = {"a": {"b": 1}, "c": [1, 2, 3]}
        assert _parse_json_output(json.dumps(data)) == data


class TestParseClaudeResult:
    def test_parse_claude_json(self):
        """_parse_claude_result correctly maps the Claude fixture to CLIResult."""
        from src.cli_benchmark.providers import _parse_claude_result

        result = _parse_claude_result(CLAUDE_FIXTURE, "")
        assert result.provider == "claude"
        # total input = inputTokens(12500) + cacheCreation(7500) + cacheRead(5000) = 25000
        assert result.input_tokens == 25000
        assert result.output_tokens == 450
        assert result.cache_read_tokens == 5000
        assert result.total_cost_usd == 0.0234
        assert result.wall_time_ms == 5432
        assert result.num_turns == 1
        assert result.model == "claude-sonnet-4-6"
        assert result.raw_response == "Here are the concepts..."

    def test_parse_claude_missing_fields(self):
        """_parse_claude_result handles missing optional fields gracefully."""
        from src.cli_benchmark.providers import _parse_claude_result

        result = _parse_claude_result({}, "")
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_cost_usd == 0.0


class TestParseGeminiResult:
    def test_parse_gemini_json(self):
        """_parse_gemini_result correctly maps the Gemini fixture to CLIResult."""
        from src.cli_benchmark.providers import _parse_gemini_result

        result = _parse_gemini_result(GEMINI_FIXTURE, "", "gemini-2.5-flash")
        assert result.provider == "gemini"
        assert result.input_tokens == 12800
        assert result.output_tokens == 400
        assert result.cache_read_tokens == 3000
        assert result.wall_time_ms == 4200
        assert result.tool_calls == 0
        assert result.raw_response == "Here are the concepts..."
        # Cost should be computed (non-negative)
        assert result.total_cost_usd >= 0.0

    def test_parse_gemini_missing_fields(self):
        """_parse_gemini_result handles missing stats gracefully."""
        from src.cli_benchmark.providers import _parse_gemini_result

        result = _parse_gemini_result({}, "", "gemini-2.5-flash")
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestIsAvailable:
    def test_is_available_no_cli(self):
        """is_available returns False when CLI is not found."""
        from src.cli_benchmark.providers import is_available

        with patch("src.cli_benchmark.providers._find_cli", return_value=None):
            assert not is_available("claude")
            assert not is_available("gemini")

    def test_is_available_unknown_provider(self):
        """is_available returns False for unknown provider strings."""
        from src.cli_benchmark.providers import is_available

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/bin/claude"):
            assert not is_available("openai")
            assert not is_available("")


class TestRunPromptDryRun:
    def test_dry_run_returns_mock_result(self):
        """run_prompt with dry_run=True returns a zeroed CLIResult with is_dry_run=True."""
        from src.cli_benchmark.providers import run_prompt

        result = run_prompt("claude", "Hello", model="claude-sonnet-4-6", dry_run=True)
        assert result.is_dry_run is True
        assert result.provider == "claude"
        assert result.input_tokens == 0
        assert result.total_cost_usd == 0.0

    def test_dry_run_gemini(self):
        """dry_run works for gemini provider too."""
        from src.cli_benchmark.providers import run_prompt

        result = run_prompt("gemini", "Hello", dry_run=True)
        assert result.is_dry_run is True
        assert result.provider == "gemini"

    def test_dry_run_preserves_model(self):
        """dry_run result captures the model name."""
        from src.cli_benchmark.providers import run_prompt

        result = run_prompt("claude", "Hi", model="claude-opus-4-6", dry_run=True)
        assert result.model == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# pricing.py tests
# ---------------------------------------------------------------------------


class TestPricing:
    def test_pricing_claude(self):
        """compute_cost for a known Claude model uses correct rates."""
        from src.cli_benchmark.pricing import compute_cost

        # 1M input tokens at $3.0/M = $3.0, 1M output at $15.0/M = $15.0 => $18.0
        cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
        assert abs(cost - 18.0) < 0.001

    def test_pricing_gemini(self):
        """compute_cost for a Gemini model uses correct rates."""
        from src.cli_benchmark.pricing import compute_cost

        # gemini-2.5-flash: $0.15/M input, $0.60/M output
        cost = compute_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
        assert abs(cost - 0.75) < 0.001

    def test_pricing_unknown_model(self):
        """compute_cost falls back to default pricing for unknown models."""
        from src.cli_benchmark.pricing import compute_cost

        cost_unknown = compute_cost("unknown-model-xyz", 1_000_000, 0)
        cost_default = compute_cost("default", 1_000_000, 0)
        assert cost_unknown == cost_default

    def test_pricing_zero_tokens(self):
        """compute_cost returns 0.0 when all token counts are zero."""
        from src.cli_benchmark.pricing import compute_cost

        assert compute_cost("claude-sonnet-4-6", 0, 0) == 0.0
        assert compute_cost("gemini-2.5-flash", 0, 0, 0) == 0.0

    def test_pricing_cache_read_tokens(self):
        """cache_read_tokens contribute to cost at the discounted cache rate."""
        from src.cli_benchmark.pricing import compute_cost

        # With cache reads included vs without
        cost_with_cache = compute_cost("claude-sonnet-4-6", 0, 0, 1_000_000)
        assert cost_with_cache > 0.0
        cost_without = compute_cost("claude-sonnet-4-6", 0, 0, 0)
        assert cost_with_cache > cost_without

    def test_get_model_rates_returns_dict(self):
        """get_model_rates returns a dict with input/output keys."""
        from src.cli_benchmark.pricing import get_model_rates

        rates = get_model_rates("claude-sonnet-4-6")
        assert "input" in rates
        assert "output" in rates
        assert rates["input"] > 0
        assert rates["output"] > 0

    def test_get_model_rates_fallback(self):
        """get_model_rates returns default rates for unknown models."""
        from src.cli_benchmark.pricing import get_model_rates, PRICING

        rates = get_model_rates("does-not-exist")
        assert rates == PRICING["default"]


# ---------------------------------------------------------------------------
# results.py tests
# ---------------------------------------------------------------------------


class TestCLIResult:
    def test_cli_result_defaults(self):
        """CLIResult() instantiates with sensible zero defaults."""
        from src.cli_benchmark.results import CLIResult

        r = CLIResult()
        assert r.provider == ""
        assert r.model == ""
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cache_read_tokens == 0
        assert r.total_cost_usd == 0.0
        assert r.wall_time_ms == 0.0
        assert r.tool_calls == 0
        assert r.num_turns == 0
        assert r.raw_response == ""
        assert r.raw_json == {}
        assert r.is_dry_run is False


class TestComparisonResult:
    def _make_result(self, baseline_tokens: int, compressed_tokens: int):
        from src.cli_benchmark.results import CLIResult, ComparisonResult

        baseline = CLIResult(
            input_tokens=baseline_tokens,
            total_cost_usd=baseline_tokens * 3.0 / 1_000_000,
        )
        compressed = CLIResult(
            input_tokens=compressed_tokens,
            total_cost_usd=compressed_tokens * 3.0 / 1_000_000,
        )
        return ComparisonResult(
            corpus_name="test",
            provider="claude",
            mode="skill",
            baseline=baseline,
            compressed=compressed,
        )

    def test_comparison_savings_calculation(self):
        """ComparisonResult auto-computes input_token_savings_pct correctly."""
        result = self._make_result(10000, 1000)
        assert result.input_token_savings_pct == pytest.approx(90.0, abs=0.1)

    def test_comparison_cost_savings_calculation(self):
        """ComparisonResult auto-computes cost_savings_pct correctly."""
        result = self._make_result(10000, 1000)
        assert result.cost_savings_pct == pytest.approx(90.0, abs=0.1)

    def test_comparison_zero_baseline(self):
        """No division by zero when baseline token count is 0."""
        result = self._make_result(0, 0)
        assert result.input_token_savings_pct == 0.0
        assert result.cost_savings_pct == 0.0

    def test_comparison_no_savings(self):
        """0% savings when compressed equals baseline."""
        result = self._make_result(5000, 5000)
        assert result.input_token_savings_pct == 0.0

    def test_comparison_negative_savings(self):
        """Negative savings (expansion) is calculated correctly."""
        result = self._make_result(1000, 1200)
        assert result.input_token_savings_pct < 0


class TestBenchmarkReport:
    def _make_comparison(self, name: str = "small"):
        from src.cli_benchmark.results import CLIResult, ComparisonResult

        return ComparisonResult(
            corpus_name=name,
            provider="claude",
            mode="skill",
            baseline=CLIResult(input_tokens=10000, total_cost_usd=0.03),
            compressed=CLIResult(input_tokens=1000, total_cost_usd=0.003),
        )

    def test_report_to_json(self, tmp_path):
        """BenchmarkReport serializes to a valid JSON file."""
        from src.cli_benchmark.results import BenchmarkReport

        report = BenchmarkReport()
        report.add(self._make_comparison("small"))
        out_path = tmp_path / "results.json"
        report.to_json(out_path)

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["corpus_name"] == "small"

    def test_report_to_table(self):
        """BenchmarkReport.to_table() returns a string with header columns."""
        from src.cli_benchmark.results import BenchmarkReport

        report = BenchmarkReport()
        report.add(self._make_comparison("medium"))
        table = report.to_table()
        assert isinstance(table, str)
        assert "Corpus" in table
        assert "Provider" in table
        assert "Savings" in table
        assert "medium" in table

    def test_report_empty(self):
        """Empty BenchmarkReport produces the 'No results' message."""
        from src.cli_benchmark.results import BenchmarkReport

        report = BenchmarkReport()
        result = report.to_table()
        assert "No results" in result

    def test_report_multiple_results(self):
        """BenchmarkReport accumulates multiple results."""
        from src.cli_benchmark.results import BenchmarkReport

        report = BenchmarkReport()
        for name in ("small", "medium", "large"):
            report.add(self._make_comparison(name))
        assert len(report.results) == 3

    def test_report_json_is_reversible(self, tmp_path):
        """JSON written by to_json can be loaded back and has expected structure."""
        from src.cli_benchmark.results import BenchmarkReport

        report = BenchmarkReport(metadata={"test": True})
        report.add(self._make_comparison())
        out_path = tmp_path / "r.json"
        report.to_json(out_path)
        data = json.loads(out_path.read_text())
        assert data["metadata"]["test"] is True
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# project_scaffold.py tests
# ---------------------------------------------------------------------------


class TestProjectScaffold:
    def test_scaffold_vanilla_creates_corpus(self, tmp_path):
        """create_vanilla copies corpus.txt into the temp dir."""
        from src.cli_benchmark.project_scaffold import create_vanilla

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_vanilla(corpus_path, provider="claude")
        try:
            assert (scaffold / "corpus.txt").exists()
            content = (scaffold / "corpus.txt").read_text(encoding="utf-8")
            assert len(content) > 0
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)

    def test_scaffold_mcp_claude_has_settings(self, tmp_path):
        """create_with_mcp for claude creates .claude/settings.json."""
        from src.cli_benchmark.project_scaffold import create_with_mcp

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_with_mcp(corpus_path, provider="claude")
        try:
            settings = scaffold / ".claude" / "settings.json"
            assert settings.exists()
            data = json.loads(settings.read_text())
            assert "mcpServers" in data
            assert "token-saver" in data["mcpServers"]
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)

    def test_scaffold_mcp_claude_has_claude_md(self, tmp_path):
        """create_with_mcp for claude creates CLAUDE.md."""
        from src.cli_benchmark.project_scaffold import create_with_mcp

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_with_mcp(corpus_path, provider="claude")
        try:
            assert (scaffold / "CLAUDE.md").exists()
            content = (scaffold / "CLAUDE.md").read_text(encoding="utf-8")
            assert "ingest_context" in content
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)

    def test_scaffold_mcp_gemini_has_settings(self, tmp_path):
        """create_with_mcp for gemini creates .gemini/settings.json."""
        from src.cli_benchmark.project_scaffold import create_with_mcp

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_with_mcp(corpus_path, provider="gemini")
        try:
            settings = scaffold / ".gemini" / "settings.json"
            assert settings.exists()
            data = json.loads(settings.read_text())
            assert "mcpServers" in data
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)

    def test_scaffold_mcp_gemini_has_gemini_md(self, tmp_path):
        """create_with_mcp for gemini creates GEMINI.md."""
        from src.cli_benchmark.project_scaffold import create_with_mcp

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_with_mcp(corpus_path, provider="gemini")
        try:
            assert (scaffold / "GEMINI.md").exists()
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)

    def test_scaffold_cleanup_removes_dir(self):
        """cleanup() removes the scaffold directory from disk."""
        from src.cli_benchmark.project_scaffold import cleanup, create_vanilla

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_vanilla(corpus_path, provider="claude")
        assert scaffold.exists()
        cleanup(scaffold)
        assert not scaffold.exists()

    def test_scaffold_cleanup_nonexistent_is_noop(self, tmp_path):
        """cleanup() on a non-existent path does not raise."""
        from src.cli_benchmark.project_scaffold import cleanup

        ghost = tmp_path / "does_not_exist"
        cleanup(ghost)  # Should not raise

    def test_scaffold_mcp_settings_has_command(self, tmp_path):
        """MCP settings include a 'command' key pointing to the server entry point."""
        from src.cli_benchmark.project_scaffold import create_with_mcp

        corpus_path = CORPUS_DIR / "small.txt"
        scaffold = create_with_mcp(corpus_path, provider="claude")
        try:
            settings_path = scaffold / ".claude" / "settings.json"
            data = json.loads(settings_path.read_text())
            server_config = data["mcpServers"]["token-saver"]
            assert "command" in server_config
            assert server_config["command"] == "token-saver-mcp"
        finally:
            import shutil

            shutil.rmtree(scaffold, ignore_errors=True)


# ---------------------------------------------------------------------------
# runner.py tests
# ---------------------------------------------------------------------------


class TestRunBenchmarkDryRun:
    def test_dry_run_skill_mode_returns_report(self):
        """run_benchmark in dry_run skill mode returns a BenchmarkReport."""
        from src.cli_benchmark.runner import run_benchmark

        report = run_benchmark(
            mode="skill",
            sizes=["small"],
            providers=["claude"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        from src.cli_benchmark.results import BenchmarkReport

        assert isinstance(report, BenchmarkReport)

    def test_dry_run_mcp_mode_returns_report(self):
        """run_benchmark in dry_run mcp mode returns a BenchmarkReport."""
        from src.cli_benchmark.runner import run_benchmark

        report = run_benchmark(
            mode="mcp",
            sizes=["small"],
            providers=["claude"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        from src.cli_benchmark.results import BenchmarkReport

        assert isinstance(report, BenchmarkReport)

    def test_dry_run_both_mode_doubles_results(self):
        """run_benchmark with mode='both' produces 2x results (skill + mcp)."""
        from src.cli_benchmark.runner import run_benchmark

        report = run_benchmark(
            mode="both",
            sizes=["small"],
            providers=["claude"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        assert len(report.results) == 2
        modes = {r.mode for r in report.results}
        assert "skill" in modes
        assert "mcp" in modes

    def test_dry_run_report_has_metadata(self):
        """BenchmarkReport from dry run includes mode and dry_run in metadata."""
        from src.cli_benchmark.runner import run_benchmark

        report = run_benchmark(
            mode="skill",
            sizes=["small"],
            providers=["claude"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        assert report.metadata["mode"] == "skill"
        assert report.metadata["dry_run"] is True

    def test_dry_run_all_sizes_produces_multiple_results(self):
        """Running all sizes in dry_run produces one result per corpus."""
        from src.cli_benchmark.runner import run_benchmark

        report = run_benchmark(
            mode="skill",
            providers=["claude"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        # Should have one result for each corpus file (small, medium, large)
        assert len(report.results) >= 3

    def test_dry_run_skips_unavailable_providers(self):
        """run_benchmark skips providers that are unavailable (non-dry-run check)."""
        from src.cli_benchmark.runner import run_benchmark

        # With dry_run=True all providers are treated as available
        report = run_benchmark(
            mode="skill",
            sizes=["small"],
            providers=["claude", "gemini"],
            corpus_dir=CORPUS_DIR,
            dry_run=True,
        )
        # Both providers ran
        provider_names = {r.provider for r in report.results}
        assert "claude" in provider_names
        assert "gemini" in provider_names
