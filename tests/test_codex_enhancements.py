"""
Tests for Codex CLI token optimization enhancements (v0.11.0).

TDD test suite covering:
1. _parse_codex_result() — JSONL parsing for Codex exec --json output
2. KNOWN_MODEL_CONTEXT_WINDOWS — gpt-5.1-codex and codex-mini entries
3. KNOWN_MODEL_COMPRESSION_TRIGGERS — Codex compression trigger at 0.80
4. pricing.py — Codex model pricing entries
5. is_available("codex") — provider availability check
6. _build_command("codex", ...) — correct CLI invocation
7. run_prompt("codex", ..., dry_run=True) — dry-run path works
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Codex exec --json outputs multiple JSON lines (not a single JSON object).
# Each line is a separate event. The turn.completed event has usage data.
# The item.completed event with agent_message has the response text.
CODEX_JSONL_FIXTURE = """{"type":"thread.started","session_id":"sess_123"}
{"type":"turn.started"}
{"type":"item.started","id":"item_1","details":{"type":"agent_message"}}
{"type":"item.updated","id":"item_1","details":{"type":"agent_message","content":[{"type":"output_text","text":"Here are the concepts..."}]}}
{"type":"item.completed","id":"item_1","details":{"type":"agent_message","content":[{"type":"output_text","text":"Here are the concepts..."}]}}
{"type":"turn.completed","usage":{"input_tokens":15000,"cached_input_tokens":8000,"output_tokens":500}}"""


# ---------------------------------------------------------------------------
# _parse_codex_result — JSONL parsing
# ---------------------------------------------------------------------------


class TestParseCodexJsonl:
    def test_parse_codex_jsonl(self):
        """Parses JSONL fixture into CLIResult with correct token counts."""
        from src.cli_benchmark.providers import _parse_codex_result

        result = _parse_codex_result(CODEX_JSONL_FIXTURE)
        assert result.provider == "codex"
        assert result.input_tokens == 15000
        assert result.output_tokens == 500
        assert result.cache_read_tokens == 8000

    def test_parse_codex_extracts_response(self):
        """raw_response contains the agent message text from item.completed."""
        from src.cli_benchmark.providers import _parse_codex_result

        result = _parse_codex_result(CODEX_JSONL_FIXTURE)
        assert "Here are the concepts..." in result.raw_response

    def test_parse_codex_empty_output(self):
        """Handles empty stdout gracefully — returns zeros, no exception."""
        from src.cli_benchmark.providers import _parse_codex_result

        result = _parse_codex_result("")
        assert result.provider == "codex"
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.cache_read_tokens == 0
        assert result.raw_response == ""

    def test_parse_codex_no_turn_completed(self):
        """Handles missing turn.completed event — returns zeros, no exception."""
        from src.cli_benchmark.providers import _parse_codex_result

        # JSONL with events but no turn.completed
        partial_jsonl = (
            '{"type":"thread.started","session_id":"sess_x"}\n'
            '{"type":"turn.started"}\n'
            '{"type":"item.completed","id":"i1","details":{"type":"agent_message",'
            '"content":[{"type":"output_text","text":"hello"}]}}'
        )
        result = _parse_codex_result(partial_jsonl)
        assert result.input_tokens == 0
        assert result.output_tokens == 0
        # Response text is still extracted from item.completed
        assert "hello" in result.raw_response

    def test_parse_codex_cost_is_non_negative(self):
        """Computed cost for Codex result is always >= 0."""
        from src.cli_benchmark.providers import _parse_codex_result

        result = _parse_codex_result(CODEX_JSONL_FIXTURE)
        assert result.total_cost_usd >= 0.0

    def test_parse_codex_model_name(self):
        """CLIResult.model is set to the Codex model string."""
        from src.cli_benchmark.providers import _parse_codex_result

        result = _parse_codex_result(CODEX_JSONL_FIXTURE)
        assert result.model == "gpt-5.1-codex"

    def test_parse_codex_invalid_json_lines_skipped(self):
        """Lines that are not valid JSON are silently skipped."""
        from src.cli_benchmark.providers import _parse_codex_result

        mixed = (
            "not json at all\n"
            '{"type":"turn.completed","usage":{"input_tokens":100,"cached_input_tokens":0,"output_tokens":10}}\n'
            "also not json"
        )
        result = _parse_codex_result(mixed)
        assert result.input_tokens == 100
        assert result.output_tokens == 10


# ---------------------------------------------------------------------------
# KNOWN_MODEL_CONTEXT_WINDOWS — Codex model entries
# ---------------------------------------------------------------------------


class TestCodexModelsInDatabase:
    def test_gpt_5_1_codex_in_database(self):
        """gpt-5.1-codex is present in KNOWN_MODEL_CONTEXT_WINDOWS."""
        from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS

        assert "gpt-5.1-codex" in KNOWN_MODEL_CONTEXT_WINDOWS

    def test_codex_mini_in_database(self):
        """codex-mini is present in KNOWN_MODEL_CONTEXT_WINDOWS."""
        from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS

        assert "codex-mini" in KNOWN_MODEL_CONTEXT_WINDOWS

    def test_gpt_5_1_codex_context_window(self):
        """gpt-5.1-codex has a 200,000 token context window."""
        from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS

        assert KNOWN_MODEL_CONTEXT_WINDOWS["gpt-5.1-codex"] == 200_000

    def test_codex_mini_context_window(self):
        """codex-mini has a 200,000 token context window."""
        from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS

        assert KNOWN_MODEL_CONTEXT_WINDOWS["codex-mini"] == 200_000


# ---------------------------------------------------------------------------
# KNOWN_MODEL_COMPRESSION_TRIGGERS — Codex compression triggers
# ---------------------------------------------------------------------------


class TestCodexCompressionTriggers:
    def test_gpt_5_1_codex_trigger(self):
        """gpt-5.1-codex has compression_trigger=0.80 (HISTORY_SOFT_CAP_RATIO)."""
        from src.constants import KNOWN_MODEL_COMPRESSION_TRIGGERS

        assert "gpt-5.1-codex" in KNOWN_MODEL_COMPRESSION_TRIGGERS
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS["gpt-5.1-codex"] == pytest.approx(0.80)

    def test_codex_mini_trigger(self):
        """codex-mini has compression_trigger=0.80."""
        from src.constants import KNOWN_MODEL_COMPRESSION_TRIGGERS

        assert "codex-mini" in KNOWN_MODEL_COMPRESSION_TRIGGERS
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS["codex-mini"] == pytest.approx(0.80)

    def test_o3_trigger(self):
        """o3 has compression_trigger=0.80 (Codex workflow model)."""
        from src.constants import KNOWN_MODEL_COMPRESSION_TRIGGERS

        assert "o3" in KNOWN_MODEL_COMPRESSION_TRIGGERS
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS["o3"] == pytest.approx(0.80)

    def test_o4_mini_trigger(self):
        """o4-mini has compression_trigger=0.80 (Codex workflow model)."""
        from src.constants import KNOWN_MODEL_COMPRESSION_TRIGGERS

        assert "o4-mini" in KNOWN_MODEL_COMPRESSION_TRIGGERS
        assert KNOWN_MODEL_COMPRESSION_TRIGGERS["o4-mini"] == pytest.approx(0.80)


# ---------------------------------------------------------------------------
# pricing.py — Codex model pricing
# ---------------------------------------------------------------------------


class TestCodexPricing:
    def test_codex_pricing_exists(self):
        """gpt-5.1-codex entry exists in PRICING table."""
        from src.cli_benchmark.pricing import PRICING

        assert "gpt-5.1-codex" in PRICING

    def test_codex_mini_pricing_exists(self):
        """codex-mini entry exists in PRICING table."""
        from src.cli_benchmark.pricing import PRICING

        assert "codex-mini" in PRICING

    def test_codex_pricing_computation(self):
        """compute_cost for gpt-5.1-codex uses correct rates ($2.50 input, $10.0 output)."""
        from src.cli_benchmark.pricing import compute_cost

        # 1M input at $2.50/M = $2.50, 1M output at $10.0/M = $10.0 => $12.50
        cost = compute_cost("gpt-5.1-codex", 1_000_000, 1_000_000)
        assert abs(cost - 12.50) < 0.001

    def test_codex_mini_pricing_computation(self):
        """compute_cost for codex-mini uses correct rates ($1.50 input, $6.0 output)."""
        from src.cli_benchmark.pricing import compute_cost

        # 1M input at $1.50/M = $1.50, 1M output at $6.0/M = $6.0 => $7.50
        cost = compute_cost("codex-mini", 1_000_000, 1_000_000)
        assert abs(cost - 7.50) < 0.001

    def test_codex_cache_read_pricing(self):
        """Codex cache read is priced at $0.625/M (25% of input rate)."""
        from src.cli_benchmark.pricing import PRICING

        rates = PRICING["gpt-5.1-codex"]
        assert rates["cache_read"] == pytest.approx(0.625)

    def test_codex_mini_cache_read_pricing(self):
        """codex-mini cache read is priced at $0.375/M (25% of input rate)."""
        from src.cli_benchmark.pricing import PRICING

        rates = PRICING["codex-mini"]
        assert rates["cache_read"] == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# is_available("codex") — provider availability
# ---------------------------------------------------------------------------


class TestCodexIsAvailable:
    def test_codex_is_available_check_present(self):
        """is_available("codex") returns True when codex binary is on PATH."""
        from src.cli_benchmark.providers import is_available

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/bin/codex"):
            assert is_available("codex") is True

    def test_codex_is_available_check_absent(self):
        """is_available("codex") returns False when codex binary is not on PATH."""
        from src.cli_benchmark.providers import is_available

        with patch("src.cli_benchmark.providers._find_cli", return_value=None):
            assert is_available("codex") is False

    def test_codex_does_not_affect_other_providers(self):
        """Adding codex support does not break claude/gemini availability checks."""
        from src.cli_benchmark.providers import is_available

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/bin/claude"):
            assert is_available("claude") is True
            assert is_available("gemini") is True


# ---------------------------------------------------------------------------
# _build_command("codex", ...) — CLI command construction
# ---------------------------------------------------------------------------


class TestCodexBuildCommand:
    def test_codex_build_command_no_model(self):
        """_build_command("codex", None) produces correct base args."""
        from src.cli_benchmark.providers import _build_command

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/local/bin/codex"):
            cmd = _build_command("codex", None)

        assert cmd[0] == "/usr/local/bin/codex"
        assert "exec" in cmd
        assert "--json" in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd

    def test_codex_build_command_with_model(self):
        """_build_command("codex", model) appends --model flag."""
        from src.cli_benchmark.providers import _build_command

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/local/bin/codex"):
            cmd = _build_command("codex", "codex-mini")

        assert "--model" in cmd
        assert "codex-mini" in cmd

    def test_codex_build_command_raises_when_not_found(self):
        """_build_command("codex", ...) raises RuntimeError when codex is not on PATH."""
        from src.cli_benchmark.providers import _build_command

        with patch("src.cli_benchmark.providers._find_cli", return_value=None):
            with pytest.raises(RuntimeError, match="codex CLI not found"):
                _build_command("codex", None)


# ---------------------------------------------------------------------------
# run_prompt("codex", ..., dry_run=True) — dry-run path
# ---------------------------------------------------------------------------


class TestCodexDryRun:
    def test_codex_dry_run(self):
        """run_prompt("codex", ..., dry_run=True) returns a zeroed CLIResult with is_dry_run=True."""
        from src.cli_benchmark.providers import run_prompt

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/local/bin/codex"):
            result = run_prompt("codex", "Summarize this.", model="gpt-5.1-codex", dry_run=True)

        assert result.is_dry_run is True
        assert result.provider == "codex"
        assert result.input_tokens == 0
        assert result.total_cost_usd == 0.0

    def test_codex_dry_run_preserves_model(self):
        """dry_run result captures the model name provided."""
        from src.cli_benchmark.providers import run_prompt

        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/local/bin/codex"):
            result = run_prompt("codex", "Hello", model="codex-mini", dry_run=True)

        assert result.model == "codex-mini"
