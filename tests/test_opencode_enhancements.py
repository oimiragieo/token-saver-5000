"""Tests for OpenCode enhancements: model database, cache strategy advisor, pricing, providers."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Model database tests
# ---------------------------------------------------------------------------


class TestModelDatabase(unittest.TestCase):
    """Tests for KNOWN_MODEL_CONTEXT_WINDOWS and KNOWN_MODEL_COMPRESSION_TRIGGERS."""

    def setUp(self) -> None:
        from src.constants import KNOWN_MODEL_CONTEXT_WINDOWS, KNOWN_MODEL_COMPRESSION_TRIGGERS

        self.windows = KNOWN_MODEL_CONTEXT_WINDOWS
        self.triggers = KNOWN_MODEL_COMPRESSION_TRIGGERS

    def test_gpt41_in_database(self) -> None:
        """gpt-4.1 has 1M+ context window."""
        assert "gpt-4.1" in self.windows
        assert self.windows["gpt-4.1"] >= 1_000_000

    def test_gpt41_mini_in_database(self) -> None:
        """gpt-4.1-mini has 200K context window."""
        assert "gpt-4.1-mini" in self.windows
        assert self.windows["gpt-4.1-mini"] == 200_000

    def test_gpt41_nano_in_database(self) -> None:
        """gpt-4.1-nano has 200K context window."""
        assert "gpt-4.1-nano" in self.windows
        assert self.windows["gpt-4.1-nano"] == 200_000

    def test_groq_models_in_database(self) -> None:
        """groq-llama-4-scout has 512K context window."""
        assert "groq-llama-4-scout" in self.windows
        assert self.windows["groq-llama-4-scout"] == 512_000
        assert "groq-llama-4-maverick" in self.windows
        assert self.windows["groq-llama-4-maverick"] == 512_000

    def test_grok_models_in_database(self) -> None:
        """grok-3 has 131K context window."""
        assert "grok-3" in self.windows
        assert self.windows["grok-3"] == 131_072
        assert "grok-3-mini" in self.windows
        assert self.windows["grok-3-mini"] == 131_072

    def test_claude_opencode_models_in_database(self) -> None:
        """OpenCode-style Claude model IDs are registered."""
        for model in ("claude-4-opus", "claude-4-sonnet", "claude-4.5-sonnet", "claude-3.7-sonnet"):
            assert model in self.windows, f"Missing model: {model}"
            assert self.windows[model] == 200_000

    def test_opencode_compression_triggers_95pct(self) -> None:
        """Most OpenCode-provider models trigger at 0.95."""
        models_at_95 = [
            "claude-4-opus",
            "claude-4-sonnet",
            "claude-4.5-sonnet",
            "claude-3.7-sonnet",
            "claude-3.5-sonnet",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "groq-llama-4-scout",
            "groq-llama-4-maverick",
            "groq-deepseek-r1",
            "groq-qwq",
            "grok-3",
            "grok-3-mini",
        ]
        for model in models_at_95:
            assert model in self.triggers, f"Missing trigger for: {model}"
            assert (
                self.triggers[model] == 0.95
            ), f"Expected 0.95 trigger for {model}, got {self.triggers[model]}"

    def test_o1_pro_trigger_80pct(self) -> None:
        """o1-pro uses 0.80 compression trigger (more conservative)."""
        assert "o1-pro" in self.triggers
        assert self.triggers["o1-pro"] == 0.80

    def test_gemini_flash_lite_in_database(self) -> None:
        """gemini-2.0-flash-lite is registered with 1M context."""
        assert "gemini-2.0-flash-lite" in self.windows
        assert self.windows["gemini-2.0-flash-lite"] == 1_000_000


# ---------------------------------------------------------------------------
# Cache strategy advisor tests
# ---------------------------------------------------------------------------


class TestCacheStrategyAdvisor(unittest.TestCase):
    """Tests for advise_cache_strategy()."""

    def setUp(self) -> None:
        from src.cache_strategy_advisor import advise_cache_strategy

        self.advise = advise_cache_strategy

    def test_anthropic_strategy_explicit(self) -> None:
        """Anthropic models use explicit caching."""
        strategy = self.advise("claude-4-sonnet")
        assert strategy.cache_type == "explicit"
        assert strategy.provider == "anthropic"

    def test_anthropic_strategy_discount_90(self) -> None:
        """Anthropic caching provides 90% discount."""
        strategy = self.advise("claude-4-opus")
        assert strategy.cache_discount_pct == 90

    def test_anthropic_supports_cache(self) -> None:
        """Anthropic models support caching."""
        strategy = self.advise("claude-3.5-sonnet")
        assert strategy.supports_cache is True

    def test_openai_strategy_automatic(self) -> None:
        """OpenAI models use automatic caching."""
        strategy = self.advise("gpt-4.1")
        assert strategy.cache_type == "automatic"
        assert strategy.provider == "openai"

    def test_openai_strategy_discount_50(self) -> None:
        """OpenAI caching provides 50% discount."""
        strategy = self.advise("gpt-4.1-mini")
        assert strategy.cache_discount_pct == 50

    def test_openai_supports_cache(self) -> None:
        """OpenAI models support caching."""
        strategy = self.advise("gpt-4o")
        assert strategy.supports_cache is True

    def test_gemini_25_strategy_implicit(self) -> None:
        """Gemini 2.5+ uses implicit caching."""
        strategy = self.advise("gemini-2.5-flash")
        assert strategy.cache_type == "implicit"
        assert strategy.provider == "google"

    def test_gemini_25_discount_90(self) -> None:
        """Gemini 2.5+ implicit caching provides 90% discount."""
        strategy = self.advise("gemini-2.5-pro")
        assert strategy.cache_discount_pct == 90

    def test_gemini_20_strategy_explicit(self) -> None:
        """Gemini 2.0 uses explicit caching with 75% discount."""
        strategy = self.advise("gemini-2.0-flash")
        assert strategy.cache_type == "explicit"
        assert strategy.cache_discount_pct == 75

    def test_gemini_31_strategy_implicit(self) -> None:
        """Gemini 3.1 uses implicit caching (matches 2.5+ branch)."""
        strategy = self.advise("gemini-3.1-flash")
        assert strategy.cache_type == "implicit"

    def test_groq_strategy_no_cache(self) -> None:
        """Groq models do not support caching."""
        strategy = self.advise("groq-llama-4-scout")
        assert strategy.supports_cache is False
        assert strategy.provider == "groq"

    def test_groq_cache_type_automatic(self) -> None:
        """Groq cache_type is 'automatic' (limited, no discount)."""
        strategy = self.advise("groq-deepseek-r1")
        assert strategy.cache_type == "automatic"
        assert strategy.cache_discount_pct == 0

    def test_grok_strategy_no_cache(self) -> None:
        """xAI Grok models do not support caching."""
        strategy = self.advise("grok-3")
        assert strategy.supports_cache is False
        assert strategy.provider == "xai"

    def test_grok_cache_type_none(self) -> None:
        """Grok cache_type is 'none'."""
        strategy = self.advise("grok-3-mini")
        assert strategy.cache_type == "none"

    def test_local_strategy_no_cache(self) -> None:
        """Local models do not support caching."""
        strategy = self.advise("local-llama3")
        assert strategy.supports_cache is False
        assert strategy.provider == "local"

    def test_ollama_strategy_no_cache(self) -> None:
        """Ollama models also map to local provider."""
        strategy = self.advise("ollama-mistral")
        assert strategy.supports_cache is False
        assert strategy.provider == "local"

    def test_unknown_strategy_fallback(self) -> None:
        """Unrecognised models return unknown cache_type."""
        strategy = self.advise("some-weird-model-xyz")
        assert strategy.cache_type == "unknown"
        assert strategy.provider == "unknown"

    def test_strategy_has_tips(self) -> None:
        """Every strategy must include at least one optimisation tip."""
        for model_id in [
            "claude-4-sonnet",
            "gpt-4.1",
            "gemini-2.5-flash",
            "groq-llama-4-scout",
            "grok-3",
            "local-llama3",
            "some-unknown-model",
        ]:
            strategy = self.advise(model_id)
            assert len(strategy.tips) >= 1, f"No tips for {model_id}"

    def test_strategy_has_client_action(self) -> None:
        """Every strategy must have a non-empty client_action string."""
        for model_id in ["claude-4-sonnet", "gpt-4.1", "gemini-2.5-flash", "grok-3"]:
            strategy = self.advise(model_id)
            assert isinstance(strategy.client_action, str)
            assert len(strategy.client_action) > 0, f"Empty client_action for {model_id}"

    def test_strategy_model_field_preserved(self) -> None:
        """The model field in the returned strategy matches the input model_id."""
        model_id = "claude-4-opus"
        strategy = self.advise(model_id)
        assert strategy.model == model_id


# ---------------------------------------------------------------------------
# Pricing tests
# ---------------------------------------------------------------------------


class TestPricing(unittest.TestCase):
    """Tests for OpenCode additions to PRICING dict."""

    def setUp(self) -> None:
        from src.cli_benchmark.pricing import PRICING, compute_cost

        self.pricing = PRICING
        self.compute_cost = compute_cost

    def test_gpt41_pricing(self) -> None:
        """gpt-4.1 input cost is $2.00 per million tokens."""
        assert "gpt-4.1" in self.pricing
        assert self.pricing["gpt-4.1"]["input"] == 2.0

    def test_gpt41_output_pricing(self) -> None:
        """gpt-4.1 output cost is $8.00 per million tokens."""
        assert self.pricing["gpt-4.1"]["output"] == 8.0

    def test_gpt41_mini_pricing(self) -> None:
        """gpt-4.1-mini input cost is $0.40 per million tokens."""
        assert "gpt-4.1-mini" in self.pricing
        assert self.pricing["gpt-4.1-mini"]["input"] == 0.40

    def test_groq_pricing(self) -> None:
        """groq-llama-4-scout input cost is $0.11 per million tokens."""
        assert "groq-llama-4-scout" in self.pricing
        assert self.pricing["groq-llama-4-scout"]["input"] == 0.11

    def test_groq_maverick_pricing(self) -> None:
        """groq-llama-4-maverick input cost is $0.50 per million tokens."""
        assert "groq-llama-4-maverick" in self.pricing
        assert self.pricing["groq-llama-4-maverick"]["input"] == 0.50

    def test_grok_pricing(self) -> None:
        """grok-3 input cost is $3.00 per million tokens."""
        assert "grok-3" in self.pricing
        assert self.pricing["grok-3"]["input"] == 3.0

    def test_grok_mini_pricing(self) -> None:
        """grok-3-mini input cost is $0.30 per million tokens."""
        assert "grok-3-mini" in self.pricing
        assert self.pricing["grok-3-mini"]["input"] == 0.30

    def test_o1_pro_pricing(self) -> None:
        """o1-pro input cost is $2.00 per million tokens."""
        assert "o1-pro" in self.pricing
        assert self.pricing["o1-pro"]["input"] == 2.0

    def test_compute_cost_gpt41(self) -> None:
        """compute_cost returns correct value for gpt-4.1."""
        # 1M input tokens at $2.00/M = $2.00
        cost = self.compute_cost("gpt-4.1", 1_000_000, 0)
        assert abs(cost - 2.0) < 0.001

    def test_groq_cache_read_zero(self) -> None:
        """Groq models have zero cache_read pricing."""
        assert self.pricing["groq-llama-4-scout"]["cache_read"] == 0.0
        assert self.pricing["groq-llama-4-maverick"]["cache_read"] == 0.0

    def test_grok_cache_read_zero(self) -> None:
        """Grok models have zero cache_read pricing."""
        assert self.pricing["grok-3"]["cache_read"] == 0.0
        assert self.pricing["grok-3-mini"]["cache_read"] == 0.0


# ---------------------------------------------------------------------------
# Benchmark provider tests
# ---------------------------------------------------------------------------


class TestBenchmarkProvider(unittest.TestCase):
    """Tests for OpenCode provider support in cli_benchmark."""

    def test_opencode_is_available_check_found(self) -> None:
        """is_available('opencode') returns True when binary is on PATH."""
        with patch("src.cli_benchmark.providers.shutil.which", return_value="/usr/bin/opencode"):
            from src.cli_benchmark.providers import is_available

            assert is_available("opencode") is True

    def test_opencode_is_available_check_not_found(self) -> None:
        """is_available('opencode') returns False when binary is absent."""
        with patch("src.cli_benchmark.providers.shutil.which", return_value=None):
            from src.cli_benchmark.providers import is_available

            assert is_available("opencode") is False

    def test_opencode_build_command_base(self) -> None:
        """_build_command('opencode', None) produces correct base args."""
        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/bin/opencode"):
            from src.cli_benchmark.providers import _build_command

            cmd = _build_command("opencode", None)
            assert cmd[0] == "/usr/bin/opencode"
            assert "-p" in cmd
            assert "-f" in cmd
            assert "json" in cmd

    def test_opencode_build_command_with_model(self) -> None:
        """_build_command includes --model flag when model is given."""
        with patch("src.cli_benchmark.providers._find_cli", return_value="/usr/bin/opencode"):
            from src.cli_benchmark.providers import _build_command

            cmd = _build_command("opencode", "gpt-4.1")
            assert "--model" in cmd
            assert "gpt-4.1" in cmd

    def test_opencode_build_command_missing_binary_raises(self) -> None:
        """_build_command raises RuntimeError when opencode is not on PATH."""
        with patch("src.cli_benchmark.providers._find_cli", return_value=None):
            from src.cli_benchmark.providers import _build_command

            with self.assertRaises(RuntimeError):
                _build_command("opencode", None)

    def test_unknown_provider_raises(self) -> None:
        """is_available returns False for unknown providers."""
        from src.cli_benchmark.providers import is_available

        assert is_available("notareal_provider") is False

    def test_runner_accepts_model_opencode(self) -> None:
        """run_benchmark signature accepts model_opencode parameter."""
        import inspect
        from src.cli_benchmark.runner import run_benchmark

        sig = inspect.signature(run_benchmark)
        assert "model_opencode" in sig.parameters


# ---------------------------------------------------------------------------
# MCP tool integration test
# ---------------------------------------------------------------------------


class TestMCPCacheStrategyTool(unittest.TestCase):
    """Tests for the advise_cache_strategy MCP tool."""

    def test_tool_registered_in_schema(self) -> None:
        """advise_cache_strategy appears in the MCP tool registry."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        names = {t.name for t in tools}
        assert "advise_cache_strategy" in names

    def test_tool_schema_has_model_id_required(self) -> None:
        """advise_cache_strategy schema requires model_id."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        tool = next(t for t in tools if t.name == "advise_cache_strategy")
        assert "model_id" in tool.inputSchema.get("required", [])

    def _call_handler_sync(self, model_id: str) -> dict:
        import asyncio
        from src.handlers.token_optimization_handlers import handle_advise_cache_strategy

        async def _inner() -> dict:
            result = await handle_advise_cache_strategy({}, {"model_id": model_id})
            return json.loads(result)

        return asyncio.run(_inner())

    def test_handler_returns_anthropic_strategy(self) -> None:
        """Handler returns correct provider for Anthropic model."""
        data = self._call_handler_sync("claude-4-sonnet")
        assert data["status"] == "success"
        assert data["provider"] == "anthropic"
        assert data["cache_type"] == "explicit"

    def test_handler_returns_openai_strategy(self) -> None:
        """Handler returns correct provider for OpenAI model."""
        data = self._call_handler_sync("gpt-4.1")
        assert data["status"] == "success"
        assert data["provider"] == "openai"
        assert data["supports_cache"] is True

    def test_handler_returns_groq_strategy(self) -> None:
        """Handler returns correct provider for Groq model."""
        data = self._call_handler_sync("groq-llama-4-scout")
        assert data["status"] == "success"
        assert data["provider"] == "groq"
        assert data["supports_cache"] is False


if __name__ == "__main__":
    unittest.main()
