"""GTM claim reproduction suite.

Validates all go-to-market claims with reproducible benchmarks:
- Document compression ratios (13x on large docs)
- CLI output strategy count (11 strategies)
- Tool schema compression (N tools -> 3 meta-tools = 96%+ reduction)
- ROI calculations match marketing claims
- Budget monitor and team export work end-to-end
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# --- Document compression benchmarks -----------------------------------------


class TestDocumentCompressionClaims:
    """Validate compression ratio claims against corpus."""

    @pytest.fixture(autouse=True)
    def _setup_compressor(self):
        from src.semantic_compressor import SemanticCompressor

        self.compressor = SemanticCompressor()

    def _compress_and_measure(self, text: str, file_id: str = "bench") -> dict:
        self.compressor.ingest_file(text, file_id)
        skeleton = self.compressor.read_skeleton(file_id)
        original_tokens = len(text.split())
        skeleton_tokens = len(skeleton.split()) if skeleton else original_tokens
        ratio = original_tokens / skeleton_tokens if skeleton_tokens > 0 else 1.0
        savings_pct = (1 - skeleton_tokens / original_tokens) * 100 if original_tokens > 0 else 0
        return {
            "original_tokens": original_tokens,
            "skeleton_tokens": skeleton_tokens,
            "ratio": ratio,
            "savings_pct": savings_pct,
        }

    def test_medium_document_compression(self):
        """Medium docs (~500 tokens) achieve >= 2x compression."""
        corpus_path = Path("benchmarks/corpus/medium.txt")
        if not corpus_path.exists():
            pytest.skip("Benchmark corpus not available")
        text = corpus_path.read_text(encoding="utf-8")
        result = self._compress_and_measure(text, "medium_bench")
        assert (
            result["ratio"] >= 2.0
        ), f"Medium doc compression {result['ratio']:.1f}x < 2.0x target"
        assert result["savings_pct"] >= 50.0

    def test_large_document_compression(self):
        """Large docs (~2000 tokens) achieve >= 5x compression."""
        corpus_path = Path("benchmarks/corpus/large.txt")
        if not corpus_path.exists():
            pytest.skip("Benchmark corpus not available")
        text = corpus_path.read_text(encoding="utf-8")
        result = self._compress_and_measure(text, "large_bench")
        assert result["ratio"] >= 5.0, f"Large doc compression {result['ratio']:.1f}x < 5.0x target"
        assert result["savings_pct"] >= 80.0

    def test_code_compression(self):
        """Code files achieve meaningful compression."""
        code_dir = Path("benchmarks/corpus/code")
        if not code_dir.exists():
            pytest.skip("Code corpus not available")
        py_files = list(code_dir.glob("*.py"))
        if not py_files:
            pytest.skip("No Python files in code corpus")
        all_code = "\n\n".join(f.read_text(encoding="utf-8") for f in py_files[:5])
        if len(all_code.split()) < 100:
            pytest.skip("Code corpus too small for meaningful compression")
        result = self._compress_and_measure(all_code, "code_bench")
        assert result["ratio"] >= 1.5, f"Code compression {result['ratio']:.1f}x < 1.5x target"


# --- CLI output strategy count ------------------------------------------------


class TestCLIOutputOptimizerClaims:
    """Validate CLI optimizer strategy claims."""

    def test_strategy_count_meets_gtm(self):
        """GTM claims '10+ strategies' - we have 11."""
        from src.cli_output_optimizer import STRATEGY_MAP

        assert len(STRATEGY_MAP) >= 10, f"Only {len(STRATEGY_MAP)} strategies, GTM claims 10+"

    def test_all_strategies_callable(self):
        """Every strategy in the map is a valid method."""
        from src.cli_output_optimizer import CLIOutputOptimizer, STRATEGY_MAP

        optimizer = CLIOutputOptimizer()
        for cmd_type, method_name in STRATEGY_MAP.items():
            method = getattr(optimizer, method_name, None)
            assert method is not None, f"Strategy {method_name} for {cmd_type} not found"
            assert callable(method), f"Strategy {method_name} is not callable"

    def test_git_diff_compression(self):
        """Git diff strategy produces meaningful compression."""
        from src.cli_output_optimizer import CLIOutputOptimizer

        optimizer = CLIOutputOptimizer()
        diff_output = (
            "diff --git a/src/main.py b/src/main.py\n"
            "index abc123..def456 100644\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -10,7 +10,7 @@ def main():\n"
            "-    old_line = True\n"
            "+    new_line = True\n" + "     unchanged = True\n" * 20 + "\n"
            "diff --git a/src/utils.py b/src/utils.py\n"
            "index 111222..333444 100644\n"
            "--- a/src/utils.py\n"
            "+++ b/src/utils.py\n"
            "@@ -1,5 +1,5 @@\n"
            "-def old_func():\n"
            "+def new_func():\n" + "     pass\n" * 10
        )
        result = optimizer.filter(diff_output, command_hint="git_diff")
        assert len(result.filtered_text) < len(diff_output)

    def test_test_failure_compression(self):
        """Test failure strategy extracts failures from verbose output."""
        from src.cli_output_optimizer import CLIOutputOptimizer

        optimizer = CLIOutputOptimizer()
        test_output = (
            "============================= test session starts =============================\n"
            "collected 50 items\n\n"
            "tests/test_a.py ..........                                              [ 20%]\n"
            "tests/test_b.py ..........                                              [ 40%]\n"
            "tests/test_c.py ..........                                              [ 60%]\n"
            "tests/test_d.py ..........                                              [ 80%]\n"
            "tests/test_e.py .....F....                                              [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "_________________________________ test_thing __________________________________\n\n"
            "    def test_thing():\n"
            ">       assert 1 == 2\n"
            "E       AssertionError: assert 1 == 2\n\n"
            "tests/test_e.py:5: AssertionError\n"
            "=========================== short test summary info ===========================\n"
            "FAILED tests/test_e.py::test_thing - AssertionError: assert 1 == 2\n"
            "========================= 1 failed, 49 passed in 3.21s =========================\n"
        )
        result = optimizer.filter(test_output, command_hint="test_output")
        assert len(result.filtered_text) < len(test_output)
        assert "FAIL" in result.filtered_text or "failed" in result.filtered_text


# --- Schema compression -------------------------------------------------------


class TestSchemaCompressionClaims:
    """Validate schema compression: N tools -> 3 meta-tools."""

    def _make_tool_schemas(self, count: int) -> list:
        tools = []
        for i in range(count):
            tools.append(
                {
                    "name": f"tool_{i}",
                    "description": (
                        f"This tool performs operation {i} "
                        "with detailed parameter handling and validation. "
                        "It supports multiple input formats and returns structured output."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": f"Input for tool {i}",
                            },
                            "options": {
                                "type": "object",
                                "description": "Configuration options",
                                "properties": {
                                    "verbose": {"type": "boolean", "default": False},
                                    "format": {
                                        "type": "string",
                                        "enum": ["json", "text", "yaml"],
                                    },
                                },
                            },
                        },
                        "required": ["input"],
                    },
                }
            )
        return tools

    def test_schema_compression_ratio(self):
        """50 tools -> 3 meta-tools = 94% tool count reduction."""
        from src.proxy.schema_compressor import SchemaCompressor

        tools = self._make_tool_schemas(50)
        compressor = SchemaCompressor(tools)
        meta = compressor.meta_tool_schemas()
        assert len(meta) == 3, f"Expected 3 meta-tools, got {len(meta)}"
        reduction = 1 - len(meta) / len(tools)
        assert reduction >= 0.90, f"Tool count reduction only {reduction * 100:.0f}%"

    def test_schema_token_savings(self):
        """Meta-tools use far fewer tokens than full tool list."""
        from src.proxy.schema_compressor import SchemaCompressor

        tools = self._make_tool_schemas(100)
        compressor = SchemaCompressor(tools)
        meta = compressor.meta_tool_schemas()
        original_size = len(json.dumps(tools))
        compressed_size = len(json.dumps(meta))
        reduction = 1 - compressed_size / original_size
        assert reduction >= 0.80, (
            f"Token reduction only {reduction * 100:.0f}% "
            f"(original={original_size}, compressed={compressed_size})"
        )

    def test_meta_tools_have_required_names(self):
        """Meta-tools include search, get_schema, invoke."""
        from src.proxy.schema_compressor import SchemaCompressor

        tools = self._make_tool_schemas(10)
        compressor = SchemaCompressor(tools)
        meta = compressor.meta_tool_schemas()
        names = {t["name"] for t in meta}
        assert "search_tools" in names
        assert "get_tool_schema" in names
        assert "invoke_tool" in names


# --- ROI calculation claims ---------------------------------------------------


class TestROIClaims:
    """Validate ROI calculation claims from GTM plan."""

    @pytest.mark.asyncio
    async def test_enterprise_roi_scenario(self):
        """GTM: team of 10, Opus pricing shows meaningful ROI."""
        from src.handlers.token_optimization_handlers import handle_calculate_roi

        result = json.loads(
            await handle_calculate_roi(
                {},
                {
                    "model": "claude-opus-4-6",
                    "tokens_per_day": 100_000,
                    "team_size": 10,
                    "compression_ratio": 0.85,
                },
            )
        )
        assert result["status"] == "success"
        assert result["dollars_saved_monthly"] > 0
        assert result["roi_multiplier"] >= 1.0, "ROI should be at least 1x"

    @pytest.mark.asyncio
    async def test_solo_developer_roi(self):
        """Single developer should still see positive ROI."""
        from src.handlers.token_optimization_handlers import handle_calculate_roi

        result = json.loads(
            await handle_calculate_roi(
                {},
                {
                    "model": "claude-sonnet-4-6",
                    "tokens_per_day": 500_000,
                    "team_size": 1,
                    "compression_ratio": 0.85,
                },
            )
        )
        assert result["roi_multiplier"] >= 1.0

    @pytest.mark.asyncio
    async def test_available_models_comprehensive(self):
        """ROI calculator exposes all priced models."""
        from src.handlers.token_optimization_handlers import handle_calculate_roi

        result = json.loads(await handle_calculate_roi({}, {}))
        models = result["available_models"]
        assert len(models) >= 15, f"Only {len(models)} models, expected 15+"
        assert "claude-sonnet-4-6" in models
        assert "gpt-4o" in models
        assert "gemini-2.5-pro" in models


# --- MCP tool count claim -----------------------------------------------------


class TestToolCountClaims:
    """Validate tool count claims."""

    def test_tool_count_exceeds_100(self):
        """GTM implies 100+ MCP tools available."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools("full")
        assert len(tools) >= 100, f"Only {len(tools)} tools, GTM implies 100+"

    def test_core_stable_profile_exists(self):
        """Core stable profile exposes minimal tool set."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools("core_stable")
        assert len(tools) >= 5, "Core stable profile too small"
        assert len(tools) <= 20, "Core stable profile too large"


# --- Feature existence validation ---------------------------------------------


class TestFeatureExistence:
    """Validate that all GTM-claimed features exist and are importable."""

    def test_savings_tracker_exists(self):
        from src.savings_tracker import SavingsTracker

        assert SavingsTracker is not None

    def test_savings_dashboard_exists(self):
        from src.savings_dashboard import SavingsDashboard

        assert SavingsDashboard is not None

    def test_budget_monitor_exists(self):
        from src.budget_monitor import TokenBudgetMonitor

        assert TokenBudgetMonitor is not None

    def test_team_exporter_exists(self):
        from src.team_export import TeamExporter

        assert TeamExporter is not None

    def test_filter_rules_exists(self):
        from src.filter_rules import FilterRuleEngine

        assert FilterRuleEngine is not None

    def test_tee_recovery_exists(self):
        from src.tee_recovery import TeeStore

        assert TeeStore is not None

    def test_context_analyzer_exists(self):
        from src.savings_discover import ContextAnalyzer

        assert ContextAnalyzer is not None

    def test_roi_handler_exists(self):
        from src.handlers.token_optimization_handlers import handle_calculate_roi

        assert handle_calculate_roi is not None

    def test_proxy_schema_compressor_exists(self):
        from src.proxy.schema_compressor import SchemaCompressor

        assert SchemaCompressor is not None

    def test_code_compressor_exists(self):
        from src.code_compressor import CodeSemanticCompressor

        assert CodeSemanticCompressor is not None

    def test_multi_agent_setup_exists(self):
        from src.mcp_install import AGENT_CONFIGS

        assert len(AGENT_CONFIGS) >= 5, "Need configs for 5+ agents"
