"""Tests for the MCP proxy CLI with savings tracking."""

from __future__ import annotations

from src.proxy_cli import main
from src.proxy.proxy_server import ProxyConfig, ProxyServer, ProxySessionMetrics
from src.proxy.response_interceptor import InterceptionStats


class TestProxyCLIDryRun:
    def test_dry_run_prints_config(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["token-saver-proxy", "echo", "hello", "--dry-run"],
        )
        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Token Saver MCP Proxy (dry-run)" in captured.out
        assert "echo" in captured.out

    def test_dry_run_with_options(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            [
                "token-saver-proxy",
                "python",
                "--dry-run",
                "--schema-compression",
                "--refiner-ratio",
                "0.5",
                "--no-meta-tokens",
            ],
        )
        result = main()
        assert result == 0
        captured = capsys.readouterr()
        assert "Schema compression: True" in captured.out
        assert "Refiner ratio: 0.5" in captured.out
        assert "Meta-tokens: False" in captured.out

    def test_dry_run_verbose_flag_accepted(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["token-saver-proxy", "echo", "--dry-run", "--verbose"],
        )
        result = main()
        assert result == 0


class TestProxySavingsTracking:
    def test_proxy_server_process_result_returns_stats(self):
        config = ProxyConfig(upstream_command="echo", upstream_args=["hello"])
        proxy = ProxyServer(config)
        text = "A " * 200
        compressed, stats = proxy.process_tool_result("test_tool", text)
        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "tokens_saved_estimate" in stats
        assert "savings_pct" in stats

    def test_proxy_config_defaults(self):
        config = ProxyConfig(upstream_command="python", upstream_args=["-m", "server"])
        assert config.provider == "unknown"
        assert config.enable_schema_compression is False
        assert config.refiner_ratio == 0.7
        assert config.enable_meta_tokens is True


class TestProxySessionMetrics:
    def test_empty_metrics(self):
        m = ProxySessionMetrics()
        assert m.total_calls == 0
        assert m.savings_pct == 0.0
        assert "0 calls" in m.summary_line()

    def test_record_accumulates(self):
        m = ProxySessionMetrics()
        m.record(
            "tool_a",
            {"original_chars": 1000, "compressed_chars": 400, "tokens_saved_estimate": 150},
        )
        m.record(
            "tool_a", {"original_chars": 500, "compressed_chars": 200, "tokens_saved_estimate": 75}
        )
        m.record(
            "tool_b",
            {"original_chars": 2000, "compressed_chars": 800, "tokens_saved_estimate": 300},
        )
        assert m.total_calls == 3
        assert m.total_original_chars == 3500
        assert m.total_compressed_chars == 1400
        assert m.total_tokens_saved == 525
        assert m.savings_pct == 60.0
        assert m.by_tool["tool_a"]["calls"] == 2
        assert m.by_tool["tool_a"]["tokens_saved"] == 225
        assert m.by_tool["tool_b"]["calls"] == 1

    def test_summary_line_format(self):
        m = ProxySessionMetrics()
        m.record("t", {"original_chars": 100, "compressed_chars": 50, "tokens_saved_estimate": 12})
        line = m.summary_line()
        assert "1 calls" in line
        assert "12 tokens saved" in line
        assert "50%" in line

    def test_proxy_server_accumulates_metrics(self):
        config = ProxyConfig(upstream_command="echo")
        proxy = ProxyServer(config)
        text = "A " * 200
        proxy.process_tool_result("tool_x", text)
        proxy.process_tool_result("tool_x", text)
        assert proxy.metrics.total_calls == 2
        assert proxy.metrics.by_tool["tool_x"]["calls"] == 2


class TestInterceptionStatsProperties:
    def test_savings_pct_positive(self):
        stats = InterceptionStats(
            original_chars=1000, compressed_chars=400, tokens_saved_estimate=150
        )
        assert stats.savings_pct == 60.0

    def test_savings_pct_zero_original(self):
        stats = InterceptionStats(original_chars=0, compressed_chars=0, tokens_saved_estimate=0)
        assert stats.savings_pct == 0.0

    def test_token_estimates(self):
        stats = InterceptionStats(
            original_chars=400, compressed_chars=200, tokens_saved_estimate=50
        )
        assert stats.original_tokens_estimate == 100
        assert stats.compressed_tokens_estimate == 50
