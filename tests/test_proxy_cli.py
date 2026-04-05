"""Tests for the MCP proxy CLI with savings tracking."""

from __future__ import annotations

from src.proxy_cli import main
from src.proxy.proxy_server import ProxyConfig, ProxyServer


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


class TestProxySavingsTracking:
    def test_proxy_server_process_result_returns_stats(self):
        config = ProxyConfig(upstream_command="echo", upstream_args=["hello"])
        proxy = ProxyServer(config)
        text = "A " * 200  # Long enough to be compressed
        compressed, stats = proxy.process_tool_result("test_tool", text)
        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "tokens_saved_estimate" in stats

    def test_proxy_config_defaults(self):
        config = ProxyConfig(upstream_command="python", upstream_args=["-m", "server"])
        assert config.provider == "unknown"
        assert config.enable_schema_compression is False
        assert config.refiner_ratio == 0.7
        assert config.enable_meta_tokens is True
