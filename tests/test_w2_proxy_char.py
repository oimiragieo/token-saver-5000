"""Characterization tests capturing baseline behavior of proxy components (Wave 2 CL 0)."""

from __future__ import annotations

import json

from src.proxy.proxy_server import ProxyConfig, ProxyServer
from src.proxy.response_interceptor import ResponseInterceptor
from src.proxy.schema_compressor import SchemaCompressor

SAMPLE_TOOLS = [
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern across the workspace.",
        "inputSchema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "long_desc_tool",
        "description": (
            "This tool performs a comprehensive analysis of the entire repository "
            "and returns a detailed report with various metrics, statistics, and "
            "recommendations for improving overall code quality, maintainability, "
            "and performance across all modules in the project."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"target": {"type": "string"}},
            "required": ["target"],
        },
    },
]


class TestBaselineSchemaCompressor:
    def test_meta_tool_schemas_returns_three_tools(self) -> None:
        comp = SchemaCompressor(SAMPLE_TOOLS)
        metas = comp.meta_tool_schemas()
        assert [t["name"] for t in metas] == ["search_tools", "get_tool_schema", "invoke_tool"]
        assert "2 available upstream tools" in metas[0]["description"]

    def test_search_get_invoke_roundtrip(self) -> None:
        comp = SchemaCompressor(SAMPLE_TOOLS)
        search_res = json.loads(comp.handle_meta_tool("search_tools", {"query": "glob"}))
        assert search_res["status"] == "success"
        assert search_res["results"][0]["name"] == "search_files"

        schema_res = json.loads(
            comp.handle_meta_tool("get_tool_schema", {"tool_name": "search_files"})
        )
        assert schema_res["status"] == "success"
        assert schema_res["tool"]["inputSchema"]["required"] == ["pattern"]

        missing_res = json.loads(comp.handle_meta_tool("get_tool_schema", {"tool_name": "nope"}))
        assert missing_res["status"] == "error"

        invoke_res = json.loads(
            comp.handle_meta_tool(
                "invoke_tool", {"tool_name": "search_files", "arguments": {"pattern": "*.py"}}
            )
        )
        assert invoke_res == {
            "_invoke_upstream": True,
            "tool_name": "search_files",
            "arguments": {"pattern": "*.py"},
        }


class TestBaselineResponseInterceptor:
    def test_short_text_passes_through(self) -> None:
        ri = ResponseInterceptor()
        text = "Short output."
        out, stats = ri.intercept_text(text)
        assert out == text
        assert stats.pipeline_stages == []
        assert stats.tokens_saved_estimate == 0

    def test_long_text_compresses(self) -> None:
        ri = ResponseInterceptor()
        text = (
            "The system basically actually processes very large amounts of redundant "
            "information in order to demonstrate that the token refiner and meta-token "
            "compressor reduce repetitive phrasing across multiple sentences. " * 4
        )
        out, stats = ri.intercept_text(text)
        assert stats.original_chars == len(text)
        assert stats.compressed_chars <= len(text)
        assert len(stats.pipeline_stages) >= 1

    def test_intercept_tool_descriptions_preserves_schema(self) -> None:
        ri = ResponseInterceptor()
        processed = ri.intercept_tool_descriptions(SAMPLE_TOOLS)
        assert len(processed) == 2
        assert processed[0]["description"] == SAMPLE_TOOLS[0]["description"]
        assert processed[1]["inputSchema"] == SAMPLE_TOOLS[1]["inputSchema"]
        assert processed[1]["name"] == "long_desc_tool"


class TestBaselineProxyServer:
    def test_default_proxy_server_behavior(self) -> None:
        proxy = ProxyServer(ProxyConfig(upstream_command="echo"))
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        tools = proxy.get_tools(SAMPLE_TOOLS)
        assert [t["name"] for t in tools] == ["search_files", "long_desc_tool"]
        assert proxy.handle_meta_tool_call("search_tools", {"query": "glob"}) is None

    def test_schema_compression_enabled_behavior(self) -> None:
        proxy = ProxyServer(ProxyConfig(upstream_command="echo", enable_schema_compression=True))
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        tools = proxy.get_tools(SAMPLE_TOOLS)
        assert [t["name"] for t in tools] == ["search_tools", "get_tool_schema", "invoke_tool"]
        raw = proxy.handle_meta_tool_call("search_tools", {"query": "glob"})
        assert raw is not None
        assert json.loads(raw)["status"] == "success"

        out, stats = proxy.process_tool_result("custom_tool", "hello " * 50)
        assert isinstance(out, str)
        assert stats["original_chars"] == 300
        assert proxy.metrics.total_calls == 1
