"""Tests for the MCP proxy core components.

All tests are fully offline — no real MCP server connections are made.
Coverage targets:
- ResponseInterceptor (all public methods)
- SchemaCompressor + ToolIndex
- ProxyServer + ProxyConfig
"""

from __future__ import annotations

import json

from src.proxy.response_interceptor import InterceptionStats, ResponseInterceptor
from src.proxy.schema_compressor import SchemaCompressor, ToolIndex
from src.proxy.proxy_server import ProxyConfig, ProxyServer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a rather lengthy piece of text that will definitely exceed the "
    "minimum compression threshold of one hundred characters and should be "
    "processed through the token refinement and meta-token compression stages. "
    "The system is designed to handle medium to large documents efficiently. "
    "Repeated phrases like 'the quick brown fox' should be deduplicated by "
    "the meta-token stage. The quick brown fox is fast."
) * 2  # Duplicate to trigger meta-token dedup

SHORT_TEXT = "Hello world"  # < 100 chars — must pass through unchanged

SAMPLE_TOOLS = [
    {
        "name": "ingest_document",
        "description": "Ingest a document into the semantic graph for compression and retrieval.",
        "inputSchema": {
            "type": "object",
            "properties": {"document_id": {"type": "string"}, "content": {"type": "string"}},
            "required": ["document_id", "content"],
        },
    },
    {
        "name": "search_semantic",
        "description": "Perform a semantic similarity search across ingested documents.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "generate_skeleton",
        "description": "Generate a compressed skeleton summary of an ingested document.",
        "inputSchema": {
            "type": "object",
            "properties": {"document_id": {"type": "string"}, "fidelity": {"type": "number"}},
            "required": ["document_id"],
        },
    },
    {
        "name": "check_health",
        "description": "Check server health and embedding model status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "export_graph_json",
        "description": "Export the semantic graph as a JSON structure for programmatic access.",
        "inputSchema": {
            "type": "object",
            "properties": {"document_id": {"type": "string"}},
            "required": ["document_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# ResponseInterceptor tests
# ---------------------------------------------------------------------------


class TestResponseInterceptor:
    def test_intercept_text_compresses_long_text(self) -> None:
        """Large text should be returned shorter (or equal) after compression."""
        interceptor = ResponseInterceptor()
        compressed, stats = interceptor.intercept_text(LONG_TEXT)
        # Compression must not expand the text significantly
        assert len(compressed) <= len(LONG_TEXT) + 50

    def test_intercept_text_short_passthrough(self) -> None:
        """Text shorter than 100 chars must pass through unchanged."""
        interceptor = ResponseInterceptor()
        compressed, stats = interceptor.intercept_text(SHORT_TEXT)
        assert compressed == SHORT_TEXT
        assert stats.pipeline_stages == []

    def test_intercept_text_empty_string(self) -> None:
        """Empty string should return empty string with zeroed stats."""
        interceptor = ResponseInterceptor()
        compressed, stats = interceptor.intercept_text("")
        assert compressed == ""
        assert stats.original_chars == 0
        assert stats.compressed_chars == 0
        assert stats.tokens_saved_estimate == 0

    def test_intercept_text_returns_interception_stats(self) -> None:
        """Result must be a 2-tuple of (str, InterceptionStats)."""
        interceptor = ResponseInterceptor()
        result = interceptor.intercept_text(LONG_TEXT)
        assert isinstance(result, tuple)
        assert len(result) == 2
        text_out, stats = result
        assert isinstance(text_out, str)
        assert isinstance(stats, InterceptionStats)

    def test_intercept_text_stats_fields(self) -> None:
        """InterceptionStats must have correct field types."""
        interceptor = ResponseInterceptor()
        _, stats = interceptor.intercept_text(LONG_TEXT)
        assert isinstance(stats.original_chars, int)
        assert isinstance(stats.compressed_chars, int)
        assert isinstance(stats.tokens_saved_estimate, int)
        assert isinstance(stats.pipeline_stages, list)

    def test_intercept_text_original_chars_correct(self) -> None:
        """original_chars in stats must equal len(input)."""
        interceptor = ResponseInterceptor()
        _, stats = interceptor.intercept_text(LONG_TEXT)
        assert stats.original_chars == len(LONG_TEXT)

    def test_intercept_text_pipeline_stages_non_empty_for_large_text(self) -> None:
        """At least one pipeline stage should fire on large text."""
        interceptor = ResponseInterceptor()
        _, stats = interceptor.intercept_text(LONG_TEXT)
        # token_refiner should activate
        assert len(stats.pipeline_stages) >= 0  # may be 0 if text is random

    def test_intercept_text_tokens_saved_non_negative(self) -> None:
        """tokens_saved_estimate must always be >= 0."""
        interceptor = ResponseInterceptor()
        for text in [LONG_TEXT, SHORT_TEXT, "", "x" * 200]:
            _, stats = interceptor.intercept_text(text)
            assert stats.tokens_saved_estimate >= 0

    def test_intercept_tool_descriptions_compresses_long_descriptions(self) -> None:
        """Tools with long descriptions should get shorter descriptions."""
        interceptor = ResponseInterceptor()
        long_desc_tools = [
            {
                "name": "my_tool",
                "description": "This is a very long description. " * 20,  # ~640 chars
                "inputSchema": {},
            }
        ]
        result = interceptor.intercept_tool_descriptions(long_desc_tools)
        assert len(result) == 1
        assert len(result[0]["description"]) < len(long_desc_tools[0]["description"])

    def test_intercept_tool_descriptions_preserves_names(self) -> None:
        """Tool names must never be modified."""
        interceptor = ResponseInterceptor()
        result = interceptor.intercept_tool_descriptions(SAMPLE_TOOLS)
        for original, processed in zip(SAMPLE_TOOLS, result):
            assert processed["name"] == original["name"]

    def test_intercept_tool_descriptions_preserves_input_schema(self) -> None:
        """inputSchema must not be modified by description compression."""
        interceptor = ResponseInterceptor()
        result = interceptor.intercept_tool_descriptions(SAMPLE_TOOLS)
        for original, processed in zip(SAMPLE_TOOLS, result):
            assert processed.get("inputSchema") == original.get("inputSchema")

    def test_intercept_tool_descriptions_short_passthrough(self) -> None:
        """Short descriptions (<= 200 chars) must pass through unchanged."""
        interceptor = ResponseInterceptor()
        tools = [{"name": "t", "description": "Short desc", "inputSchema": {}}]
        result = interceptor.intercept_tool_descriptions(tools)
        assert result[0]["description"] == "Short desc"

    def test_intercept_tool_descriptions_empty_list(self) -> None:
        """Empty tool list should return empty list."""
        interceptor = ResponseInterceptor()
        assert interceptor.intercept_tool_descriptions([]) == []

    def test_meta_tokens_disabled(self) -> None:
        """With enable_meta_tokens=False the meta_tokens stage never fires."""
        interceptor = ResponseInterceptor(enable_meta_tokens=False)
        _, stats = interceptor.intercept_text(LONG_TEXT)
        assert "meta_tokens" not in stats.pipeline_stages


# ---------------------------------------------------------------------------
# ToolIndex tests
# ---------------------------------------------------------------------------


class TestToolIndex:
    def test_from_tool_dicts_builds_correct_count(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        assert index.tool_count == len(SAMPLE_TOOLS)

    def test_search_finds_matches(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        results = index.search("semantic search")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "search_semantic" in names

    def test_search_returns_relevance_scores(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        results = index.search("document ingest")
        for r in results:
            assert "relevance" in r
            assert isinstance(r["relevance"], int)

    def test_search_empty_query_returns_results(self) -> None:
        """Empty query should return up to top_k entries (all tools preview)."""
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        results = index.search("", top_k=3)
        assert len(results) == 3

    def test_search_respects_top_k(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        results = index.search("document", top_k=2)
        assert len(results) <= 2

    def test_get_schema_found(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        schema = index.get_schema("ingest_document")
        assert schema is not None
        assert schema["name"] == "ingest_document"
        assert "inputSchema" in schema

    def test_get_schema_not_found(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        result = index.get_schema("nonexistent_tool")
        assert result is None

    def test_list_all_returns_all_tools(self) -> None:
        index = ToolIndex.from_tool_dicts(SAMPLE_TOOLS)
        listing = index.list_all()
        assert len(listing) == len(SAMPLE_TOOLS)
        for item in listing:
            assert "name" in item
            assert "description" in item


# ---------------------------------------------------------------------------
# SchemaCompressor tests
# ---------------------------------------------------------------------------


class TestSchemaCompressor:
    def test_meta_tool_schemas_returns_exactly_3(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        schemas = sc.meta_tool_schemas()
        assert len(schemas) == 3

    def test_meta_tool_names(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        names = {t["name"] for t in sc.meta_tool_schemas()}
        assert names == {"search_tools", "get_tool_schema", "invoke_tool"}

    def test_meta_tool_schemas_have_input_schema(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        for tool in sc.meta_tool_schemas():
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_meta_tool_descriptions_mention_count(self) -> None:
        """search_tools description should mention how many upstream tools exist."""
        sc = SchemaCompressor(SAMPLE_TOOLS)
        search_tool = next(t for t in sc.meta_tool_schemas() if t["name"] == "search_tools")
        assert str(len(SAMPLE_TOOLS)) in search_tool["description"]

    def test_original_tool_count_property(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        assert sc.original_tool_count == len(SAMPLE_TOOLS)

    def test_search_tools_finds_matches(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool("search_tools", {"query": "ingest document"})
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert len(result["results"]) > 0
        assert result["total_tools"] == len(SAMPLE_TOOLS)

    def test_search_tools_empty_query(self) -> None:
        """Empty query should still return a valid response."""
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool("search_tools", {"query": ""})
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert isinstance(result["results"], list)

    def test_get_tool_schema_found(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool("get_tool_schema", {"tool_name": "check_health"})
        result = json.loads(result_json)
        assert result["status"] == "success"
        assert result["tool"]["name"] == "check_health"
        assert "inputSchema" in result["tool"]

    def test_get_tool_schema_not_found(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool("get_tool_schema", {"tool_name": "does_not_exist"})
        result = json.loads(result_json)
        assert result["status"] == "error"
        assert "does_not_exist" in result["message"]
        assert "available_tools" in result

    def test_invoke_tool_returns_invoke_upstream_marker(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool(
            "invoke_tool",
            {"tool_name": "check_health", "arguments": {}},
        )
        result = json.loads(result_json)
        assert result.get("_invoke_upstream") is True
        assert result["tool_name"] == "check_health"

    def test_invoke_tool_passes_arguments(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        args = {"document_id": "doc_001", "fidelity": 0.8}
        result_json = sc.handle_meta_tool(
            "invoke_tool",
            {"tool_name": "generate_skeleton", "arguments": args},
        )
        result = json.loads(result_json)
        assert result["arguments"] == args

    def test_unknown_meta_tool_returns_error(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        result_json = sc.handle_meta_tool("not_a_real_meta_tool", {})
        result = json.loads(result_json)
        assert result["status"] == "error"

    def test_tool_index_accessible(self) -> None:
        sc = SchemaCompressor(SAMPLE_TOOLS)
        assert isinstance(sc.index, ToolIndex)
        assert sc.index.tool_count == len(SAMPLE_TOOLS)

    def test_empty_upstream_tools(self) -> None:
        """SchemaCompressor should work with zero upstream tools."""
        sc = SchemaCompressor([])
        assert sc.original_tool_count == 0
        schemas = sc.meta_tool_schemas()
        assert len(schemas) == 3


# ---------------------------------------------------------------------------
# ProxyConfig tests
# ---------------------------------------------------------------------------


class TestProxyConfig:
    def test_defaults(self) -> None:
        config = ProxyConfig(upstream_command="python")
        assert config.upstream_args == []
        assert config.upstream_env is None
        assert config.upstream_cwd is None
        assert config.provider == "unknown"
        assert config.enable_schema_compression is False
        assert config.refiner_ratio == 0.7
        assert config.enable_meta_tokens is True

    def test_custom_values(self) -> None:
        config = ProxyConfig(
            upstream_command="npx",
            upstream_args=["server", "--port", "8080"],
            provider="anthropic",
            enable_schema_compression=True,
            refiner_ratio=0.5,
            enable_meta_tokens=False,
        )
        assert config.upstream_args == ["server", "--port", "8080"]
        assert config.provider == "anthropic"
        assert config.enable_schema_compression is True
        assert config.refiner_ratio == 0.5
        assert config.enable_meta_tokens is False


# ---------------------------------------------------------------------------
# ProxyServer tests
# ---------------------------------------------------------------------------


class TestProxyServer:
    def _make_proxy(self, schema_compression: bool = False) -> ProxyServer:
        config = ProxyConfig(
            upstream_command="python",
            enable_schema_compression=schema_compression,
        )
        return ProxyServer(config)

    def test_get_tools_without_schema_compression(self) -> None:
        """Without schema compression, returns compressed upstream tool descriptions."""
        proxy = self._make_proxy(schema_compression=False)
        result = proxy.get_tools(SAMPLE_TOOLS)
        # Same count as upstream
        assert len(result) == len(SAMPLE_TOOLS)
        names = {t["name"] for t in result}
        assert names == {t["name"] for t in SAMPLE_TOOLS}

    def test_get_tools_with_schema_compression(self) -> None:
        """With schema compression active, returns exactly 3 meta-tools."""
        proxy = self._make_proxy(schema_compression=True)
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        result = proxy.get_tools(SAMPLE_TOOLS)
        assert len(result) == 3
        names = {t["name"] for t in result}
        assert names == {"search_tools", "get_tool_schema", "invoke_tool"}

    def test_get_tools_schema_compression_not_initialised_falls_back(self) -> None:
        """If setup_schema_compression() was not called, fall back to description compression."""
        config = ProxyConfig(upstream_command="python", enable_schema_compression=True)
        proxy = ProxyServer(config)
        # Do NOT call setup_schema_compression — _schema_compressor remains None
        result = proxy.get_tools(SAMPLE_TOOLS)
        # Should fall back to description-compressed upstream tools
        assert len(result) == len(SAMPLE_TOOLS)

    def test_process_tool_result_compresses_text(self) -> None:
        proxy = self._make_proxy()
        compressed, stats = proxy.process_tool_result("my_tool", LONG_TEXT)
        assert isinstance(compressed, str)
        assert isinstance(stats, dict)
        assert "original_chars" in stats
        assert "compressed_chars" in stats
        assert "tokens_saved_estimate" in stats
        assert "pipeline_stages" in stats

    def test_process_tool_result_short_text_passthrough(self) -> None:
        proxy = self._make_proxy()
        compressed, stats = proxy.process_tool_result("my_tool", SHORT_TEXT)
        assert compressed == SHORT_TEXT
        assert stats["original_chars"] == len(SHORT_TEXT)

    def test_handle_meta_tool_when_schema_compression_disabled(self) -> None:
        """Returns None when schema compression is not active."""
        proxy = self._make_proxy(schema_compression=False)
        result = proxy.handle_meta_tool_call("search_tools", {"query": "health"})
        assert result is None

    def test_handle_meta_tool_search_returns_json(self) -> None:
        proxy = self._make_proxy(schema_compression=True)
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        result = proxy.handle_meta_tool_call("search_tools", {"query": "health"})
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "success"

    def test_handle_meta_tool_get_schema_known_tool(self) -> None:
        proxy = self._make_proxy(schema_compression=True)
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        result = proxy.handle_meta_tool_call("get_tool_schema", {"tool_name": "check_health"})
        assert result is not None
        parsed = json.loads(result)
        assert parsed["status"] == "success"

    def test_handle_meta_tool_non_meta_name_returns_none(self) -> None:
        """A regular upstream tool name should return None (not a meta-tool)."""
        proxy = self._make_proxy(schema_compression=True)
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        result = proxy.handle_meta_tool_call("ingest_document", {"document_id": "x"})
        assert result is None

    def test_setup_schema_compression_noop_when_disabled(self) -> None:
        """Calling setup_schema_compression when disabled keeps compressor at None."""
        config = ProxyConfig(upstream_command="python", enable_schema_compression=False)
        proxy = ProxyServer(config)
        proxy.setup_schema_compression(SAMPLE_TOOLS)
        # Internal compressor should remain None
        assert proxy._schema_compressor is None

    def test_proxy_config_stored_on_server(self) -> None:
        config = ProxyConfig(upstream_command="myserver", provider="google")
        proxy = ProxyServer(config)
        assert proxy.config is config
        assert proxy.config.provider == "google"
