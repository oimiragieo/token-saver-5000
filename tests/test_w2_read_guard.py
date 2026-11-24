"""Tests for Byte-Exact File-Read Guard and ProxyServer Wave 2 wiring (CL 3c & CL 4)."""

from __future__ import annotations

import json

from src.proxy.proxy_server import ProxyConfig, ProxyServer
from src.proxy.response_interceptor import ResponseInterceptor
from src.proxy.schema_levels import SchemaCompressionLevel
from src.proxy_cli import main

LONG_SOURCE = (
    "def compute_exact_checksum(payload: bytes) -> str:\n"
    "    # This comment and whitespace must stay byte-exact for edit_file!\n"
    "    return payload.hex()\n"
) * 10

TOOLS = [
    {
        "name": "read_file",
        "description": "Read a file byte-for-byte.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "search_logs",
        "description": "Search application logs.",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    },
]


class TestByteExactReadGuard:
    def test_read_tools_never_compressed_unless_forced(self) -> None:
        ri = ResponseInterceptor()
        for tool in (
            "read",
            "read_file",
            "view",
            "open",
            "cat",
            "show",
            "contents",
            "fs:read_file",
        ):
            out, stats = ri.intercept_text(LONG_SOURCE, tool_name=tool)
            assert out == LONG_SOURCE
            assert stats.tokens_saved_estimate == 0
            assert stats.pipeline_stages == ["passthrough:read_guard"]

        forced, forced_stats = ri.intercept_text(
            LONG_SOURCE, tool_name="read_file", force_compress=True
        )
        assert forced_stats.pipeline_stages != ["passthrough:read_guard"]
        assert len(forced) <= len(LONG_SOURCE)

    def test_error_structured_and_non_text_never_compressed(self) -> None:
        ri = ResponseInterceptor()

        err_out, err_stats = ri.intercept_text(LONG_SOURCE, tool_name="search_logs", is_error=True)
        assert err_out == LONG_SOURCE
        assert err_stats.pipeline_stages == ["passthrough:error_guard"]

        struct_out, struct_stats = ri.intercept_text(
            LONG_SOURCE, tool_name="search_logs", structured_content={"ok": True}
        )
        assert struct_out == LONG_SOURCE
        assert struct_stats.pipeline_stages == ["passthrough:structured_guard"]

        mcp_res = {
            "content": [
                {"type": "image", "data": "base64==", "mimeType": "image/png"},
                {"type": "text", "text": LONG_SOURCE},
            ]
        }
        out_res, res_stats = ri.intercept_mcp_result(mcp_res, tool_name="search_logs")
        assert out_res["content"][0] == mcp_res["content"][0]
        assert "passthrough:non_text_guard" in res_stats.pipeline_stages


class TestProxyServerWave2Wiring:
    def test_dual_path_router_and_prebinding_validation(self) -> None:
        config = ProxyConfig(
            upstream_command="echo",
            schema_level=SchemaCompressionLevel.DUAL_PATH_ROUTER,
            pin_tools=["search_logs"],
        )
        proxy = ProxyServer(config)
        proxy.setup_schema_compression(TOOLS)

        tools = proxy.get_tools(TOOLS)
        names = [t["name"] for t in tools]
        assert "read_file" in names
        assert "search_logs" in names
        assert "internal_router" in names

        # Pre-binding validation rejects missing required field
        bad_gate = proxy.validate_call("search_logs", {"limit": "not-int"})
        assert bad_gate["allowed"] is False
        assert bad_gate["reason"] == "invalid_arguments"

        # internal_router routes valid call
        routed = json.loads(
            proxy.handle_meta_tool_call(
                "internal_router",
                {
                    "action": "route",
                    "tool_name": "search_logs",
                    "arguments": {"query": "timeout", "limit": 10},
                },
            )
            or "{}"
        )
        assert routed["_invoke_upstream"] is True

        # Byte-exact read guard enforced through ProxyServer
        out, stats = proxy.process_tool_result("read_file", LONG_SOURCE)
        assert out == LONG_SOURCE
        assert stats["pipeline_stages"] == ["passthrough:read_guard"]


class TestProxyCLIWave2Flags:
    def test_cli_dry_run_accepts_wave2_flags(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "token-saver-proxy",
                "python",
                "--dry-run",
                "--schema-level",
                "DUAL_PATH_ROUTER",
                "--max-schema-tokens",
                "2048",
                "--pin-tools",
                "read_file,bash",
            ],
        )
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Schema level: DUAL_PATH_ROUTER" in out
        assert "Max schema tokens: 2048" in out
        assert "Pinned tools: read_file,bash" in out

    def test_cli_dry_run_accepts_blacklist_and_whitelist(self, capsys, monkeypatch) -> None:
        monkeypatch.setattr(
            "sys.argv",
            [
                "token-saver-proxy",
                "python",
                "--dry-run",
                "--schema-level",
                "DUAL_PATH_ROUTER",
                "--blacklist",
                "danger_tool,drop_db",
                "--whitelist",
                "safe_tool",
            ],
        )
        rc = main()
        assert rc == 0
        out = capsys.readouterr().out
        assert "Blacklist: danger_tool,drop_db" in out
        assert "Whitelist: safe_tool" in out


class TestDualPathBoundaryAndCallToolResult:
    def test_dual_path_rejects_direct_non_resident_calls_and_defers_effects(self) -> None:
        tools = [
            {
                "name": "read_file",
                "description": "Read a file.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
            {
                "name": "auth_login",
                "description": "Login.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
                "x-effects": ["authenticated"],
            },
        ]
        config = ProxyConfig(
            upstream_command="echo",
            schema_level=SchemaCompressionLevel.DUAL_PATH_ROUTER,
            validate_arguments=False,
        )
        proxy = ProxyServer(config)
        proxy.setup_schema_compression(tools)

        # Direct call to non-resident long-tail tool is rejected
        direct_gate = proxy.validate_call("auth_login", {"token": "abc"})
        assert direct_gate["allowed"] is False
        assert direct_gate["reason"] == "non_resident_tool"
        assert "internal_router" in direct_gate["error_message"]

        # Resident tool + internal_router allowed directly
        assert proxy.validate_call("read_file", {"path": "x"})["allowed"] is True
        assert proxy.validate_call("internal_router", {})["allowed"] is True

        # Route via internal_router with validate_arguments=False
        routed = json.loads(
            proxy.handle_meta_tool_call(
                "internal_router",
                {"action": "route", "tool_name": "auth_login", "arguments": {}},
            )
            or "{}"
        )
        assert routed["_invoke_upstream"] is True
        assert proxy.validate_call("auth_login", {}, via_internal_router=True)["allowed"] is True
        assert "authenticated" not in proxy._cache_router.state

        # Error result does not commit state effects; success result does
        proxy.process_tool_result("auth_login", "failed", is_error=True)
        assert "authenticated" not in proxy._cache_router.state
        proxy.process_tool_result("auth_login", "ok", is_error=False)
        assert "authenticated" in proxy._cache_router.state

    def test_dispatch_tool_call_preserves_call_tool_result_semantics(self) -> None:
        import asyncio
        from mcp.types import Annotations, CallToolResult, ImageContent, TextContent
        from src.proxy_cli import _dispatch_tool_call

        config = ProxyConfig(upstream_command="echo")
        proxy = ProxyServer(config)
        proxy.setup_schema_compression(TOOLS)

        img = ImageContent(type="image", data="YmFzZTY0", mimeType="image/png")
        txt = TextContent(
            type="text",
            text=LONG_SOURCE,
            annotations=Annotations(audience=["assistant"]),
            _meta={"trace": "t1"},
        )
        upstream_result = CallToolResult(
            content=[img, txt],
            structuredContent={"ok": True},
            isError=True,
            _meta={"req": "r1"},
        )

        class _FakeSession:
            async def call_tool(self, name: str, arguments: dict):
                return upstream_result

        res = asyncio.run(
            _dispatch_tool_call(proxy, _FakeSession(), "search_logs", {"query": "err"})
        )
        assert isinstance(res, CallToolResult)
        assert res.isError is True
        assert res.structuredContent == {"ok": True}
        assert res.meta == {"req": "r1"}
        assert res.content[0] == img
        assert res.content[1].text == LONG_SOURCE
        assert res.content[1].annotations == Annotations(audience=["assistant"])
        assert res.content[1].meta == {"trace": "t1"}
