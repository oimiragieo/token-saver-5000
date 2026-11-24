"""Regression tests for Wave 2 Daybreak Blue & Sol audit blockers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from src.proxy.proxy_server import ProxyConfig, ProxyServer
from src.proxy.schema_compressor import SchemaCompressor, compact_schema_l0
from src.proxy.schema_levels import RouterPolicy, SchemaCompressionLevel
from src.proxy_cli import _dispatch_tool_call, _tool_to_dict


def _sample_tools() -> list[dict]:
    return [
        {
            "name": "read_file",
            "description": "Read a file.",
            "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
        {
            "name": "danger",
            "description": "Dangerous bare tool.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "other:danger",
            "description": "Dangerous namespaced tool.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "safe",
            "description": "Safe bare tool.",
            "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
        },
        {
            "name": "other:safe",
            "description": "Safe namespaced tool.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "extra_tool",
            "description": "Unlisted extra tool.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


class TestSymmetricPolicyNormalization:
    def test_namespaced_blacklist_and_whitelist_are_symmetric(self) -> None:
        policy = RouterPolicy(blacklist={"mcp:danger"}, whitelist={"mcp:safe"})
        assert policy.is_blacklisted("mcp:danger") is True
        assert policy.is_blacklisted("danger") is True
        assert policy.is_blacklisted("other:danger") is True
        assert policy.is_blacklisted("safe") is False

        assert policy.is_whitelisted("mcp:safe") is True
        assert policy.is_whitelisted("safe") is True
        assert policy.is_whitelisted("other:safe") is True
        assert policy.is_whitelisted("extra_tool") is False

    @pytest.mark.parametrize(
        "level",
        [
            SchemaCompressionLevel.L0_LOSSLESS,
            SchemaCompressionLevel.L1_SUMMARY,
            SchemaCompressionLevel.L3_ADAPTIVE_PINNED,
        ],
    )
    def test_policy_enforced_uniformly_across_l0_l1_l3(self, level: SchemaCompressionLevel) -> None:
        policy = RouterPolicy(blacklist={"mcp:danger"}, whitelist={"mcp:safe"})
        comp = SchemaCompressor(_sample_tools(), level=level, policy=policy)
        names = {t["name"] for t in comp.get_tools()}
        assert "danger" not in names
        assert "other:danger" not in names
        assert "extra_tool" not in names
        assert "read_file" in names
        assert "safe" in names

    def test_policy_enforced_in_l2_meta_tools_and_proxy_validate_call(self) -> None:
        config = ProxyConfig(
            upstream_command="echo",
            schema_level=SchemaCompressionLevel.L2_LAZY,
            blacklist=["mcp:danger"],
            whitelist=["mcp:safe"],
        )
        proxy = ProxyServer(config)
        proxy.setup_schema_compression(_sample_tools())

        search = json.loads(proxy.handle_meta_tool_call("search_tools", {"query": ""}) or "{}")
        found = {r["name"] for r in search["results"]}
        assert "danger" not in found
        assert "other:danger" not in found
        assert "extra_tool" not in found
        assert "safe" in found

        assert proxy.validate_call("danger", {})["reason"] == "blacklisted"
        assert proxy.validate_call("other:danger", {})["reason"] == "blacklisted"
        assert proxy.validate_call("extra_tool", {})["reason"] == "not_whitelisted"

        invoke_blocked = json.loads(
            proxy.handle_meta_tool_call("invoke_tool", {"tool_name": "danger", "arguments": {}})
            or "{}"
        )
        assert invoke_blocked["status"] == "error"
        assert invoke_blocked["reason"] == "blacklisted"

        invoke_unlisted = json.loads(
            proxy.handle_meta_tool_call("invoke_tool", {"tool_name": "extra_tool", "arguments": {}})
            or "{}"
        )
        assert invoke_unlisted["status"] == "error"
        assert invoke_unlisted["reason"] == "not_whitelisted"


class TestSearchToolsTopKBounds:
    @pytest.mark.parametrize("bad_top_k", [True, False, "not-an-int", 0, -5, 1.5, None])
    def test_search_tools_rejects_invalid_top_k_cleanly(self, bad_top_k) -> None:
        comp = SchemaCompressor(_sample_tools(), level=SchemaCompressionLevel.L2_LAZY)
        res = json.loads(
            comp.handle_meta_tool("search_tools", {"query": "safe", "top_k": bad_top_k})
        )
        assert res["status"] == "error"
        assert res["reason"] == "invalid_top_k"

    def test_search_tools_clamps_upper_bound(self) -> None:
        many_tools = [
            {
                "name": f"tool_{i}",
                "description": "Searchable tool.",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for i in range(30)
        ]
        comp = SchemaCompressor(many_tools, level=SchemaCompressionLevel.L2_LAZY)
        res = json.loads(comp.handle_meta_tool("search_tools", {"query": "tool", "top_k": 500}))
        assert res["status"] == "success"
        assert len(res["results"]) == 20


class TestInactiveMetaToolsAndMalformedRouterArgs:
    @pytest.mark.parametrize(
        ("level", "inactive_tool"),
        [
            (SchemaCompressionLevel.L0_LOSSLESS, "internal_router"),
            (SchemaCompressionLevel.L0_LOSSLESS, "search_tools"),
            (SchemaCompressionLevel.L1_SUMMARY, "internal_router"),
            (SchemaCompressionLevel.L2_LAZY, "internal_router"),
            (SchemaCompressionLevel.L3_ADAPTIVE_PINNED, "internal_router"),
            (SchemaCompressionLevel.DUAL_PATH_ROUTER, "search_tools"),
        ],
    )
    def test_inactive_meta_tool_rejected(
        self, level: SchemaCompressionLevel, inactive_tool: str
    ) -> None:
        proxy = ProxyServer(ProxyConfig(upstream_command="echo", schema_level=level))
        proxy.setup_schema_compression(_sample_tools())
        gate = proxy.validate_call(inactive_tool, {})
        assert gate["allowed"] is False
        assert gate["reason"] == "unknown_tool"

    def test_malformed_internal_router_arguments_do_not_crash(self) -> None:
        proxy = ProxyServer(
            ProxyConfig(
                upstream_command="echo",
                schema_level=SchemaCompressionLevel.DUAL_PATH_ROUTER,
            )
        )
        proxy.setup_schema_compression(_sample_tools())

        for bad_args in (
            None,
            ["discover"],
            {"action": 123},
            {"action": "discover", "query": ["bad"]},
            {"action": "inspect", "tool_name": 42},
            {"action": "route", "tool_name": ""},
            {"action": "route", "tool_name": "safe", "arguments": ["bad"]},
        ):
            res = json.loads(proxy.handle_meta_tool_call("internal_router", bad_args) or "{}")
            assert res["status"] == "error"


class TestLosslessL0LiteralPreservation:
    def test_l0_preserves_literal_data_in_enum_const_default_examples(self) -> None:
        raw = {
            "name": "literal_tool",
            "description": "Test literal preservation.",
            "inputSchema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$comment": "top-level comment",
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "enum": [{"$schema": "keep", "$comment": "keep", "title": "keep"}],
                        "const": {"$comment": "const_keep", "description": "  keep   spaces  "},
                        "default": {"title": "default_keep", "description": ""},
                        "examples": [{"$schema": "ex_keep", "title": "ex_title"}],
                    }
                },
            },
        }
        out = compact_schema_l0(raw)
        schema = out["inputSchema"]
        assert "$schema" not in schema
        assert "$comment" not in schema
        prop = schema["properties"]["payload"]
        assert prop["enum"] == [{"$schema": "keep", "$comment": "keep", "title": "keep"}]
        assert prop["const"] == {"$comment": "const_keep", "description": "  keep   spaces  "}
        assert prop["default"] == {"title": "default_keep", "description": ""}
        assert prop["examples"] == [{"$schema": "ex_keep", "title": "ex_title"}]


class TestCallToolResultAndPreconditionTransport:
    @pytest.mark.asyncio
    async def test_dispatch_preserves_call_tool_result_model_extra(self) -> None:
        proxy = ProxyServer(ProxyConfig(upstream_command="echo"))
        proxy.setup_schema_compression(_sample_tools())

        upstream_result = CallToolResult(
            content=[TextContent(type="text", text="hello")],
            structuredContent={"ok": True},
            isError=False,
            customExtension="keep-me",
        )

        class _FakeSession:
            async def call_tool(self, name: str, arguments: dict):
                return upstream_result

        out = await _dispatch_tool_call(proxy, _FakeSession(), "safe", {"x": 1})
        assert out.model_extra == {"customExtension": "keep-me"}
        assert out.structuredContent == {"ok": True}
        assert out.isError is False

    @pytest.mark.asyncio
    async def test_tool_to_dict_transports_preconditions_across_dispatch(self) -> None:
        auth_tool = Tool(
            name="auth_login",
            description="Authenticate session.",
            inputSchema={"type": "object", "properties": {}},
            **{"x-effects": ["authenticated"]},
        )
        protected_tool = Tool(
            name="protected_op",
            description="Requires auth.",
            inputSchema={
                "type": "object",
                "properties": {"val": {"type": "string"}},
                "x-preconditions": ["authenticated"],
            },
        )
        upstream_tools = [_tool_to_dict(auth_tool), _tool_to_dict(protected_tool)]

        proxy = ProxyServer(
            ProxyConfig(
                upstream_command="echo",
                schema_level=SchemaCompressionLevel.DUAL_PATH_ROUTER,
            )
        )
        proxy.setup_schema_compression(upstream_tools)

        class _FakeSession:
            async def call_tool(self, name: str, arguments: dict):
                return SimpleNamespace(
                    content=[TextContent(type="text", text=f"ran:{name}")],
                    isError=False,
                    structuredContent=None,
                )

        blocked = await _dispatch_tool_call(
            proxy,
            _FakeSession(),
            "internal_router",
            {"action": "route", "tool_name": "protected_op", "arguments": {"val": "a"}},
        )
        assert blocked.isError is True
        assert "precondition_unmet" in blocked.content[0].text

        authed = await _dispatch_tool_call(
            proxy,
            _FakeSession(),
            "internal_router",
            {"action": "route", "tool_name": "auth_login", "arguments": {}},
        )
        assert authed.isError is False

        allowed = await _dispatch_tool_call(
            proxy,
            _FakeSession(),
            "internal_router",
            {"action": "route", "tool_name": "protected_op", "arguments": {"val": "a"}},
        )
        assert allowed.isError is False
        assert allowed.content[0].text == "ran:protected_op"


class TestSol62RemainingFixes:
    def test_l0_preserves_defs_and_pattern_properties_identifiers(self) -> None:
        from src.proxy.schema_compressor import compact_schema_l0

        tool = {
            "name": "complex_schema",
            "description": "Use complex schema.",
            "inputSchema": {
                "type": "object",
                "$defs": {
                    "description": {"type": "string", "$comment": "strip me"},
                    "$comment": {"type": "integer"},
                },
                "patternProperties": {
                    "^title$": {"type": "string", "title": "^title$"},
                    "^description$": {"type": "boolean"},
                },
            },
        }
        out = compact_schema_l0(tool)
        assert "description" in out["inputSchema"]["$defs"]
        assert "$comment" in out["inputSchema"]["$defs"]
        assert "$comment" not in out["inputSchema"]["$defs"]["description"]
        assert "^title$" in out["inputSchema"]["patternProperties"]
        assert "^description$" in out["inputSchema"]["patternProperties"]

    def test_dict_to_tool_forwards_preconditions_and_effects(self) -> None:
        from mcp.types import Tool
        from src.proxy_cli import _dict_to_tool

        tool = _dict_to_tool(
            {
                "name": "deploy",
                "description": "Deploy.",
                "inputSchema": {"type": "object"},
                "x-preconditions": ["authenticated"],
                "x-effects": ["deployed"],
            },
            Tool,
        )
        roundtrip = _tool_to_dict(tool)
        assert roundtrip["x-preconditions"] == ["authenticated"]
        assert roundtrip["x-effects"] == ["deployed"]
