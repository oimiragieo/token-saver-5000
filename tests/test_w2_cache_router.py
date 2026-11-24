"""Tests for CacheRouter 4-stage filtering, state gating, and prefix stability (Wave 2 CL 3b)."""

from __future__ import annotations

import hashlib
import json

from src.proxy.cache_router import INTERNAL_ROUTER_TOOL_NAME, CacheRouter
from src.proxy.schema_levels import RouterPolicy


def _make_80_tools() -> list[dict]:
    core = [
        {
            "name": "read_file",
            "description": "Read a file from disk.",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
        {
            "name": "bash",
            "description": "Execute a shell command.",
            "inputSchema": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    ]
    long_tail = [
        {
            "name": f"billing_invoice_{i}" if i == 42 else f"service_tool_{i}",
            "description": (
                "Generate and export PDF billing invoice for customer account."
                if i == 42
                else f"Perform specialized operation number {i} on subsystem {i % 7}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"target_id": {"type": "string"}},
                "required": ["target_id"],
            },
        }
        for i in range(78)
    ]
    return core + long_tail


class TestCacheRouterFourStageFilter:
    def test_blacklist_whitelist_bm25_and_yellowlist_fallback(self) -> None:
        tools = _make_80_tools() + [
            {
                "name": "danger_drop_db",
                "description": "Drop production database invoice.",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
        policy = RouterPolicy(blacklist={"danger_drop_db", "bash"})
        router = CacheRouter(tools, policy=policy)

        # Blacklisted danger_drop_db excluded; bash protected because core execution tool
        results = router.filter_tools(query="billing invoice pdf", top_k=5)
        names = [t["name"] for t in results]
        assert "danger_drop_db" not in names
        assert "billing_invoice_42" in names

        # Yellowlist soft-avoidance with top-1 fallback when all candidates yellowlisted
        only_core = router.filter_tools(
            query="read file",
            top_k=1,
            yellowlist={"read_file", "bash"},
        )
        assert len(only_core) == 1
        assert only_core[0]["name"] == "read_file"

    def test_whitelist_restricts_non_core_tools(self) -> None:
        tools = _make_80_tools()
        policy = RouterPolicy(whitelist={"service_tool_3"})
        router = CacheRouter(tools, policy=policy)
        filtered = router.filter_tools(query="operation subsystem", top_k=10)
        names = {t["name"] for t in filtered}
        assert "service_tool_3" in names
        assert "service_tool_4" not in names


class TestStatePreconditionsAndHallucinationGate:
    def test_precondition_gating_and_hallucination_rejection(self) -> None:
        tools = [
            {
                "name": "auth_login",
                "description": "Authenticate session.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"token": {"type": "string"}},
                    "required": ["token"],
                },
                "x-effects": ["authenticated"],
            },
            {
                "name": "deploy_prod",
                "description": "Deploy to production.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"version": {"type": "string"}},
                    "required": ["version"],
                },
                "x-preconditions": ["authenticated"],
            },
        ]
        router = CacheRouter(tools)

        # Unknown tool rejected
        unknown = router.validate_call("nonexistent_tool", {})
        assert unknown["allowed"] is False
        assert unknown["reason"] == "unknown_tool"

        # Precondition unmet before auth_login
        blocked = router.validate_call("deploy_prod", {"version": "1.0.0"})
        assert blocked["allowed"] is False
        assert blocked["reason"] == "precondition_unmet"
        assert blocked["missing_preconditions"] == ["authenticated"]

        # Route auth_login -> does NOT advance state until execution succeeds
        routed_auth = router.route_internal_call(
            {"action": "route", "tool_name": "auth_login", "arguments": {"token": "abc"}}
        )
        assert routed_auth["_invoke_upstream"] is True
        assert "authenticated" not in router.state

        # Failed execution must not commit effects
        router.commit_tool_effects("auth_login", succeeded=False)
        assert "authenticated" not in router.state

        # Successful execution commits effects
        router.commit_tool_effects("auth_login", succeeded=True)
        assert "authenticated" in router.state

        # Invalid arguments still rejected even when precondition satisfied
        bad_args = router.validate_call("deploy_prod", {})
        assert bad_args["allowed"] is False
        assert bad_args["reason"] == "invalid_arguments"

        # Valid call now passes
        ok = router.validate_call("deploy_prod", {"version": "1.0.0"})
        assert ok["allowed"] is True


class TestResidentPrefixStability20Turns:
    def test_resident_tools_byte_identical_across_20_turns_on_80_tool_server(self) -> None:
        tools = _make_80_tools()
        assert len(tools) == 80
        router = CacheRouter(tools)

        initial_resident = router.get_resident_tools()
        initial_bytes = json.dumps(initial_resident, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        initial_sha = hashlib.sha256(initial_bytes).hexdigest()
        assert initial_sha == router.resident_sha256
        assert any(t["name"] == INTERNAL_ROUTER_TOOL_NAME for t in initial_resident)

        for turn in range(20):
            disc = router.route_internal_call(
                {"action": "discover", "query": f"subsystem {turn % 7}", "top_k": 5}
            )
            assert disc["status"] == "success"
            insp = router.route_internal_call(
                {"action": "inspect", "tool_name": f"service_tool_{turn}"}
            )
            assert insp["status"] == "success"
            routed = router.route_internal_call(
                {
                    "action": "route",
                    "tool_name": f"service_tool_{turn}",
                    "arguments": {"target_id": f"id-{turn}"},
                }
            )
            assert routed["_invoke_upstream"] is True

            turn_resident = router.get_resident_tools()
            turn_bytes = json.dumps(turn_resident, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
            assert turn_bytes == initial_bytes
            assert hashlib.sha256(turn_bytes).hexdigest() == initial_sha


class TestSecurityHardeningCacheRouter:
    def test_namespaced_blacklist_whitelist_and_core_tool_normalization(self) -> None:
        policy = RouterPolicy(blacklist={"danger"}, whitelist={"allowed_op"})
        assert policy.is_blacklisted("danger") is True
        assert policy.is_blacklisted("ns.danger") is True
        assert policy.is_blacklisted("mcp:DANGER") is True
        assert policy.is_core_tool("bash") is True
        assert policy.is_core_tool("ns.bash") is True
        assert policy.is_blacklisted("ns.bash") is False
        assert policy.is_whitelisted("allowed_op") is True
        assert policy.is_whitelisted("ns.allowed_op") is True
        assert policy.is_whitelisted("other_op") is False

    def test_whitelist_enforced_in_validate_call_discover_inspect_and_route(self) -> None:
        tools = [
            {"name": "bash", "description": "Shell.", "inputSchema": {"type": "object"}},
            {"name": "allowed_op", "description": "Allowed.", "inputSchema": {"type": "object"}},
            {"name": "blocked_op", "description": "Blocked.", "inputSchema": {"type": "object"}},
        ]
        router = CacheRouter(tools, policy=RouterPolicy(whitelist={"allowed_op"}))

        assert router.validate_call("blocked_op", {})["reason"] == "not_whitelisted"
        assert router.validate_call("allowed_op", {})["allowed"] is True

        disc = router.route_internal_call({"action": "discover", "query": "Blocked"})
        assert all(r["name"] != "blocked_op" for r in disc["results"])

        insp = router.route_internal_call({"action": "inspect", "tool_name": "blocked_op"})
        assert insp["status"] == "error"
        assert insp["reason"] == "not_whitelisted"

        routed = router.route_internal_call(
            {"action": "route", "tool_name": "blocked_op", "arguments": {}}
        )
        assert routed["status"] == "error"
        assert routed["reason"] == "not_whitelisted"

    def test_internal_router_rejects_tunneling_core_tools(self) -> None:
        tools = _make_80_tools()
        router = CacheRouter(tools)
        res = router.route_internal_call(
            {"action": "route", "tool_name": "bash", "arguments": {"command": "ls"}}
        )
        assert res["status"] == "error"
        assert res["reason"] == "resident_tool_direct_only"
        assert "directly" in res["error_message"].lower()

    def test_top_k_validation_and_clamping(self) -> None:
        router = CacheRouter(_make_80_tools())
        bool_res = router.route_internal_call(
            {"action": "discover", "query": "subsystem", "top_k": True}
        )
        assert bool_res["status"] == "error"
        assert bool_res["reason"] == "invalid_top_k"

        bad_res = router.route_internal_call(
            {"action": "discover", "query": "subsystem", "top_k": "not-an-int"}
        )
        assert bad_res["status"] == "error"
        assert bad_res["reason"] == "invalid_top_k"

        clamped_hi = router.route_internal_call(
            {"action": "discover", "query": "subsystem", "top_k": 500}
        )
        assert clamped_hi["status"] == "success"
        assert len(clamped_hi["results"]) <= 20

    def test_validate_arguments_false_respected_in_dual_router(self) -> None:
        tools = _make_80_tools()
        router = CacheRouter(tools)
        res = router.route_internal_call(
            {
                "action": "route",
                "tool_name": "service_tool_0",
                "arguments": {},  # missing required target_id
            },
            validate_arguments=False,
        )
        assert res["_invoke_upstream"] is True
