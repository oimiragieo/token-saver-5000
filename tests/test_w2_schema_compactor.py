"""Tests for L0-L3 Schema Compactor and <0.5ms Pre-Binding Validator (Wave 2 CL 3a)."""

from __future__ import annotations

import time

from src.proxy.schema_compressor import (
    SchemaCompressor,
    compact_schema_l0,
    compact_schema_l1,
    validate_tool_arguments,
)
from src.proxy.schema_levels import RouterPolicy, SchemaCompressionLevel


class TestL0AndL1Compaction:
    def test_l0_strips_metadata_and_normalizes_whitespace(self) -> None:
        raw = {
            "name": "query_db",
            "description": "  Run   a   SQL\n  query.  ",
            "inputSchema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$comment": "internal note",
                "title": "QueryDbInput",
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "title": "Sql Query",
                        "description": "   ",
                    },
                    "limit": {
                        "type": "integer",
                        "title": "Custom Meaningful Title",
                        "description": "  Max   rows  ",
                        "minimum": 1,
                        "maximum": 100,
                    },
                },
                "required": ["sql_query"],
            },
        }
        out = compact_schema_l0(raw)
        assert out["description"] == "Run a SQL query."
        schema = out["inputSchema"]
        assert "$schema" not in schema
        assert "$comment" not in schema
        assert "title" not in schema
        assert "title" not in schema["properties"]["sql_query"]
        assert "description" not in schema["properties"]["sql_query"]
        assert schema["properties"]["limit"]["title"] == "Custom Meaningful Title"
        assert schema["properties"]["limit"]["description"] == "Max rows"
        assert schema["required"] == ["sql_query"]

    def test_l1_keeps_verbatim_first_sentence_and_up_to_two_constraints(self) -> None:
        raw = {
            "name": "deploy",
            "description": (
                "Deploy a service to the cluster. "
                "It supports rolling updates across regions. "
                "The tag must be valid semver. "
                "It can also emit telemetry events. "
                "Never deploy on Friday without approval. "
                "Only admins can target prod."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        }
        out = compact_schema_l1(raw)
        assert out["description"] == (
            "Deploy a service to the cluster. "
            "The tag must be valid semver. "
            "Never deploy on Friday without approval."
        )


class TestL3AdaptivePinned:
    def test_l3_prefix_sha256_never_shifts_when_tail_tools_unlocked(self) -> None:
        tools = [
            {
                "name": "read_file",
                "description": "Read a file.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "bash",
                "description": "Run shell command.",
                "inputSchema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        ] + [
            {
                "name": f"extra_tool_{i}",
                "description": f"Extra tool number {i} for testing.",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
            for i in range(15)
        ]

        comp = SchemaCompressor(
            tools,
            level=SchemaCompressionLevel.L3_ADAPTIVE_PINNED,
            policy=RouterPolicy(max_schema_tokens=5000),
            top_k_pinned=3,
        )
        turn0 = comp.get_tools()
        sha_turn0 = comp.prefix_sha256
        prefix_len = len(turn0)
        names_turn0 = [t["name"] for t in turn0]
        assert "read_file" in names_turn0
        assert "bash" in names_turn0
        assert "search_tools" in names_turn0

        # Unlock two tail tools across subsequent turns
        assert comp.unlock_tool("extra_tool_10") is True
        assert comp.unlock_tool("extra_tool_12") is True
        assert comp.unlock_tool("extra_tool_10") is False  # idempotent

        turn2 = comp.get_tools()
        assert comp.prefix_sha256 == sha_turn0
        assert turn2[:prefix_len] == turn0
        assert [t["name"] for t in turn2[prefix_len:]] == ["extra_tool_10", "extra_tool_12"]


class TestPreBindingValidator:
    SCHEMA = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "minimum": 1, "maximum": 10},
            "flag": {"type": "boolean"},
            "mode": {"type": "string", "enum": ["fast", "safe"], "minLength": 4},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        },
        "required": ["count", "mode"],
        "additionalProperties": False,
    }

    def test_valid_arguments_pass(self) -> None:
        res = validate_tool_arguments(
            "run",
            {"count": 3, "flag": True, "mode": "fast", "tags": ["a"]},
            self.SCHEMA,
        )
        assert res["valid"] is True
        assert res["errors"] == []

    def test_bool_is_rejected_for_integer_and_number(self) -> None:
        res = validate_tool_arguments("run", {"count": True, "mode": "fast"}, self.SCHEMA)
        assert res["valid"] is False
        assert any("expected type 'integer'" in e for e in res["errors"])

    def test_missing_required_enum_bounds_and_additional_props(self) -> None:
        res = validate_tool_arguments(
            "run",
            {"count": 99, "mode": "bad", "tags": ["a", "b", "c"], "extra": 1},
            self.SCHEMA,
        )
        assert res["valid"] is False
        msg = res["error_message"]
        assert "maximum" in msg
        assert "enum" in msg
        assert "minLength" in msg
        assert "maxItems" in msg
        assert "additional property" in msg

    def test_enum_distinguishes_bool_from_int(self) -> None:
        schema = {
            "type": "object",
            "properties": {"choice": {"enum": [1, 2]}},
            "required": ["choice"],
        }
        assert validate_tool_arguments("t", {"choice": True}, schema)["valid"] is False
        assert validate_tool_arguments("t", {"choice": 1}, schema)["valid"] is True

    def test_validation_is_sub_half_millisecond_and_ignores_redos_pattern(self) -> None:
        evil_schema = {
            "type": "object",
            "properties": {"text": {"type": "string", "pattern": "^(a+)+$"}},
            "required": ["text"],
        }
        evil_input = {"text": "a" * 5000 + "!"}
        t0 = time.perf_counter()
        res = validate_tool_arguments("evil", evil_input, evil_schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res["valid"] is True
        assert elapsed_ms < 0.5

    def test_depth_gt_16_fails_closed(self) -> None:
        # Build nested object schema and payload of depth 18 (> 16)
        schema: dict = {"type": "object", "properties": {"v": {"type": "integer"}}}
        payload: dict = {"v": 1}
        for _ in range(17):
            schema = {"type": "object", "properties": {"next": schema}}
            payload = {"next": payload}

        res = validate_tool_arguments("deep", payload, schema)
        assert res["valid"] is False
        assert any("maximum validation depth" in e for e in res["errors"])

    def test_300k_array_max_items_fails_fast_under_5ms(self) -> None:
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "integer"}, "maxItems": 1}},
            "required": ["items"],
        }
        payload = {"items": list(range(300_000))}
        t0 = time.perf_counter()
        res = validate_tool_arguments("huge_arr", payload, schema)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert res["valid"] is False
        assert any("maxItems" in e for e in res["errors"])
        assert elapsed_ms < 5.0

    def test_node_budget_exceeded_fails_closed(self) -> None:
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "integer"}}},
        }
        payload = {"items": list(range(1500))}
        res = validate_tool_arguments("wide_arr", payload, schema)
        assert res["valid"] is False
        assert any("validation node budget exceeded" in e for e in res["errors"])
