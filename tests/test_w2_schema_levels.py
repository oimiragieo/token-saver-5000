"""Tests for SchemaCompressionLevel and RouterPolicy contracts (Wave 2 CL 1)."""

from __future__ import annotations

import pytest

from src.proxy.schema_levels import (
    DEFAULT_BYTE_EXACT_READ_TOOLS,
    DEFAULT_CORE_PINNED_TOOLS,
    RouterPolicy,
    SchemaCompressionLevel,
    parse_schema_level,
)


class TestSchemaCompressionLevel:
    def test_all_five_levels_defined(self) -> None:
        assert [lvl.value for lvl in SchemaCompressionLevel] == [
            "L0_LOSSLESS",
            "L1_SUMMARY",
            "L2_LAZY",
            "L3_ADAPTIVE_PINNED",
            "DUAL_PATH_ROUTER",
        ]

    def test_parse_schema_level_aliases(self) -> None:
        assert parse_schema_level("l0") is SchemaCompressionLevel.L0_LOSSLESS
        assert parse_schema_level("L1_SUMMARY") is SchemaCompressionLevel.L1_SUMMARY
        assert parse_schema_level("l2-lazy") is SchemaCompressionLevel.L2_LAZY
        assert parse_schema_level("adaptive") is SchemaCompressionLevel.L3_ADAPTIVE_PINNED
        assert parse_schema_level("dual_path_router") is SchemaCompressionLevel.DUAL_PATH_ROUTER
        assert (
            parse_schema_level(SchemaCompressionLevel.L0_LOSSLESS)
            is SchemaCompressionLevel.L0_LOSSLESS
        )

    def test_parse_schema_level_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schema compression level"):
            parse_schema_level("L99_UNKNOWN")


class TestRouterPolicy:
    def test_core_execution_tools_never_stubbed_or_blacklisted(self) -> None:
        policy = RouterPolicy(
            blacklist={"bash", "read_file", "apply_patch", "custom_danger"},
            core_pinned_tools={"my_pinned"},
        )
        for core in ("shell", "bash", "read", "read_file", "edit", "apply_patch"):
            assert core in DEFAULT_CORE_PINNED_TOOLS
            assert policy.is_core_pinned(core)
            assert not policy.is_blacklisted(core)

        assert policy.is_core_pinned("my_pinned")
        assert policy.is_blacklisted("custom_danger")

    def test_byte_exact_read_tools_defaults_and_namespaced(self) -> None:
        policy = RouterPolicy()
        for tool in DEFAULT_BYTE_EXACT_READ_TOOLS:
            assert policy.is_byte_exact_read(tool)
        assert policy.is_byte_exact_read("fs:read_file")
        assert policy.is_byte_exact_read("mcp.view")
        assert not policy.is_byte_exact_read("search_files")

    def test_max_schema_tokens_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_schema_tokens"):
            RouterPolicy(max_schema_tokens=0)
