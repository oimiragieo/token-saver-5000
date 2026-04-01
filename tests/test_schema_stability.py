"""
Tests for schema stability in src/handlers/mcp_core.py.

These tests verify that the MCP tool schemas are stable, deterministic,
and free of dynamic content that would cause schema drift across invocations.

A stable schema is critical because:
- Clients cache tool schemas at session start
- Schema drift forces expensive re-negotiation
- Determinism is required for schema hashing / change detection
"""

import hashlib
import json
import re

import pytest

from src.handlers.mcp_core import setup_mcp_tools


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def all_tools():
    """Run setup_mcp_tools() once per module — it must be side-effect-free."""
    return setup_mcp_tools()


@pytest.fixture(scope="module")
def tool_names(all_tools):
    """Sorted list of tool names from the full profile."""
    return [tool.name for tool in all_tools]


# ============================================================================
# Ordering
# ============================================================================


def test_tool_schemas_are_alphabetically_ordered(all_tools):
    """Tools returned by setup_mcp_tools() are in alphabetical order by name."""
    names = [tool.name for tool in all_tools]
    assert names == sorted(names), (
        "Tool names are not alphabetically sorted. "
        f"First out-of-order pair: "
        f"{next((a, b) for a, b in zip(names, sorted(names)) if a != b)}"
    )


# ============================================================================
# Determinism
# ============================================================================


def test_schema_output_is_deterministic():
    """Calling setup_mcp_tools() twice returns identical output."""
    tools_a = setup_mcp_tools()
    tools_b = setup_mcp_tools()

    names_a = [t.name for t in tools_a]
    names_b = [t.name for t in tools_b]
    assert names_a == names_b

    # Compare descriptions for each tool by name
    schema_a = {t.name: t.description for t in tools_a}
    schema_b = {t.name: t.description for t in tools_b}
    assert schema_a == schema_b


def test_schema_hash_stable():
    """SHA256 of all serialized tool names+descriptions is consistent across calls."""

    def _hash_tools():
        tools = setup_mcp_tools()
        payload = json.dumps(
            [{"name": t.name, "description": t.description} for t in tools],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    hash_a = _hash_tools()
    hash_b = _hash_tools()
    assert hash_a == hash_b


# ============================================================================
# No dynamic/runtime content in descriptions
# ============================================================================

# Patterns that indicate dynamic content that should NOT appear in stable schemas
_DYNAMIC_PATTERNS = [
    r"\bv\d+\.\d+",  # version strings like v0.10, v1.2
    r"\bcurrently\b",  # "currently supports"
    r"\bas of\b",  # "as of version"
    r"\btoday\b",  # date-relative language
    r"\bnow\b",  # "now supports"
    r"\d{4}-\d{2}-\d{2}",  # ISO dates like 2024-01-01
    r"\brecently\b",  # relative recency claims
]


def test_tool_descriptions_contain_no_dynamic_content(all_tools):
    """Tool descriptions must not contain version numbers, dates, or temporal language."""
    violations = []
    for tool in all_tools:
        desc = tool.description or ""
        for pattern in _DYNAMIC_PATTERNS:
            match = re.search(pattern, desc, re.IGNORECASE)
            if match:
                violations.append(
                    f"Tool '{tool.name}': description contains dynamic pattern "
                    f"{pattern!r} (matched: {match.group()!r})"
                )

    assert not violations, "Dynamic content found in tool descriptions:\n" + "\n".join(violations)


def test_no_description_contains_runtime_value(all_tools):
    """No description references runtime-computed values (counts, percentages, tool counts)."""
    # Look for patterns like "49 tools", "85% compression", "currently 44"
    count_pattern = re.compile(r"\b\d+\s+tool", re.IGNORECASE)
    violations = []
    for tool in all_tools:
        desc = tool.description or ""
        if count_pattern.search(desc):
            violations.append(f"Tool '{tool.name}': description contains a tool count reference")

    assert not violations, "Tool count references found:\n" + "\n".join(violations)


# ============================================================================
# Schema completeness
# ============================================================================


def test_all_tools_have_non_empty_description(all_tools):
    """Every tool has a non-empty description string."""
    empty = [t.name for t in all_tools if not (t.description or "").strip()]
    assert not empty, f"Tools with empty descriptions: {empty}"


def test_all_tools_have_input_schema(all_tools):
    """Every tool has an inputSchema dict."""
    missing = [t.name for t in all_tools if not t.inputSchema]
    assert not missing, f"Tools missing inputSchema: {missing}"


def test_all_tools_have_unique_names(all_tools):
    """No two tools share the same name."""
    names = [t.name for t in all_tools]
    duplicates = [name for name in names if names.count(name) > 1]
    assert not duplicates, f"Duplicate tool names: {set(duplicates)}"


# ============================================================================
# Core stable profile
# ============================================================================


def test_core_stable_profile_contains_required_tools():
    """core_stable profile must contain the 7 essential tools."""
    core_tools = setup_mcp_tools(profile="core_stable")
    core_names = {t.name for t in core_tools}

    required = {
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "modulate_region",
        "get_stats",
        "list_documents",
        "delete_document",
    }
    missing = required - core_names
    assert not missing, f"core_stable profile missing required tools: {missing}"


def test_core_stable_is_subset_of_full():
    """core_stable profile tools are a subset of full profile tools."""
    full_names = {t.name for t in setup_mcp_tools(profile="full")}
    core_names = {t.name for t in setup_mcp_tools(profile="core_stable")}
    assert core_names.issubset(full_names)


def test_full_profile_has_more_tools_than_core_stable():
    """full profile contains more tools than core_stable."""
    full_count = len(setup_mcp_tools(profile="full"))
    core_count = len(setup_mcp_tools(profile="core_stable"))
    assert full_count > core_count
