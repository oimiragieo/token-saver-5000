"""Phase 0 MCP contract tests for core stable runtime surfaces."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.handlers import compression_handlers, mcp_core

SCOPE_FIELDS = {"workspace_id", "user_id", "agent_id", "session_id"}


def _make_mock_skeleton():
    return SimpleNamespace(
        file_id="phase0_doc",
        total_nodes=4,
        total_tokens=200,
        skeleton_tokens=60,
        compression_ratio=3.3,
        skeleton_text="Stable skeleton output",
        node_map={"phase0_doc_0": "Anchor summary"},
    )


def test_core_stable_schema_contracts_cover_expected_fields():
    """The core-stable MCP profile should keep its minimum contract surface."""
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools(profile="core_stable")}

    assert set(tools) == {
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "modulate_region",
        "get_stats",
        "list_documents",
        "delete_document",
    }

    assert {"text", "file_id", "metadata", "skeleton_ratio", "chunking_strategy"}.issubset(
        tools["ingest_context"].inputSchema["properties"]
    )
    assert {"file_id"} == set(tools["read_skeleton"].inputSchema["required"])
    assert {"selection_mode", "query", "top_k", "min_similarity", "anchored_keywords"}.issubset(
        tools["read_skeleton"].inputSchema["properties"]
    )
    assert {"query"} <= set(tools["search_semantic"].inputSchema["required"])
    assert {"file_id", "top_k", "evidence_aware", "min_similarity"}.issubset(
        tools["search_semantic"].inputSchema["properties"]
    )
    for tool_name in (
        "ingest_context",
        "read_skeleton",
        "search_semantic",
        "get_stats",
        "list_documents",
        "delete_document",
    ):
        assert SCOPE_FIELDS.issubset(tools[tool_name].inputSchema["properties"])


def test_file_sync_tools_accept_scope_fields():
    """Tenant scope should also be exposed on document-adjacent file sync tools."""
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools()}

    for tool_name in (
        "check_file_sync",
        "diff_cached_file",
        "refresh_document",
        "get_version_history",
    ):
        assert SCOPE_FIELDS.issubset(tools[tool_name].inputSchema["properties"])


def test_secondary_document_tools_accept_scope_fields():
    """Secondary document flows should expose the same tenant scope contract."""
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools()}

    for tool_name in (
        "check_blind_spots",
        "detect_hallucination",
        "export_graph_json",
        "visualize_graph_html",
        "export_graph_graphml",
        "explain_compression_decision",
        "diff_reingest",
        "ingest_directory",
        "scar_compress",
        "multimodal_ingest",
        "ingest_multimodal",
        "search_multimodal",
        "create_handoff_bundle",
        "list_handoff_bundles",
        "get_handoff_bundle",
        "replay_handoff_bundle",
    ):
        assert SCOPE_FIELDS.issubset(tools[tool_name].inputSchema["properties"])


def test_model_optimization_tools_expose_expected_contracts():
    """Model-aware tools should expose their minimum schema surface."""
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools()}

    assert {"model"} <= set(tools["get_provider_profile"].inputSchema["required"])
    assert {"model", "original_tokens", "compressed_tokens"} <= set(
        tools["estimate_model_cost"].inputSchema["required"]
    )
    assert {"model", "text", "use_case", "num_nodes"} <= set(
        tools["optimize_for_model"].inputSchema["required"]
    )
    assert {"model", "api_response"} <= set(
        tools["capture_cache_telemetry"].inputSchema["required"]
    )
    assert {"model", "harness"} <= set(tools["assess_cache_compatibility"].inputSchema["required"])


@pytest.mark.asyncio
async def test_ingest_runtime_output_matches_canonical_contract():
    skeleton = _make_mock_skeleton()
    compressor = Mock(
        ingest_file_async=AsyncMock(return_value=skeleton),
        graphs={"phase0_doc": Mock()},
        file_metadata={},
        chunks={},
    )
    resource_manager = Mock(
        check_document_size_async=AsyncMock(return_value=(True, "")),
        register_document_async=AsyncMock(),
    )
    sync_manager = Mock(export_metadata=Mock(return_value=[]), register_file=Mock())
    version_manager = Mock(add_version_async=AsyncMock())
    persistence = Mock(
        save_document=Mock(return_value=True), save_file_sync_metadata=Mock(return_value=True)
    )
    context = {
        "compressor": compressor,
        "resource_manager": resource_manager,
        "sync_manager": sync_manager,
        "version_manager": version_manager,
        "persistence": persistence,
        "retrieval_history": {},
        "path_validator": Mock(),
    }

    with patch("src.handlers.compression_handlers_ingest.CompressionAdvisor") as advisor_cls:
        advisor = Mock()
        advisor.estimate_compression.return_value = SimpleNamespace(
            compression_ratio=3.1, original_tokens=200, estimated_compressed=64
        )
        advisor_cls.return_value = advisor
        payload = json.loads(
            await compression_handlers.handle_ingest(
                context,
                {
                    "text": "This document is long enough for semantic compression characterization.",
                    "file_id": "phase0_doc",
                },
            )
        )

    assert {
        "status",
        "file_id",
        "total_nodes",
        "total_tokens",
        "skeleton_tokens",
        "compression_ratio",
        "token_savings",
        "token_savings_percent",
        "estimate",
        "message",
        "cost_savings",
    } <= set(payload)


@pytest.mark.asyncio
async def test_read_and_search_runtime_outputs_match_canonical_contracts():
    skeleton = _make_mock_skeleton()
    node = SimpleNamespace(
        text="Prompt caching keeps stable prefixes.", importance=0.77, metadata={"tokens": 12}
    )
    compressor = Mock(
        _generate_skeleton=Mock(return_value=skeleton),
        search_semantic_with_scores=Mock(return_value=[("phase0_doc_0", 0.91)]),
        chunks={"phase0_doc_0": node},
        _generate_summary=Mock(return_value="Prompt caching summary"),
        graphs={"phase0_doc": Mock()},
    )
    context = {
        "compressor": compressor,
        "sync_manager": Mock(file_metadata={}),
    }

    read_payload = json.loads(
        await compression_handlers.handle_read_skeleton(
            context,
            {"file_id": "phase0_doc"},
        )
    )
    assert {
        "file_id",
        "total_nodes",
        "total_tokens",
        "skeleton_tokens",
        "compression_ratio",
        "skeleton_text",
        "cache_stable_prefix",
        "node_map",
        "selection_mode",
    } <= set(read_payload)

    search_payload = json.loads(
        await compression_handlers.handle_search_semantic(
            {"compressor": compressor},
            {"query": "stable prefixes", "file_id": "phase0_doc", "top_k": 1},
        )
    )
    assert {
        "query",
        "file_id",
        "evidence_aware",
        "total_results",
        "results",
        "tip",
        "score_explanation",
    } <= set(search_payload)
    assert {"node_id", "similarity", "importance", "summary", "tokens"} <= set(
        search_payload["results"][0]
    )
