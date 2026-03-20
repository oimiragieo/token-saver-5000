import json

import pytest

from src.handlers import mcp_core
from src.handlers.help_handlers import handle_tool_help


def test_capture_cache_telemetry_tool_schema_contract():
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools()}
    tool = tools["capture_cache_telemetry"]

    assert {"model", "api_response"} <= set(tool.inputSchema["required"])
    assert {
        "model",
        "api_response",
        "expected_cache_hit",
        "file_id",
        "prompt_id",
        "actual_rendered_prefix",
        "session_id",
    } <= set(tool.inputSchema["properties"])


def test_diagnose_cache_miss_tool_schema_contract():
    tools = {tool.name: tool for tool in mcp_core.setup_mcp_tools()}
    tool = tools["diagnose_cache_miss"]

    assert {"prompt_id", "model", "actual_rendered_prefix", "api_response"} <= set(
        tool.inputSchema["required"]
    )


@pytest.mark.asyncio
async def test_capture_cache_telemetry_help_documents_normalized_fields():
    payload = json.loads(
        await handle_tool_help({}, {"tool_name": "capture_cache_telemetry", "verbose": True})
    )

    assert payload["category"] == "Model Optimization"
    assert "telemetry.cache_read_field" in payload["output_fields"]
    assert "telemetry.cache_hit_detected" in payload["output_fields"]
    assert "telemetry.warning" in payload["output_fields"]
    assert "telemetry.observability.cached_input_tokens" in payload["output_fields"]
    assert "telemetry.observability.validation_status" in payload["output_fields"]
    assert "telemetry.validation.status" in payload["output_fields"]
    assert "telemetry.validation.stale_expectation.current_version" in payload["output_fields"]
    assert "telemetry.validation.sibling_coherence.coherence_valid" in payload["output_fields"]
    assert "telemetry.validation.cache_creation_churn.churn_detected" in payload["output_fields"]
    assert "telemetry.validation.section_interleaving.layout_changed" in payload["output_fields"]
    assert "telemetry.validation.diagnostic.probable_cause" in payload["output_fields"]
    assert (
        "telemetry.validation.diagnostic.section_interleaving.layout_changed"
        in payload["output_fields"]
    )
    assert (
        "telemetry.validation.diagnostic.semantic_equivalence.semantic_match"
        in payload["output_fields"]
    )
    assert (
        "telemetry.validation.diagnostic.partial_reuse.partial_reuse_detected"
        in payload["output_fields"]
    )
    assert "telemetry.validation.prefix_integrity.prefix_changed" in payload["output_fields"]
    assert "telemetry.validation.prefix_integrity.actual_prefix_hash" in payload["output_fields"]
    assert (
        "telemetry.validation.prefix_integrity.semantic_equivalence.drift_type"
        in payload["output_fields"]
    )
    assert "telemetry.validation.prefix_integrity.trend.drift_frequency" in payload["output_fields"]
    assert (
        "telemetry.validation.prefix_integrity.trend.systematic_drift_detected"
        in payload["output_fields"]
    )
    assert "telemetry.session_metrics.cache_hits" in payload["output_fields"]
    assert "telemetry.cache_health[].label" in payload["output_fields"]
    assert "telemetry.cache_health[].degraded" in payload["output_fields"]
    assert "telemetry.cache_health[].coherence.skew_detected" in payload["output_fields"]


@pytest.mark.asyncio
async def test_diagnose_cache_miss_help_documents_output_fields():
    payload = json.loads(
        await handle_tool_help({}, {"tool_name": "diagnose_cache_miss", "verbose": True})
    )

    assert payload["category"] == "Model Optimization"
    assert "diagnostic.probable_cause" in payload["output_fields"]
    assert "diagnostic.suggested_remediation" in payload["output_fields"]
    assert "diagnostic.section_interleaving.layout_changed" in payload["output_fields"]
    assert "diagnostic.semantic_equivalence.semantic_match" in payload["output_fields"]
    assert "diagnostic.partial_reuse.partial_reuse_detected" in payload["output_fields"]
