import hashlib
import json

import pytest

from src.cache_diagnostics import (
    detect_section_interleaving,
    diagnose_cache_miss,
    extract_section_order,
)
from src.handlers.mcp_core import route_tool_call
from src.prompt_cache_middleware import PromptCacheMiddleware


def setup_function():
    PromptCacheMiddleware.reset()


def test_diagnose_cache_miss_detects_framework_uuid_injection():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix_hash": hashlib.sha256(
            "[system_instructions]\nBe accurate.".encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix=(
            '[system_instructions]\nBe accurate.\n{"id":"550e8400-e29b-41d4-a716-446655440000"}'
        ),
        model="gpt-5.4",
        api_response={"usage": {"prompt_tokens": 500, "completion_tokens": 100}},
    )

    assert diagnostic["prefix_changed"] is True
    assert diagnostic["probable_cause"] == "framework_uuid_injection"
    assert diagnostic["framework_signature"]["has_uuid"] is True
    assert "disable dynamic id injection" in diagnostic["suggested_remediation"].lower()


def test_diagnose_cache_miss_detects_timestamp_in_prefix():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix_hash": hashlib.sha256(
            "[system_instructions]\nBe accurate.".encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix=(
            '[system_instructions]\nBe accurate.\n[metadata]\n{"timestamp":"2026-03-16T20:00:00Z"}'
        ),
        model="claude-sonnet-4.6",
        api_response={
            "usage": {"input_tokens": 500, "output_tokens": 100, "cache_read_input_tokens": 0}
        },
    )

    assert diagnostic["probable_cause"] == "volatile_timestamp_in_prefix"
    assert "move timestamps" in diagnostic["suggested_remediation"].lower()


def test_diagnose_cache_miss_reports_provider_eviction_when_prefix_matches():
    stable_prefix = "[system_instructions]\nBe accurate."
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix_hash": hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest(),
        "expected_cache_hit": True,
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix=stable_prefix,
        model="claude-sonnet-4.6",
        api_response={
            "usage": {"input_tokens": 500, "output_tokens": 100, "cache_read_input_tokens": 0}
        },
    )

    assert diagnostic["prefix_changed"] is False
    assert diagnostic["probable_cause"] == "provider_cache_eviction_or_miss"
    assert diagnostic["metrics"]["actual_cached_tokens"] == 0


def test_extract_section_order_reads_rendered_prompt_headers():
    rendered_prompt = (
        "[tool_definitions]\ncall_tool()\n\n"
        "[system_instructions]\nBe accurate.\n\n"
        "[rag_context]\nStatic docs.\n\n"
        "[user_query]\nWhat changed?"
    )

    assert extract_section_order(rendered_prompt) == [
        "tool_definitions",
        "system_instructions",
        "rag_context",
        "user_query",
    ]


def test_detect_section_interleaving_reports_added_stable_sections():
    section_interleaving = detect_section_interleaving(
        expected_section_order=["system_instructions", "rag_context", "user_query"],
        actual_rendered_prefix=(
            "[system_instructions]\nBe accurate.\n\n"
            "[rag_context]\nStatic docs.\n\n"
            "[few_shot_examples]\nQ: hi\nA: hello\n\n"
            "[user_query]\nWhat changed?"
        ),
    )

    assert section_interleaving["layout_changed"] is True
    assert section_interleaving["sections_reordered"] is False
    assert section_interleaving["unexpected_stable_sections"] == ["few_shot_examples"]
    assert "canonical order" in section_interleaving["warning"].lower()


def test_diagnose_cache_miss_detects_section_interleaving_before_uuid_injection():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix_hash": hashlib.sha256(
            "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.".encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
        "expected_section_order": ["system_instructions", "rag_context", "user_query"],
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix=(
            "[system_instructions]\nBe accurate.\n\n"
            "[few_shot_examples]\n550e8400-e29b-41d4-a716-446655440000\n\n"
            "[rag_context]\nStatic docs.\n\n"
            "[user_query]\nWhat changed?"
        ),
        model="gpt-5.4",
        api_response={"usage": {"prompt_tokens": 500, "completion_tokens": 100}},
    )

    assert diagnostic["prefix_changed"] is True
    assert diagnostic["probable_cause"] == "section_interleaving"
    assert diagnostic["section_interleaving"]["layout_changed"] is True
    assert diagnostic["section_interleaving"]["unexpected_stable_sections"] == ["few_shot_examples"]
    assert "canonical section ordering" in diagnostic["suggested_remediation"].lower()


def test_diagnose_cache_miss_detects_whitespace_only_semantic_drift():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
        "cacheable_prefix_hash": hashlib.sha256(
            "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.".encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
        "expected_section_order": ["system_instructions", "rag_context", "user_query"],
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix=(
            "[system_instructions]\r\nBe accurate.   \r\n\r\n\r\n"
            "[rag_context]\r\nStatic docs.\r\n"
        ),
        model="claude-sonnet-4.6",
        api_response={
            "usage": {"input_tokens": 500, "output_tokens": 100, "cache_read_input_tokens": 0}
        },
    )

    assert diagnostic["probable_cause"] == "semantic_equivalence_drift"
    assert diagnostic["semantic_equivalence"]["semantic_match"] is True
    assert diagnostic["semantic_equivalence"]["drift_type"] == "whitespace_only"
    assert "canonical whitespace" in diagnostic["suggested_remediation"].lower()


def test_diagnose_cache_miss_detects_json_serialization_semantic_drift():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix": '[system_instructions]\nBe accurate.\n\n[rag_context]\n{"b":2,"a":1}',
        "cacheable_prefix_hash": hashlib.sha256(
            '[system_instructions]\nBe accurate.\n\n[rag_context]\n{"b":2,"a":1}'.encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
        "expected_section_order": ["system_instructions", "rag_context", "user_query"],
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix='[system_instructions]\nBe accurate.\n\n[rag_context]\n{"a":1,"b":2}',
        model="gpt-5.4",
        api_response={"usage": {"prompt_tokens": 500, "completion_tokens": 100}},
    )

    assert diagnostic["probable_cause"] == "semantic_equivalence_drift"
    assert diagnostic["semantic_equivalence"]["semantic_match"] is True
    assert diagnostic["semantic_equivalence"]["drift_type"] == "json_serialization"
    assert diagnostic["semantic_equivalence"]["json_serialization_changed"] is True


def test_diagnose_cache_miss_detects_partial_cache_reuse_underperformance():
    expectation = {
        "template_name": "review-default",
        "cacheable_prefix": "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
        "cacheable_prefix_hash": hashlib.sha256(
            "[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.".encode("utf-8")
        ).hexdigest(),
        "expected_cache_hit": True,
        "expected_section_order": ["system_instructions", "rag_context", "user_query"],
    }

    diagnostic = diagnose_cache_miss(
        expectation=expectation,
        actual_rendered_prefix="[system_instructions]\nBe accurate.\n\n[rag_context]\nStatic docs.",
        model="claude-sonnet-4.6",
        api_response={
            "usage": {
                "input_tokens": 500,
                "output_tokens": 100,
                "cache_read_input_tokens": 80,
            }
        },
    )

    assert diagnostic["probable_cause"] == "partial_cache_reuse_underperformance"
    assert diagnostic["partial_reuse"]["partial_reuse_detected"] is True
    assert diagnostic["partial_reuse"]["actual_cached_tokens"] == 80
    assert diagnostic["partial_reuse"]["expected_cached_tokens"] == 500
    assert diagnostic["partial_reuse"]["cache_shortfall_tokens"] == 420
    assert "partial cache reuse" in diagnostic["suggested_remediation"].lower()


@pytest.mark.asyncio
async def test_diagnose_cache_miss_handler_returns_actionable_diagnostic():
    from src.handlers.model_handlers import handle_diagnose_cache_miss

    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await handle_diagnose_cache_miss(
            {},
            {
                "prompt_id": prompt_id,
                "model": "gpt-5.4",
                "actual_rendered_prefix": (
                    '[system_instructions]\nBe accurate.\n{"id":"550e8400-e29b-41d4-a716-446655440000"}'
                ),
                "api_response": {
                    "usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 100,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    }
                },
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["diagnostic"]["probable_cause"] == "framework_uuid_injection"
    assert payload["diagnostic"]["metrics"]["actual_cached_tokens"] == 0


@pytest.mark.asyncio
async def test_route_tool_call_dispatches_diagnose_cache_miss():
    prompt_id = PromptCacheMiddleware.record_expectation(
        "review-default",
        {
            "template": {"name": "review-default"},
            "resolved_version": {"version": 1},
            "resolved_labels": ["production"],
            "audit": {"score": 99, "is_cache_friendly": True},
            "cacheable_prefix": "[system_instructions]\nBe accurate.",
            "volatile_suffix": "",
        },
    )

    payload = json.loads(
        await route_tool_call(
            "diagnose_cache_miss",
            {
                "prompt_id": prompt_id,
                "model": "claude-sonnet-4.6",
                "actual_rendered_prefix": (
                    '[system_instructions]\nBe accurate.\n[metadata]\n{"timestamp":"2026-03-16T20:00:00Z"}'
                ),
                "api_response": {
                    "usage": {
                        "input_tokens": 500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 0,
                    }
                },
            },
            {},
        )
    )

    assert payload["status"] == "success"
    assert payload["diagnostic"]["probable_cause"] == "volatile_timestamp_in_prefix"
