import json

import pytest

from src.prompt_cache_audit import audit_prompt_cacheability
from src.prompt_cache_stability_guard import evaluate_prompt_stability


def test_audit_prompt_cacheability_accepts_canonical_order():
    audit = audit_prompt_cacheability(
        [
            {"name": "tool_definitions", "content": "tool A\ntool B"},
            {"name": "system_instructions", "content": "You are a precise assistant."},
            {"name": "rag_context", "content": "Static product docs."},
            {"name": "chat_history", "content": "User asked about billing before."},
            {"name": "metadata", "content": '{"workspace_id":"acme"}'},
            {"name": "user_query", "content": "How do retries work?"},
        ]
    )

    assert audit["is_cache_friendly"] is True
    assert audit["score"] == 100
    assert audit["issues"] == []
    assert "[tool_definitions]" in audit["cacheable_prefix"]
    assert audit["volatile_suffix"].startswith("[metadata]")


def test_audit_prompt_cacheability_flags_out_of_order_query():
    audit = audit_prompt_cacheability(
        [
            {"name": "tool_definitions", "content": "tool A"},
            {"name": "user_query", "content": "What changed?"},
            {"name": "rag_context", "content": "Static docs."},
        ]
    )

    assert audit["is_cache_friendly"] is False
    assert audit["score"] < 100
    assert any(issue["code"] == "section_order_violation" for issue in audit["issues"])


def test_audit_prompt_cacheability_flags_volatile_ids_in_stable_prefix():
    audit = audit_prompt_cacheability(
        [
            {
                "name": "system_instructions",
                "content": "Assistant bootstrap. request_id=550e8400-e29b-41d4-a716-446655440000",
            },
            {"name": "user_query", "content": "Help"},
        ]
    )

    assert audit["is_cache_friendly"] is False
    assert any(issue["code"] == "volatile_content_in_stable_section" for issue in audit["issues"])


@pytest.mark.asyncio
async def test_audit_prompt_cacheability_handler_returns_structured_result():
    from src.handlers.prompt_handlers import handle_audit_prompt_cacheability

    payload = json.loads(
        await handle_audit_prompt_cacheability(
            {},
            {
                "sections": [
                    {"name": "system_instructions", "content": "Be accurate."},
                    {"name": "metadata", "content": '{"timestamp":"2026-03-16T00:00:00Z"}'},
                    {"name": "user_query", "content": "Explain auth."},
                ]
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["audit"]["recommended_order"][0] == "tool_definitions"
    assert payload["audit"]["volatile_suffix"].startswith("[metadata]")


def test_evaluate_prompt_stability_returns_prefix_fingerprint():
    stability = evaluate_prompt_stability(
        [
            {"name": "tool_definitions", "content": "tool A"},
            {"name": "system_instructions", "content": "Be accurate."},
            {"name": "rag_context", "content": "Static context."},
            {"name": "user_query", "content": "What changed?"},
        ]
    )

    assert stability["is_stable"] is True
    assert stability["stable_prefix_hash"]
    assert stability["order_valid"] is True
    assert stability["violations"] == []


def test_evaluate_prompt_stability_flags_volatile_content_for_enforcement():
    stability = evaluate_prompt_stability(
        [
            {
                "name": "tool_definitions",
                "content": 'tool schema {"request_id":"550e8400-e29b-41d4-a716-446655440000"}',
            },
            {"name": "system_instructions", "content": "Be accurate."},
            {"name": "user_query", "content": "What changed?"},
        ]
    )

    assert stability["is_stable"] is False
    assert any(
        violation["code"] == "volatile_content_in_stable_prefix"
        for violation in stability["violations"]
    )
