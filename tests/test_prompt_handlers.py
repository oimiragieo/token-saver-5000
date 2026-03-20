"""Tests for prompt registry MCP handlers."""

import json
from unittest.mock import Mock, patch

import pytest

from src.prompt_registry import PromptRegistry


@pytest.fixture
def prompt_context():
    PromptRegistry.reset_singleton()
    return {"prompt_registry": PromptRegistry(seed_defaults=False)}


@pytest.mark.asyncio
async def test_create_and_list_prompt_templates(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_list_prompt_templates,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            created = json.loads(
                await handle_create_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "description": "Review prompt",
                        "system_prompt": "You are a reviewer.",
                        "user_prompt_template": "Review {diff}",
                        "variables": ["diff"],
                        "deployment_label": "staging",
                    },
                )
            )
            listed = json.loads(await handle_list_prompt_templates(prompt_context, {}))

    assert created["status"] == "success"
    assert created["resolved_version"]["version"] == 1
    assert created["stable_prefix_analysis"]["impact"] == "initial_prefix_created"
    assert listed["total_templates"] == 1
    assert listed["templates"][0]["deployment_labels"]["staging"] == 1


@pytest.mark.asyncio
async def test_update_deploy_get_and_compare_prompt_templates(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_compare_prompt_versions,
        handle_create_prompt_template,
        handle_deploy_prompt_version,
        handle_get_prompt_template,
        handle_update_prompt_template,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                },
            )
            updated = json.loads(
                await handle_update_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "system_prompt": "You are a reviewer. Prioritize bugs.",
                        "change_note": "bug focus",
                    },
                )
            )
            deployed = json.loads(
                await handle_deploy_prompt_version(
                    prompt_context,
                    {
                        "name": "review-default",
                        "version": 2,
                        "deployment_label": "production",
                        "allow_stable_prefix_change": True,
                    },
                )
            )
            fetched = json.loads(
                await handle_get_prompt_template(
                    prompt_context,
                    {"name": "review-default", "deployment_label": "production"},
                )
            )
            compared = json.loads(
                await handle_compare_prompt_versions(
                    prompt_context,
                    {"name": "review-default", "version_a": 1, "version_b": 2},
                )
            )

    assert updated["resolved_version"]["version"] == 2
    assert deployed["deployment_labels"]["production"] == 2
    assert fetched["resolved_version"]["version"] == 2
    assert compared["changed_fields"] == ["system_prompt"]
    assert compared["stable_prefix_analysis"]["stable_prefix_changed"] is True
    assert updated["stable_prefix_analysis"]["stable_prefix_changed"] is True
    assert "bug focus" == updated["resolved_version"]["change_note"]


@pytest.mark.asyncio
async def test_deploy_prompt_version_handler_reports_prefix_change(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_render_prompt_template,
        handle_deploy_prompt_version,
        handle_update_prompt_template,
    )
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                    "deployment_label": "production",
                },
            )
            await handle_update_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "system_prompt": "You are a reviewer. Prioritize correctness.",
                },
            )
            rendered = json.loads(
                await handle_render_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "deployment_label": "production",
                        "variables": {"diff": "a.py"},
                    },
                )
            )
            deployed = json.loads(
                await handle_deploy_prompt_version(
                    prompt_context,
                    {
                        "name": "review-default",
                        "version": 2,
                        "deployment_label": "production",
                        "allow_stable_prefix_change": True,
                    },
                )
            )

    assert deployed["status"] == "success"
    assert deployed["previous_version"] == 1
    assert deployed["stable_prefix_analysis"]["stable_prefix_changed"] is True
    assert deployed["expectation_invalidations"]["stale_expectations"] == 1
    assert rendered["rendered"]["prompt_id"]


@pytest.mark.asyncio
async def test_deploy_prompt_version_handler_reports_prefix_collisions(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_deploy_prompt_version,
        handle_render_prompt_template,
    )
    from src.prompt_cache_middleware import PromptCacheMiddleware

    PromptCacheMiddleware.reset()

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                    "deployment_label": "production",
                },
            )
            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-security",
                    "description": "Security review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review security for {diff}",
                    "variables": ["diff"],
                },
            )
            await handle_render_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "deployment_label": "production",
                    "variables": {"diff": "a.py"},
                },
            )
            await handle_render_prompt_template(
                prompt_context,
                {
                    "name": "review-security",
                    "variables": {"diff": "b.py"},
                },
            )
            deployed = json.loads(
                await handle_deploy_prompt_version(
                    prompt_context,
                    {
                        "name": "review-security",
                        "version": 1,
                        "deployment_label": "production",
                    },
                )
            )

    assert deployed["status"] == "success"
    assert len(deployed["prefix_collisions"]) == 1
    assert deployed["prefix_collisions"][0]["template_name"] == "review-default"


@pytest.mark.asyncio
async def test_deploy_prompt_version_handler_rejects_unacknowledged_prefix_change(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_deploy_prompt_version,
        handle_update_prompt_template,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                    "deployment_label": "production",
                },
            )
            await handle_update_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "system_prompt": "You are a reviewer. Prioritize correctness.",
                },
            )

            with pytest.raises(ValueError) as exc:
                await handle_deploy_prompt_version(
                    prompt_context,
                    {
                        "name": "review-default",
                        "version": 2,
                        "deployment_label": "production",
                    },
                )

    assert "allow_stable_prefix_change=True" in str(exc.value)


@pytest.mark.asyncio
async def test_audit_prompt_cacheability_handler_flags_ordering_issues(prompt_context):
    from src.handlers.prompt_handlers import handle_audit_prompt_cacheability

    payload = json.loads(
        await handle_audit_prompt_cacheability(
            prompt_context,
            {
                "sections": [
                    {"name": "user_query", "content": "What changed?"},
                    {"name": "system_instructions", "content": "Be accurate."},
                ]
            },
        )
    )

    assert payload["status"] == "success"
    assert payload["audit"]["is_cache_friendly"] is False
    assert any(issue["code"] == "section_order_violation" for issue in payload["audit"]["issues"])


@pytest.mark.asyncio
async def test_render_prompt_template_handler_returns_stability_guard(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_render_prompt_template,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                },
            )
            rendered = json.loads(
                await handle_render_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "variables": {"diff": "a.py"},
                        "rag_context": "Static docs.",
                    },
                )
            )

    assert rendered["status"] == "success"
    assert rendered["rendered"]["stability_guard"]["is_stable"] is True
    assert rendered["rendered"]["stability_guard"]["stable_prefix_hash"]


@pytest.mark.asyncio
async def test_render_prompt_template_handler_can_enforce_stability(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_render_prompt_template,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                },
            )
            with pytest.raises(ValueError) as exc:
                await handle_render_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "variables": {"diff": "a.py"},
                        "tool_definitions": 'tool schema {"request_id":"550e8400-e29b-41d4-a716-446655440000"}',
                        "enforce_stability": True,
                    },
                )

    assert "stability guard" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_render_prompt_template_handler_returns_rendered_sections(prompt_context):
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_render_prompt_template,
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            await handle_create_prompt_template(
                prompt_context,
                {
                    "name": "review-default",
                    "description": "Review prompt",
                    "system_prompt": "You are a reviewer.",
                    "user_prompt_template": "Review {diff}",
                    "variables": ["diff"],
                },
            )
            rendered = json.loads(
                await handle_render_prompt_template(
                    prompt_context,
                    {
                        "name": "review-default",
                        "variables": {"diff": "print('hi')"},
                        "metadata": {"request_id": "req-123"},
                    },
                )
            )

    assert rendered["status"] == "success"
    assert isinstance(rendered["rendered"]["prompt_id"], str)
    assert rendered["rendered"]["sections"][-1]["name"] == "user_query"
    assert rendered["rendered"]["audit"]["is_cache_friendly"] is True
    assert "Review print('hi')" in rendered["rendered"]["rendered_prompt"]
