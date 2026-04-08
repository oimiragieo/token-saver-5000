"""Phase 10 cross-surface launch-readiness tests."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.connector_registry import ConnectorRegistry
from src.dataset_registry import DatasetRegistry
from src.experiment_tracker import ExperimentTracker
from src.memory_api import MemoryAPI
from src.prompt_registry import PromptRegistry


@pytest.fixture
def platform_context():
    MemoryAPI.reset_singleton()
    PromptRegistry.reset_singleton()
    DatasetRegistry.reset_singleton()
    ExperimentTracker.reset_singleton()
    ConnectorRegistry.reset_singleton()

    compressor = Mock()
    compressor.graphs = {}
    compressor.file_metadata = {}
    compressor.chunks = {}
    compressor.ingest_file_async = AsyncMock(
        return_value=SimpleNamespace(total_tokens=100, skeleton_tokens=20, compression_ratio=5.0)
    )

    dataset_registry = DatasetRegistry(seed_defaults=False)

    return {
        "memory_api": MemoryAPI(),
        "prompt_registry": PromptRegistry(seed_defaults=False),
        "dataset_registry": dataset_registry,
        "experiment_tracker": ExperimentTracker(dataset_registry=dataset_registry),
        "connector_registry": ConnectorRegistry.get_registry(),
        "compressor": compressor,
        "resource_manager": Mock(
            check_connector_batch_async=AsyncMock(return_value=(True, None)),
            check_document_size_async=AsyncMock(return_value=(True, None)),
            register_document_async=AsyncMock(),
        ),
        "sync_manager": Mock(register_file=Mock(), export_metadata=Mock(return_value={})),
        "version_manager": Mock(add_version_async=AsyncMock()),
        "persistence": Mock(
            save_document=Mock(return_value=True), save_file_sync_metadata=Mock(return_value=True)
        ),
        "retrieval_history": {},
    }


@pytest.mark.asyncio
async def test_platform_workflow_spans_memory_prompt_experiment_connector_and_model(
    platform_context,
):
    from src.handlers.connector_handlers import (
        handle_create_connector_feed,
        handle_sync_connector_feed,
    )
    from src.handlers.experiment_handlers import handle_create_dataset, handle_run_experiment
    from src.handlers.memory_handlers import (
        handle_add_memory,
        handle_compile_knowledge,
        handle_get_user_profile,
        handle_ingest_transcript,
        handle_lint_knowledge,
    )
    from src.handlers.model_handlers import handle_optimize_for_model
    from src.handlers.prompt_handlers import (
        handle_create_prompt_template,
        handle_get_prompt_template,
    )

    memory_result = json.loads(
        await handle_add_memory(
            platform_context,
            {
                "text": "Prefer aggressive prompt-cache stability and concise summaries.",
                "workspace_id": "acme",
                "user_id": "alice",
            },
        )
    )
    profile_result = json.loads(
        await handle_get_user_profile(
            platform_context, {"workspace_id": "acme", "user_id": "alice"}
        )
    )

    # Knowledge management: ingest transcript, compile, and lint
    transcript_result = json.loads(
        await handle_ingest_transcript(
            platform_context,
            {
                "text": (
                    "We decided to use prompt-cache stability for all templates. "
                    "Watch out for volatile content in the stable prefix."
                ),
                "workspace_id": "acme",
                "user_id": "alice",
            },
        )
    )
    compile_result = json.loads(
        await handle_compile_knowledge(
            platform_context, {"workspace_id": "acme", "user_id": "alice"}
        )
    )
    lint_result = json.loads(
        await handle_lint_knowledge(platform_context, {"workspace_id": "acme", "user_id": "alice"})
    )

    with patch("src.handlers.prompt_handlers.get_metrics") as mock_metrics:
        mock_metrics.return_value = Mock()
        with patch("src.handlers.prompt_handlers.get_observability") as mock_observe:
            mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
            mock_observe.return_value.trace.return_value.__exit__.return_value = None

            prompt_created = json.loads(
                await handle_create_prompt_template(
                    platform_context,
                    {
                        "name": "cache-stable-review",
                        "description": "Prompt for scoped code review handoffs",
                        "system_prompt": "Keep static instructions stable and push the volatile query last.",
                        "user_prompt_template": "Profile: {profile}\nDiff: {diff}",
                        "variables": ["profile", "diff"],
                        "deployment_label": "production",
                    },
                )
            )
            prompt_fetched = json.loads(
                await handle_get_prompt_template(
                    platform_context,
                    {"name": "cache-stable-review", "deployment_label": "production"},
                )
            )

    with patch("src.handlers.experiment_handlers.get_observability") as mock_observe:
        mock_observe.return_value.trace.return_value.__enter__.return_value = Mock()
        mock_observe.return_value.trace.return_value.__exit__.return_value = None
        dataset_created = json.loads(
            await handle_create_dataset(
                platform_context,
                {
                    "name": "phase10-release-gate",
                    "description": "Launch-readiness benchmark subset",
                    "cases": [
                        {
                            "case_id": "tiny",
                            "name": "Tiny case",
                            "text": "Stable prefixes improve provider cache hit rates. " * 20,
                            "min_compression_ratio": 1.1,
                            "min_token_savings_pct": 5.0,
                            "query": "stable prefixes",
                        }
                    ],
                },
            )
        )
        experiment_run = json.loads(
            await handle_run_experiment(
                platform_context,
                {"dataset_name": "phase10-release-gate", "mode": "query_guided"},
            )
        )
    connector_created = json.loads(
        await handle_create_connector_feed(
            platform_context,
            {
                "name": "docs-web",
                "connector_type": "web",
                "config": {"pages": [{"url": "https://example.com/docs", "content": "Docs body"}]},
            },
        )
    )
    connector_sync = json.loads(
        await handle_sync_connector_feed(
            platform_context,
            {"name": "docs-web", "workspace_id": "acme", "user_id": "alice"},
        )
    )
    model_plan = json.loads(
        await handle_optimize_for_model(
            platform_context,
            {
                "model": "claude-sonnet-4.6",
                "text": "Stable prefixes improve provider cache hit rates.",
                "use_case": "question_answering",
                "num_nodes": 4,
            },
        )
    )

    assert memory_result["status"] == "success"
    assert profile_result["profile"]["user_id"] == "alice"
    assert transcript_result["status"] == "success"
    assert compile_result["status"] == "success"
    assert lint_result["status"] == "success"
    assert "checks_run" in lint_result
    assert prompt_created["status"] == "success"
    assert prompt_fetched["resolved_version"]["version"] == 1
    assert dataset_created["status"] == "success"
    assert experiment_run["status"] == "success"
    assert connector_created["status"] == "success"
    assert connector_sync["ingested_documents"] == 1
    assert model_plan["status"] == "success"
    assert connector_sync["results"][0]["doc_id"].startswith("scope__w=acme")


@pytest.mark.asyncio
async def test_workspace_scopes_isolate_memory_and_connector_outputs(platform_context):
    from src.handlers.connector_handlers import (
        handle_create_connector_feed,
        handle_sync_connector_feed,
    )
    from src.handlers.memory_handlers import handle_add_memory, handle_search_memory

    await handle_add_memory(
        platform_context,
        {
            "text": "Acme prefers pytest fixtures.",
            "workspace_id": "acme",
            "user_id": "alice",
        },
    )
    await handle_add_memory(
        platform_context,
        {
            "text": "Beta prefers unittest classes.",
            "workspace_id": "beta",
            "user_id": "bob",
        },
    )

    acme_search = json.loads(
        await handle_search_memory(
            platform_context,
            {"query": "pytest", "workspace_id": "acme", "user_id": "alice"},
        )
    )
    beta_search = json.loads(
        await handle_search_memory(
            platform_context,
            {"query": "pytest", "workspace_id": "beta", "user_id": "bob"},
        )
    )

    await handle_create_connector_feed(
        platform_context,
        {
            "name": "shared-web",
            "connector_type": "web",
            "config": {"pages": [{"url": "https://example.com/docs", "content": "Shared docs"}]},
        },
    )
    acme_sync = json.loads(
        await handle_sync_connector_feed(
            platform_context, {"name": "shared-web", "workspace_id": "acme"}
        )
    )
    beta_sync = json.loads(
        await handle_sync_connector_feed(
            platform_context, {"name": "shared-web", "workspace_id": "beta"}
        )
    )

    assert len(acme_search["results"]) == 1
    assert all(result["workspace_id"] == "acme" for result in acme_search["results"])
    assert all(result["workspace_id"] == "beta" for result in beta_search["results"])
    assert "__w=acme" in acme_sync["results"][0]["doc_id"]
    assert "__w=beta" in beta_sync["results"][0]["doc_id"]
    assert acme_sync["results"][0]["doc_id"] != beta_sync["results"][0]["doc_id"]
