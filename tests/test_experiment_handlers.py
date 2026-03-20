"""Tests for dataset and experiment MCP handlers."""

import json

import pytest

from src.dataset_registry import DatasetRegistry
from src.experiment_tracker import ExperimentTracker
from src.handlers.mcp_core import route_tool_call, setup_mcp_tools


@pytest.fixture
def experiment_context():
    DatasetRegistry.reset_singleton()
    ExperimentTracker.reset_singleton()
    datasets = DatasetRegistry(seed_defaults=False)
    tracker = ExperimentTracker(dataset_registry=datasets)
    return {"dataset_registry": datasets, "experiment_tracker": tracker}


@pytest.mark.asyncio
async def test_create_list_run_get_and_compare_experiment(experiment_context):
    from src.handlers.experiment_handlers import (
        handle_compare_experiment_runs,
        handle_create_dataset,
        handle_get_experiment_run,
        handle_list_datasets,
        handle_run_experiment,
    )

    created = json.loads(
        await handle_create_dataset(
            experiment_context,
            {
                "name": "release-gate",
                "description": "Release benchmark subset",
                "cases": [
                    {
                        "case_id": "tiny",
                        "name": "Tiny case",
                        "text": "Prompt caching works best with stable prefixes. " * 20,
                        "min_compression_ratio": 1.1,
                        "min_token_savings_pct": 5.0,
                        "query": "stable prefixes",
                    }
                ],
            },
        )
    )
    listed = json.loads(await handle_list_datasets(experiment_context, {}))
    baseline = json.loads(
        await handle_run_experiment(
            experiment_context, {"dataset_name": "release-gate", "mode": "baseline"}
        )
    )
    query_guided = json.loads(
        await handle_run_experiment(
            experiment_context, {"dataset_name": "release-gate", "mode": "query_guided"}
        )
    )
    fetched = json.loads(
        await handle_get_experiment_run(experiment_context, {"run_id": baseline["run"]["run_id"]})
    )
    comparison = json.loads(
        await handle_compare_experiment_runs(
            experiment_context,
            {
                "run_id_a": baseline["run"]["run_id"],
                "run_id_b": query_guided["run"]["run_id"],
            },
        )
    )

    assert created["status"] == "success"
    assert listed["total_datasets"] == 1
    assert fetched["run"]["run_id"] == baseline["run"]["run_id"]
    assert "avg_compression_ratio" in comparison["comparison"]["deltas"]


@pytest.mark.asyncio
async def test_experiment_tools_are_registered_and_routable(experiment_context):
    tools = {tool.name: tool for tool in setup_mcp_tools()}
    assert {
        "create_dataset",
        "list_datasets",
        "run_experiment",
        "get_experiment_run",
        "compare_experiment_runs",
    }.issubset(tools)

    created = json.loads(
        await route_tool_call(
            "create_dataset",
            {
                "name": "release-gate",
                "description": "Release benchmark subset",
                "cases": [
                    {
                        "case_id": "tiny",
                        "name": "Tiny case",
                        "text": "Prompt caching works best with stable prefixes. " * 20,
                        "min_compression_ratio": 1.1,
                        "min_token_savings_pct": 5.0,
                    }
                ],
            },
            experiment_context,
        )
    )
    assert created["status"] == "success"
