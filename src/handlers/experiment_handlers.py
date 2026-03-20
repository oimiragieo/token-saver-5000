"""Handlers for datasets and tracked experiment runs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from ..dataset_registry import DatasetRegistry
from ..experiment_tracker import ExperimentTracker
from ..metrics import get_metrics
from ..observability import get_observability

logger = logging.getLogger("semantic-modulator")


def _dataset_registry(context: dict[str, Any]) -> DatasetRegistry:
    return context.get("dataset_registry") or DatasetRegistry.get_registry()


def _experiment_tracker(context: dict[str, Any]) -> ExperimentTracker:
    return context.get("experiment_tracker") or ExperimentTracker.get_tracker()


def _record_metrics(start_time: float, status: str, error_type: str | None = None) -> None:
    metrics = get_metrics()
    try:
        metrics.record_latency("experiment_tracker", time.perf_counter() - start_time, "NONE")
        metrics.increment_documents_processed("experiment_tracker", "NONE", status)
        if error_type is not None:
            metrics.increment_errors(error_type, "experiment_tracker")
    except Exception as exc:
        logger.warning(f"Experiment tracker metrics update failed: {exc}")


def _required_string(args: dict[str, Any], field: str, tool_name: str) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires a non-empty '{field}' field")
    return value.strip()


def _optional_string(args: dict[str, Any], field: str) -> str | None:
    value = args.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field}' must be a string when provided")
    return value.strip()


def get_dataset_output_fields() -> list[str]:
    return [
        "status",
        "dataset.name",
        "dataset.case_count",
        "dataset.source",
        "datasets[].name",
        "datasets[].case_count",
    ]


def get_experiment_output_fields() -> list[str]:
    return [
        "status",
        "run.run_id",
        "run.dataset_name",
        "run.summary.total_cases",
        "run.summary.avg_compression_ratio",
        "run.verification_pass_rate",
        "run.avg_reward",
        "comparison.deltas.avg_compression_ratio",
        "comparison.deltas.avg_reward",
    ]


async def handle_create_dataset(context: dict[str, Any], args: dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _dataset_registry(context)
    observe = get_observability()
    name = _required_string(args, "name", "create_dataset")
    description = _required_string(args, "description", "create_dataset")
    cases = args.get("cases")
    source_path = _optional_string(args, "source_path")
    metadata = args.get("metadata") or {}
    if metadata and not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided")
    if cases is not None and not isinstance(cases, list):
        raise ValueError("'cases' must be a list when provided")

    try:
        with observe.trace("experiment_tracker.create_dataset", dataset_name=name):
            dataset = registry.create_dataset(
                name=name,
                description=description,
                cases=cases,
                source_path=source_path,
                metadata=metadata,
            )
            _record_metrics(start_time, "success")
            return json.dumps(
                {
                    "status": "success",
                    "dataset": dataset,
                    "message": f"Created dataset '{name}'",
                },
                indent=2,
            )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_list_datasets(context: dict[str, Any], args: dict[str, Any]) -> str:
    start_time = time.perf_counter()
    registry = _dataset_registry(context)
    try:
        datasets = registry.list_datasets()
        _record_metrics(start_time, "success")
        return json.dumps(
            {
                "status": "success",
                "total_datasets": len(datasets),
                "datasets": datasets,
            },
            indent=2,
        )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_run_experiment(context: dict[str, Any], args: dict[str, Any]) -> str:
    start_time = time.perf_counter()
    tracker = _experiment_tracker(context)
    observe = get_observability()
    dataset_name = _required_string(args, "dataset_name", "run_experiment")
    mode = _optional_string(args, "mode") or "baseline"
    case_ids = args.get("case_ids") or []
    if not isinstance(case_ids, list) or not all(isinstance(item, str) for item in case_ids):
        raise ValueError("'case_ids' must be a list of strings when provided")
    similarity_threshold = args.get("similarity_threshold", 0.75)
    skeleton_ratio = args.get("skeleton_ratio", 0.2)
    if not isinstance(similarity_threshold, (int, float)):
        raise ValueError("'similarity_threshold' must be numeric")
    if not isinstance(skeleton_ratio, (int, float)):
        raise ValueError("'skeleton_ratio' must be numeric")
    metadata = args.get("metadata") or {}
    if metadata and not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided")

    try:
        with observe.trace("experiment_tracker.run", dataset_name=dataset_name, mode=mode):
            run = await asyncio.to_thread(
                tracker.run_experiment,
                dataset_name=dataset_name,
                mode=mode,
                case_ids=case_ids,
                similarity_threshold=float(similarity_threshold),
                skeleton_ratio=float(skeleton_ratio),
                baseline_run_id=_optional_string(args, "baseline_run_id"),
                metadata=metadata,
            )
            _record_metrics(start_time, "success")
            return json.dumps(
                {
                    "status": "success",
                    "run": run,
                    "message": f"Completed experiment run '{run['run_id']}'",
                },
                indent=2,
            )
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_get_experiment_run(context: dict[str, Any], args: dict[str, Any]) -> str:
    start_time = time.perf_counter()
    tracker = _experiment_tracker(context)
    run_id = _required_string(args, "run_id", "get_experiment_run")
    try:
        run = tracker.get_run(run_id)
        _record_metrics(start_time, "success")
        return json.dumps({"status": "success", "run": run}, indent=2)
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise


async def handle_compare_experiment_runs(context: dict[str, Any], args: dict[str, Any]) -> str:
    start_time = time.perf_counter()
    tracker = _experiment_tracker(context)
    run_id_a = _required_string(args, "run_id_a", "compare_experiment_runs")
    run_id_b = _required_string(args, "run_id_b", "compare_experiment_runs")
    try:
        comparison = tracker.compare_runs(run_id_a, run_id_b)
        _record_metrics(start_time, "success")
        return json.dumps({"status": "success", "comparison": comparison}, indent=2)
    except Exception as exc:
        _record_metrics(start_time, "failure", type(exc).__name__)
        raise
