"""BoundedDict cap enforcement for ExperimentTracker._runs (A1)."""

from __future__ import annotations

import pytest

from src.dataset_registry import DatasetRegistry
from src.experiment_tracker import ExperimentTracker


def _make_dataset_registry(dataset_name: str) -> DatasetRegistry:
    registry = DatasetRegistry(seed_defaults=False)
    registry.create_dataset(
        name=dataset_name,
        description="Cap-test dataset for ExperimentTracker BoundedDict eviction.",
        cases=[
            {
                "case_id": "case-1",
                "name": "case-1",
                "text": "The quick brown fox jumps over the lazy dog. " * 20,
                "min_compression_ratio": 0.0,
                "min_token_savings_pct": 0.0,
            }
        ],
    )
    return registry


def test_experiment_tracker_evicts_oldest_run_past_cap():
    dataset_name = "cap-test-dataset"
    registry = _make_dataset_registry(dataset_name)
    tracker = ExperimentTracker(dataset_registry=registry, max_runs=2)

    run_1 = tracker.run_experiment(dataset_name=dataset_name)
    run_2 = tracker.run_experiment(dataset_name=dataset_name)
    run_3 = tracker.run_experiment(dataset_name=dataset_name)

    run_id_1 = run_1["run_id"]
    run_id_2 = run_2["run_id"]
    run_id_3 = run_3["run_id"]

    # The first-inserted run is now gone -- evicted by the max_runs=2 cap.
    with pytest.raises(ValueError, match=f"Unknown experiment run '{run_id_1}'"):
        tracker.get_run(run_id_1)

    # The two most recent runs remain, retrievable via the real public API.
    assert tracker.get_run(run_id_2)["run_id"] == run_id_2
    assert tracker.get_run(run_id_3)["run_id"] == run_id_3
