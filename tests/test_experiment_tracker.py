"""Tests for tracked experiment runs."""

from src.dataset_registry import DatasetRegistry
from src.experiment_tracker import ExperimentTracker


def setup_function():
    DatasetRegistry.reset_singleton()
    ExperimentTracker.reset_singleton()


def test_run_and_compare_experiments():
    datasets = DatasetRegistry(seed_defaults=False)
    datasets.create_dataset(
        name="release-gate",
        description="Release benchmark subset",
        cases=[
            {
                "case_id": "tiny",
                "name": "Tiny case",
                "text": "Prompt caching works best with stable prefixes. " * 20,
                "min_compression_ratio": 1.1,
                "min_token_savings_pct": 5.0,
                "query": "stable prefixes",
            }
        ],
    )
    tracker = ExperimentTracker(dataset_registry=datasets)

    baseline = tracker.run_experiment(dataset_name="release-gate", mode="baseline")
    query_guided = tracker.run_experiment(dataset_name="release-gate", mode="query_guided")
    comparison = tracker.compare_runs(baseline["run_id"], query_guided["run_id"])

    assert baseline["summary"]["total_cases"] == 1
    assert 0.0 <= baseline["verification_pass_rate"] <= 1.0
    assert baseline["case_reports"][0]["benchmark"]["case_id"] == "tiny"
    assert "avg_reward" in comparison["deltas"]


def test_get_unknown_run_raises():
    tracker = ExperimentTracker(dataset_registry=DatasetRegistry(seed_defaults=False))

    try:
        tracker.get_run("missing")
    except ValueError as exc:
        assert "Unknown experiment run" in str(exc)
    else:
        raise AssertionError("Expected missing run to raise ValueError")
