"""Golden regression tests for experiment report shapes."""

from src.dataset_registry import DatasetRegistry
from src.experiment_tracker import ExperimentTracker


def setup_function():
    DatasetRegistry.reset_singleton()
    ExperimentTracker.reset_singleton()


def test_experiment_run_shape_is_stable():
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
            }
        ],
    )
    tracker = ExperimentTracker(dataset_registry=datasets)
    run = tracker.run_experiment(dataset_name="release-gate")

    assert run == {
        "run_id": run["run_id"],
        "dataset_name": "release-gate",
        "mode": "baseline",
        "status": "completed",
        "created_at": run["created_at"],
        "completed_at": run["completed_at"],
        "parameters": run["parameters"],
        "summary": run["summary"],
        "case_reports": run["case_reports"],
        "verification_pass_rate": run["verification_pass_rate"],
        "avg_reward": run["avg_reward"],
        "avg_reward_components": run["avg_reward_components"],
        "baseline_run_id": None,
        "metadata": {},
    }
    assert run["summary"]["total_cases"] == 1
