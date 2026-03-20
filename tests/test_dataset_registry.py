"""Tests for named experiment datasets."""

from src.dataset_registry import DatasetRegistry


def setup_function():
    DatasetRegistry.reset_singleton()


def test_seeded_benchmark_dataset_exists():
    registry = DatasetRegistry.get_registry()
    datasets = registry.list_datasets()

    assert any(dataset["name"] == "benchmark-corpus" for dataset in datasets)


def test_create_and_get_inline_dataset():
    registry = DatasetRegistry(seed_defaults=False)
    created = registry.create_dataset(
        name="release-gate",
        description="Release benchmark subset",
        cases=[
            {
                "case_id": "tiny",
                "name": "Tiny case",
                "text": "Prompt caching works best with stable prefixes. " * 8,
                "min_compression_ratio": 1.1,
                "min_token_savings_pct": 5.0,
                "query": "stable prefixes",
            }
        ],
    )
    fetched = registry.get_dataset("release-gate").to_dict(include_cases=True)

    assert created["case_count"] == 1
    assert fetched["cases"][0]["case_id"] == "tiny"
