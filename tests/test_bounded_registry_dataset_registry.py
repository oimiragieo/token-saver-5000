"""Bounded-registry eviction proof for DatasetRegistry (A1)."""

from __future__ import annotations

import pytest

from src.dataset_registry import DatasetRegistry


def _case(i: int) -> dict:
    return {
        "case_id": f"case-{i}",
        "name": f"case {i}",
        "text": f"text {i}",
        "min_compression_ratio": 1.0,
        "min_token_savings_pct": 0.0,
    }


def test_dataset_registry_evicts_oldest_past_max_datasets():
    registry = DatasetRegistry(seed_defaults=False, max_datasets=2)

    registry.create_dataset(name="ds-0", description="d0", cases=[_case(0)])
    registry.create_dataset(name="ds-1", description="d1", cases=[_case(1)])
    registry.create_dataset(name="ds-2", description="d2", cases=[_case(2)])

    with pytest.raises(ValueError):
        registry.get_dataset("ds-0")

    assert registry.get_dataset("ds-1").name == "ds-1"
    assert registry.get_dataset("ds-2").name == "ds-2"
