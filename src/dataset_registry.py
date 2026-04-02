"""Named dataset registry for experiment runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import sys
from datetime import datetime, timezone

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Optional

from .benchmark_harness import BenchmarkCase, default_corpus_path, load_benchmark_cases


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _case_from_mapping(raw: dict[str, Any]) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=raw["case_id"],
        name=raw["name"],
        text=raw["text"],
        min_compression_ratio=float(raw["min_compression_ratio"]),
        min_token_savings_pct=float(raw["min_token_savings_pct"]),
        query=raw.get("query"),
    )


@dataclass
class DatasetRecord:
    """Stored experiment dataset."""

    name: str
    description: str
    created_at: str
    source: str
    cases: list[BenchmarkCase]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_cases: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "source": self.source,
            "case_count": len(self.cases),
            "metadata": deepcopy(self.metadata),
        }
        if include_cases:
            payload["cases"] = [case.__dict__.copy() for case in self.cases]
        return payload


class DatasetRegistry:
    """Thread-safe in-memory registry of named benchmark/eval datasets."""

    _instance: Optional["DatasetRegistry"] = None

    def __init__(self, seed_defaults: bool = True):
        self._lock = RLock()
        self._datasets: dict[str, DatasetRecord] = {}
        if seed_defaults:
            self._seed_default_benchmark_dataset()

    @classmethod
    def get_registry(cls) -> "DatasetRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def _seed_default_benchmark_dataset(self) -> None:
        name = "benchmark-corpus"
        if name in self._datasets:
            return
        cases = load_benchmark_cases(default_corpus_path())
        self._datasets[name] = DatasetRecord(
            name=name,
            description="Default benchmark corpus seeded from test fixtures.",
            created_at=_utc_now(),
            source=str(default_corpus_path()),
            cases=cases,
            metadata={"seeded": True},
        )

    def create_dataset(
        self,
        *,
        name: str,
        description: str,
        cases: Optional[Iterable[dict[str, Any]]] = None,
        source_path: str | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            if name in self._datasets:
                raise ValueError(f"Dataset '{name}' already exists")

            if source_path:
                normalized_cases = load_benchmark_cases(Path(source_path))
                source = source_path
            else:
                if not cases:
                    raise ValueError("create_dataset requires either 'cases' or 'source_path'")
                normalized_cases = [_case_from_mapping(case) for case in cases]
                source = "inline"

            record = DatasetRecord(
                name=name,
                description=description,
                created_at=_utc_now(),
                source=source,
                cases=normalized_cases,
                metadata=deepcopy(metadata or {}),
            )
            self._datasets[name] = record
            return record.to_dict(include_cases=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._datasets[name].to_dict() for name in sorted(self._datasets)]

    def get_dataset(self, name: str) -> DatasetRecord:
        with self._lock:
            if name not in self._datasets:
                raise ValueError(f"Unknown dataset '{name}'")
            return self._datasets[name]
