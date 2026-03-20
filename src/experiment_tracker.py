"""Tracked experiment runs over benchmark and evaluation datasets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from statistics import mean
from threading import RLock
from typing import Any, Optional

from .benchmark_harness import (
    BenchmarkExecutionDetail,
    filter_cases,
    run_benchmark_cases,
    run_benchmark_cases_detailed,
    summary_to_dict,
)
from .compression_rewards import CompressionRewardCalculator
from .compression_verifier import CompressionVerifier
from .dataset_registry import DatasetRegistry


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class ExperimentRunRecord:
    """Stored experiment run."""

    run_id: str
    dataset_name: str
    mode: str
    status: str
    created_at: str
    completed_at: str | None
    parameters: dict[str, Any]
    summary: dict[str, Any]
    case_reports: list[dict[str, Any]]
    verification_pass_rate: float
    avg_reward: float
    avg_reward_components: dict[str, float]
    baseline_run_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "mode": self.mode,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "parameters": deepcopy(self.parameters),
            "summary": deepcopy(self.summary),
            "case_reports": deepcopy(self.case_reports),
            "verification_pass_rate": self.verification_pass_rate,
            "avg_reward": self.avg_reward,
            "avg_reward_components": deepcopy(self.avg_reward_components),
            "baseline_run_id": self.baseline_run_id,
            "metadata": deepcopy(self.metadata),
        }


class ExperimentTracker:
    """Experiment run tracker backed by benchmark datasets."""

    _instance: Optional["ExperimentTracker"] = None

    def __init__(self, dataset_registry: DatasetRegistry | None = None):
        self._lock = RLock()
        self._dataset_registry = dataset_registry or DatasetRegistry.get_registry()
        self._runs: dict[str, ExperimentRunRecord] = {}
        self._counter = 0

    @classmethod
    def get_tracker(cls) -> "ExperimentTracker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def run_experiment(
        self,
        *,
        dataset_name: str,
        mode: str = "baseline",
        case_ids: list[str] | None = None,
        similarity_threshold: float = 0.75,
        skeleton_ratio: float = 0.2,
        baseline_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        dataset = self._dataset_registry.get_dataset(dataset_name)
        selected_cases = filter_cases(dataset.cases, case_ids)
        if not selected_cases:
            raise ValueError(f"No cases selected for dataset '{dataset_name}'")

        with self._lock:
            self._counter += 1
            run_id = f"exp_{self._counter}"

        details = run_benchmark_cases_detailed(
            selected_cases,
            mode=mode,
            similarity_threshold=similarity_threshold,
            skeleton_ratio=skeleton_ratio,
        )
        summary = run_benchmark_cases(
            selected_cases,
            mode=mode,
            similarity_threshold=similarity_threshold,
            skeleton_ratio=skeleton_ratio,
        )

        verifier = CompressionVerifier()
        reward_calculator = CompressionRewardCalculator()
        case_reports: list[dict[str, Any]] = []
        verification_scores: list[float] = []
        reward_scores: list[float] = []
        component_names = ["schema", "semantic", "fidelity", "composition", "memory"]
        component_aggregates: dict[str, list[float]] = {name: [] for name in component_names}

        for detail in details:
            case_report = self._build_case_report(
                detail=detail,
                verifier=verifier,
                reward_calculator=reward_calculator,
            )
            case_reports.append(case_report)
            verification_scores.append(1.0 if case_report["verification"]["verified"] else 0.0)
            reward_scores.append(case_report["reward"]["total_reward"])
            for component in component_names:
                component_aggregates[component].append(
                    case_report["reward"]["component_scores"][component]
                )

        avg_reward_components = {
            component: round(mean(values), 4) if values else 0.0
            for component, values in component_aggregates.items()
        }
        record = ExperimentRunRecord(
            run_id=run_id,
            dataset_name=dataset_name,
            mode=mode,
            status="completed",
            created_at=_utc_now(),
            completed_at=_utc_now(),
            parameters={
                "mode": mode,
                "case_ids": case_ids or [],
                "similarity_threshold": similarity_threshold,
                "skeleton_ratio": skeleton_ratio,
            },
            summary=summary_to_dict(summary),
            case_reports=case_reports,
            verification_pass_rate=(
                round(mean(verification_scores), 4) if verification_scores else 0.0
            ),
            avg_reward=round(mean(reward_scores), 4) if reward_scores else 0.0,
            avg_reward_components=avg_reward_components,
            baseline_run_id=baseline_run_id,
            metadata=deepcopy(metadata or {}),
        )

        with self._lock:
            self._runs[run_id] = record
        return record.to_dict()

    def _build_case_report(
        self,
        *,
        detail: BenchmarkExecutionDetail,
        verifier: CompressionVerifier,
        reward_calculator: CompressionRewardCalculator,
    ) -> dict[str, Any]:
        verification = verifier.verify_compression_operation(
            document=detail.case.text,
            skeleton_text=detail.skeleton_text,
            node_map=detail.node_map,
            original_tokens=detail.result.original_tokens,
            skeleton_tokens=detail.result.skeleton_tokens,
            fidelity_level="BALANCED",
            compression_ratio=detail.result.compression_ratio,
        )
        reward = reward_calculator.calculate(
            input_text=detail.case.text,
            output_text=detail.skeleton_text,
            input_tokens=detail.result.original_tokens,
            output_tokens=detail.result.skeleton_tokens,
            fidelity_level="BALANCED",
            node_map=detail.node_map,
        )
        return {
            "case": detail.case.__dict__.copy(),
            "benchmark": detail.result.__dict__.copy() | {"passed": detail.result.passed},
            "verification": verification.to_dict(),
            "reward": reward.to_dict(),
        }

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if run_id not in self._runs:
                raise ValueError(f"Unknown experiment run '{run_id}'")
            return self._runs[run_id].to_dict()

    def compare_runs(self, run_id_a: str, run_id_b: str) -> dict[str, Any]:
        left = self.get_run(run_id_a)
        right = self.get_run(run_id_b)
        left_summary = left["summary"]
        right_summary = right["summary"]
        return {
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "dataset_name_a": left["dataset_name"],
            "dataset_name_b": right["dataset_name"],
            "mode_a": left["mode"],
            "mode_b": right["mode"],
            "deltas": {
                "passed_cases": right_summary["passed_cases"] - left_summary["passed_cases"],
                "failed_cases": right_summary["failed_cases"] - left_summary["failed_cases"],
                "avg_compression_ratio": round(
                    right_summary["avg_compression_ratio"] - left_summary["avg_compression_ratio"],
                    4,
                ),
                "avg_token_savings_pct": round(
                    right_summary["avg_token_savings_pct"] - left_summary["avg_token_savings_pct"],
                    4,
                ),
                "verification_pass_rate": round(
                    right["verification_pass_rate"] - left["verification_pass_rate"], 4
                ),
                "avg_reward": round(right["avg_reward"] - left["avg_reward"], 4),
            },
        }
