"""Tests for the composable read-skeleton compression pipeline."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from src.compression_pipeline import run_read_skeleton_pipeline


def _skeleton(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        file_id="doc1",
        total_nodes=5,
        total_tokens=100,
        skeleton_tokens=20,
        compression_ratio=5.0,
        skeleton_text=f"{name} skeleton",
        node_map={"doc1_n0": name},
    )


def test_read_skeleton_pipeline_runs_baseline_then_evidence_stages():
    compressor = Mock()
    compressor._generate_skeleton.side_effect = [
        _skeleton("baseline"),
        _skeleton("query_guided"),
        _skeleton("evidence_aware"),
    ]
    compressor.retrieve_evidence.return_value = SimpleNamespace(
        sufficient=True,
        best_score=0.9,
        threshold=0.35,
        used_expanded_search=False,
        message="ok",
        node_ids=["doc1_n1"],
    )

    result = run_read_skeleton_pipeline(
        compressor=compressor,
        file_id="doc1",
        selection_mode="evidence_aware",
        query="retry behavior",
        top_k=2,
        min_similarity=0.4,
        excluded_node_ids=set(),
    )

    assert [stage["name"] for stage in result["stages"]] == [
        "baseline",
        "query_guided",
        "evidence_aware",
    ]
    assert result["final_stage"] == "evidence_aware"
    assert result["evidence"]["sufficient"] is True
