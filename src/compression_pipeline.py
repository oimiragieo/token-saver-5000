"""Composable multi-pass pipeline for read-skeleton compression flows."""

from __future__ import annotations

import inspect
from typing import Any


def _generate_skeleton(
    compressor: Any,
    file_id: str,
    *,
    query: str | None = None,
    anchor_node_ids: set[str] | None = None,
    exclude_node_ids: set[str] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if query is not None:
        kwargs["query"] = query
    if anchor_node_ids is not None:
        kwargs["anchor_node_ids"] = anchor_node_ids
    try:
        params = inspect.signature(compressor._generate_skeleton).parameters
    except (TypeError, ValueError):
        params = {}
    if exclude_node_ids and "exclude_node_ids" in params:
        kwargs["exclude_node_ids"] = exclude_node_ids
    return compressor._generate_skeleton(file_id, **kwargs)


def _stage_payload(
    name: str,
    skeleton: Any,
    *,
    query: str | None = None,
    anchor_node_ids: set[str] | None = None,
    evidence_used: bool = False,
) -> dict[str, Any]:
    return {
        "name": name,
        "query": query,
        "anchor_node_count": len(anchor_node_ids or set()),
        "evidence_used": evidence_used,
        "total_nodes": skeleton.total_nodes,
        "skeleton_tokens": skeleton.skeleton_tokens,
        "compression_ratio": skeleton.compression_ratio,
    }


def run_read_skeleton_pipeline(
    *,
    compressor: Any,
    file_id: str,
    selection_mode: str,
    query: str | None,
    top_k: int,
    min_similarity: float,
    anchor_node_ids: set[str] | None = None,
    excluded_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run baseline/query/evidence stages and return the final skeleton plus trace."""
    stages: list[dict[str, Any]] = []
    working_anchors = set(anchor_node_ids or set())
    evidence_info: dict[str, Any] | None = None

    baseline = _generate_skeleton(
        compressor,
        file_id,
        anchor_node_ids=working_anchors or None,
        exclude_node_ids=excluded_node_ids,
    )
    stages.append(
        _stage_payload(
            "baseline",
            baseline,
            anchor_node_ids=working_anchors,
        )
    )

    if selection_mode == "baseline":
        return {
            "final_skeleton": baseline,
            "final_stage": "baseline",
            "stage_count": len(stages),
            "stages": stages,
            "evidence": None,
        }

    query_guided = _generate_skeleton(
        compressor,
        file_id,
        query=query,
        anchor_node_ids=working_anchors or None,
        exclude_node_ids=excluded_node_ids,
    )
    stages.append(
        _stage_payload(
            "query_guided",
            query_guided,
            query=query,
            anchor_node_ids=working_anchors,
        )
    )

    if selection_mode == "query_guided":
        return {
            "final_skeleton": query_guided,
            "final_stage": "query_guided",
            "stage_count": len(stages),
            "stages": stages,
            "evidence": None,
        }

    evidence = compressor.retrieve_evidence(
        query=query,
        file_id=file_id,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    working_anchors.update(evidence.node_ids)
    evidence_aware = _generate_skeleton(
        compressor,
        file_id,
        query=query,
        anchor_node_ids=working_anchors,
        exclude_node_ids=excluded_node_ids,
    )
    evidence_info = {
        "sufficient": evidence.sufficient,
        "best_score": round(evidence.best_score, 3),
        "threshold": evidence.threshold,
        "used_expanded_search": evidence.used_expanded_search,
        "message": evidence.message,
        "node_ids": list(evidence.node_ids),
    }
    stages.append(
        _stage_payload(
            "evidence_aware",
            evidence_aware,
            query=query,
            anchor_node_ids=working_anchors,
            evidence_used=True,
        )
    )
    return {
        "final_skeleton": evidence_aware,
        "final_stage": "evidence_aware",
        "stage_count": len(stages),
        "stages": stages,
        "evidence": evidence_info,
    }
