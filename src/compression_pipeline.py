"""Composable multi-pass pipeline for read-skeleton compression flows."""

from __future__ import annotations

import inspect
import re
from typing import Any

# F3: Heuristics for auto-detecting structured audit/report documents.
# The auto selection_mode resolves to evidence_aware (using the H1 text as a
# synthetic query) for structured docs, and falls back to baseline otherwise.
_AUTO_H2_RE = re.compile(r"^## ", re.MULTILINE)
_AUTO_FINDING_RE = re.compile(r"^\d+\.", re.MULTILINE)
_AUTO_VERDICT_RE = re.compile(
    r"\b(verdict|conclusion|summary|finding|result|status|critical|high|p0|blocker)\b",
    re.IGNORECASE,
)
_AUTO_H1_RE = re.compile(r"^# (.+)", re.MULTILINE)


def _resolve_auto_mode(text: str) -> tuple[str, str]:
    """Return (resolved_mode, selection_mode_resolved_label) for selection_mode='auto'.

    Structured detection criteria (all three must pass):
    - 3+ H2 headings  (## …)
    - 3+ numbered findings  (1. …)
    - At least one verdict-like keyword

    When detected, resolves to evidence_aware and extracts the H1 heading as
    the synthetic query.  Falls back to baseline when text is None/empty or the
    criteria are not met.
    """
    if not text:
        return "baseline", "auto-detected: baseline"

    h2_count = len(_AUTO_H2_RE.findall(text))
    finding_count = len(_AUTO_FINDING_RE.findall(text))
    has_verdict = bool(_AUTO_VERDICT_RE.search(text))

    if h2_count >= 3 and finding_count >= 3 and has_verdict:
        return "evidence_aware", "auto-detected: evidence_aware"
    return "baseline", "auto-detected: baseline"


def _extract_h1_query(text: str | None) -> str | None:
    """Return the first H1 heading text, or None if not found."""
    if not text:
        return None
    m = _AUTO_H1_RE.search(text)
    return m.group(1).strip() if m else None


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
    raw_text: str | None = None,
) -> dict[str, Any]:
    """Run baseline/query/evidence stages and return the final skeleton plus trace.

    When selection_mode='auto', the pipeline inspects ``raw_text`` (the
    original ingested document) to decide whether to use evidence_aware
    (structured audit/report docs) or baseline (plain prose).  The resolved
    mode is reported in the ``selection_mode_resolved`` key of the return dict.
    """
    # F3: Resolve auto selection mode before any skeleton generation.
    selection_mode_resolved: str = selection_mode
    if selection_mode == "auto":
        resolved, selection_mode_resolved = _resolve_auto_mode(raw_text or "")
        selection_mode = resolved
        # For evidence_aware, synthesise a query from the H1 heading when the
        # caller didn't supply one explicitly.
        if selection_mode == "evidence_aware" and not query:
            query = _extract_h1_query(raw_text) or "key findings and verdict"
    else:
        selection_mode_resolved = selection_mode

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
            "selection_mode_resolved": selection_mode_resolved,
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
            "selection_mode_resolved": selection_mode_resolved,
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
        "selection_mode_resolved": selection_mode_resolved,
    }
