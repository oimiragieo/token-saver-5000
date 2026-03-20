"""Build token-efficient, auditable handoff bundle artifacts."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from .context_blocks import build_context_block
from .evidence_bundle import EvidenceBundle, QualityMetrics
from .toon_serializer import TOONSerializer, estimate_token_savings


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _search_results(
    compressor: Any, query: str | None, scoped_file_id: str, top_k: int
) -> list[dict[str, Any]]:
    if not query:
        return []

    results: list[dict[str, Any]] = []
    for node_id, similarity in compressor.search_semantic_with_scores(query, scoped_file_id, top_k):
        node = compressor.chunks[node_id]
        results.append(
            {
                "node_id": node_id,
                "similarity": round(float(similarity), 3),
                "importance": round(float(getattr(node, "importance", 0.0)), 3),
                "summary": compressor._generate_summary(node.text, max_length=100),
            }
        )
    return results


def _context_block(
    compressor: Any, scoped_file_id: str, visible_file_id: str, skeleton_text: str
) -> dict[str, Any]:
    temporal_graph = getattr(compressor, "_temporal_graph", None)
    active_facts = (
        temporal_graph.get_active_facts(scoped_file_id, include_invalidated=False)
        if temporal_graph is not None
        else []
    )
    recent_events = (
        temporal_graph.search_timeline(
            doc_id=scoped_file_id,
            include_invalidated=False,
            limit=10,
        )
        if temporal_graph is not None
        else []
    )
    access_tracker = getattr(compressor, "_access_tracker", None)
    replay_log = getattr(compressor, "_compression_replay", None)
    access_info = (
        access_tracker.get_access_info(scoped_file_id) if access_tracker is not None else None
    )
    compression_history = replay_log.get_history(scoped_file_id) if replay_log is not None else []
    block = build_context_block(
        doc_id=scoped_file_id,
        active_facts=active_facts,
        recent_events=recent_events,
        access_info=access_info,
        compression_history=compression_history,
        skeleton_text=skeleton_text,
        max_facts=5,
    )
    block["doc_id"] = visible_file_id
    return block


def _replay_text(
    *,
    visible_file_id: str,
    summary: str,
    query: str | None,
    skeleton_text: str,
    search_results: list[dict[str, Any]],
) -> str:
    lines = [
        f"=== HANDOFF BUNDLE: {visible_file_id} ===",
        f"Summary: {summary}",
    ]
    if query:
        lines.append(f"Focus query: {query}")
    lines.extend(["", "Skeleton:", skeleton_text])
    if search_results:
        lines.extend(["", "Relevant nodes:"])
        for result in search_results:
            lines.append(
                f"- {result['node_id']} (similarity={result['similarity']}, importance={result['importance']}): {result['summary']}"
            )
    return "\n".join(lines)


def distill_handoff_bundle(
    *,
    compressor: Any,
    visible_file_id: str,
    scoped_file_id: str,
    query: str | None = None,
    top_k: int = 5,
    metadata: dict[str, Any] | None = None,
    bundle_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    resolved_scoped_file_id = scoped_file_id
    if resolved_scoped_file_id not in compressor.graphs:
        if visible_file_id in compressor.graphs:
            resolved_scoped_file_id = visible_file_id
        else:
            candidates = [
                doc_id
                for doc_id in compressor.graphs
                if doc_id == visible_file_id or doc_id.endswith(f"__f={visible_file_id}")
            ]
            if len(candidates) == 1:
                resolved_scoped_file_id = candidates[0]
            else:
                raise ValueError(f"Document '{visible_file_id}' not found")

    timestamp = created_at or _utc_now()
    skeleton = compressor._generate_skeleton(resolved_scoped_file_id, query=query)
    search_results = _search_results(compressor, query, resolved_scoped_file_id, top_k)
    summary = f"Distilled handoff for '{visible_file_id}'"
    replay_text = _replay_text(
        visible_file_id=visible_file_id,
        summary=summary,
        query=query,
        skeleton_text=skeleton.skeleton_text,
        search_results=search_results,
    )
    context_block = _context_block(
        compressor,
        resolved_scoped_file_id,
        visible_file_id,
        skeleton.skeleton_text,
    )
    artifact = {
        "bundle_version": "1.0",
        "bundle_id": bundle_id,
        "doc_id": visible_file_id,
        "scoped_doc_id": resolved_scoped_file_id,
        "created_at": timestamp,
        "query": query,
        "summary": summary,
        "metadata": deepcopy(metadata or {}),
        "skeleton": {
            "text": skeleton.skeleton_text,
            "cache_stable_prefix": skeleton.skeleton_text,
            "total_nodes": skeleton.total_nodes,
            "total_tokens": skeleton.total_tokens,
            "skeleton_tokens": skeleton.skeleton_tokens,
            "compression_ratio": round(float(skeleton.compression_ratio), 3),
            "node_map": dict(skeleton.node_map),
        },
        "search_results": search_results,
        "context_block": context_block,
        "replay_text": replay_text,
    }
    artifact_json = json.dumps(artifact, indent=2, sort_keys=True)
    artifact_toon = TOONSerializer().serialize_handoff_bundle(artifact)
    replay_tokens = (
        compressor._count_tokens(replay_text)
        if hasattr(compressor, "_count_tokens")
        else max(1, len(replay_text.split()))
    )
    evidence_bundle = EvidenceBundle.create(
        operation="handoff_bundle",
        input_data=json.dumps(
            {
                "doc_id": resolved_scoped_file_id,
                "query": query,
                "metadata": metadata or {},
            },
            sort_keys=True,
        ),
        output_data=replay_text,
        input_token_count=skeleton.total_tokens,
        output_token_count=replay_tokens,
        parameters={"doc_id": resolved_scoped_file_id, "query": query, "top_k": top_k},
        quality_metrics=QualityMetrics(
            compression_ratio=float(skeleton.compression_ratio),
            token_reduction=round(
                (
                    ((skeleton.total_tokens - replay_tokens) / skeleton.total_tokens * 100)
                    if skeleton.total_tokens
                    else 0.0
                ),
                2,
            ),
        ),
        metadata={"visible_doc_id": visible_file_id},
    )
    return {
        "artifact": artifact,
        "artifact_toon": artifact_toon,
        "replay_text": replay_text,
        "token_estimate": estimate_token_savings(artifact_json, artifact_toon),
        "evidence_bundle": evidence_bundle,
    }
