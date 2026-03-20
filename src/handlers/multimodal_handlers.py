"""Stable handlers for production multimodal ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..identity_scope import compose_scoped_file_id
from ..multimodal_ingestion import (
    MultimodalIngestionService,
    build_multimodal_request,
    estimate_multimodal_payload_bytes,
)


def _service(context: dict[str, Any]) -> MultimodalIngestionService:
    return context.get("multimodal_ingestion") or MultimodalIngestionService.get_service()


def _scoped_doc_id(args: dict[str, Any]) -> str:
    return compose_scoped_file_id(
        args["doc_id"],
        workspace_id=args.get("workspace_id"),
        user_id=args.get("user_id"),
        agent_id=args.get("agent_id"),
        session_id=args.get("session_id"),
    )


def get_multimodal_output_fields() -> list[str]:
    return [
        "status",
        "doc_id",
        "project.doc_id",
        "project.content_types",
        "project.asset_count",
        "project.stats.total_nodes",
        "results[].node_id",
        "results[].score",
        "results[].modality",
    ]


async def handle_ingest_multimodal(context: dict[str, Any], args: dict[str, Any]) -> str:
    doc_id = args.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("ingest_multimodal requires non-empty 'doc_id'")

    content_items, manifest = build_multimodal_request(
        args,
        path_validator=context.get("path_validator"),
    )
    payload_bytes = estimate_multimodal_payload_bytes(content_items)
    scoped_doc_id = _scoped_doc_id(args)

    allowed, error = await context["resource_manager"].check_multimodal_batch_async(
        scoped_doc_id,
        len(content_items),
        payload_bytes,
    )
    if not allowed:
        raise ValueError(error)

    project = _service(context).ingest(scoped_doc_id, content_items, manifest)
    project["doc_id"] = doc_id
    return json.dumps({"status": "success", "doc_id": doc_id, "project": project}, indent=2)


async def handle_search_multimodal(context: dict[str, Any], args: dict[str, Any]) -> str:
    doc_id = args.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ValueError("search_multimodal requires non-empty 'doc_id'")

    query = args.get("query")
    image_query_path = args.get("image_query_path")
    query_type = args.get("query_type", "text")
    if query_type == "image":
        if not image_query_path:
            raise ValueError("search_multimodal requires 'image_query_path' for image queries")
        if context.get("path_validator") is None:
            raise ValueError("PathValidator not configured - cannot safely handle image queries")
        validated_path = context["path_validator"].validate(image_query_path)
        query_payload: str | bytes = context.get(
            "read_binary_file", lambda path: Path(path).read_bytes()
        )(validated_path)
    else:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("search_multimodal requires non-empty 'query' for text/code queries")
        query_payload = query.strip()

    scoped_doc_id = _scoped_doc_id(args)
    results = _service(context).search(
        scoped_doc_id,
        query_payload,
        query_type=query_type,
        top_k=args.get("top_k", 5),
        filter_modality=args.get("filter_modality"),
    )
    results["doc_id"] = doc_id
    return json.dumps({"status": "success", **results}, indent=2)
