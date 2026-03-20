"""Stable handlers for structured handoff bundles."""

from __future__ import annotations

import json
from typing import Any

from ..bundle_distiller import distill_handoff_bundle
from ..evidence_bundle import get_evidence_store
from ..identity_scope import compose_scoped_file_id
from ..observability import get_observability
from ..structured_bundles import HandoffBundleRegistry


def _scope_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": args.get("workspace_id"),
        "user_id": args.get("user_id"),
        "agent_id": args.get("agent_id"),
        "session_id": args.get("session_id"),
    }


def _scoped_file_id(file_id: str, args: dict[str, Any]) -> str:
    return compose_scoped_file_id(file_id, **_scope_kwargs(args))


def _registry(context: dict[str, Any]) -> HandoffBundleRegistry:
    return context.get("handoff_bundle_registry") or HandoffBundleRegistry.get_registry()


def get_bundle_output_fields() -> list[str]:
    return [
        "status",
        "bundle.bundle_id",
        "bundle.doc_id",
        "bundle.audit.bundle_hash",
        "bundle.artifact.summary",
        "bundle.artifact.skeleton.compression_ratio",
        "bundle.artifact.search_results[].node_id",
        "bundle.artifact.context_block.summary",
        "replay.bundle_id",
        "replay.replay_text",
    ]


async def handle_create_handoff_bundle(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    file_id = args.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        raise ValueError("create_handoff_bundle requires non-empty 'file_id'")

    scoped_file_id = _scoped_file_id(file_id, args)
    with observe.trace("handoff_bundle.create", file_id=file_id, **_scope_kwargs(args)):
        payload = distill_handoff_bundle(
            compressor=context["compressor"],
            visible_file_id=file_id,
            scoped_file_id=scoped_file_id,
            query=args.get("query"),
            top_k=args.get("top_k", 5),
            metadata=args.get("metadata") or {},
            bundle_id=args.get("bundle_id"),
            created_at=args.get("created_at"),
        )
        evidence_store = context.get("evidence_store") or get_evidence_store()
        evidence_store.append(payload["evidence_bundle"])
        bundle = _registry(context).create_bundle(
            bundle_id=args.get("bundle_id"),
            scoped_doc_id=scoped_file_id,
            artifact=payload["artifact"],
            replay_text=payload["replay_text"],
            evidence_bundle=payload["evidence_bundle"],
            metadata=args.get("metadata") or {},
            created_at=args.get("created_at"),
        )
    bundle["token_estimate"] = payload["token_estimate"]
    return json.dumps({"status": "success", "bundle": bundle}, indent=2)


async def handle_list_handoff_bundles(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    with observe.trace(
        "handoff_bundle.list", file_id=args.get("file_id", ""), **_scope_kwargs(args)
    ):
        bundles = _registry(context).list_bundles(
            file_id=args.get("file_id"), **_scope_kwargs(args)
        )
    return json.dumps(
        {"status": "success", "total_bundles": len(bundles), "bundles": bundles}, indent=2
    )


async def handle_get_handoff_bundle(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    bundle_id = args.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise ValueError("get_handoff_bundle requires non-empty 'bundle_id'")
    with observe.trace("handoff_bundle.get", bundle_id=bundle_id, **_scope_kwargs(args)):
        bundle = _registry(context).get_bundle(bundle_id, **_scope_kwargs(args))
    return json.dumps({"status": "success", "bundle": bundle}, indent=2)


async def handle_replay_handoff_bundle(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    bundle_id = args.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        raise ValueError("replay_handoff_bundle requires non-empty 'bundle_id'")
    with observe.trace("handoff_bundle.replay", bundle_id=bundle_id, **_scope_kwargs(args)):
        replay = _registry(context).replay_bundle(bundle_id, **_scope_kwargs(args))
    return json.dumps({"status": "success", "replay": replay}, indent=2)
