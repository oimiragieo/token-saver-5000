"""Handlers for managed connector feeds."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import networkx as nx

from ..connector_registry import ConnectorRegistry
from ..identity_scope import compose_scoped_file_id
from ..observability import get_observability


def _registry(context: dict[str, Any]) -> ConnectorRegistry:
    return context.get("connector_registry") or ConnectorRegistry.get_registry()


def _required_string(args: dict[str, Any], field: str, tool_name: str) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires a non-empty '{field}' field")
    return value.strip()


def _scope_args(args: dict[str, Any]) -> dict[str, str | None]:
    return {
        "workspace_id": args.get("workspace_id"),
        "user_id": args.get("user_id"),
        "agent_id": args.get("agent_id"),
        "session_id": args.get("session_id"),
    }


def get_connector_output_fields() -> list[str]:
    return [
        "status",
        "connector_types[].connector_type",
        "feed.name",
        "feed.connector_type",
        "feeds[].name",
        "feeds[].last_synced_at",
        "results[].file_id",
        "results[].doc_id",
        "results[].compression_ratio",
    ]


async def handle_list_connector_types(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    with observe.trace("connector_registry.list_types"):
        types = _registry(context).list_connector_types()
    return json.dumps({"status": "success", "connector_types": types}, indent=2)


async def handle_create_connector_feed(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    name = _required_string(args, "name", "create_connector_feed")
    connector_type = _required_string(args, "connector_type", "create_connector_feed")
    config = args.get("config")
    if not isinstance(config, dict):
        raise ValueError("create_connector_feed requires object 'config'")
    metadata = args.get("metadata") or {}
    if metadata and not isinstance(metadata, dict):
        raise ValueError("'metadata' must be an object when provided")
    with observe.trace(
        "connector_registry.create_feed", feed_name=name, connector_type=connector_type
    ):
        feed = _registry(context).create_feed(
            name=name,
            connector_type=connector_type,
            config=config,
            metadata=metadata,
        )
    return json.dumps(
        {"status": "success", "feed": feed, "message": f"Created connector feed '{name}'"},
        indent=2,
    )


async def handle_list_connector_feeds(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    with observe.trace("connector_registry.list_feeds"):
        feeds = _registry(context).list_feeds()
    return json.dumps(
        {"status": "success", "total_feeds": len(feeds), "feeds": feeds},
        indent=2,
    )


async def handle_get_connector_feed(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    name = _required_string(args, "name", "get_connector_feed")
    with observe.trace("connector_registry.get_feed", feed_name=name):
        feed = _registry(context).get_feed(name)
    return json.dumps({"status": "success", "feed": feed}, indent=2)


async def handle_sync_connector_feed(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    feed_name = _required_string(args, "name", "sync_connector_feed")
    with observe.trace("connector_registry.sync_feed", feed_name=feed_name, **_scope_args(args)):
        feed, documents = _registry(context).collect_documents(feed_name)

        total_size_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
        allowed, reason = await context["resource_manager"].check_connector_batch_async(
            feed_name, len(documents), total_size_bytes
        )
        if not allowed:
            raise ValueError(reason)

        results: list[dict[str, Any]] = []
        ingested_ids: list[str] = []
        for document in documents:
            visible_file_id = f"{feed_name}:{document.file_id}"
            scoped_file_id = compose_scoped_file_id(visible_file_id, **_scope_args(args))
            text_size = len(document.text.encode("utf-8"))

            allowed_doc, error = await context["resource_manager"].check_document_size_async(
                scoped_file_id, text_size
            )
            if not allowed_doc:
                results.append({"file_id": visible_file_id, "success": False, "error": error})
                continue

            metadata = dict(document.metadata)
            metadata.update({"connector_feed": feed_name, "source_id": document.source_id})
            skeleton = await context["compressor"].ingest_file_async(
                document.text, scoped_file_id, metadata
            )
            await context["resource_manager"].register_document_async(scoped_file_id, text_size)

            graph = context["compressor"].graphs.get(scoped_file_id)
            if isinstance(graph, nx.Graph):
                graph_data = nx.node_link_data(graph, edges="links")
                context["persistence"].save_document(
                    file_id=scoped_file_id,
                    chunks={
                        k: v
                        for k, v in context["compressor"].chunks.items()
                        if k.startswith(scoped_file_id)
                    },
                    graph_data=graph_data,
                    metadata=context["compressor"].file_metadata.get(scoped_file_id, {}),
                )

            context["sync_manager"].register_file(
                scoped_file_id,
                None,
                document.text,
                source_type=feed["connector_type"],
                source_id=document.source_id,
            )

            checksum = hashlib.md5(document.text.encode()).hexdigest()
            await context["version_manager"].add_version_async(
                doc_id=scoped_file_id,
                content=document.text,
                checksum=checksum,
                file_path=None,
                metadata=metadata,
                compression_stats={
                    "total_tokens": skeleton.total_tokens,
                    "skeleton_tokens": skeleton.skeleton_tokens,
                    "compression_ratio": skeleton.compression_ratio,
                },
            )

            context["retrieval_history"][scoped_file_id] = []
            ingested_ids.append(scoped_file_id)
            results.append(
                {
                    "file_id": visible_file_id,
                    "doc_id": scoped_file_id,
                    "success": True,
                    "compression_ratio": skeleton.compression_ratio,
                    "source_id": document.source_id,
                }
            )

        context["persistence"].save_file_sync_metadata(context["sync_manager"].export_metadata())
        updated_feed = _registry(context).mark_synced(feed_name, ingested_ids)
    return json.dumps(
        {
            "status": "success",
            "feed": updated_feed,
            "total_documents": len(documents),
            "ingested_documents": len(ingested_ids),
            "results": results,
        },
        indent=2,
    )
