"""Handlers for temporal context, timelines, and fact lifecycle management."""

from __future__ import annotations

import json
from typing import Any

from ..context_blocks import build_context_block
from ..identity_scope import compose_scoped_file_id, display_file_id, scope_matches
from ..temporal_graph import TemporalGraph


def _scope_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    return {
        "workspace_id": args.get("workspace_id"),
        "user_id": args.get("user_id"),
        "agent_id": args.get("agent_id"),
        "session_id": args.get("session_id"),
    }


def _scoped_file_id(file_id: str, args: dict[str, Any]) -> str:
    return compose_scoped_file_id(file_id, **_scope_kwargs(args))


def _temporal_graph(context: dict[str, Any]) -> TemporalGraph:
    graph = getattr(context["compressor"], "_temporal_graph", None)
    if graph is None:
        raise ValueError("Temporal graph is not enabled on this compressor instance")
    return graph


def get_temporal_output_fields() -> list[str]:
    return [
        "status",
        "context_block.doc_id",
        "context_block.summary",
        "context_block.active_fact_count",
        "context_block.active_facts[].fact_id",
        "events[].event_type",
        "events[].timestamp",
        "facts[].fact_id",
        "facts[].active",
        "fact.fact_id",
        "fact.invalidation_reason",
    ]


async def handle_get_context_block(context: dict[str, Any], args: dict[str, Any]) -> str:
    file_id = args["file_id"]
    scoped_file_id = _scoped_file_id(file_id, args)
    include_invalidated = args.get("include_invalidated", False)
    as_of = args.get("as_of")

    if scoped_file_id not in context["compressor"].graphs:
        raise ValueError(f"Document '{file_id}' not found")

    temporal_graph = _temporal_graph(context)
    excluded_node_ids = (
        set()
        if include_invalidated
        else temporal_graph.get_invalidated_fact_ids(scoped_file_id, as_of=as_of)
    )
    skeleton = context["compressor"]._generate_skeleton(
        scoped_file_id,
        query=args.get("query"),
        exclude_node_ids=excluded_node_ids,
    )
    active_facts = temporal_graph.get_active_facts(
        scoped_file_id,
        as_of=as_of,
        include_invalidated=include_invalidated,
    )
    recent_events = temporal_graph.search_timeline(
        doc_id=scoped_file_id,
        until=as_of,
        include_invalidated=include_invalidated,
        limit=args.get("limit", 10),
    )
    access_info = context["compressor"]._access_tracker.get_access_info(scoped_file_id)
    compression_history = context["compressor"]._compression_replay.get_history(scoped_file_id)
    context_block = build_context_block(
        doc_id=scoped_file_id,
        active_facts=active_facts,
        recent_events=recent_events,
        access_info=access_info,
        compression_history=compression_history,
        skeleton_text=skeleton.skeleton_text,
        max_facts=args.get("max_facts", 5),
    )
    context_block["doc_id"] = file_id
    return json.dumps({"status": "success", "context_block": context_block}, indent=2)


async def handle_search_timeline(context: dict[str, Any], args: dict[str, Any]) -> str:
    temporal_graph = _temporal_graph(context)
    file_id = args.get("file_id")
    doc_id = _scoped_file_id(file_id, args) if file_id else None
    since = args.get("since")
    until = args.get("until")
    limit = args.get("limit", 25)
    events = temporal_graph.search_timeline(
        query=args.get("query"),
        doc_id=doc_id,
        fact_id=args.get("fact_id"),
        event_types=args.get("event_types"),
        since=since,
        until=until,
        include_invalidated=args.get("include_invalidated", True),
        limit=limit,
    )
    tracker = getattr(context["compressor"], "_access_tracker", None)
    if tracker is not None:
        for event in tracker.get_access_timeline(doc_id=doc_id, limit=limit):
            events.append(
                {
                    "event_id": None,
                    "event_type": event["event_type"],
                    "timestamp": event["timestamp"],
                    "timestamp_unix": event["timestamp"],
                    "doc_id": event["doc_id"],
                    "fact_id": None,
                    "summary": f"{event['event_type']} for {event['doc_id']}",
                    "metadata": event.get("metadata", {}),
                }
            )
    replay = getattr(context["compressor"], "_compression_replay", None)
    if replay is not None:
        for entry in replay.get_timeline(doc_id=doc_id, limit=limit):
            events.append(
                {
                    "event_id": None,
                    "event_type": entry["event_type"],
                    "timestamp": entry["timestamp"],
                    "timestamp_unix": entry["timestamp"],
                    "doc_id": entry["doc_id"],
                    "fact_id": None,
                    "summary": f"compression ratio {entry['ratio']:.2f}x",
                    "metadata": entry.get("metadata", {}),
                }
            )

    if doc_id is None and any(_scope_kwargs(args).values()):
        events = [
            event
            for event in events
            if event["doc_id"] and scope_matches(event["doc_id"], **_scope_kwargs(args))
        ]

    if args.get("query"):
        query_text = str(args["query"]).lower()
        filtered_events = []
        for event in events:
            searchable = " ".join(
                [
                    str(event.get("event_type") or ""),
                    str(event.get("doc_id") or ""),
                    str(event.get("summary") or ""),
                    json.dumps(event.get("metadata") or {}, sort_keys=True),
                ]
            ).lower()
            if query_text in searchable:
                filtered_events.append(event)
        events = filtered_events

    events.sort(key=lambda event: event.get("timestamp_unix", 0), reverse=True)
    events = events[:limit]

    for event in events:
        if event.get("doc_id"):
            event["file_id"] = display_file_id(event["doc_id"])

    return json.dumps(
        {
            "status": "success",
            "total_events": len(events),
            "events": events,
        },
        indent=2,
    )


async def handle_list_fact_history(context: dict[str, Any], args: dict[str, Any]) -> str:
    temporal_graph = _temporal_graph(context)
    file_id = args.get("file_id")
    doc_id = _scoped_file_id(file_id, args) if file_id else None
    facts = temporal_graph.list_fact_history(
        doc_id=doc_id,
        fact_id=args.get("fact_id"),
        include_invalidated=args.get("include_invalidated", True),
        as_of=args.get("as_of"),
    )
    for fact in facts:
        fact["file_id"] = display_file_id(fact["doc_id"])
    return json.dumps(
        {
            "status": "success",
            "total_facts": len(facts),
            "facts": facts,
        },
        indent=2,
    )


async def handle_invalidate_fact(context: dict[str, Any], args: dict[str, Any]) -> str:
    temporal_graph = _temporal_graph(context)
    invalidated = temporal_graph.invalidate_fact(
        args["fact_id"],
        reason=args["reason"],
        metadata={"requested_by": "invalidate_fact"},
        timestamp=args.get("timestamp"),
    )
    invalidated["file_id"] = display_file_id(invalidated["doc_id"])
    return json.dumps(
        {
            "status": "success",
            "fact": invalidated,
            "message": f"Invalidated fact '{args['fact_id']}'",
        },
        indent=2,
    )
