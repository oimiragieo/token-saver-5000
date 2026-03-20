"""Handlers for explicit memory and personalization APIs."""

from __future__ import annotations

import json
from typing import Any

from ..memory_api import MemoryAPI
from ..observability import get_observability


def _memory_api(context: dict[str, Any]) -> MemoryAPI:
    return context.get("memory_api") or MemoryAPI.get_api()


def _scope_args(args: dict[str, Any]) -> dict[str, str | None]:
    return {
        "workspace_id": args.get("workspace_id"),
        "user_id": args.get("user_id"),
        "agent_id": args.get("agent_id"),
        "session_id": args.get("session_id"),
    }


def _required_string(args: dict[str, Any], field: str, tool_name: str) -> str:
    value = args.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires a non-empty '{field}' field")
    return value.strip()


def _optional_string(args: dict[str, Any], field: str) -> str | None:
    value = args.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field}' must be a string when provided")
    return value.strip()


def get_memory_output_fields() -> list[str]:
    return [
        "status",
        "memory.memory_id",
        "memory.text",
        "memory.category",
        "memory.source",
        "memory.user_id",
        "memory.metadata",
        "results[].memory_id",
        "results[].score",
        "memories[].memory_id",
    ]


def get_user_profile_output_fields() -> list[str]:
    return [
        "status",
        "profile.user_id",
        "profile.memory_count",
        "profile.category_breakdown",
        "profile.preferences",
        "profile.recurring_topics",
        "profile.recent_memories",
        "profile.profile_summary",
        "summary.user_id",
        "summary.memory_count",
        "summary.summary",
    ]


async def handle_add_memory(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    with observe.trace("memory_api.add", **_scope_args(args)):
        memory = _memory_api(context).add_memory(
            text=_required_string(args, "text", "add_memory"),
            category=_optional_string(args, "category"),
            source=_optional_string(args, "source") or "manual",
            file_id=_optional_string(args, "file_id"),
            metadata=args.get("metadata") or {},
            **_scope_args(args),
        )
    if not isinstance(memory["metadata"], dict):
        raise ValueError("'metadata' must be an object when provided")
    return json.dumps(
        {
            "status": "success",
            "memory": memory,
            "message": f"Added memory '{memory['memory_id']}'",
        },
        indent=2,
    )


async def handle_search_memory(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    query = _required_string(args, "query", "search_memory")
    top_k = args.get("top_k", 5)
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("'top_k' must be a positive integer")
    with observe.trace("memory_api.search", query=query, top_k=top_k, **_scope_args(args)):
        results = _memory_api(context).search_memory(
            query=query,
            top_k=top_k,
            category=_optional_string(args, "category"),
            **_scope_args(args),
        )
    return json.dumps(
        {
            "status": "success",
            "query": query,
            "total_results": len(results),
            "results": results,
        },
        indent=2,
    )


async def handle_list_memories(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    limit = args.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("'limit' must be a positive integer when provided")
    with observe.trace("memory_api.list", limit=limit or 0, **_scope_args(args)):
        memories = _memory_api(context).list_memories(
            category=_optional_string(args, "category"),
            limit=limit,
            **_scope_args(args),
        )
    return json.dumps(
        {
            "status": "success",
            "total_memories": len(memories),
            "memories": memories,
        },
        indent=2,
    )


async def handle_delete_memory(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    memory_id = _required_string(args, "memory_id", "delete_memory")
    with observe.trace("memory_api.delete", memory_id=memory_id, **_scope_args(args)):
        deleted = _memory_api(context).delete_memory(memory_id, **_scope_args(args))
    return json.dumps(
        {
            "status": "success",
            "deleted_memory": deleted,
            "message": f"Deleted memory '{memory_id}'",
        },
        indent=2,
    )


async def handle_summarize_user_memory(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    user_id = _required_string(args, "user_id", "summarize_user_memory")
    with observe.trace(
        "memory_api.summarize", user_id=user_id, workspace_id=args.get("workspace_id")
    ):
        summary = _memory_api(context).summarize_user_memory(
            user_id=user_id,
            workspace_id=args.get("workspace_id"),
            agent_id=args.get("agent_id"),
            session_id=args.get("session_id"),
        )
    return json.dumps({"status": "success", "summary": summary}, indent=2)


async def handle_get_user_profile(context: dict[str, Any], args: dict[str, Any]) -> str:
    observe = get_observability()
    user_id = _required_string(args, "user_id", "get_user_profile")
    with observe.trace(
        "memory_api.profile", user_id=user_id, workspace_id=args.get("workspace_id")
    ):
        profile = _memory_api(context).get_user_profile(
            user_id=user_id,
            workspace_id=args.get("workspace_id"),
            agent_id=args.get("agent_id"),
            session_id=args.get("session_id"),
        )
    return json.dumps({"status": "success", "profile": profile}, indent=2)
