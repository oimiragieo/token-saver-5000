"""Handlers for explicit memory, personalization, and knowledge management APIs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..memory_api import MemoryAPI
from ..path_validator import PathValidator
from ..knowledge_compiler import KnowledgeCompiler
from ..knowledge_lint import KnowledgeLinter
from ..transcript_extractor import ingest_transcript
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


# ---------------------------------------------------------------------------
# Transcript ingestion (Phase 1)
# ---------------------------------------------------------------------------


async def handle_ingest_transcript(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Extract insights from a conversation transcript and store as memories."""
    observe = get_observability()
    text = _required_string(args, "text", "ingest_transcript")
    mode = args.get("mode", "all")
    if mode not in ("all", "decisions", "patterns"):
        raise ValueError("'mode' must be one of: all, decisions, patterns")
    source = args.get("source", "transcript")

    with observe.trace("transcript.ingest", mode=mode, **_scope_args(args)):
        result = ingest_transcript(
            text,
            mode=mode,
            source=source if isinstance(source, str) else "transcript",
            memory_api=_memory_api(context),
            **_scope_args(args),
        )

    return json.dumps(
        {
            "status": "success",
            "total_sentences": result.total_sentences,
            "extracted_count": result.extracted_count,
            "stored_count": result.stored_count,
            "insights": result.insights,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Knowledge compilation (Phase 2)
# ---------------------------------------------------------------------------


async def handle_compile_knowledge(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Compile flat memories into cross-linked concept articles + index."""
    observe = get_observability()
    write_files = bool(args.get("write_files", False))
    output_dir = _optional_string(args, "output_dir")

    validated_output_dir: Path | None = None
    if write_files and output_dir:
        path_validator = context.get("path_validator")
        if path_validator is None:
            path_validator = PathValidator(allowed_base_dirs=[os.getcwd(), os.path.expanduser("~")])
        try:
            output_dir = path_validator.validate(output_dir)
        except ValueError as exc:
            raise ValueError(f"Invalid output_dir: {exc}") from exc
        validated_output_dir = Path(output_dir)

    with observe.trace("knowledge.compile", write_files=write_files, **_scope_args(args)):
        compiler = KnowledgeCompiler(
            output_dir=validated_output_dir,
        )
        result = compiler.compile_from_api(
            memory_api=_memory_api(context),
            **_scope_args(args),
            write_files=write_files,
        )

    articles_summary = [
        {"title": a.title, "category": a.category, "entry_count": len(a.memories)}
        for a in result.articles
    ]

    return json.dumps(
        {
            "status": "success",
            "total_memories": result.total_memories,
            "deduplicated": result.deduplicated,
            "articles_count": len(result.articles),
            "articles": articles_summary,
            "index_markdown": result.index_markdown,
            "output_dir": result.output_dir if write_files else None,
        },
        indent=2,
    )


async def handle_get_knowledge_index(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Return the compiled knowledge index for index-first retrieval."""
    observe = get_observability()

    with observe.trace("knowledge.get_index", **_scope_args(args)):
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(
            memory_api=_memory_api(context),
            **_scope_args(args),
            write_files=False,
        )

    return json.dumps(
        {
            "status": "success",
            "index_markdown": result.index_markdown,
            "articles_count": len(result.articles),
            "total_memories": result.total_memories,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Knowledge lint (Phase 3)
# ---------------------------------------------------------------------------


async def handle_lint_knowledge(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Run quality checks on stored memories and return a lint report."""
    observe = get_observability()
    stale_days = args.get("stale_days", 30)
    if not isinstance(stale_days, int) or stale_days <= 0:
        raise ValueError("'stale_days' must be a positive integer")

    with observe.trace("knowledge.lint", stale_days=stale_days, **_scope_args(args)):
        linter = KnowledgeLinter(stale_days=stale_days)
        report = linter.lint_from_api(
            memory_api=_memory_api(context),
            **_scope_args(args),
        )

    return json.dumps(
        {
            "status": "success",
            **report.to_dict(),
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Index-first retrieval (Phase 4)
# ---------------------------------------------------------------------------


async def handle_search_memory_index(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Search memories via the compiled knowledge index.

    Returns the full index markdown plus any articles matching the query,
    enabling index-guided retrieval for small knowledge bases (<500 entries).
    """
    observe = get_observability()
    query = _required_string(args, "query", "search_memory_index")

    with observe.trace("memory.search_index", query=query, **_scope_args(args)):
        compiler = KnowledgeCompiler()
        result = compiler.compile_from_api(
            memory_api=_memory_api(context),
            **_scope_args(args),
            write_files=False,
        )

        # Filter articles whose content matches the query
        query_lower = query.lower()
        matched = []
        for article in result.articles:
            article_text = " ".join(m.get("text", "") for m in article.memories).lower()
            if query_lower in article_text or any(w in article_text for w in query_lower.split()):
                matched.append(
                    {
                        "title": article.title,
                        "category": article.category,
                        "entries": article.memories,
                        "markdown": article.to_markdown(),
                    }
                )

    return json.dumps(
        {
            "status": "success",
            "query": query,
            "index_markdown": result.index_markdown,
            "matched_articles": len(matched),
            "articles": matched,
        },
        indent=2,
    )
