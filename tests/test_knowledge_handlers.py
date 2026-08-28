"""Tests for knowledge management MCP handlers (ingest_transcript, compile, lint, index search)."""

import json

import pytest

from src.path_validator import PathValidator
from src.memory_api import MemoryAPI


@pytest.fixture
def memory_context():
    MemoryAPI.reset_singleton()
    ctx = {"memory_api": MemoryAPI()}
    yield ctx
    MemoryAPI.reset_singleton()


# ---------------------------------------------------------------------------
# ingest_transcript handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_transcript_handler(memory_context):
    from src.handlers.memory_handlers import handle_ingest_transcript

    result = json.loads(
        await handle_ingest_transcript(
            memory_context,
            {
                "text": "We decided to use Redis for caching. Watch out for eviction policies.",
                "mode": "all",
            },
        )
    )
    assert result["status"] == "success"
    assert result["total_sentences"] >= 1
    assert result["stored_count"] >= 1
    assert isinstance(result["insights"], list)


@pytest.mark.asyncio
async def test_ingest_transcript_mode_decisions(memory_context):
    from src.handlers.memory_handlers import handle_ingest_transcript

    result = json.loads(
        await handle_ingest_transcript(
            memory_context,
            {"text": "We decided to use TypeScript. Bug in the auth flow.", "mode": "decisions"},
        )
    )
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_ingest_transcript_empty_text_rejected(memory_context):
    from src.handlers.memory_handlers import handle_ingest_transcript

    with pytest.raises(ValueError, match="non-empty"):
        await handle_ingest_transcript(memory_context, {"text": ""})


@pytest.mark.asyncio
async def test_ingest_transcript_invalid_mode_rejected(memory_context):
    from src.handlers.memory_handlers import handle_ingest_transcript

    with pytest.raises(ValueError, match="mode"):
        await handle_ingest_transcript(
            memory_context, {"text": "Some transcript text here.", "mode": "invalid"}
        )


@pytest.mark.asyncio
async def test_ingest_transcript_with_scoping(memory_context):
    from src.handlers.memory_handlers import handle_ingest_transcript

    result = json.loads(
        await handle_ingest_transcript(
            memory_context,
            {
                "text": "We decided to migrate to GraphQL for the API layer.",
                "workspace_id": "acme",
                "user_id": "bob",
            },
        )
    )
    assert result["status"] == "success"
    # Verify scoped storage
    memories = memory_context["memory_api"].list_memories(workspace_id="acme", user_id="bob")
    assert len(memories) >= result["stored_count"]


# ---------------------------------------------------------------------------
# compile_knowledge handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compile_knowledge_handler(memory_context):
    from src.handlers.memory_handlers import handle_add_memory, handle_compile_knowledge

    await handle_add_memory(memory_context, {"text": "Always lint before commits"})
    await handle_add_memory(memory_context, {"text": "Critical bug in auth flow"})

    result = json.loads(await handle_compile_knowledge(memory_context, {}))
    assert result["status"] == "success"
    assert result["total_memories"] == 2
    assert result["articles_count"] >= 1
    assert "index_markdown" in result


@pytest.mark.asyncio
async def test_compile_knowledge_empty(memory_context):
    from src.handlers.memory_handlers import handle_compile_knowledge

    result = json.loads(await handle_compile_knowledge(memory_context, {}))
    assert result["status"] == "success"
    assert result["total_memories"] == 0
    assert result["articles_count"] == 0


@pytest.mark.asyncio
async def test_compile_knowledge_write_files(memory_context, tmp_path):
    from src.handlers.memory_handlers import handle_add_memory, handle_compile_knowledge

    await handle_add_memory(memory_context, {"text": "Use dependency injection pattern always"})
    result = json.loads(
        await handle_compile_knowledge(
            memory_context,
            {"write_files": True, "output_dir": str(tmp_path / "compiled")},
        )
    )
    assert result["status"] == "success"
    assert result["output_dir"] is not None


@pytest.mark.asyncio
async def test_compile_knowledge_rejects_traversal_output_dir(memory_context, tmp_path):
    from src.handlers.memory_handlers import handle_add_memory, handle_compile_knowledge

    memory_context["path_validator"] = PathValidator(allowed_base_dirs=[str(tmp_path.resolve())])
    await handle_add_memory(memory_context, {"text": "Scoped memory for compile test"})

    with pytest.raises(ValueError, match="output_dir"):
        await handle_compile_knowledge(
            memory_context,
            {
                "write_files": True,
                "output_dir": str(tmp_path / ".." / "outside_compile"),
            },
        )


# ---------------------------------------------------------------------------
# get_knowledge_index handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_knowledge_index_handler(memory_context):
    from src.handlers.memory_handlers import handle_add_memory, handle_get_knowledge_index

    await handle_add_memory(memory_context, {"text": "Decided to use FastAPI"})
    result = json.loads(await handle_get_knowledge_index(memory_context, {}))
    assert result["status"] == "success"
    assert "Knowledge Index" in result["index_markdown"]
    assert result["total_memories"] == 1


# ---------------------------------------------------------------------------
# lint_knowledge handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lint_knowledge_handler(memory_context):
    from src.handlers.memory_handlers import handle_add_memory, handle_lint_knowledge

    await handle_add_memory(memory_context, {"text": "Always use mocks in tests"})
    await handle_add_memory(memory_context, {"text": "Never use mocks in integration tests"})

    result = json.loads(await handle_lint_knowledge(memory_context, {}))
    assert result["status"] == "success"
    assert "stale" in result["checks_run"]
    assert "duplicates" in result["checks_run"]
    assert "contradictions" in result["checks_run"]


@pytest.mark.asyncio
async def test_lint_knowledge_invalid_stale_days(memory_context):
    from src.handlers.memory_handlers import handle_lint_knowledge

    with pytest.raises(ValueError, match="stale_days"):
        await handle_lint_knowledge(memory_context, {"stale_days": -1})


# ---------------------------------------------------------------------------
# search_memory_index handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_memory_index_handler(memory_context):
    from src.handlers.memory_handlers import handle_add_memory, handle_search_memory_index

    await handle_add_memory(
        memory_context, {"text": "PostgreSQL is our primary database", "category": "decision"}
    )
    await handle_add_memory(
        memory_context, {"text": "Watch out for N+1 queries", "category": "gotcha"}
    )

    result = json.loads(await handle_search_memory_index(memory_context, {"query": "database"}))
    assert result["status"] == "success"
    assert "index_markdown" in result
    assert result["matched_articles"] >= 1


@pytest.mark.asyncio
async def test_search_memory_index_no_match(memory_context):
    from src.handlers.memory_handlers import handle_add_memory, handle_search_memory_index

    await handle_add_memory(memory_context, {"text": "Use TypeScript for frontend"})
    result = json.loads(
        await handle_search_memory_index(memory_context, {"query": "quantum physics"})
    )
    assert result["status"] == "success"
    assert result["matched_articles"] == 0


@pytest.mark.asyncio
async def test_search_memory_index_requires_query(memory_context):
    from src.handlers.memory_handlers import handle_search_memory_index

    with pytest.raises(ValueError, match="non-empty"):
        await handle_search_memory_index(memory_context, {"query": ""})


# ---------------------------------------------------------------------------
# MCP routing integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_knowledge_tools_registered_in_router():
    from src.handlers.mcp_core import setup_mcp_tools

    tools = setup_mcp_tools("full")
    tool_names = {t.name for t in tools}
    assert "ingest_transcript" in tool_names
    assert "compile_knowledge" in tool_names
    assert "get_knowledge_index" in tool_names
    assert "lint_knowledge" in tool_names
    assert "search_memory_index" in tool_names
