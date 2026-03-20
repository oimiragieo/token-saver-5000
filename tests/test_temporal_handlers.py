"""Tests for temporal MCP handlers and retrieval filtering."""

import json

import pytest


@pytest.mark.asyncio
async def test_temporal_handlers_round_trip(handler_context, sample_text_short):
    from src.handlers.compression_handlers import (
        handle_ingest,
        handle_read_skeleton,
        handle_search_semantic,
    )
    from src.handlers.temporal_handlers import (
        handle_get_context_block,
        handle_invalidate_fact,
        handle_list_fact_history,
        handle_search_timeline,
    )

    await handle_ingest(handler_context, {"text": sample_text_short, "file_id": "temporal_doc"})

    facts_before = json.loads(
        await handle_list_fact_history(handler_context, {"file_id": "temporal_doc"})
    )
    fact_id = facts_before["facts"][0]["fact_id"]

    invalidated = json.loads(
        await handle_invalidate_fact(
            handler_context,
            {"fact_id": fact_id, "reason": "superseded by updated knowledge"},
        )
    )
    facts_after = json.loads(
        await handle_list_fact_history(
            handler_context,
            {"file_id": "temporal_doc", "include_invalidated": False},
        )
    )
    context_block = json.loads(
        await handle_get_context_block(handler_context, {"file_id": "temporal_doc"})
    )
    skeleton = json.loads(await handle_read_skeleton(handler_context, {"file_id": "temporal_doc"}))
    search = json.loads(
        await handle_search_semantic(
            handler_context,
            {"query": "machine learning", "file_id": "temporal_doc"},
        )
    )
    timeline = json.loads(
        await handle_search_timeline(handler_context, {"file_id": "temporal_doc"})
    )

    assert invalidated["status"] == "success"
    assert all(item["fact_id"] != fact_id for item in facts_after["facts"])
    assert context_block["context_block"]["active_fact_count"] == len(facts_after["facts"])
    assert fact_id not in skeleton["node_map"]
    assert all(result["node_id"] != fact_id for result in search["results"])
    assert timeline["total_events"] > 0


@pytest.mark.asyncio
async def test_temporal_handlers_include_invalidated_when_requested(
    handler_context, sample_text_short
):
    from src.handlers.compression_handlers import handle_ingest, handle_read_skeleton
    from src.handlers.temporal_handlers import handle_invalidate_fact, handle_list_fact_history

    await handle_ingest(handler_context, {"text": sample_text_short, "file_id": "temporal_doc_2"})
    facts_before = json.loads(
        await handle_list_fact_history(handler_context, {"file_id": "temporal_doc_2"})
    )
    fact_id = facts_before["facts"][0]["fact_id"]
    await handle_invalidate_fact(
        handler_context,
        {"fact_id": fact_id, "reason": "historical only"},
    )

    skeleton = json.loads(
        await handle_read_skeleton(
            handler_context,
            {"file_id": "temporal_doc_2", "include_invalidated": True},
        )
    )

    assert fact_id in skeleton["node_map"]
