"""Phase 0 characterization tests for core MCP compression flows."""

import json

import pytest

from src.handlers import compression_handlers


@pytest.mark.asyncio
async def test_core_compression_flow_characterization(
    characterization_context_factory, parity_corpus
):
    """Characterize the current ingest -> skeleton -> search -> stats workflow."""
    context = characterization_context_factory()
    document = parity_corpus["documents"][0]

    ingest_payload = json.loads(
        await compression_handlers.handle_ingest(
            context,
            {"text": document["text"], "file_id": document["file_id"]},
        )
    )

    assert ingest_payload["status"] == "success"
    assert ingest_payload["file_id"] == document["file_id"]
    assert ingest_payload["total_nodes"] > 0
    assert ingest_payload["total_tokens"] >= ingest_payload["skeleton_tokens"]
    assert "estimate" in ingest_payload
    assert "cost_savings" in ingest_payload
    assert document["file_id"] in context["retrieval_history"]
    assert context["retrieval_history"][document["file_id"]] == []

    skeleton_payload = json.loads(
        await compression_handlers.handle_read_skeleton(
            context,
            {"file_id": document["file_id"]},
        )
    )

    assert skeleton_payload["file_id"] == document["file_id"]
    assert skeleton_payload["selection_mode"] == "baseline"
    assert skeleton_payload["cache_stable_prefix"] == skeleton_payload["skeleton_text"]
    assert isinstance(skeleton_payload["node_map"], dict)

    search_payload = json.loads(
        await compression_handlers.handle_search_semantic(
            context,
            {
                "file_id": document["file_id"],
                "query": document["query"],
                "top_k": 2,
            },
        )
    )

    assert search_payload["file_id"] == document["file_id"]
    assert search_payload["query"] == document["query"]
    assert search_payload["evidence_aware"] is False
    assert search_payload["total_results"] >= 1
    assert search_payload["results"]
    assert "similarity" in search_payload["results"][0]
    assert "importance" in search_payload["results"][0]
    assert "score_explanation" in search_payload

    stats_output = await compression_handlers.handle_get_stats(
        context,
        {"file_id": document["file_id"]},
    )

    assert f"Document Statistics: {document['file_id']}" in stats_output
    assert "Compression ratio:" in stats_output
    assert "Token savings:" in stats_output


@pytest.mark.asyncio
async def test_query_guided_read_uses_baseline_cache_prefix(
    characterization_context_factory, parity_corpus
):
    """Characterize the cache-stable-prefix contract for query-guided reads."""
    context = characterization_context_factory()
    document = parity_corpus["documents"][0]

    await compression_handlers.handle_ingest(
        context,
        {"text": document["text"], "file_id": document["file_id"]},
    )

    baseline_payload = json.loads(
        await compression_handlers.handle_read_skeleton(
            context,
            {"file_id": document["file_id"]},
        )
    )
    query_payload = json.loads(
        await compression_handlers.handle_read_skeleton(
            context,
            {
                "file_id": document["file_id"],
                "selection_mode": "query_guided",
                "query": document["query"],
            },
        )
    )

    assert query_payload["selection_mode"] == "query_guided"
    assert query_payload["query"] == document["query"]
    assert query_payload["cache_stable_prefix"] == baseline_payload["skeleton_text"]
    assert query_payload["skeleton_text"]
