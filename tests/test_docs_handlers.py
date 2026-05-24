import json
from unittest.mock import AsyncMock, patch

import pytest

from src.handlers import docs_handlers as dh

FIXTURE_LLMS = """
# GotContext Docs

General overview of gotcontext.ai documentation.

## MCP Server

The gotcontext MCP server lets Claude Code and other MCP clients connect to the
gotcontext API. Use it to ingest context, search semantic memory, read skeletons,
and manage agent context without opening a browser.

## MCP Server Setup

Install the MCP server, configure the API key, and verify that the tool catalog
lists gotcontext tools. The MCP server endpoint supports session scoped requests.

## Authentication

Authentication uses gotcontext API keys. Pass the key through the MCP client
configuration or the Authorization bearer header. Keep keys private, rotate
legacy keys when scope controls are available, and use project or workspace
scoping when agents need bounded access to gotcontext resources.

## API Keys

API keys identify the caller and may eventually carry fine grained scopes.
Use separate keys per automation environment.

## Projects

Projects group context, knowledge, budgets, and usage reporting for a team.

## Billing

Billing docs explain plan limits, invoices, overage previews, and portal links.

## Knowledge Base

Knowledge base tools ingest private project facts and query durable memory.

## Compression

Compression docs cover ingest_context, read_skeleton, modulate_region, and
token savings reporting.

## CLI Output

filter_cli_output strips noise from command output, test logs, and diffs.

## Troubleshooting

Troubleshooting docs cover setup errors, MCP server connection failures, and
schema validation issues.
""".strip()


@pytest.fixture(autouse=True)
def docs_fixture():
    chunks = dh._parse_llms_txt(FIXTURE_LLMS)
    by_slug, by_url = dh._build_doc_maps(chunks)
    with (
        patch.object(dh, "_DOC_CHUNKS", chunks),
        patch.object(dh, "_DOC_BY_SLUG", by_slug),
        patch.object(dh, "_DOC_BY_URL", by_url),
        patch.object(dh, "_SESSION_SEEN_URLS", {}),
    ):
        yield


@pytest.mark.asyncio
async def test_search_returns_results_above_threshold():
    response = await dh.handle_gc_search_docs({}, {"query": "mcp server"})
    data = json.loads(response)

    assert len(data["results"]) >= 1
    assert data["results"][0]["score"] > 0


@pytest.mark.asyncio
async def test_search_top_k_respected():
    response = await dh.handle_gc_search_docs({}, {"query": "mcp server", "top_k": 2})
    data = json.loads(response)

    assert len(data["results"]) <= 2


@pytest.mark.asyncio
async def test_read_doc_by_slug_returns_markdown():
    response = await dh.handle_gc_read_doc({}, {"url_or_slug": "authentication"})
    data = json.loads(response)

    assert len(data["markdown"]) >= 100
    assert data["source_url"].endswith("#authentication")
    assert data["length_tokens"] > 0


@pytest.mark.asyncio
async def test_read_doc_truncates_long_responses():
    long_doc = (
        "# GotContext Docs\n\n"
        + "preface " * 5200
        + "\n\n## Authentication\n\n"
        + "authentication scoped api key bearer token " * 1200
    )
    with patch.object(dh, "_fetch_url_markdown", new=AsyncMock(return_value=long_doc)):
        response = await dh.handle_gc_read_doc(
            {},
            {"url_or_slug": "https://gotcontext.ai/docs#authentication"},
        )

    data = json.loads(response)
    assert "length_tokens" in data
    assert data["length_tokens"] > 5000
    assert data["truncated"] is True
    assert dh._count_tokens(data["markdown"]) <= dh.MAX_DOC_TOKENS + 2


@pytest.mark.asyncio
async def test_session_dedup_does_not_repeat_urls():
    args = {"query": "mcp server", "top_k": 1, "session_id": "docs-dedup-session"}
    first = json.loads(await dh.handle_gc_search_docs({}, args))
    second = json.loads(await dh.handle_gc_search_docs({}, args))

    assert first["results"]
    assert second["results"]
    assert first["results"][0]["url"] != second["results"][0]["url"]
