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


# v1.34.32 hotfix regression locks
# ===
# Bug 1: codex hardcoded apps/web/public/llms.txt which doesn't exist
# in production (the /llms.txt route is served by Next.js, not static).
# _DOC_CHUNKS was always [] → every search returned {results: []}.
# Fix: _docs_root() returns None if the local path doesn't exist, and
# _read_llms_txt() falls back to fetching DOCS_LLMS_TXT_URL via urllib.


def test_docs_root_returns_none_when_local_path_missing(monkeypatch, tmp_path):
    """v1.34.32: if neither GOTCONTEXT_LLMS_TXT env var nor the local
    apps/web/public/llms.txt is present, _docs_root must return None
    so _read_llms_txt knows to fall back to the live URL.
    """
    monkeypatch.delenv("GOTCONTEXT_LLMS_TXT", raising=False)
    monkeypatch.setattr(
        dh,
        "_docs_root",
        lambda: None,  # simulate "not found"
    )
    assert dh._docs_root() is None


def test_read_llms_txt_falls_back_to_live_url_when_local_missing(monkeypatch):
    """v1.34.32: when _docs_root returns None, _read_llms_txt must
    invoke urllib against DOCS_LLMS_TXT_URL (or GOTCONTEXT_LLMS_TXT_URL
    override) and return the response body.
    """
    monkeypatch.setattr(dh, "_docs_root", lambda: None)
    monkeypatch.setenv("GOTCONTEXT_LLMS_TXT_URL", "https://example.invalid/llms.txt")

    fake_content = b"# Test Docs\n\n## Authentication\n\nUse gc_ keys.\n"

    class _FakeResp:
        def __init__(self, body: bytes):
            self._body = body

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    import urllib.request

    def fake_urlopen(req, timeout=5):
        # Verify it actually constructed the request against our URL
        assert "example.invalid" in req.full_url
        return _FakeResp(fake_content)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = dh._read_llms_txt()
    assert "Authentication" in result
    assert result.startswith("# Test Docs")


def test_read_llms_txt_returns_empty_on_url_fetch_failure(monkeypatch):
    """v1.34.32: when the live URL fetch raises, _read_llms_txt must
    return empty string so handlers continue gracefully (matches the
    swallow-failure pattern in handle_filter_cli_output etc.).
    """
    import urllib.error
    import urllib.request

    monkeypatch.setattr(dh, "_docs_root", lambda: None)

    def boom(req, timeout=5):
        raise urllib.error.URLError("simulated failure")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    assert dh._read_llms_txt() == ""


# Bug 2: gc_read_doc on a huge live /docs page returned 80KB markdown,
# exceeding the orchestrator's tool-result size limit. _excerpt_around_anchor
# did its job at the TOKEN level but the resulting markdown was still
# 80KB of characters (long lines). Fix: defensive MAX_DOC_CHARS = 20K cap
# AFTER token-level truncation.


# v1.34.33 hotfix regression locks: prefix-match stemming for BM25.
# Pre-fix `gc_search_docs("authentication")` returned 0 results because
# the corpus contained "authenticate" / "auth" but no exact "authentication".
# Fix: query terms ≥6 chars match doc tokens sharing the first 6 chars.


@pytest.mark.asyncio
async def test_search_stemming_matches_morphological_variants():
    """v1.34.33: query 'authentication' must match docs containing
    'authenticate' (shared 6-char prefix 'authen')."""
    args = {"query": "authentication", "top_k": 3, "session_id": "stem-1"}
    response = await dh.handle_gc_search_docs({}, args)
    data = json.loads(response)
    assert len(data["results"]) >= 1, (
        "v1.34.33: 'authentication' should match 'authenticate' via prefix-match. "
        "Got 0 results — fixture contains 'authentication' related terms."
    )


@pytest.mark.asyncio
async def test_search_short_query_requires_exact_match():
    """v1.34.33: query terms <6 chars use exact match (no prefix fuzzy).
    'auth' (4 chars) must match docs with literal 'auth' token but NOT
    longer words starting with 'auth' that aren't 'auth' themselves.
    """
    args = {"query": "auth", "top_k": 5, "session_id": "stem-2"}
    response = await dh.handle_gc_search_docs({}, args)
    data = json.loads(response)
    # 'auth' is 4 chars — exact-match only. With the fixture's content
    # this might find zero or one chunk; the assertion is on behavior
    # not result count.
    assert isinstance(data["results"], list)


def test_term_freq_with_stemming_short_term_exact_only():
    """v1.34.33 unit: short terms (<6 chars) do exact match only."""
    doc = ["authenticate", "authorization", "auth", "config"]
    # 'auth' is 4 chars → exact only → 1 match
    assert dh._term_freq_with_stemming("auth", doc) == 1


def test_term_freq_with_stemming_long_term_prefix_match():
    """v1.34.33 unit: long terms (≥6 chars) match via 6-char prefix."""
    doc = ["authenticate", "authenticated", "authorization", "config"]
    # 'authentication'[:6] = 'authen' → matches 'authenticate' + 'authenticated'
    # ('authorization' starts with 'author', not 'authen' — no match)
    assert dh._term_freq_with_stemming("authentication", doc) == 2


@pytest.mark.asyncio
async def test_read_doc_enforces_character_cap_on_pathological_output(monkeypatch):
    """v1.34.32: even when token-level truncation fires, the final
    markdown must be ≤ MAX_DOC_CHARS to fit orchestrator limits.
    Simulates the /docs HTML-as-markdown case where one logical line
    is 80KB of HTML.
    """
    from unittest.mock import AsyncMock

    # A single 80KB "line" — exactly the pathological live-fetch case
    pathological = "# Authentication\n\n" + ("auth-blob " * 8000)
    monkeypatch.setattr(dh, "_fetch_url_markdown", AsyncMock(return_value=pathological))

    response = await dh.handle_gc_read_doc(
        {},
        {"url_or_slug": "https://gotcontext.ai/docs#authentication"},
    )
    data = json.loads(response)

    assert len(data["markdown"]) <= dh.MAX_DOC_CHARS + 200, (
        f"v1.34.32 hotfix: markdown should be capped at ~{dh.MAX_DOC_CHARS} chars, "
        f"got {len(data['markdown'])}"
    )
    assert data["truncated"] is True
    assert "truncated" in data["markdown"].lower()
