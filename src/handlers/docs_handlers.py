"""GotContext documentation MCP handlers.

The docs index is built once at module import time from the monorepo's
apps/web/public/llms.txt and then kept in memory until the process restarts.
Fly redeploys restart the process, which is the cache invalidation boundary.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DOCS_BASE_URL = "https://gotcontext.ai/docs"
DOCS_LLMS_TXT_URL = "https://gotcontext.ai/llms.txt"
MAX_DOC_TOKENS = 5000
# v1.34.32 hotfix: defensive character cap on truncated markdown.
# Even after _excerpt_around_anchor trims to MAX_DOC_TOKENS, the
# orchestrator tool-result wrapper has its own size limit; cap at
# ~20K chars so a 5K-token excerpt with very long words still fits.
MAX_DOC_CHARS = 20_000


@dataclass(frozen=True)
class DocChunk:
    title: str
    markdown: str
    slug: str
    url: str


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "docs"


def _tokenize(value: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", value.lower())


def _count_tokens(value: str) -> int:
    return len(re.findall(r"\S+", value))


def _trim_words(value: str, limit: int = 240) -> str:
    squashed = re.sub(r"\s+", " ", value).strip()
    if len(squashed) <= limit:
        return squashed
    return squashed[: limit - 3].rstrip() + "..."


def _docs_root() -> Path | None:
    """Return a local llms.txt path if one is configured/available, else None.

    v1.34.32 hotfix: the original spec assumed `apps/web/public/llms.txt`
    existed as a static file, but the production `/llms.txt` is served by a
    Next.js route handler (`apps/web/src/app/llms.txt/route.ts`) — there is
    no static file. Returning None tells `_read_llms_txt` to fall back to
    fetching the live URL.
    """
    env_path = os.environ.get("GOTCONTEXT_LLMS_TXT")
    if env_path:
        return Path(env_path)
    # Best-effort local-dev path (works when running pytest inside the
    # monorepo with apps/web/public populated). Returns None if missing,
    # which triggers the live-URL fallback in _read_llms_txt.
    candidate = Path(__file__).resolve().parents[3] / "apps" / "web" / "public" / "llms.txt"
    return candidate if candidate.exists() else None


def _read_llms_txt() -> str:
    """Load llms.txt content; prefer local file, fall back to live URL fetch.

    v1.34.32 hotfix: synchronous urllib fetch at module init (no asyncio
    available pre-event-loop). 5s timeout. Empty string on failure (handler
    returns empty results gracefully — already the swallow-failure pattern).
    """
    local = _docs_root()
    if local is not None:
        try:
            return local.read_text(encoding="utf-8")
        except Exception as exc:
            logger.warning("docs MCP: local llms.txt read failed (%s); trying live URL", exc)

    # Live-URL fallback — production case where the Fly container has no
    # apps/web/public/llms.txt baked in. Uses urllib (stdlib) so this works
    # at module-import time before the asyncio loop exists.
    import urllib.request
    import urllib.error

    url = os.environ.get("GOTCONTEXT_LLMS_TXT_URL", DOCS_LLMS_TXT_URL)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "gotcontext-mcp-docs-handler/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            logger.info("docs MCP: loaded llms.txt from %s (%d bytes)", url, len(content))
            return content
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        logger.warning("docs MCP: live llms.txt fetch from %s failed: %s", url, exc)
        return ""


def _extract_first_url(markdown: str, slug: str) -> str:
    match = re.search(r"https?://[^\s)\]>\"']+", markdown)
    if match:
        return match.group(0).rstrip(".,")
    return f"{DOCS_BASE_URL}#{slug}"


def _parse_llms_txt(content: str) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    title = "GotContext documentation"
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, title
        markdown = "\n".join(buffer).strip()
        if not markdown:
            return
        slug = _slugify(title)
        chunks.append(
            DocChunk(
                title=title,
                markdown=markdown,
                slug=slug,
                url=_extract_first_url(markdown, slug),
            )
        )
        buffer = []

    for raw_line in content.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw_line)
        if heading:
            flush()
            title = heading.group(2).strip()
        buffer.append(raw_line)
    flush()

    if not chunks and content.strip():
        slug = "docs"
        chunks.append(
            DocChunk(
                title="GotContext documentation",
                markdown=content.strip(),
                slug=slug,
                url=f"{DOCS_BASE_URL}#{slug}",
            )
        )
    return chunks


def _build_doc_maps(chunks: list[DocChunk]) -> tuple[dict[str, DocChunk], dict[str, DocChunk]]:
    by_slug: dict[str, DocChunk] = {}
    by_url: dict[str, DocChunk] = {}
    for chunk in chunks:
        by_slug.setdefault(chunk.slug, chunk)
        by_url.setdefault(chunk.url, chunk)
        parsed = urlparse(chunk.url)
        if parsed.fragment:
            by_slug.setdefault(parsed.fragment, chunk)
    return by_slug, by_url


def _idf(term: str, tokenized_docs: list[list[str]]) -> float:
    docs_with_term = sum(1 for doc in tokenized_docs if term in doc)
    return math.log(1 + (len(tokenized_docs) - docs_with_term + 0.5) / (docs_with_term + 0.5))


# v1.34.33 hotfix: minimum prefix length for stemming match. Query terms
# shorter than this require exact match (avoids false positives on short
# common words). 6 chars is the sweet spot: "auth" (4) → exact only;
# "authentication" (14) → matches "authenticate" via "authen" prefix.
_PREFIX_MATCH_MIN_LEN = 6


def _term_freq_with_stemming(term: str, doc: list[str]) -> int:
    """Count occurrences of `term` in `doc`, with prefix-match stemming.

    v1.34.33 hotfix: docs MCP must match morphological variants.
    Pre-fix `gc_search_docs(query="authentication")` returned 0 results
    because the corpus has "authenticate" / "auth" / "auth-gated" but
    not the exact string "authentication" — common English suffixes
    (-ation, -ing, -ed, etc.) made the exact-match BM25 brittle.

    Rule: if query term ≥ 6 chars, match any doc token sharing the
    first 6 chars as prefix (e.g., "authentication"[:6]="authen" matches
    "authenticate", "authenticated", "authenticating"). Short query terms
    (<6 chars) require exact match to avoid false positives.
    """
    exact = doc.count(term)
    if len(term) < _PREFIX_MATCH_MIN_LEN:
        return exact
    prefix = term[:_PREFIX_MATCH_MIN_LEN]
    return sum(1 for tok in doc if tok == term or tok.startswith(prefix))


def _bm25_scores(query: str, chunks: list[DocChunk]) -> list[tuple[DocChunk, float]]:
    query_terms = _tokenize(query)
    if not query_terms or not chunks:
        return []

    documents = [_tokenize(f"{chunk.title}\n{chunk.markdown}") for chunk in chunks]
    avgdl = sum(len(doc) for doc in documents) / max(1, len(documents))
    k1 = 1.5
    b = 0.75
    scored: list[tuple[DocChunk, float]] = []

    for chunk, doc in zip(chunks, documents):
        if not doc:
            continue
        score = 0.0
        doc_len = len(doc)
        for term in query_terms:
            freq = _term_freq_with_stemming(term, doc)
            if freq == 0:
                continue
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1))
            score += _idf(term, documents) * numerator / denominator
        if score > 0:
            scored.append((chunk, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def _normalize_top_k(value: Any) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        top_k = 5
    return max(1, min(top_k, 20))


def _normalize_slug_or_url(value: str) -> tuple[str, str | None]:
    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme and parsed.netloc:
        return value, parsed.fragment or None
    slug = _slugify(value)
    return f"{DOCS_BASE_URL}#{slug}", slug


def _excerpt_around_anchor(markdown: str, anchor_slug: str | None, max_tokens: int) -> str:
    words = re.findall(r"\S+", markdown)
    if len(words) <= max_tokens:
        return markdown

    if anchor_slug:
        lines = markdown.splitlines()
        token_count = 0
        center_token = 0
        for line in lines:
            heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
            if heading and _slugify(heading.group(1)) == anchor_slug:
                center_token = token_count
                break
            token_count += _count_tokens(line)
        else:
            anchor_terms = set(_tokenize(anchor_slug.replace("-", " ")))
            best_score = -1
            best_index = 0
            token_count = 0
            for line in lines:
                line_terms = set(_tokenize(line))
                score = len(anchor_terms & line_terms)
                if score > best_score:
                    best_score = score
                    best_index = token_count
                token_count += _count_tokens(line)
            center_token = best_index
    else:
        center_token = 0

    half_window = max_tokens // 2
    start = max(0, center_token - half_window)
    end = min(len(words), start + max_tokens)
    start = max(0, end - max_tokens)
    prefix = "[...]\n" if start > 0 else ""
    suffix = "\n[...]" if end < len(words) else ""
    return prefix + " ".join(words[start:end]) + suffix


async def _fetch_url_markdown(url: str) -> str:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=15) as response:
            response.raise_for_status()
            return await response.text()


_LLMS_TXT = _read_llms_txt()
_DOC_CHUNKS = _parse_llms_txt(_LLMS_TXT)
_DOC_BY_SLUG, _DOC_BY_URL = _build_doc_maps(_DOC_CHUNKS)
_SESSION_SEEN_URLS: dict[str, set[str]] = {}


async def handle_gc_search_docs(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Search gotcontext product and API docs."""

    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string", "results": []})

    top_k = _normalize_top_k(args.get("top_k", 5))
    session_id = args.get("session_id")

    try:
        ranked = _bm25_scores(query, _DOC_CHUNKS)
        seen_urls = _SESSION_SEEN_URLS.setdefault(session_id, set()) if session_id else set()
        results = []
        for chunk, score in ranked:
            if session_id and chunk.url in seen_urls:
                continue
            results.append(
                {
                    "title": chunk.title,
                    "snippet": _trim_words(chunk.markdown),
                    "url": chunk.url,
                    "score": round(score, 4),
                }
            )
            if len(results) >= top_k:
                break

        if session_id:
            seen_urls.update(result["url"] for result in results)

        return json.dumps({"results": results})
    except Exception as exc:
        logger.warning("gc_search_docs failed: %s", exc)
        return json.dumps({"error": "docs search failed", "results": []})


async def handle_gc_read_doc(context: dict[str, Any], args: dict[str, Any]) -> str:
    """Read a gotcontext docs page by URL or slug."""

    url_or_slug = args.get("url_or_slug", "")
    if not isinstance(url_or_slug, str) or not url_or_slug.strip():
        return json.dumps({"error": "url_or_slug must be a non-empty string"})

    try:
        source_url, anchor_slug = _normalize_slug_or_url(url_or_slug)
        chunk = _DOC_BY_SLUG.get(anchor_slug or "") or _DOC_BY_URL.get(source_url)

        if chunk and not urlparse(url_or_slug).scheme:
            markdown = chunk.markdown
            source_url = chunk.url
        else:
            markdown = await _fetch_url_markdown(source_url)

        original_tokens = _count_tokens(markdown)
        truncated = original_tokens > MAX_DOC_TOKENS
        if truncated:
            markdown = _excerpt_around_anchor(markdown, anchor_slug, MAX_DOC_TOKENS)

        # v1.34.32 hotfix: defensive character cap. The token-based truncation
        # above can still produce 80KB+ output when fetching HTML pages that
        # have very long lines (the live /docs page is one big React-rendered
        # blob). Hard-cap at MAX_DOC_CHARS to fit within orchestrator tool-result
        # limits regardless of token semantics.
        if len(markdown) > MAX_DOC_CHARS:
            markdown = (
                markdown[:MAX_DOC_CHARS]
                + f"\n\n... (truncated at {MAX_DOC_CHARS} chars; call gc_search_docs for navigation)"
            )
            truncated = True

        return json.dumps(
            {
                "markdown": markdown,
                "source_url": source_url,
                "length_tokens": original_tokens,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        logger.warning("gc_read_doc failed: %s", exc)
        return json.dumps({"error": "docs read failed", "source_url": url_or_slug})
