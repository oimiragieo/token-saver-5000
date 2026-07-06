"""Tests for SSRF-hardened URL fetcher (src/url_fetcher.py).

Coverage: one test per SSRF mitigation + happy path + error propagation.
"""

import pytest
import httpx

from src.url_fetcher import URLFetchError, fetch_url

# ---------------------------------------------------------------------------
# Helpers — minimal MockTransport factory
# ---------------------------------------------------------------------------


def _make_transport(
    *,
    status_code: int = 200,
    body: bytes = b"hello world",
    content_type: str = "text/plain; charset=utf-8",
    content_length: str | None = None,
    location: str | None = None,
) -> httpx.MockTransport:
    """Return an httpx.MockTransport that returns a canned response."""
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = content_length
    if location is not None:
        headers["location"] = location

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            headers=headers,
            content=body,
        )

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_url_happy_path_returns_decoded_text():
    """fetch_url returns decoded text when all mitigations pass."""
    transport = _make_transport(body=b"Hello, world!", content_type="text/plain")
    result = await fetch_url("https://example.com/doc.txt", _transport=transport)
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_fetch_url_returns_json_content():
    """fetch_url accepts application/json content type."""
    transport = _make_transport(body=b'{"key": "value"}', content_type="application/json")
    result = await fetch_url("https://example.com/data.json", _transport=transport)
    assert result == '{"key": "value"}'


# ---------------------------------------------------------------------------
# Mitigation 1: HTTPS-ONLY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mitigation_1_http_scheme_rejected():
    """Plain http:// URLs must be rejected before any network call."""
    transport = _make_transport()
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("http://example.com/doc.txt", _transport=transport)
    assert exc_info.value.code == "scheme_not_allowed"
    assert "https" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_mitigation_1_ftp_scheme_rejected():
    """ftp:// URLs must be rejected."""
    transport = _make_transport()
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("ftp://example.com/file.txt", _transport=transport)
    assert exc_info.value.code == "scheme_not_allowed"


@pytest.mark.asyncio
async def test_mitigation_1_no_hostname_rejected():
    """URLs with no hostname must be rejected."""
    transport = _make_transport()
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https:///path/only", _transport=transport)
    assert exc_info.value.code == "private_ip"


# ---------------------------------------------------------------------------
# Mitigation 3: NO REDIRECTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mitigation_3_301_redirect_rejected():
    """301 redirects must be blocked."""
    transport = _make_transport(status_code=301, location="https://example.com/new", body=b"")
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/old", _transport=transport)
    assert exc_info.value.code == "redirect_blocked"


@pytest.mark.asyncio
async def test_mitigation_3_302_redirect_rejected():
    """302 redirects must be blocked."""
    transport = _make_transport(status_code=302, location="https://example.com/new", body=b"")
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/old", _transport=transport)
    assert exc_info.value.code == "redirect_blocked"


@pytest.mark.asyncio
async def test_mitigation_3_307_redirect_rejected():
    """307 Temporary Redirect must be blocked."""
    transport = _make_transport(status_code=307, location="https://example.com/new", body=b"")
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/old", _transport=transport)
    assert exc_info.value.code == "redirect_blocked"


# ---------------------------------------------------------------------------
# Mitigation 4: SIZE CAP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mitigation_4_content_length_header_too_large():
    """Responses with Content-Length > 10 MB must be rejected before download."""
    ten_mb_plus_one = 10 * 1024 * 1024 + 1
    transport = _make_transport(
        content_length=str(ten_mb_plus_one),
        # Body can be small — the header check should fire first
        body=b"x",
    )
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/large.txt", _transport=transport)
    assert exc_info.value.code == "too_large"


@pytest.mark.asyncio
async def test_mitigation_4_body_too_large_no_content_length():
    """Body bytes > 10 MB must be rejected even without a Content-Length header."""
    big_body = b"x" * (10 * 1024 * 1024 + 1)
    transport = _make_transport(body=big_body)
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/big.txt", _transport=transport)
    assert exc_info.value.code == "too_large"


@pytest.mark.asyncio
async def test_mitigation_4_exactly_10_mb_is_allowed():
    """Body of exactly 10 MB must pass the size cap."""
    exact_body = b"x" * (10 * 1024 * 1024)
    transport = _make_transport(body=exact_body)
    result = await fetch_url("https://example.com/exact.txt", _transport=transport)
    assert len(result) == 10 * 1024 * 1024


@pytest.mark.asyncio
async def test_mitigation_4_size_cap_aborts_stream_without_buffering_whole_body():
    """H2 (supply-chain-hardening): an oversized body with NO Content-Length must
    abort DURING the stream — the fetcher must NOT read the entire response into
    memory before checking the cap. A malicious server on an attacker-controlled
    file_url that omits Content-Length and streams multi-GB of text/plain would
    otherwise OOM the host before the post-buffer len() check ever runs.
    """
    chunk = b"x" * (1024 * 1024)  # 1 MB
    produced = {"bytes": 0}

    async def _body():
        # Would total 32 MB if the consumer drains the whole stream.
        for _ in range(32):
            produced["bytes"] += len(chunk)
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        # NOTE: deliberately NO content-length header — forces the streaming path.
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/plain"},
            content=_body(),
        )

    transport = httpx.MockTransport(handler)
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/stream.txt", _transport=transport)
    assert exc_info.value.code == "too_large"
    # The cap must fire mid-stream — well before the full 32 MB is pulled into memory.
    assert produced["bytes"] <= 12 * 1024 * 1024, (
        f"buffered {produced['bytes'] // (1024 * 1024)} MB before the cap fired — "
        "fetch_url must stream + abort at the cap, not read the whole body into memory"
    )


# ---------------------------------------------------------------------------
# Mitigation 6: CONTENT-TYPE ALLOWLIST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mitigation_6_binary_content_type_rejected():
    """application/octet-stream must be rejected."""
    transport = _make_transport(content_type="application/octet-stream")
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/file.bin", _transport=transport)
    assert exc_info.value.code == "content_type_not_allowed"


@pytest.mark.asyncio
async def test_mitigation_6_image_content_type_rejected():
    """image/png must be rejected."""
    transport = _make_transport(content_type="image/png")
    with pytest.raises(URLFetchError) as exc_info:
        await fetch_url("https://example.com/img.png", _transport=transport)
    assert exc_info.value.code == "content_type_not_allowed"


@pytest.mark.asyncio
async def test_mitigation_6_text_html_allowed():
    """text/html; charset=utf-8 must be accepted (text/* prefix)."""
    transport = _make_transport(content_type="text/html; charset=utf-8")
    result = await fetch_url("https://example.com/page.html", _transport=transport)
    assert result == "hello world"


@pytest.mark.asyncio
async def test_mitigation_6_application_xml_allowed():
    """application/xml must be accepted."""
    transport = _make_transport(content_type="application/xml", body=b"<root/>")
    result = await fetch_url("https://example.com/data.xml", _transport=transport)
    assert result == "<root/>"


@pytest.mark.asyncio
async def test_mitigation_6_application_yaml_allowed():
    """application/yaml must be accepted."""
    transport = _make_transport(content_type="application/yaml", body=b"key: value")
    result = await fetch_url("https://example.com/conf.yaml", _transport=transport)
    assert result == "key: value"


@pytest.mark.asyncio
async def test_mitigation_6_application_x_yaml_allowed():
    """application/x-yaml must be accepted."""
    transport = _make_transport(content_type="application/x-yaml", body=b"key: value")
    result = await fetch_url("https://example.com/conf.yml", _transport=transport)
    assert result == "key: value"


# ---------------------------------------------------------------------------
# Non-2xx responses (not an SSRF mitigation but important hygiene)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_2xx_status_raises_error():
    """4xx responses must raise URLFetchError."""
    transport = _make_transport(status_code=404, body=b"not found")
    with pytest.raises(URLFetchError):
        await fetch_url("https://example.com/missing.txt", _transport=transport)


@pytest.mark.asyncio
async def test_server_error_raises_error():
    """5xx responses must raise URLFetchError."""
    transport = _make_transport(status_code=500, body=b"oops")
    with pytest.raises(URLFetchError):
        await fetch_url("https://example.com/broken.txt", _transport=transport)


# ---------------------------------------------------------------------------
# Error attributes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_urlfetcherror_has_code_attribute():
    """URLFetchError must expose the 'code' attribute set at construction."""
    transport = _make_transport(content_type="image/jpeg")
    try:
        await fetch_url("https://example.com/photo.jpg", _transport=transport)
        pytest.fail("Expected URLFetchError")
    except URLFetchError as exc:
        assert hasattr(exc, "code")
        assert exc.code == "content_type_not_allowed"


def test_urlfetcherror_construction():
    """URLFetchError stores message and code correctly."""
    err = URLFetchError("test message", code="scheme_not_allowed")
    assert str(err) == "test message"
    assert err.code == "scheme_not_allowed"
