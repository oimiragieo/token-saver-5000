"""Phase 2 Chunk 6B SSRF regression fence for token-saver url_fetcher.py.

Numbered tests T11-T17 cover size-cap and additional SSRF fence scenarios
for the library-level fetch_url function.
"""

import pytest
import httpx

from src.url_fetcher import URLFetchError, fetch_url


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


class TestClassESizeCap:
    """T11-T16: size-cap fence — large responses MUST raise URLFetchError(code='too_large')."""

    @pytest.mark.asyncio
    async def test_T11_content_length_1_byte_over_cap_raises(self):
        """Content-Length exactly 1 byte over 10 MB must be rejected immediately."""
        ten_mb_plus_one = 10 * 1024 * 1024 + 1
        transport = _make_transport(
            content_length=str(ten_mb_plus_one),
            body=b"x",
        )
        with pytest.raises(URLFetchError) as exc_info:
            await fetch_url("https://example.com/big.bin", _transport=transport)
        assert exc_info.value.code == "too_large"

    @pytest.mark.asyncio
    async def test_T12_content_length_100_mb_raises(self):
        """Content-Length of 100 MB must be rejected immediately."""
        hundred_mb = 100 * 1024 * 1024
        transport = _make_transport(
            content_length=str(hundred_mb),
            body=b"x",
        )
        with pytest.raises(URLFetchError) as exc_info:
            await fetch_url("https://example.com/huge.bin", _transport=transport)
        assert exc_info.value.code == "too_large"

    @pytest.mark.asyncio
    async def test_T13_body_streaming_over_10_mb_raises(self):
        """Body that exceeds 10 MB during streaming (no Content-Length) must raise."""
        big_body = b"y" * (10 * 1024 * 1024 + 1)
        transport = _make_transport(body=big_body)
        with pytest.raises(URLFetchError) as exc_info:
            await fetch_url("https://example.com/stream.txt", _transport=transport)
        assert exc_info.value.code == "too_large"

    @pytest.mark.asyncio
    async def test_T14_exact_10_mb_body_is_allowed(self):
        """Body of exactly 10 MB must NOT raise (boundary check)."""
        exact_body = b"z" * (10 * 1024 * 1024)
        transport = _make_transport(body=exact_body)
        result = await fetch_url("https://example.com/exact.txt", _transport=transport)
        assert len(result) == 10 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_T15_large_response_truncated_or_allowed(self):
        """
        Fence hardening (v1.35.3 P2): a response exceeding 10 MB MUST raise
        URLFetchError with code 'too_large'.  The old implementation used a
        try/except soft-assert that allowed the call to silently succeed when
        the size-cap branch was not reached. This test asserts the hard raise.
        """
        ten_mb_plus_one = 10 * 1024 * 1024 + 1
        transport = _make_transport(
            content_length=str(ten_mb_plus_one),
            body=b"x" * min(ten_mb_plus_one, 64),  # header check fires first
        )
        with pytest.raises(URLFetchError) as exc_info:
            await fetch_url("https://example.com/large_fence.txt", _transport=transport)
        assert exc_info.value.code == "too_large"

    @pytest.mark.asyncio
    async def test_T16_size_cap_code_is_exactly_too_large(self):
        """URLFetchError raised for size-cap must carry code=='too_large' (not a subcode)."""
        big_body = b"a" * (10 * 1024 * 1024 + 512)
        transport = _make_transport(body=big_body)
        exc = None
        try:
            await fetch_url("https://example.com/body_size.txt", _transport=transport)
        except URLFetchError as e:
            exc = e
        assert exc is not None, "Expected URLFetchError to be raised"
        assert exc.code == "too_large", f"Expected code='too_large', got {exc.code!r}"


class TestClassFSchemeAndRedirectFence:
    """T17: scheme + redirect fence regression — catches any reversion in prior guards."""

    @pytest.mark.asyncio
    async def test_T17_http_scheme_still_blocked(self):
        """Plain http:// must still raise URLFetchError after Chunk 6A/6B changes."""
        transport = _make_transport()
        with pytest.raises(URLFetchError) as exc_info:
            await fetch_url("http://example.com/insecure.txt", _transport=transport)
        assert exc_info.value.code == "scheme_not_allowed"
