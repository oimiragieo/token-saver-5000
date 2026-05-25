"""Tests for the 4 new SSRF hardening mitigations added in Phase 2 Chunk 6A.

Scope: NEW protections only — do NOT re-test the 6 mitigations already covered in
test_url_fetcher.py (HTTPS-only, public-IP DNS check, no redirects, size cap, timeout,
content-type allowlist).

The 4 gaps this file covers:
  Gap 1 — IPv4-mapped IPv6 normalisation  (::ffff:127.0.0.1 collapses to 127.0.0.1)
  Gap 2 — DNS rebinding detection         (IP set changes between check and connect)
  Gap 3 — Blocked-hostname pre-check      (169.254.169.254, metadata.google.internal, …)
  Gap 4 — *.internal TLD block            (Fly.io / GCP internal networking)

All tests that exercise DNS-level behaviour use unittest.mock.patch to avoid real socket
calls.  Tests for _normalize_ip_address and _is_blocked_hostname are synchronous unit
tests on the helper functions directly.
"""

import socket
from unittest.mock import patch

import httpx
import pytest

from src.url_fetcher import (
    URLFetchError,
    _is_blocked_hostname,
    _is_private_ip,
    _normalize_ip_address,
    fetch_url,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport(
    *,
    status_code: int = 200,
    body: bytes = b"hello world",
    content_type: str = "text/plain; charset=utf-8",
) -> httpx.MockTransport:
    headers = {"content-type": content_type}
    return httpx.MockTransport(
        handler=lambda req: httpx.Response(status_code, headers=headers, content=body)
    )


def _fake_getaddrinfo_for(ip: str):
    """Return a getaddrinfo-compatible callable that always resolves to *ip*."""

    def _impl(host, port, *args, **kwargs):
        family = socket.AF_INET if ":" not in ip else socket.AF_INET6
        return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]

    return _impl


# ---------------------------------------------------------------------------
# Gap 1 — IPv4-mapped IPv6 normalisation
# ---------------------------------------------------------------------------


class TestIPv4MappedIPv6Normalisation:
    """::ffff:<private> addresses must be detected as private."""

    def test_normalize_ipv4_mapped_loopback(self):
        """::ffff:127.0.0.1 must collapse to 127.0.0.1 (loopback)."""
        ip = _normalize_ip_address("::ffff:127.0.0.1")
        assert str(ip) == "127.0.0.1"
        assert ip.is_loopback

    def test_normalize_ipv4_mapped_private_rfc1918(self):
        """::ffff:192.168.1.1 must collapse to 192.168.1.1 (RFC 1918)."""
        ip = _normalize_ip_address("::ffff:192.168.1.1")
        assert str(ip) == "192.168.1.1"
        assert ip.is_private

    def test_is_private_ip_rejects_ipv4_mapped_loopback(self):
        """_is_private_ip must return True for the ::ffff:127.0.0.1 string."""
        assert _is_private_ip("::ffff:127.0.0.1") is True

    def test_is_private_ip_rejects_ipv4_mapped_rfc1918(self):
        """_is_private_ip must return True for the ::ffff:10.0.0.1 string."""
        assert _is_private_ip("::ffff:10.0.0.1") is True

    def test_normalize_pure_ipv6_unchanged(self):
        """A real IPv6 address with no IPv4 mapping must not be collapsed."""
        ip = _normalize_ip_address("2001:db8::1")
        assert ip.version == 6

    @pytest.mark.asyncio
    async def test_fetch_url_rejects_when_dns_resolves_to_ipv4_mapped_loopback(self):
        """fetch_url must raise private_ip when host resolves to ::ffff:127.0.0.1."""
        with patch("src.url_fetcher.socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::ffff:127.0.0.1", 0))
            ]
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://example.com/data.txt")
        assert exc_info.value.code == "private_ip"


# ---------------------------------------------------------------------------
# Gap 2 — DNS rebinding detection
# ---------------------------------------------------------------------------


class TestDNSRebindingDetection:
    """IP set must not change between the pre-connect check and the connection."""

    @pytest.mark.asyncio
    async def test_fetch_url_raises_on_dns_rebinding(self):
        """If the second DNS resolution returns a different IP, raise dns_rebinding."""
        call_count = 0

        def _alternating_gai(host, port, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: resolves to a public IP
                return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]
            else:
                # Second call: switches to a private IP (DNS rebinding attack)
                return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))]

        with patch("src.url_fetcher.socket.getaddrinfo", side_effect=_alternating_gai):
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://example.com/data.txt")

        assert exc_info.value.code in ("private_ip", "dns_rebinding")

    @pytest.mark.asyncio
    async def test_fetch_url_raises_on_ip_set_change_even_if_both_public(self):
        """IP set change (even public→public) must raise dns_rebinding."""
        call_count = 0

        def _changing_gai(host, port, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            ip = "93.184.216.34" if call_count == 1 else "1.1.1.1"
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

        with patch("src.url_fetcher.socket.getaddrinfo", side_effect=_changing_gai):
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://example.com/data.txt")

        assert exc_info.value.code == "dns_rebinding"

    @pytest.mark.asyncio
    async def test_fetch_url_succeeds_when_ip_set_stable(self):
        """fetch_url must succeed when both DNS resolutions return the same IP."""
        stable_ip = "93.184.216.34"

        with patch(
            "src.url_fetcher.socket.getaddrinfo",
            side_effect=_fake_getaddrinfo_for(stable_ip),
        ):
            transport = _make_transport()
            result = await fetch_url("https://example.com/data.txt", _transport=transport)

        assert result == "hello world"


# ---------------------------------------------------------------------------
# Gap 3 — Blocked-hostname pre-check
# ---------------------------------------------------------------------------


class TestBlockedHostnamePreCheck:
    """Metadata endpoints and known internal hostnames must be blocked before DNS."""

    def test_is_blocked_hostname_aws_imds(self):
        assert _is_blocked_hostname("169.254.169.254") is True

    def test_is_blocked_hostname_nip_io_bypass(self):
        assert _is_blocked_hostname("169.254.169.254.nip.io") is True

    def test_is_blocked_hostname_gcp_metadata(self):
        assert _is_blocked_hostname("metadata.google.internal") is True

    def test_is_blocked_hostname_aws_metadata(self):
        assert _is_blocked_hostname("metadata.aws.internal") is True

    def test_is_blocked_hostname_bare_metadata(self):
        assert _is_blocked_hostname("metadata") is True

    def test_is_blocked_hostname_case_insensitive(self):
        assert _is_blocked_hostname("Metadata.Google.Internal") is True

    def test_is_blocked_hostname_public_not_blocked(self):
        assert _is_blocked_hostname("example.com") is False

    @pytest.mark.asyncio
    async def test_fetch_url_raises_before_dns_for_aws_imds(self):
        """169.254.169.254 must be blocked at the hostname check, never reaching DNS."""
        with patch("src.url_fetcher.socket.getaddrinfo") as mock_gai:
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://169.254.169.254/latest/meta-data/")
        mock_gai.assert_not_called()
        assert exc_info.value.code == "blocked_hostname"

    @pytest.mark.asyncio
    async def test_fetch_url_raises_for_gcp_metadata_hostname(self):
        """metadata.google.internal must be caught before DNS resolution."""
        with patch("src.url_fetcher.socket.getaddrinfo") as mock_gai:
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://metadata.google.internal/computeMetadata/v1/")
        mock_gai.assert_not_called()
        assert exc_info.value.code == "blocked_hostname"


# ---------------------------------------------------------------------------
# Gap 4 — *.internal TLD block
# ---------------------------------------------------------------------------


class TestInternalTLDBlock:
    """Any hostname under the .internal TLD must be blocked (Fly.io, GCP, private DNS)."""

    def test_is_blocked_hostname_fly_internal(self):
        assert _is_blocked_hostname("my-app.internal") is True

    def test_is_blocked_hostname_deep_internal(self):
        assert _is_blocked_hostname("db.cluster.production.internal") is True

    def test_is_blocked_hostname_bare_internal(self):
        assert _is_blocked_hostname("internal") is True

    @pytest.mark.asyncio
    async def test_fetch_url_raises_for_fly_internal_hostname(self):
        """my-app.internal must be blocked before DNS resolution."""
        with patch("src.url_fetcher.socket.getaddrinfo") as mock_gai:
            with pytest.raises(URLFetchError) as exc_info:
                await fetch_url("https://my-app.internal/api/v1/data")
        mock_gai.assert_not_called()
        assert exc_info.value.code == "blocked_hostname"
