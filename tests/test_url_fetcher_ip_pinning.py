"""#282 — DNS-rebind TOCTOU close: pin the validated IP to the connection.

The pre-existing SSRF fence resolves + checks the host twice (``_resolve_and_check_host``
then ``_check_dns_rebinding``) but then hands the ORIGINAL URL to httpx, which performs
its OWN third DNS resolution at connect time — an attacker who flips the record in the
window between the second check and httpx's connect can still land on a private IP.

The fix pins the exact validated IP into the outgoing request (URL host rewritten to the
IP) while preserving the original hostname for the ``Host`` header AND the TLS SNI /
certificate check via httpx's ``sni_hostname`` request extension. There is no third
resolution: the IP that was validated is the IP connected to.

Reference pattern: langchain-core ``SSRFSafeTransport`` + httpx docs ``sni_hostname``
extension (both converge on URL-host→IP rewrite + sni_hostname + preserved Host).
"""

import socket
from unittest.mock import patch

import httpx
import pytest

from src.url_fetcher import _pin_request, fetch_url


def _fake_getaddrinfo_for(ip: str):
    def _impl(host, port, *args, **kwargs):
        family = socket.AF_INET if ":" not in ip else socket.AF_INET6
        return [(family, socket.SOCK_STREAM, 0, "", (ip, 0))]

    return _impl


# ---------------------------------------------------------------------------
# Unit: _pin_request — the TLS-critical rewrite in isolation
# ---------------------------------------------------------------------------


class TestPinRequest:
    def test_rewrites_host_to_ip_preserves_sni_and_host_header(self):
        url, headers, ext = _pin_request(
            "https://example.com/data.txt", "example.com", {"93.184.216.34"}
        )
        assert url.host == "93.184.216.34"  # connect target is the validated IP
        assert headers["Host"] == "example.com"  # Host header = original hostname
        assert ext["sni_hostname"] == "example.com"  # TLS SNI / cert check = hostname
        assert str(url).startswith("https://93.184.216.34/data.txt")

    def test_deterministic_pick_among_multiple_validated_ips(self):
        url, _, _ = _pin_request("https://x.com/", "x.com", {"8.8.8.8", "1.1.1.1", "9.9.9.9"})
        assert url.host == "1.1.1.1"  # sorted()[0] — stable across processes

    def test_nondefault_port_included_in_host_header(self):
        _, headers, _ = _pin_request("https://x.com:8443/p", "x.com", {"1.2.3.4"})
        assert headers["Host"] == "x.com:8443"

    def test_default_port_omitted_from_host_header(self):
        _, headers, _ = _pin_request("https://x.com/p", "x.com", {"1.2.3.4"})
        assert headers["Host"] == "x.com"

    def test_ipv6_validated_ip_pinned(self):
        url, _, _ = _pin_request("https://x.com/", "x.com", {"2606:4700::1111"})
        assert url.host == "2606:4700::1111"

    def test_ipv6_source_host_header_bracketed_with_port(self):
        # droid #282 LOW: an IPv6-literal source URL with a non-default port must
        # produce a well-formed bracketed Host authority ([ipv6]:port), not ipv6:port.
        _, headers, _ = _pin_request(
            "https://[2606:4700::1111]:8443/p", "2606:4700::1111", {"1.2.3.4"}
        )
        assert headers["Host"] == "[2606:4700::1111]:8443"

    def test_ipv6_source_host_header_bracketed_default_port(self):
        _, headers, _ = _pin_request("https://[2606:4700::1111]/p", "2606:4700::1111", {"1.2.3.4"})
        assert headers["Host"] == "[2606:4700::1111]"

    def test_sni_hostname_is_str_type(self):
        # httpcore passes sni_hostname to ssl server_hostname, which requires str.
        _, _, ext = _pin_request("https://x.com/", "x.com", {"1.2.3.4"})
        assert isinstance(ext["sni_hostname"], str)


# ---------------------------------------------------------------------------
# Integration: fetch_url wires the pin into the outgoing request
# ---------------------------------------------------------------------------


class TestFetchUrlPinsConnection:
    @pytest.mark.asyncio
    async def test_fetch_url_pins_validated_ip_to_connection(self):
        captured: dict = {}

        def handler(req: httpx.Request) -> httpx.Response:
            captured["host"] = req.url.host
            captured["sni"] = req.extensions.get("sni_hostname")
            captured["host_header"] = req.headers.get("host")
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"ok")

        transport = httpx.MockTransport(handler)
        with patch(
            "src.url_fetcher.socket.getaddrinfo",
            side_effect=_fake_getaddrinfo_for("93.184.216.34"),
        ):
            result = await fetch_url(
                "https://example.com/x.txt", _transport=transport, _skip_dns=False
            )

        assert captured["host"] == "93.184.216.34"  # connection targets the validated IP
        assert captured["sni"] == "example.com"  # TLS verifies against the hostname
        assert captured["host_header"] == "example.com"  # Host header preserved
        assert result == "ok"
