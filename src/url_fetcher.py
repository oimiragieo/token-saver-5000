"""SSRF-hardened URL fetcher for ingest_context file_url parameter.

Implements 8 mitigations against Server-Side Request Forgery (SSRF):
  1. HTTPS-ONLY — rejects any non-https scheme
  2. BLOCKED HOSTNAMES — denies metadata endpoints, *.internal TLD, bare 'metadata'
  3. PUBLIC IP ONLY — blocks RFC1918, loopback, link-local, IPv6-private after DNS resolution
     (includes IPv4-mapped IPv6 normalisation — ::ffff:127.0.0.1 collapses to 127.0.0.1)
  4. DNS REBINDING — re-resolves after pre-connect check; rejects if IP set changes,
     then PINS the validated IP into the connection (URL host -> IP + sni_hostname) so
     httpx performs no third connect-time resolution an attacker could rebind (TOCTOU close)
  5. NO REDIRECTS — follow_redirects=False
  6. SIZE CAP — 10 MB hard limit checked via Content-Length header + streaming counter
  7. TIMEOUT — connect=10s, read=30s
  8. CONTENT-TYPE ALLOWLIST — text/*, application/json, application/xml, application/yaml, application/x-yaml

Protection logic (items 2-4) is vendored from api/app/services/url_validator.py (2026-05-24,
Phase 2 Chunk 6A).  token-saver-5000 tests run with PYTHONPATH=. from inside this directory,
so importing from api/ at module-load time would fail.  Vendoring keeps this module
standalone for tests and in production (where both paths are on PYTHONPATH anyway).
"""

import ipaddress
import socket
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB

_ALLOWED_CONTENT_TYPE_PREFIXES = (
    "text/",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
)

_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

# Vendored from api/app/services/url_validator.py — _METADATA_HOSTNAMES
# Plus bare 'metadata' hostname used by some cloud environments.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "169.254.169.254",  # AWS/GCP/Azure IMDS endpoint
        "169.254.169.254.nip.io",  # nip.io wildcard bypass
        "metadata.google.internal",  # GCP metadata server
        "metadata.aws.internal",  # AWS metadata server
        "metadata",  # bare hostname (some private clouds)
    }
)


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class URLFetchError(Exception):
    """Raised when URL fetching fails due to SSRF mitigation or network error.

    Attributes:
        code: Machine-readable error code for upstream handlers.
    """

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Private helpers (vendored from api/app/services/url_validator.py)
# ---------------------------------------------------------------------------


def _normalize_ip_address(ip_str: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    """Return an ip_address object, collapsing IPv4-mapped IPv6 to pure IPv4.

    ::ffff:127.0.0.1 maps to 127.0.0.1 so is_loopback() fires correctly.
    Without this normalisation, ipaddress.ip_address('::ffff:127.0.0.1').is_loopback
    returns False (Python stdlib bug-by-design).
    """
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def _is_private_ip(addr: str) -> bool:
    """Return True if *addr* resolves to a private/reserved IP address.

    Handles IPv4-mapped IPv6 by normalising first.
    """
    try:
        ip = _normalize_ip_address(addr)
    except ValueError:
        # addr is a hostname — not expected here but treat as non-private
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
        or ip.is_multicast
    )


def _is_blocked_hostname(host: str) -> bool:
    """Return True if *host* is an explicitly blocked metadata/internal hostname.

    Blocks:
    - Any host in _BLOCKED_HOSTNAMES (exact match, case-insensitive)
    - Any host with a .internal TLD (covers Fly.io internal networking and GCP)
    """
    lower = host.lower()
    if lower in _BLOCKED_HOSTNAMES:
        return True
    # Block *.internal TLD — Fly.io, GCP, and most private-cloud internal DNS zones
    if lower.endswith(".internal") or lower == "internal":
        return True
    return False


def _resolve_host(host: str) -> set[str]:
    """DNS-resolve *host* and return the set of IP address strings.

    Raises:
        URLFetchError(code="private_ip") if DNS resolution fails.
    """
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise URLFetchError(
            f"DNS resolution failed for host '{host}': {exc}",
            code="private_ip",
        ) from exc
    return {sockaddr[0] for _family, _type, _proto, _canonname, sockaddr in results}


def _check_resolved_ips(host: str, ips: set[str]) -> None:
    """Raise URLFetchError if any IP in *ips* is private/reserved.

    Raises:
        URLFetchError(code="private_ip") if any address is private.
    """
    for ip_addr in ips:
        if _is_private_ip(ip_addr):
            raise URLFetchError(
                f"Host '{host}' resolves to a private/reserved IP address ({ip_addr}). "
                "Fetching private network resources is not allowed.",
                code="private_ip",
            )


def _resolve_and_check_host(host: str) -> set[str]:
    """DNS-resolve *host*, assert all IPs are public, return the IP set.

    Returns the resolved IP set for use in the optional DNS-rebinding check.

    Raises:
        URLFetchError(code="private_ip") if any resolved IP is private/reserved or
        DNS resolution fails.
    """
    ips = _resolve_host(host)
    _check_resolved_ips(host, ips)
    return ips


def _check_dns_rebinding(host: str, first_ips: set[str]) -> None:
    """Re-resolve *host* and raise if the IP set has changed (DNS rebinding).

    Attackers can make a hostname resolve to a public IP during the check, then
    switch the DNS record to 127.0.0.1 just before the actual connection is made.
    A second resolution with comparison catches this race.

    Raises:
        URLFetchError(code="dns_rebinding") if IPs changed between resolutions.
        URLFetchError(code="private_ip") if second resolution yields a private IP.
    """
    second_ips = _resolve_host(host)
    _check_resolved_ips(host, second_ips)
    if second_ips != first_ips:
        raise URLFetchError(
            f"DNS rebinding detected for host '{host}': IP set changed between "
            f"pre-connect check ({sorted(first_ips)}) and connection "
            f"({sorted(second_ips)}). Request blocked.",
            code="dns_rebinding",
        )


def _pin_request(
    url: str, host: str, validated_ips: set[str]
) -> tuple[httpx.URL, dict[str, str], dict[str, str]]:
    """Pin the connection to a validated IP, preserving Host + TLS SNI (mitigation 4b).

    Rewrites the URL host to a deterministic validated IP so httpx connects to the exact
    IP that passed the SSRF check. There is NO third DNS resolution at connect time, which
    closes the DNS-rebinding TOCTOU: the gap between ``_check_dns_rebinding`` (resolution #2)
    and httpx's own connect-time resolution (#3) an attacker could rebind through.

    The original hostname is preserved for:
      - the ``Host`` header (virtual-host routing), and
      - the TLS SNI + certificate hostname check, via httpx's ``sni_hostname`` request
        extension — so certificate verification still targets the real hostname, not the IP.
        A naive URL-host->IP rewrite WITHOUT ``sni_hostname`` would break TLS verification
        (cert is issued for the hostname, not the IP) or, worse, weaken it.

    ``sni_hostname`` is a ``str`` because httpcore forwards it to the stdlib ssl
    ``server_hostname`` argument, which rejects bytes.

    Returns ``(pinned_url, headers, extensions)`` for
    ``client.stream("GET", pinned_url, headers=headers, extensions=extensions)``.
    """
    pinned_ip = sorted(validated_ips)[0]  # deterministic across processes
    parsed = httpx.URL(url)
    pinned_url = parsed.copy_with(host=pinned_ip)
    port = parsed.port  # None for the default port (443/80); the int otherwise
    # Bracket an IPv6-literal host so the Host authority is well-formed (RFC 3986
    # host = IP-literal in brackets): [::1]:8443, not ::1:8443. A ':' in host
    # reliably means IPv6 — DNS hostnames never contain colons.
    header_host = f"[{host}]" if ":" in host else host
    host_header = f"{header_host}:{port}" if port else header_host
    headers = {"Host": host_header}
    extensions: dict[str, str] = {}
    if parsed.scheme == "https":
        extensions["sni_hostname"] = host
    return pinned_url, headers, extensions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_url(
    url: str,
    *,
    _transport: Optional[httpx.AsyncBaseTransport] = None,
    _skip_dns: Optional[bool] = None,
) -> str:
    """Fetch *url* with SSRF hardening and return the response body as text.

    The ``_transport`` parameter is for testing only (allows MockTransport injection).

    Args:
        url: The URL to fetch. Must use https scheme.
        _transport: Optional httpx transport override (test hook only). When set, DNS
            resolution + IP pinning are skipped by default (the mock has no real socket).
        _skip_dns: Test hook to force-enable or force-disable DNS resolution + IP pinning
            independent of ``_transport``. When ``None`` (default) it follows
            ``_transport is not None``. Pass ``False`` alongside a capturing ``_transport``
            to exercise the pin path (resolve -> validate -> rewrite URL to IP + sni_hostname)
            without a real network call.

    Returns:
        The response body decoded as UTF-8 text (or best-effort charset decoding).

    Raises:
        URLFetchError: If any SSRF mitigation is triggered or a network error occurs.
    """
    # --- Mitigation 1: HTTPS-ONLY ---
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise URLFetchError(
            f"Only https:// URLs are allowed. Got scheme '{parsed.scheme}'.",
            code="scheme_not_allowed",
        )

    host = parsed.hostname or ""
    if not host:
        raise URLFetchError(
            "URL has no resolvable hostname.",
            code="private_ip",
        )

    # --- Mitigation 2: BLOCKED HOSTNAMES (pre-DNS, hostname-level check) ---
    if _is_blocked_hostname(host):
        raise URLFetchError(
            f"Host '{host}' is a blocked metadata or internal-network hostname. "
            "Fetching from this host is not allowed.",
            code="blocked_hostname",
        )

    # --- Mitigations 3 + 4: PUBLIC IP ONLY + DNS REBINDING + PIN ---
    # Skip DNS resolution + pinning when a mock transport is injected (unit tests),
    # unless a test explicitly forces it via _skip_dns=False.
    skip_dns = _skip_dns if _skip_dns is not None else (_transport is not None)
    first_ips: Optional[set[str]] = None
    if not skip_dns:
        first_ips = _resolve_and_check_host(host)

    # --- Build client ---
    client_kwargs: dict = {
        "follow_redirects": False,  # Mitigation 5: NO REDIRECTS
        "timeout": _TIMEOUT,  # Mitigation 7: TIMEOUT
    }
    if _transport is not None:
        client_kwargs["transport"] = _transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        # --- Mitigation 4: DNS REBINDING — second resolution just before connect ---
        # --- Mitigation 4b: PIN the validated IP into the request so httpx does not
        #     perform its own (rebind-able) third resolution at connect time. The URL
        #     host becomes the validated IP; the original hostname is preserved for the
        #     Host header and the TLS SNI/certificate check (sni_hostname extension).
        request_url: httpx.URL | str = url
        request_headers: Optional[dict[str, str]] = None
        request_extensions: Optional[dict[str, str]] = None
        if not skip_dns and first_ips is not None:
            _check_dns_rebinding(host, first_ips)
            request_url, request_headers, request_extensions = _pin_request(url, host, first_ips)

        try:
            # Mitigation 6 depends on STREAMING, not a buffered .get(): a malicious
            # server that omits Content-Length (or lies) could stream unbounded bytes
            # into memory before any post-download len() check runs — an OOM/DoS on an
            # attacker-controlled file_url. client.stream() gives us the headers up front
            # (redirect/status/content-type/content-length checks below) while the body is
            # consumed incrementally so we can abort AT the cap, capping memory at _MAX_BYTES.
            async with client.stream(
                "GET", request_url, headers=request_headers, extensions=request_extensions
            ) as response:
                # --- Mitigation 5: NO REDIRECTS — reject 3xx ---
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    raise URLFetchError(
                        f"Redirects are not followed. Got HTTP {response.status_code} from '{url}'.",
                        code="redirect_blocked",
                    )

                # Surface non-2xx as generic errors (not an SSRF mitigation, but good hygiene)
                if response.status_code < 200 or response.status_code >= 300:
                    raise URLFetchError(
                        f"HTTP {response.status_code} from '{url}'.",
                        code="private_ip",
                    )

                # --- Mitigation 8: CONTENT-TYPE ALLOWLIST (header available pre-body) ---
                content_type = response.headers.get("content-type", "")
                # Strip parameters (e.g. "text/html; charset=utf-8" → "text/html")
                ct_base = content_type.split(";")[0].strip().lower()
                if not any(ct_base.startswith(p) for p in _ALLOWED_CONTENT_TYPE_PREFIXES):
                    raise URLFetchError(
                        f"Content-Type '{ct_base}' is not in the allowlist "
                        f"({', '.join(_ALLOWED_CONTENT_TYPE_PREFIXES)}). "
                        "Only text and structured-data content types are accepted.",
                        code="content_type_not_allowed",
                    )

                # --- Mitigation 6a: SIZE CAP — Content-Length header pre-check (fast reject) ---
                content_length_header = response.headers.get("content-length")
                if content_length_header is not None:
                    try:
                        declared_size = int(content_length_header)
                        if declared_size > _MAX_BYTES:
                            raise URLFetchError(
                                f"Response Content-Length {declared_size:,} bytes exceeds "
                                f"the {_MAX_BYTES // (1024 * 1024)} MB limit.",
                                code="too_large",
                            )
                    except ValueError:
                        pass  # Malformed header — the streaming counter below still bounds us

                # --- Mitigation 6b: SIZE CAP — streaming byte counter (bounds memory) ---
                # Accumulate chunks and abort the moment we cross the cap, so a lying or
                # absent Content-Length can never buffer more than _MAX_BYTES into memory.
                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_BYTES:
                        raise URLFetchError(
                            f"Response body exceeds the {_MAX_BYTES // (1024 * 1024)} MB "
                            "limit (streaming abort).",
                            code="too_large",
                        )
                    chunks.append(chunk)
                body_bytes = b"".join(chunks)
                # Charset comes from the Content-Type header (available without the body).
                encoding = response.encoding or "utf-8"
        except httpx.TimeoutException as exc:
            raise URLFetchError(
                f"Request to '{url}' timed out: {exc}",
                code="timeout",
            ) from exc
        except httpx.RequestError as exc:
            raise URLFetchError(
                f"Network error fetching '{url}': {exc}",
                code="private_ip",
            ) from exc

    # Decode: prefer charset from Content-Type, fall back to UTF-8 replacement.
    try:
        return body_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return body_bytes.decode("utf-8", errors="replace")
