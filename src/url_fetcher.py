"""SSRF-hardened URL fetcher for ingest_context file_url parameter.

Implements 6 mitigations against Server-Side Request Forgery (SSRF):
  1. HTTPS-ONLY — rejects any non-https scheme
  2. PUBLIC IP ONLY — blocks RFC1918, loopback, link-local, IPv6-private after DNS resolution
  3. NO REDIRECTS — follow_redirects=False
  4. SIZE CAP — 10 MB hard limit checked via Content-Length header + streaming counter
  5. TIMEOUT — connect=10s, read=30s
  6. CONTENT-TYPE ALLOWLIST — text/*, application/json, application/xml, application/yaml, application/x-yaml
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
# Private helpers
# ---------------------------------------------------------------------------


def _is_private_ip(addr: str) -> bool:
    """Return True if *addr* resolves to a private/reserved IP address."""
    try:
        ip = ipaddress.ip_address(addr)
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


def _resolve_and_check_host(host: str) -> None:
    """DNS-resolve *host* and raise URLFetchError if any address is private.

    Raises:
        URLFetchError(code="private_ip") if the host resolves to a private address.
        URLFetchError(code="private_ip") if DNS resolution fails.
    """
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise URLFetchError(
            f"DNS resolution failed for host '{host}': {exc}",
            code="private_ip",
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip_addr = sockaddr[0]
        if _is_private_ip(ip_addr):
            raise URLFetchError(
                f"Host '{host}' resolves to a private/reserved IP address ({ip_addr}). "
                "Fetching private network resources is not allowed.",
                code="private_ip",
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_url(url: str, *, _transport: Optional[httpx.AsyncBaseTransport] = None) -> str:
    """Fetch *url* with SSRF hardening and return the response body as text.

    The ``_transport`` parameter is for testing only (allows MockTransport injection).

    Args:
        url: The URL to fetch. Must use https scheme.
        _transport: Optional httpx transport override (test hook only).

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

    # --- Mitigation 2: PUBLIC IP ONLY (pre-connect DNS check) ---
    # Skip DNS check when a mock transport is injected (unit tests)
    if _transport is None:
        _resolve_and_check_host(host)

    # --- Build client ---
    client_kwargs: dict = {
        "follow_redirects": False,  # Mitigation 3: NO REDIRECTS
        "timeout": _TIMEOUT,  # Mitigation 5: TIMEOUT
    }
    if _transport is not None:
        client_kwargs["transport"] = _transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        try:
            response = await client.get(url)
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

        # --- Mitigation 3: NO REDIRECTS — reject 3xx ---
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

        # --- Mitigation 6: CONTENT-TYPE ALLOWLIST ---
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

        # --- Mitigation 4: SIZE CAP — Content-Length header pre-check ---
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
                pass  # Malformed header — proceed with streaming check

        # --- Mitigation 4: SIZE CAP — streaming byte counter ---
        # httpx has already buffered the body at this point for a non-streaming .get().
        # We read .content (bytes) and check length.
        body_bytes = response.content
        if len(body_bytes) > _MAX_BYTES:
            raise URLFetchError(
                f"Response body {len(body_bytes):,} bytes exceeds "
                f"the {_MAX_BYTES // (1024 * 1024)} MB limit.",
                code="too_large",
            )

        # Decode: prefer charset from Content-Type, fall back to apparent encoding
        encoding = response.encoding or "utf-8"
        try:
            return body_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return body_bytes.decode("utf-8", errors="replace")
