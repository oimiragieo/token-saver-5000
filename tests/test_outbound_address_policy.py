"""Public-only address policy exercised through the real engine fetch adapter."""

import socket

import httpx
import pytest

from src.url_fetcher import URLFetchError, fetch_url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "100.64.0.1",
        "100.127.255.254",
        "::ffff:100.64.0.1",
        "4000::1",
        "224.0.0.1",
        "ff02::1",
        "127.0.0.1",
        "169.254.169.254",
        "0.0.0.0",
        "10.0.0.1",
        "::1",
        "2001:db8::1",
        "fec0::",
        "fec0::1",
        "feff:ffff:ffff:ffff:ffff:ffff:ffff:ffff",
    ],
)
@pytest.mark.parametrize("second_pass", [False, True])
async def test_fetch_denies_every_nonpublic_dns_answer(monkeypatch, address, second_pass):
    calls = []
    resolutions = []

    def resolve(*args, **kwargs):
        resolutions.append(args)
        ips = ["8.8.8.8"] if second_pass and len(resolutions) == 1 else ["8.8.8.8", address]
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 443)) for ip in ips]

    def respond(request):
        calls.append(request)
        return httpx.Response(200, text="public", headers={"content-type": "text/plain"})

    monkeypatch.setattr("src.url_fetcher.socket.getaddrinfo", resolve)
    with pytest.raises(URLFetchError) as error:
        await fetch_url(
            "https://fetch.example/doc", _transport=httpx.MockTransport(respond), _skip_dns=False
        )
    assert error.value.code == "private_ip"
    assert len(resolutions) == (2 if second_pass else 1)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"])
async def test_fetch_public_control_keeps_host_sni_and_pinning(monkeypatch, address):
    requests = []
    monkeypatch.setattr(
        "src.url_fetcher.socket.getaddrinfo",
        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 443))],
    )

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text="public", headers={"content-type": "text/plain"})

    assert (
        await fetch_url(
            "https://fetch.example/doc", _transport=httpx.MockTransport(respond), _skip_dns=False
        )
        == "public"
    )
    assert len(requests) == 1
    assert requests[0].url.host == address
    assert requests[0].headers["host"] == "fetch.example"
    assert requests[0].extensions["sni_hostname"] == "fetch.example"


@pytest.mark.asyncio
async def test_site_local_only_dns_answer_never_reaches_transport(monkeypatch):
    requests = []
    monkeypatch.setattr(
        "src.url_fetcher.socket.getaddrinfo",
        lambda *a, **k: [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fec0::1", 443, 0, 0))],
    )

    def respond(request):
        requests.append(request)
        return httpx.Response(200, text="site-local", headers={"content-type": "text/plain"})

    refusal = None
    try:
        await fetch_url(
            "https://fetch.example/doc", _transport=httpx.MockTransport(respond), _skip_dns=False
        )
    except URLFetchError as exc:
        refusal = exc
    assert requests == [], "site-local resolution reached the mock HTTP transport"
    assert refusal is not None and refusal.code == "private_ip"
