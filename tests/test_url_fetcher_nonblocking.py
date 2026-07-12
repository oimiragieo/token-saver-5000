"""#219: DNS resolution in ``fetch_url`` must run OFF the event loop.

``socket.getaddrinfo`` is a synchronous, blocking call. If it runs directly on the
asyncio event loop, a slow or hostile authoritative nameserver stalls the ENTIRE
worker's event loop (not just its own request) for the resolver timeout — a handful
of concurrent slow-DNS ``ingest_context(file_url=...)`` calls would wedge the whole
Fly machine's API. The fix wraps the two resolution call sites in
``asyncio.to_thread``. This test locks the non-blocking property: it is RED if the
resolution runs on the loop and GREEN with the to_thread wrap.
"""

import asyncio
import time

import httpx
import pytest

import src.url_fetcher as uf


@pytest.mark.asyncio
async def test_dns_resolution_does_not_block_event_loop(monkeypatch):
    # Simulate a slow authoritative nameserver: the sync resolver sleeps 0.5s.
    def _slow_resolve(host):
        time.sleep(0.5)
        return {"93.184.216.34"}  # a public IP so the SSRF check passes

    monkeypatch.setattr(uf, "_resolve_and_check_host", _slow_resolve)
    monkeypatch.setattr(uf, "_check_dns_rebinding", lambda host, ips: None)
    monkeypatch.setattr(uf, "_pin_request", lambda url, host, ips: (url, None, None))

    ticks = 0

    async def _ticker():
        nonlocal ticks
        for _ in range(60):
            await asyncio.sleep(0.01)
            ticks += 1

    def _handler(request):
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="ok")

    transport = httpx.MockTransport(_handler)

    ticker = asyncio.create_task(_ticker())
    # Force the DNS path (_skip_dns=False) despite the mock transport. fetch_url
    # blocks ~0.5s in _slow_resolve; if that ran ON the loop the ticker would be
    # frozen and make ~0 progress. With to_thread it keeps ticking concurrently.
    body = await uf.fetch_url("https://example.com/x", _transport=transport, _skip_dns=False)
    assert body == "ok"
    assert ticks >= 25, f"event loop was stalled during DNS resolution (ticks={ticks})"
    ticker.cancel()
