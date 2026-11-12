"""Overall fetch deadline, cancellation and bounded daemon-DNS lifecycle regressions."""

import asyncio
import gc
import subprocess
import sys
import threading
import weakref
from types import SimpleNamespace

import httpx
import pytest

import src.url_fetcher as uf


async def _until(predicate):
    while not predicate():
        await asyncio.sleep(0)


@pytest.fixture
def lane(monkeypatch):
    slots, release, workers, failures = threading.BoundedSemaphore(4), threading.Event(), [], []
    original_thread = threading.Thread

    def thread(*args, **kwargs):
        worker = original_thread(*args, **kwargs)
        workers.append(worker)
        assert kwargs["daemon"] is True
        return worker

    monkeypatch.setattr(uf, "_DNS_SLOTS", slots)
    monkeypatch.setattr(uf, "Thread", thread)
    monkeypatch.setattr(threading, "excepthook", failures.append)
    yield slots, release, workers
    release.set()
    for worker in workers:
        if worker.ident is not None:
            worker.join(1)
            assert not worker.is_alive()
    assert all(slots.acquire(blocking=False) for _ in range(4)), "restore all four DNS slots"
    assert not slots.acquire(blocking=False)
    assert failures == [], "worker must not raise on late publication or double-release"


class _Body(httpx.AsyncByteStream):
    def __init__(self, stall=False):
        self.stall, self.closed, self.count = stall, False, 0
        self.entered = threading.Event()

    async def __aiter__(self):
        self.entered.set()
        while True:
            self.count += 1
            yield b"plain"
            if not self.stall:
                return
            await asyncio.sleep(0.001)

    async def aclose(self):
        self.closed = True


def _transport(body):
    return httpx.MockTransport(
        lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, stream=body)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["first", "second", "body"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_external_cancel_at_each_phase(monkeypatch, lane, phase, cancel):
    _, release, _ = lane
    body = _Body(stall=True)
    entered = body.entered if phase == "body" else threading.Event()

    def block(*args):
        entered.set()
        release.wait()
        return {"8.8.8.8"}

    monkeypatch.setattr(uf, "_TOTAL_TIMEOUT", 30.0 if cancel else 0.05)
    monkeypatch.setattr(
        uf, "_resolve_and_check_host", block if phase == "first" else lambda h: {"8.8.8.8"}
    )
    monkeypatch.setattr(uf, "_check_dns_rebinding", block if phase == "second" else lambda *a: None)

    task = asyncio.create_task(
        uf.fetch_url("https://example.com/x", _transport=_transport(body), _skip_dns=False)
    )
    await _until(entered.is_set)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else uf.URLFetchError) as error:
        await task
    if not cancel:
        assert error.value.code == "timeout"
    if phase == "body":
        assert body.closed
        assert body.count >= (1 if cancel else 2), "deadline must stop continuing partial reads"


@pytest.mark.asyncio
async def test_plain_success_closes_body():
    body = _Body()
    assert await uf.fetch_url("https://example.com/x", _transport=_transport(body)) == "plain"
    assert body.closed


@pytest.mark.asyncio
async def test_public_deadline_stops_finite_drip(monkeypatch):
    monkeypatch.setattr(uf, "_TOTAL_TIMEOUT", 0.01, raising=False)

    class FiniteBody(_Body):
        async def __aiter__(self):
            for _ in range(10):
                self.count += 1
                yield b"plain"
                await asyncio.sleep(0.01)

    body = FiniteBody()
    with pytest.raises(uf.URLFetchError) as error:
        await uf.fetch_url("https://example.com/finite", _transport=_transport(body))
    assert error.value.code == "timeout"
    assert 1 <= body.count < 10 and body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_abandoned_dns_holds_capacity_but_not_default_executor(monkeypatch, lane, cancel):
    slots, release, _ = lane
    entered = []

    def block(host):
        entered.append(host)
        release.wait()
        return {"8.8.8.8"}

    monkeypatch.setattr(uf, "_resolve_and_check_host", block)
    monkeypatch.setattr(uf, "_TOTAL_TIMEOUT", 30.0 if cancel else 0.05)
    tasks = [asyncio.create_task(uf.fetch_url("https://example.com/x")) for _ in range(4)]
    await _until(lambda: len(entered) == 4)
    if cancel:
        for task in tasks:
            task.cancel()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    expected = asyncio.CancelledError if cancel else uf.URLFetchError
    assert all(isinstance(result, expected) for result in results)
    if not cancel:
        assert all(result.code == "timeout" for result in results)
    assert not slots.acquire(blocking=False), "caller completion must not free blocked DNS slots"
    with pytest.raises(uf.URLFetchError) as error:
        await uf._run_dns(block, "fifth")
    assert error.value.code == "timeout" and len(entered) == 4
    assert await asyncio.to_thread(lambda: "executor available") == "executor available"


@pytest.mark.asyncio
async def test_worker_exception_and_start_failure_restore_slots(monkeypatch, lane):
    error = uf.URLFetchError("DNS failed", code="private_ip")

    def fail():
        raise error

    with pytest.raises(uf.URLFetchError) as caught:
        await uf._run_dns(fail)
    assert caught.value is error

    def start():
        raise RuntimeError("start failed")

    monkeypatch.setattr(uf, "Thread", lambda **kw: SimpleNamespace(start=start))
    with pytest.raises(RuntimeError, match="start failed"):
        await uf._run_dns(lambda: None)


@pytest.mark.asyncio
@pytest.mark.parametrize("before_queue", [False, True])
async def test_cancelled_future_ignores_queued_publication(monkeypatch, lane, before_queue):
    slots, release, workers = lane
    entered, queued = threading.Event(), []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop, "call_soon_threadsafe", lambda callback, *args: queued.append((callback, args))
    )

    def resolve():
        entered.set()
        release.wait()
        return {"8.8.8.8"}

    task = asyncio.create_task(uf._run_dns(resolve))
    await _until(entered.is_set)
    if before_queue:
        task.cancel()
    release.set()
    await _until(lambda: bool(queued))
    if not before_queue:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    workers[0].join(1)
    assert not workers[0].is_alive()
    assert all(
        slots.acquire(blocking=False) for _ in range(4)
    ), "release cannot await callback delivery"
    for _ in range(4):
        slots.release()
    callback, args = queued[0]
    assert args == ({"8.8.8.8"}, None), "publication captures outcome as arguments"
    callback(*args)  # Delivery after future cancellation must perform no invalid-state write.


def test_closed_origin_loop_releases_slot(lane):
    _, release, _ = lane
    entered = threading.Event()
    loop = asyncio.new_event_loop()

    def resolve():
        entered.set()
        release.wait()

    async def cancel_observer():
        task = asyncio.create_task(uf._run_dns(resolve))
        await _until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        loop.run_until_complete(cancel_observer())
    finally:
        loop.close()
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_completed_child_external_cancel_race_keeps_cancel(monkeypatch, fail):
    loop, warnings, child_refs = asyncio.get_running_loop(), [], []
    original_handler = loop.get_exception_handler()
    owner = None
    loop.set_exception_handler(lambda _, context: warnings.append(context))

    async def completed(*args, **kwargs):
        child_refs.append(weakref.ref(asyncio.current_task()))
        owner.cancel()
        if fail:
            raise RuntimeError("child completed with error")
        return "must not replace cancellation"

    monkeypatch.setattr(uf, "_fetch_url", completed)
    try:
        owner = asyncio.create_task(uf.fetch_url("https://example.com/x"))
        with pytest.raises(asyncio.CancelledError):
            await owner
        owner = None
        await asyncio.sleep(0)
        gc.collect()
        assert child_refs[0]() is None, "prove child was collected before checking warning absence"
        assert warnings == []
    finally:
        loop.set_exception_handler(original_handler)


def test_permanently_blocked_dns_child_exits_without_executor_join():
    script = """import asyncio, threading
from src import url_fetcher as uf
uf._TOTAL_TIMEOUT = 0.01
def resolve(host):
    print('DNS_ENTERED', flush=True)
    threading.Event().wait()
uf._resolve_and_check_host = resolve
try:
    asyncio.run(uf.fetch_url('https://example.com/x'))
except uf.URLFetchError:
    print('DNS_TIMEOUT_WITH_DAEMON_EXIT', flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=5, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "DNS_ENTERED" in result.stdout and "DNS_TIMEOUT_WITH_DAEMON_EXIT" in result.stdout
