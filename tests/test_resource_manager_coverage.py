from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

import src.resource_manager as rm
from src.resource_manager import ResourceLimits, ResourceManager


def test_document_limit_checks_and_register_unregister():
    manager = ResourceManager(
        ResourceLimits(max_document_size_mb=1.0, max_total_storage_mb=10.0, max_documents=1)
    )
    ok, err = manager.check_document_size("a", int(0.5 * 1024 * 1024))
    assert ok is True
    assert err is None
    manager.register_document("a", int(0.5 * 1024 * 1024))

    ok2, err2 = manager.check_document_size("b", int(0.5 * 1024 * 1024))
    assert ok2 is False
    assert "Too many documents" in err2

    ok3, err3 = manager.check_document_size("a", int(3 * 1024 * 1024))
    assert ok3 is False
    assert "Document too large" in err3

    manager.unregister_document("a")
    assert manager.document_sizes == {}


def test_check_then_register_race_over_admits_past_the_storage_limit():
    """#236 rank11: two concurrent ingests using check-then-register (the old,
    non-atomic pattern) can BOTH pass admission before either registers,
    because neither has written to document_sizes yet when the other checks.

    This is the regression the atomic `check_and_reserve_document_size`
    closes below — the same race demonstrated here as a control.
    """
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=10.0, max_documents=1000))

    # Both ingests of a 6MB document check admission (under the 10MB cap
    # individually) before either has registered — the classic TOCTOU window.
    ok_a, err_a = manager.check_document_size("doc_a", int(6 * 1024 * 1024))
    ok_b, err_b = manager.check_document_size("doc_b", int(6 * 1024 * 1024))
    assert ok_a is True and err_a is None
    assert ok_b is True and err_b is None  # <-- the race: should have been rejected

    manager.register_document("doc_a", int(6 * 1024 * 1024))
    manager.register_document("doc_b", int(6 * 1024 * 1024))

    # The limit is blown: 12MB registered against a 10MB cap.
    assert sum(manager.document_sizes.values()) > manager.limits.max_total_storage_mb


def test_check_and_reserve_document_size_closes_the_admission_race():
    """The atomic check-and-reserve method must admit the FIRST of two
    concurrent 6MB ingests against a 10MB cap, then REJECT the second —
    because the first call's reservation is visible to the second before it
    computes its own candidate total.
    """
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=10.0, max_documents=1000))

    ok_a, err_a = manager.check_and_reserve_document_size("doc_a", int(6 * 1024 * 1024))
    assert ok_a is True and err_a is None

    ok_b, err_b = manager.check_and_reserve_document_size("doc_b", int(6 * 1024 * 1024))
    assert ok_b is False
    assert "Total storage limit exceeded" in err_b

    # Only the admitted document counts against the tenant.
    assert sum(manager.document_sizes.values()) <= manager.limits.max_total_storage_mb
    assert "doc_b" not in manager.document_sizes


def test_check_and_reserve_document_size_thread_race_admits_only_one():
    """Real concurrent callers (two OS threads racing check_and_reserve) must
    never both be admitted when doing so would exceed the cap — proving the
    lock genuinely serializes the check+write, not just the single-threaded
    call order used by the test above.
    """
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=10.0, max_documents=1000))
    results: dict[str, tuple[bool, object]] = {}
    barrier = threading.Barrier(2)

    def _attempt(doc_id: str) -> None:
        barrier.wait(timeout=5)
        results[doc_id] = manager.check_and_reserve_document_size(doc_id, int(6 * 1024 * 1024))

    threads = [
        threading.Thread(target=_attempt, args=("doc_a",)),
        threading.Thread(target=_attempt, args=("doc_b",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    admitted = [doc_id for doc_id, (ok, _err) in results.items() if ok]
    assert (
        len(admitted) == 1
    ), f"exactly one of two racing 6MB ingests must be admitted, got {results}"
    assert sum(manager.document_sizes.values()) <= manager.limits.max_total_storage_mb


@pytest.mark.asyncio
async def test_check_and_reserve_document_size_async_race_admits_only_one():
    """The async wrapper (the real call site used by handlers) preserves the
    same admission guarantee under asyncio.gather concurrency.
    """
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=10.0, max_documents=1000))

    results = await asyncio.gather(
        manager.check_and_reserve_document_size_async("doc_a", int(6 * 1024 * 1024)),
        manager.check_and_reserve_document_size_async("doc_b", int(6 * 1024 * 1024)),
    )
    admitted = [ok for ok, _err in results]
    assert admitted.count(True) == 1
    assert sum(manager.document_sizes.values()) <= manager.limits.max_total_storage_mb


def test_check_and_reserve_document_size_release_on_failure_frees_the_slot():
    """Callers must release a reservation (via unregister_document) when the
    subsequent ingest fails — otherwise the atomic reserve would leak space
    for documents that never actually landed.
    """
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=10.0, max_documents=1000))

    ok, err = manager.check_and_reserve_document_size("doc_a", int(6 * 1024 * 1024))
    assert ok is True and err is None

    # Simulate the caller's ingest failing and releasing the reservation.
    manager.unregister_document("doc_a")

    # A second, equally large document must now be admittable.
    ok2, err2 = manager.check_and_reserve_document_size("doc_b", int(6 * 1024 * 1024))
    assert ok2 is True and err2 is None


def test_health_summary_stats_and_cleanup_recommendation():
    manager = ResourceManager(
        ResourceLimits(
            max_document_size_mb=5.0,
            max_total_storage_mb=10.0,
            max_documents=10,
            max_memory_mb=512.0,
            warn_threshold=0.5,
        )
    )
    manager.register_document("a", int(4 * 1024 * 1024))
    manager.register_document("b", int(3 * 1024 * 1024))

    health = manager.check_health()
    assert "metrics" in health
    assert health["metrics"]["storage_mb"] > 0

    summary = manager.get_usage_summary()
    assert "Resource Usage Summary" in summary
    assert "Status:" in summary

    stats = manager.get_stats()
    assert "largest_documents" in stats
    assert len(stats["documents"]) == 2

    suggestion = manager.suggest_cleanup()
    assert suggestion is not None
    assert "Cleanup Suggestions" in suggestion


def test_memory_health_paths_with_monkeypatched_psutil(monkeypatch):
    manager = ResourceManager(ResourceLimits(max_total_storage_mb=1.0, max_memory_mb=100.0))
    manager.register_document("a", int(2 * 1024 * 1024))

    class _Proc:
        def memory_info(self):
            return SimpleNamespace(rss=200 * 1024 * 1024)

        def memory_percent(self):
            return 50.0

    class _Psutil:
        @staticmethod
        def Process():
            return _Proc()

        @staticmethod
        def virtual_memory():
            return SimpleNamespace(available=1024 * 1024 * 100, percent=65.0)

    monkeypatch.setattr(rm, "PSUTIL_AVAILABLE", True)
    monkeypatch.setattr(rm, "psutil", _Psutil)

    stats = manager.get_memory_usage()
    assert stats["process_memory_mb"] > 0
    healthy, warning = manager.check_memory_health()
    assert healthy is False
    assert "Storage limit exceeded" in warning


@pytest.mark.asyncio
async def test_async_wrappers_delegate():
    manager = ResourceManager(ResourceLimits(max_document_size_mb=2.0, max_total_storage_mb=10.0))
    ok, err = await manager.check_document_size_async("doc", int(1024 * 1024))
    assert ok is True
    assert err is None

    await manager.register_document_async("doc", int(1024 * 1024))
    health = await manager.check_health_async()
    assert "healthy" in health

    await manager.unregister_document_async("doc")
    assert "doc" not in manager.document_sizes
