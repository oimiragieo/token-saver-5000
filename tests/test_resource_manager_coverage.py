from __future__ import annotations

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
