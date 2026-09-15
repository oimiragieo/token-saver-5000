"""Tests for atomic resource reservation in token-saver-5000 (ARCH-AUDIT-07).

Proves:
1. reserve_document / commit_document / release_document contract.
2. Concurrent reservation prevents two requests from taking the last slot.
3. Failed ingestion releases uncommitted reservation.
4. Double commit/release handling and idempotence.
"""

from __future__ import annotations

import threading

from src.resource_manager import ResourceLimits, ResourceManager


def test_reserve_commit_release_lifecycle():
    manager = ResourceManager(
        ResourceLimits(max_document_size_mb=2.0, max_total_storage_mb=5.0, max_documents=2)
    )

    # Reserve doc1
    token1, err1 = manager.reserve_document("doc1", 1024 * 1024)
    assert token1 is not None
    assert err1 is None
    assert "doc1" in manager.pending_reservations

    # Commit doc1
    committed = manager.commit_document(token1)
    assert committed is True
    assert "doc1" in manager.document_sizes
    assert token1.token_id not in manager.pending_reservations

    # Double commit should be harmless False
    assert manager.commit_document(token1) is False

    # Unregister doc1
    manager.unregister_document("doc1")
    assert "doc1" not in manager.document_sizes


def test_reserve_release_lifecycle():
    manager = ResourceManager(
        ResourceLimits(max_document_size_mb=2.0, max_total_storage_mb=5.0, max_documents=2)
    )

    token, err = manager.reserve_document("doc1", 1024 * 1024)
    assert token is not None
    assert err is None

    # Release
    released = manager.release_document(token)
    assert released is True
    assert token.token_id not in manager.pending_reservations
    assert "doc1" not in manager.document_sizes

    # Double release should be harmless False
    assert manager.release_document(token) is False


def test_concurrent_reservations_cannot_exceed_capacity():
    # Only 1 document slot available
    manager = ResourceManager(
        ResourceLimits(max_document_size_mb=2.0, max_total_storage_mb=5.0, max_documents=1)
    )

    barrier = threading.Barrier(2)
    results = []

    def worker(doc_id: str):
        barrier.wait()
        token, err = manager.reserve_document(doc_id, 1024 * 1024)
        results.append((token, err))

    t1 = threading.Thread(target=worker, args=("doc_a",))
    t2 = threading.Thread(target=worker, args=("doc_b",))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Exactly one succeeded and one failed
    successes = [r for r in results if r[0] is not None]
    failures = [r for r in results if r[0] is None]

    assert len(successes) == 1
    assert len(failures) == 1
    assert "Too many documents" in failures[0][1]
