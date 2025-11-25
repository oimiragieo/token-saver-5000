#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Profiling Script for Token Saver 5000

Uses Python's built-in tracemalloc (recommended for production as of 2025)
to identify memory leaks and growth patterns.

**Research-Backed Approach:**
Based on 2025 best practices from Python community:
- tracemalloc for snapshot comparisons
- Focus on gradual leaks in long-running operations
- Track object growth across operations
- Identify retention issues in singletons

**What We're Looking For:**
1. Memory growth after document ingestion cycles
2. Singleton retention issues (EmbeddingManager, ACE contexts)
3. Graph/chunk storage leaks
4. File sync metadata accumulation

Usage:
    python scripts/profile_memory.py

    # With specific test:
    python scripts/profile_memory.py --test ingestion
    python scripts/profile_memory.py --test singleton
    python scripts/profile_memory.py --test full

Version: 0.4.1 (Week 3 Day 11)
"""

import argparse
import gc
import io
import sys
import tracemalloc
from pathlib import Path
from typing import List, Tuple

# Configure stdout for UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, ".")

from src.semantic_compressor import SemanticCompressor
from src.embeddings import EmbeddingManager
from src.ace_framework import ACEFramework
from src.file_sync_manager import FileSyncManager
from src.version_manager import VersionManager


def format_memory(size_bytes: int) -> str:
    """Format bytes as human-readable string"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_snapshot_diff(
    snapshot1: tracemalloc.Snapshot,
    snapshot2: tracemalloc.Snapshot,
    title: str,
    top_n: int = 10,
) -> None:
    """
    Print difference between two memory snapshots.

    Shows top N memory allocations that grew between snapshots.
    """
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

    top_stats = snapshot2.compare_to(snapshot1, "lineno")

    print(f"Top {top_n} memory allocations (growth):\n")
    for index, stat in enumerate(top_stats[:top_n], 1):
        print(f"{index:2d}. {stat.traceback.format()[0]}")
        print(f"    Size: {format_memory(stat.size)} ({stat.size_diff:+d} bytes)")
        print(f"    Count: {stat.count} objects ({stat.count_diff:+d} diff)")
        print()


def test_singleton_memory_leak() -> Tuple[int, int]:
    """
    Test for memory leaks in singleton EmbeddingManager.

    Potential Issue:
    - EmbeddingManager._model_cache might retain models indefinitely
    - Multiple get_text_embedder() calls should reuse cached model

    Expected Behavior:
    - First call loads model (~80MB)
    - Subsequent calls use cache (minimal growth)

    Returns:
        (initial_memory_kb, final_memory_kb)
    """
    print("\n" + "=" * 70)
    print("  TEST 1: Singleton EmbeddingManager Memory Retention")
    print("=" * 70)

    tracemalloc.start()
    gc.collect()
    snapshot1 = tracemalloc.take_snapshot()

    # First load - should allocate model
    print("\n1. Loading embedding model (first time)...")
    manager1 = EmbeddingManager()
    model1 = manager1.get_text_embedder()
    snapshot2 = tracemalloc.take_snapshot()

    # Get memory stats
    stats = manager1.get_cache_stats()
    print(f"   Models cached: {stats['num_models']}")
    print(f"   Estimated memory: {stats['estimated_memory_mb']}MB")

    # Second load - should reuse cache
    print("\n2. Getting same model (should use cache)...")
    manager2 = EmbeddingManager()
    model2 = manager2.get_text_embedder()
    snapshot3 = tracemalloc.take_snapshot()

    # Verify singleton behavior
    assert manager1 is manager2, "EmbeddingManager not a singleton!"
    assert model1 is model2, "Model not cached!"
    print("   ✅ Singleton verified - same instance returned")

    # Check for growth
    print_snapshot_diff(snapshot2, snapshot3, "Memory Growth After Cache Reuse", top_n=5)

    # Third load different model - should add to cache
    print("\n3. Loading different model (code embedder)...")
    model3 = manager1.get_code_embedder()
    snapshot4 = tracemalloc.take_snapshot()

    stats_after = manager1.get_cache_stats()
    print(f"   Models cached: {stats_after['num_models']}")
    print(f"   Estimated memory: {stats_after['estimated_memory_mb']}MB")

    print_snapshot_diff(snapshot3, snapshot4, "Memory Growth After Second Model", top_n=5)

    # Calculate totals
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n📊 Total Memory Usage:")
    print(f"   Current: {format_memory(current)}")
    print(f"   Peak: {format_memory(peak)}")

    tracemalloc.stop()

    return current // 1024, peak // 1024  # Return KB


def delete_document_from_compressor(compressor: SemanticCompressor, file_id: str) -> None:
    """
    Helper to delete document from compressor (mimics MCP handler logic).
    """
    # Remove chunks
    chunks_to_delete = [k for k in compressor.chunks.keys() if k.startswith(file_id)]
    for chunk_id in chunks_to_delete:
        del compressor.chunks[chunk_id]

    # Remove graph
    if file_id in compressor.graphs:
        del compressor.graphs[file_id]

    # Remove metadata
    if file_id in compressor.file_metadata:
        del compressor.file_metadata[file_id]


def test_document_ingestion_leak() -> Tuple[int, int]:
    """
    Test for memory leaks during repeated document ingestion.

    Potential Issues:
    - Graph nodes/edges not cleaned up
    - Chunk dictionaries growing unbounded
    - Embeddings retained after deletion

    Expected Behavior:
    - Memory grows with each document
    - delete_document() should free memory
    - Multiple ingest/delete cycles should stabilize

    Returns:
        (baseline_memory_kb, after_cycles_memory_kb)
    """
    print("\n" + "=" * 70)
    print("  TEST 2: Document Ingestion/Deletion Memory Cycles")
    print("=" * 70)

    tracemalloc.start()
    gc.collect()

    compressor = SemanticCompressor()
    snapshot_baseline = tracemalloc.take_snapshot()

    # Test document
    test_doc = "Quantum computing uses qubits for parallel computation. " * 50

    # Cycle 1: Ingest
    print("\n1. Ingesting document (cycle 1)...")
    compressor.ingest_file(test_doc, "test_doc_1")
    gc.collect()
    snapshot_after_ingest1 = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_baseline, snapshot_after_ingest1, "Memory After First Ingestion", top_n=5
    )

    # Cycle 1: Delete
    print("\n2. Deleting document (cycle 1)...")
    delete_document_from_compressor(compressor, "test_doc_1")
    gc.collect()
    snapshot_after_delete1 = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_ingest1,
        snapshot_after_delete1,
        "Memory After First Deletion (should decrease)",
        top_n=5,
    )

    # Cycle 2-5: Repeat to check for accumulation
    print("\n3. Running 4 more ingest/delete cycles...")
    for i in range(2, 6):
        compressor.ingest_file(test_doc, f"test_doc_{i}")
        delete_document_from_compressor(compressor, f"test_doc_{i}")
        gc.collect()

    snapshot_after_cycles = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_delete1,
        snapshot_after_cycles,
        "Memory After 4 More Cycles (should be stable)",
        top_n=5,
    )

    # Final check
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n📊 Total Memory Usage:")
    print(f"   Baseline: {format_memory(snapshot_baseline.statistics('lineno')[0].size)}")
    print(f"   Current: {format_memory(current)}")
    print(f"   Peak: {format_memory(peak)}")

    tracemalloc.stop()

    return current // 1024, peak // 1024


def test_ace_context_persistence() -> Tuple[int, int]:
    """
    Test for memory leaks in ACE framework context storage.

    Potential Issues:
    - ace_contexts dict growing unbounded
    - Embeddings not freed when contexts deleted
    - Bullet history accumulation

    Expected Behavior:
    - Each context has bounded memory
    - Old contexts should be cleanable

    Returns:
        (baseline_memory_kb, after_operations_memory_kb)
    """
    print("\n" + "=" * 70)
    print("  TEST 3: ACE Framework Context Persistence")
    print("=" * 70)

    tracemalloc.start()
    gc.collect()

    ace = ACEFramework()
    ace_contexts = {}  # Simulated context storage

    snapshot_baseline = tracemalloc.take_snapshot()

    # Create first context
    print("\n1. Creating first ACE context...")
    from src.ace_framework import ACEContext, ACEBullet, BulletType

    context1 = ACEContext(context_id="test_context_1")

    # Add bullets with proper ACEBullet objects (v0.4.1 fix)
    print("   Adding 10 bullets to context...")
    for i in range(10):
        text = f"Principle {i}: Follow best practices for code quality"
        # Generate embedding for this bullet
        embedding = ace.text_model.encode(text)

        # Create proper ACEBullet object
        bullet = ACEBullet(
            text=text,
            bullet_type=BulletType.PRINCIPLE,
            embedding=embedding,
            confidence=0.8,
            source="test",
        )

        # Use correct API: context.add_bullet() not ace.grow_context()
        context1.add_bullet(bullet, delta_description=f"Test bullet {i}")

    ace_contexts["test_context_1"] = context1

    gc.collect()
    snapshot_after_context1 = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_baseline,
        snapshot_after_context1,
        "Memory After First Context (10 bullets)",
        top_n=5,
    )

    # Create 9 more contexts
    print("\n2. Creating 9 more ACE contexts...")
    for ctx_idx in range(2, 11):
        context = ACEContext(context_id=f"test_context_{ctx_idx}")

        # Add 10 bullets to each context
        for i in range(10):
            text = f"Principle {i}: Follow best practices for code quality"
            embedding = ace.text_model.encode(text)

            bullet = ACEBullet(
                text=text,
                bullet_type=BulletType.PRINCIPLE,
                embedding=embedding,
                confidence=0.8,
                source="test",
            )

            context.add_bullet(bullet, delta_description=f"Test bullet {i}")

        ace_contexts[f"test_context_{ctx_idx}"] = context

    gc.collect()
    snapshot_after_10_contexts = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_context1,
        snapshot_after_10_contexts,
        "Memory After 10 Total Contexts (100 bullets)",
        top_n=5,
    )

    # Delete half the contexts
    print("\n3. Deleting 5 contexts (50 bullets)...")
    for i in range(6, 11):
        del ace_contexts[f"test_context_{i}"]

    gc.collect()
    snapshot_after_deletion = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_10_contexts,
        snapshot_after_deletion,
        "Memory After Deleting 5 Contexts (should decrease)",
        top_n=5,
    )

    # Final check
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n📊 Total Memory Usage:")
    print(f"   Baseline: {format_memory(snapshot_baseline.statistics('lineno')[0].size)}")
    print(f"   Current: {format_memory(current)}")
    print(f"   Peak: {format_memory(peak)}")

    tracemalloc.stop()

    return current // 1024, peak // 1024


def test_file_sync_metadata() -> Tuple[int, int]:
    """
    Test for memory leaks in file sync metadata tracking.

    Potential Issues:
    - file_metadata dict growing unbounded
    - File paths and checksums retained indefinitely
    - Thread lock overhead

    Expected Behavior:
    - Memory grows linearly with number of tracked files
    - Metadata cleanup should free resources

    Returns:
        (baseline_memory_kb, after_operations_memory_kb)
    """
    print("\n" + "=" * 70)
    print("  TEST 4: File Sync Metadata Memory")
    print("=" * 70)

    tracemalloc.start()
    gc.collect()

    sync_manager = FileSyncManager()
    snapshot_baseline = tracemalloc.take_snapshot()

    # Register first batch of files
    print("\n1. Registering 50 files...")
    test_content = "Quantum computing test document. " * 100  # ~350 tokens

    for i in range(50):
        doc_id = f"test_doc_{i}"
        file_path = f"/fake/path/document_{i}.txt"
        sync_manager.register_file(doc_id, file_path, test_content)

    gc.collect()
    snapshot_after_50_files = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_baseline, snapshot_after_50_files, "Memory After Registering 50 Files", top_n=5
    )

    # Register another 50 files
    print("\n2. Registering 50 more files (100 total)...")
    for i in range(50, 100):
        doc_id = f"test_doc_{i}"
        file_path = f"/fake/path/document_{i}.txt"
        sync_manager.register_file(doc_id, file_path, test_content)

    gc.collect()
    snapshot_after_100_files = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_50_files,
        snapshot_after_100_files,
        "Memory After 100 Total Files (should be linear growth)",
        top_n=5,
    )

    # Remove half the metadata
    print("\n3. Removing 50 file metadata entries...")
    for i in range(50):
        doc_id = f"test_doc_{i}"
        if doc_id in sync_manager.file_metadata:
            del sync_manager.file_metadata[doc_id]

    gc.collect()
    snapshot_after_cleanup = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_100_files,
        snapshot_after_cleanup,
        "Memory After Removing 50 Entries (should decrease)",
        top_n=5,
    )

    # Final check
    current, peak = tracemalloc.get_traced_memory()
    print(f"\n📊 Total Memory Usage:")
    print(f"   Baseline: {format_memory(snapshot_baseline.statistics('lineno')[0].size)}")
    print(f"   Current: {format_memory(current)} ({len(sync_manager.file_metadata)} files tracked)")
    print(f"   Peak: {format_memory(peak)}")

    tracemalloc.stop()

    return current // 1024, peak // 1024


def test_version_history_accumulation() -> Tuple[int, int]:
    """
    Test for memory leaks in version history storage.

    Potential Issues:
    - versions dict growing unbounded
    - Full document content stored for each version
    - No pruning of old versions

    Expected Behavior:
    - Memory grows with number of versions
    - Each version stores full content (expected overhead)
    - Version pruning should free memory

    Returns:
        (baseline_memory_kb, after_operations_memory_kb)
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Version History Memory Accumulation")
    print("=" * 70)

    tracemalloc.start()
    gc.collect()

    version_manager = VersionManager(storage_dir=".test_versions")
    snapshot_baseline = tracemalloc.take_snapshot()

    # Create first document with 10 versions
    print("\n1. Creating document with 10 versions...")
    doc_id = "test_doc_versioned"
    base_content = "Version content base. " * 50  # ~100 tokens

    for version_num in range(1, 11):
        content = base_content + f"Version {version_num} changes. " * 10
        checksum = f"checksum_{version_num}"
        version_manager.add_version(
            doc_id=doc_id,
            content=content,
            checksum=checksum,
            file_path=f"/fake/path/doc_v{version_num}.txt",
            metadata={"version": version_num},
            compression_stats={"original_tokens": 150, "compressed_tokens": 20},
        )

    gc.collect()
    snapshot_after_10_versions = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_baseline,
        snapshot_after_10_versions,
        "Memory After 10 Versions of Same Document",
        top_n=5,
    )

    # Add 40 more versions (50 total)
    print("\n2. Adding 40 more versions (50 total)...")
    for version_num in range(11, 51):
        content = base_content + f"Version {version_num} changes. " * 10
        checksum = f"checksum_{version_num}"
        version_manager.add_version(
            doc_id=doc_id,
            content=content,
            checksum=checksum,
            file_path=f"/fake/path/doc_v{version_num}.txt",
            metadata={"version": version_num},
            compression_stats={"original_tokens": 150, "compressed_tokens": 20},
        )

    gc.collect()
    snapshot_after_50_versions = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_10_versions,
        snapshot_after_50_versions,
        "Memory After 50 Total Versions (should grow linearly)",
        top_n=5,
    )

    # Prune old versions (keep only last 10)
    print("\n3. Pruning to last 10 versions...")
    if doc_id in version_manager.versions:
        version_manager.versions[doc_id] = version_manager.versions[doc_id][-10:]

    gc.collect()
    snapshot_after_pruning = tracemalloc.take_snapshot()

    print_snapshot_diff(
        snapshot_after_50_versions,
        snapshot_after_pruning,
        "Memory After Pruning to 10 Versions (should decrease significantly)",
        top_n=5,
    )

    # Final check
    current, peak = tracemalloc.get_traced_memory()
    version_count = len(version_manager.versions.get(doc_id, []))
    print(f"\n📊 Total Memory Usage:")
    print(f"   Baseline: {format_memory(snapshot_baseline.statistics('lineno')[0].size)}")
    print(f"   Current: {format_memory(current)} ({version_count} versions retained)")
    print(f"   Peak: {format_memory(peak)}")

    tracemalloc.stop()

    # Cleanup test directory
    import shutil

    if Path(".test_versions").exists():
        shutil.rmtree(".test_versions")

    return current // 1024, peak // 1024


def run_full_profile() -> None:
    """
    Run all memory tests and generate comprehensive report.
    """
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE MEMORY PROFILING REPORT")
    print("  Token Saver 5000 v0.4.1")
    print("=" * 70)

    results = {}

    # Test 1: Singleton
    try:
        current, peak = test_singleton_memory_leak()
        results["singleton"] = {"status": "✅ PASS", "current": current, "peak": peak}
    except Exception as e:
        results["singleton"] = {"status": f"❌ FAIL: {e}", "current": 0, "peak": 0}

    # Test 2: Document Cycles
    try:
        current, peak = test_document_ingestion_leak()
        results["ingestion"] = {"status": "✅ PASS", "current": current, "peak": peak}
    except Exception as e:
        results["ingestion"] = {"status": f"❌ FAIL: {e}", "current": 0, "peak": 0}

    # Test 3: ACE Contexts
    try:
        current, peak = test_ace_context_persistence()
        results["ace"] = {"status": "✅ PASS", "current": current, "peak": peak}
    except Exception as e:
        results["ace"] = {"status": f"❌ FAIL: {e}", "current": 0, "peak": 0}

    # Test 4: File Sync Metadata
    try:
        current, peak = test_file_sync_metadata()
        results["file_sync"] = {"status": "✅ PASS", "current": current, "peak": peak}
    except Exception as e:
        results["file_sync"] = {"status": f"❌ FAIL: {e}", "current": 0, "peak": 0}

    # Test 5: Version History
    try:
        current, peak = test_version_history_accumulation()
        results["version_history"] = {"status": "✅ PASS", "current": current, "peak": peak}
    except Exception as e:
        results["version_history"] = {"status": f"❌ FAIL: {e}", "current": 0, "peak": 0}

    # Summary
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        print(f"\n{test_name.upper()} Test:")
        print(f"  Status: {result['status']}")
        if result["current"] > 0:
            print(f"  Current Memory: {result['current']} KB")
            print(f"  Peak Memory: {result['peak']} KB")

    print("\n" + "=" * 70)
    print("  RECOMMENDATIONS")
    print("=" * 70)
    print(
        """
1. EmbeddingManager Singleton: Monitor cache growth over time
2. Document Cycles: Ensure delete_document() fully cleans up
3. ACE Contexts: Implement context pruning for long-running servers
4. File Sync Metadata: Implement cleanup for deleted documents
5. Version History: Add automatic pruning (keep last N versions)
6. Consider adding memory limits to prevent unbounded growth

Configuration Suggestions (src/constants.py):
- MAX_ACE_CONTEXTS = 100  # Maximum ACE contexts to retain
- MAX_VERSION_HISTORY = 50  # Maximum versions per document
- MAX_FILE_SYNC_ENTRIES = 1000  # Maximum tracked files
- MAX_MEMORY_MB = 1024  # Global memory limit (1GB)

Next Steps:
- Review identified leaks
- Implement fixes in Week 3 Day 13-14
- Re-run this profiler after fixes
"""
    )


def main():
    parser = argparse.ArgumentParser(description="Profile memory usage in Token Saver 5000")
    parser.add_argument(
        "--test",
        choices=["singleton", "ingestion", "ace", "file_sync", "version_history", "full"],
        default="full",
        help="Which test to run (default: full)",
    )

    args = parser.parse_args()

    if args.test == "singleton":
        test_singleton_memory_leak()
    elif args.test == "ingestion":
        test_document_ingestion_leak()
    elif args.test == "ace":
        test_ace_context_persistence()
    elif args.test == "file_sync":
        test_file_sync_metadata()
    elif args.test == "version_history":
        test_version_history_accumulation()
    else:
        run_full_profile()


if __name__ == "__main__":
    main()
