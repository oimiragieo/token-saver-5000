#!/usr/bin/env python3
"""
Performance Benchmarking Script for Token Saver 5000

Measures and reports performance metrics across different document sizes,
compression ratios, and feature sets.

Usage:
    python benchmark.py
    python benchmark.py --quick  # Fast benchmark (fewer iterations)
    python benchmark.py --full   # Comprehensive benchmark
"""

import time
import sys
import argparse
import statistics
from typing import List, Tuple
from dataclasses import dataclass
import psutil
import os


@dataclass
class BenchmarkResult:
    """Performance benchmark results"""

    name: str
    avg_time: float
    min_time: float
    max_time: float
    std_dev: float
    avg_memory_mb: float
    throughput: float  # operations per second
    iterations: int


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""

    def __init__(self, iterations: int = 10):
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []

    def measure_time_and_memory(self, func, *args, **kwargs) -> Tuple[float, float]:
        """Measure execution time and memory usage"""
        process = psutil.Process(os.getpid())

        # Get initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Time the function
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()

        # Get final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        elapsed = end_time - start_time
        memory_used = mem_after - mem_before

        return elapsed, memory_used

    def run_benchmark(self, name: str, func, *args, **kwargs) -> BenchmarkResult:
        """Run a benchmark multiple times and collect statistics"""
        times = []
        memories = []

        print(f"\n{'='*70}")
        print(f"Benchmarking: {name}")
        print(f"{'='*70}")
        print(f"Running {self.iterations} iterations...", end="", flush=True)

        for i in range(self.iterations):
            elapsed, memory = self.measure_time_and_memory(func, *args, **kwargs)
            times.append(elapsed)
            memories.append(memory)

            if (i + 1) % (self.iterations // 10 or 1) == 0:
                print(f" {i+1}", end="", flush=True)

        print(" Done!")

        # Calculate statistics
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        avg_memory = statistics.mean(memories)
        throughput = 1.0 / avg_time if avg_time > 0 else 0

        result = BenchmarkResult(
            name=name,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            std_dev=std_dev,
            avg_memory_mb=avg_memory,
            throughput=throughput,
            iterations=self.iterations,
        )

        self.results.append(result)
        self._print_result(result)

        return result

    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result"""
        print("\n📊 Results:")
        print(f"   Avg Time:     {result.avg_time*1000:8.2f} ms")
        print(f"   Min Time:     {result.min_time*1000:8.2f} ms")
        print(f"   Max Time:     {result.max_time*1000:8.2f} ms")
        print(f"   Std Dev:      {result.std_dev*1000:8.2f} ms")
        print(f"   Throughput:   {result.throughput:8.2f} ops/sec")
        print(f"   Memory:       {result.avg_memory_mb:8.2f} MB")

    def print_summary(self):
        """Print summary of all benchmarks"""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Benchmark':<40} {'Avg Time':>12} {'Throughput':>12} {'Memory':>10}")
        print("-" * 70)

        for result in self.results:
            print(
                f"{result.name:<40} {result.avg_time*1000:10.2f} ms "
                f"{result.throughput:10.2f} op/s {result.avg_memory_mb:8.2f} MB"
            )

        print("=" * 70)


def benchmark_document_ingestion(iterations: int = 10):
    """Benchmark document ingestion with different sizes"""
    from src.semantic_compressor import SemanticCompressor

    bench = PerformanceBenchmark(iterations)

    # Small document (~100 tokens)
    small_doc = "Quantum computing uses qubits. " * 10

    # Medium document (~500 tokens)
    medium_doc = (
        """
    Quantum computing is a revolutionary technology that uses quantum mechanics.
    Unlike classical computers that use bits (0 or 1), quantum computers use qubits.
    Qubits can exist in superposition, meaning they can be both 0 and 1 simultaneously.

    This property enables quantum computers to solve certain problems exponentially faster.
    Applications include cryptography, drug discovery, and optimization problems.
    """
        * 20
    )

    # Large document (~2000 tokens)
    large_doc = (
        """
    Quantum Error Correction is essential for practical quantum computing.
    Physical qubits are prone to errors from decoherence and noise.
    Error correction codes like surface codes can protect quantum information.

    The threshold theorem states that if error rates are below a certain threshold,
    arbitrary long quantum computations are possible with polynomial overhead.
    Current experiments aim to demonstrate fault-tolerant quantum computation.

    Key challenges include:
    - Maintaining long coherence times
    - Implementing high-fidelity gates
    - Scaling to many physical qubits
    - Fast syndrome extraction

    Recent progress has been encouraging, with several groups demonstrating
    improved error rates and longer coherence times.
    """
        * 30
    )

    compressor = SemanticCompressor()

    # Benchmark small document
    bench.run_benchmark(
        "Small Doc Ingestion (~100 tokens)",
        lambda: compressor.ingest_file(small_doc, "small_bench"),
    )

    # Benchmark medium document
    bench.run_benchmark(
        "Medium Doc Ingestion (~500 tokens)",
        lambda: compressor.ingest_file(medium_doc, "medium_bench"),
    )

    # Benchmark large document
    bench.run_benchmark(
        "Large Doc Ingestion (~2000 tokens)",
        lambda: compressor.ingest_file(large_doc, "large_bench"),
    )

    return bench


def benchmark_search_performance(iterations: int = 10):
    """Benchmark semantic search with varying corpus sizes"""
    from src.semantic_compressor import SemanticCompressor

    bench = PerformanceBenchmark(iterations)
    compressor = SemanticCompressor()

    # Create a corpus with multiple documents
    docs = {
        f"doc_{i}": f"Document {i} about quantum computing, error correction, and qubits. " * 10
        for i in range(10)
    }

    # Ingest all documents
    for doc_id, text in docs.items():
        compressor.ingest_file(text, doc_id)

    query = "quantum error correction"

    # Benchmark search in single document
    bench.run_benchmark(
        "Semantic Search (1 document)",
        lambda: compressor.search_semantic(query, "doc_0", top_k=5),
    )

    # Benchmark search across all documents
    bench.run_benchmark(
        "Semantic Search (10 documents)",
        lambda: compressor.search_semantic(query, None, top_k=5),
    )

    return bench


def benchmark_fidelity_modulation(iterations: int = 10):
    """Benchmark different fidelity levels"""
    from src.semantic_compressor import SemanticCompressor, FidelityLevel

    bench = PerformanceBenchmark(iterations)
    compressor = SemanticCompressor()

    # Prepare document
    doc = (
        """
    Quantum computing represents a paradigm shift in computation.
    By harnessing quantum mechanics, these systems can solve problems
    that are intractable for classical computers.
    """
        * 20
    )

    compressor.ingest_file(doc, "fidelity_test")
    node_ids = [nid for nid in compressor.chunks.keys() if nid.startswith("fidelity_test")][:5]

    # Benchmark different fidelity levels
    for fidelity in [
        FidelityLevel.ABSTRACT,
        FidelityLevel.STRUCTURE,
        FidelityLevel.RAW,
    ]:
        bench.run_benchmark(
            f"Fidelity: {fidelity.name}",
            lambda f=fidelity: compressor.modulate_region(node_ids, f),
        )

    return bench


def benchmark_compression_ratios(iterations: int = 10):
    """Benchmark and report compression ratios"""
    from src.semantic_compressor import SemanticCompressor

    print("\n" + "=" * 70)
    print("COMPRESSION RATIO ANALYSIS")
    print("=" * 70)

    compressor = SemanticCompressor()

    test_cases = [
        ("Small (100 tokens)", "Quantum computing " * 20, iterations),
        ("Medium (500 tokens)", "Quantum error correction " * 100, iterations),
        ("Large (2000 tokens)", "Surface codes for quantum computing " * 300, iterations),
    ]

    results = []

    for name, doc, iters in test_cases:
        ratios = []
        token_savings = []

        for i in range(iters):
            result = compressor.ingest_file(doc, f"compress_test_{i}")
            ratios.append(result.compression_ratio)
            savings = (1 - result.skeleton_tokens / result.total_tokens) * 100
            token_savings.append(savings)

        avg_ratio = statistics.mean(ratios)
        avg_savings = statistics.mean(token_savings)

        print(f"\n{name}:")
        print(f"   Avg Compression Ratio: {avg_ratio:.2f}x")
        print(f"   Avg Token Savings:     {avg_savings:.1f}%")
        print(f"   Min Ratio:             {min(ratios):.2f}x")
        print(f"   Max Ratio:             {max(ratios):.2f}x")

        results.append((name, avg_ratio, avg_savings))

    return results


def benchmark_scar_features(iterations: int = 5):
    """Benchmark SCAR-enhanced features"""
    from src.semantic_compressor import SemanticCompressor
    from src.scar_compressor import SCAREnhancedCompressor

    bench = PerformanceBenchmark(iterations)
    base_compressor = SemanticCompressor()

    doc = "Quantum error correction threshold theorem " * 100
    base_compressor.ingest_file(doc, "scar_test")

    scar_compressor = SCAREnhancedCompressor(
        base_compressor, use_learnable_compression=True, use_alignment_guidance=True
    )

    # Benchmark standard search
    query = "error threshold"
    bench.run_benchmark(
        "Standard Semantic Search",
        lambda: base_compressor.search_semantic(query, "scar_test", top_k=5),
    )

    # Benchmark SCAR alignment search
    bench.run_benchmark(
        "SCAR Alignment-Guided Search",
        lambda: scar_compressor.search_with_alignment(
            query, "scar_test", top_k=5, alignment_weight=0.5
        ),
    )

    return bench


def main():
    """Run all benchmarks"""
    parser = argparse.ArgumentParser(description="Token Saver 5000 Performance Benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick benchmark (5 iterations)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full benchmark (20 iterations)",
    )
    args = parser.parse_args()

    # Determine iterations
    if args.quick:
        iterations = 5
    elif args.full:
        iterations = 20
    else:
        iterations = 10

    print("=" * 70)
    print("TOKEN SAVER 5000 - PERFORMANCE BENCHMARK")
    print("=" * 70)
    print(f"Iterations per benchmark: {iterations}")
    print(f"Python version: {sys.version}")
    print("=" * 70)

    try:
        # Run all benchmarks
        print("\n🔬 Phase 1: Document Ingestion Benchmarks")
        ingestion_bench = benchmark_document_ingestion(iterations)

        print("\n🔍 Phase 2: Search Performance Benchmarks")
        search_bench = benchmark_search_performance(iterations)

        print("\n📊 Phase 3: Fidelity Modulation Benchmarks")
        fidelity_bench = benchmark_fidelity_modulation(iterations)

        print("\n📉 Phase 4: Compression Ratio Analysis")
        compression_results = benchmark_compression_ratios(iterations)

        print("\n🔥 Phase 5: SCAR Features Benchmarks")
        scar_bench = benchmark_scar_features(max(iterations // 2, 3))

        # Print combined summary
        print("\n" + "=" * 70)
        print("COMPLETE BENCHMARK SUMMARY")
        print("=" * 70)

        all_results = (
            ingestion_bench.results
            + search_bench.results
            + fidelity_bench.results
            + scar_bench.results
        )

        print(f"{'Benchmark':<45} {'Avg Time':>12} {'Throughput':>12}")
        print("-" * 70)

        for result in all_results:
            print(
                f"{result.name:<45} {result.avg_time*1000:10.2f} ms "
                f"{result.throughput:10.2f} op/s"
            )

        print("=" * 70)
        print("\n✅ Benchmark Complete!")
        print(f"\nTotal benchmarks run: {len(all_results)}")
        print(f"Total iterations: {sum(r.iterations for r in all_results)}")

        # Performance insights
        print("\n💡 Performance Insights:")
        print(
            f"   • Document ingestion: {ingestion_bench.results[0].avg_time*1000:.1f} ms (small doc)"
        )
        print(f"   • Semantic search: {search_bench.results[0].avg_time*1000:.1f} ms (single doc)")
        print(
            f"   • Fidelity modulation: {fidelity_bench.results[0].avg_time*1000:.1f} ms (abstract)"
        )
        print(f"   • Avg compression ratio: {compression_results[1][1]:.1f}x (medium docs)")

        return 0

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
