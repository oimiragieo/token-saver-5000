"""
SCAR-Enhanced Semantic Compression Demo

Demonstrates the integration of concepts from:
"Semantic Context Matters: Improving Conditioning for Autoregressive Models"
(arXiv:2511.14063v1)

This example shows:
1. Learnable semantic compression (SCAR Section 3.2)
2. Semantic alignment guidance (SCAR Section 3.3)
3. Adaptive fidelity modulation based on alignment scores
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.semantic_compressor import SemanticCompressor
from src.scar_compressor import SCAREnhancedCompressor


def main():
    print("=" * 80)
    print("SCAR-ENHANCED SEMANTIC COMPRESSION DEMO")
    print("Adapting arXiv:2511.14063v1 to Text/Document Compression")
    print("=" * 80)

    # Sample technical document
    technical_paper = """
    Introduction to Quantum Error Correction

    Quantum computers promise exponential speedups for certain computational tasks.
    However, quantum systems are inherently fragile, suffering from decoherence and noise.
    Error correction is essential for practical quantum computing.

    Surface Codes: A Leading Approach

    Surface codes are topological error-correcting codes defined on a 2D lattice.
    Each physical qubit interacts only with its nearest neighbors, making them
    experimentally feasible with current technology.

    The key advantage of surface codes is their high error threshold, approximately 1%.
    This threshold is significantly higher than earlier approaches like Shor codes.
    However, the overhead is substantial: protecting one logical qubit requires
    hundreds of physical qubits arranged in a square lattice.

    Error Detection and Correction

    Surface codes use stabilizer measurements to detect errors without collapsing
    the quantum state. Syndrome extraction identifies error locations through
    repeated measurements of plaquette and vertex operators.

    The minimum distance of the code determines how many errors can be corrected.
    For an L×L surface code, the code distance is L, allowing correction of
    (L-1)/2 errors per error correction cycle.

    Practical Challenges

    Several challenges remain for practical implementation:
    1. High qubit overhead - requires ~1000 physical qubits per logical qubit
    2. Fast syndrome extraction - measurements must be faster than decoherence
    3. Decoder efficiency - classical processing of syndrome data in real-time
    4. Connectivity requirements - implementing nearest-neighbor interactions

    Recent Advances

    Recent work has explored code concatenation and magic state distillation
    to reduce overhead. Lattice surgery techniques enable logical gate operations
    without excessive qubit movement.

    Color codes and other topological variants offer alternative trade-offs
    between overhead and error threshold. The optimal choice depends on the
    specific hardware platform and target application.

    Conclusion

    Surface codes represent a practical path toward fault-tolerant quantum computing.
    While significant challenges remain, ongoing research continues to improve
    their efficiency and reduce overhead requirements.
    """

    # Initialize base compressor
    print("\n[1] Initializing Base Semantic Compressor...")
    base_compressor = SemanticCompressor(similarity_threshold=0.75, skeleton_ratio=0.2)

    # Initialize SCAR enhancements
    print("[2] Initializing SCAR Enhancements...")
    scar = SCAREnhancedCompressor(
        base_compressor=base_compressor,
        use_learnable_compression=True,
        use_alignment_guidance=True,
        compression_ratio=4.0,  # 4× compression like SCAR paper
    )

    # Ingest document
    print("\n[3] Ingesting technical paper...")
    skeleton = base_compressor.ingest_file(
        text=technical_paper,
        file_id="quantum_ec",
        metadata={
            "title": "Introduction to Quantum Error Correction",
            "domain": "quantum_computing",
        },
    )

    print("\n📊 Compression Statistics:")
    print(f"   Original tokens: {skeleton.total_tokens:,}")
    print(f"   Skeleton tokens: {skeleton.skeleton_tokens:,}")
    print(f"   Compression ratio: {skeleton.compression_ratio:.1f}×")
    print(f"   Total nodes: {skeleton.total_nodes}")

    # Display skeleton
    print("\n" + "=" * 80)
    print("BASELINE: Standard Skeleton View")
    print("=" * 80)
    print(base_compressor.read_skeleton("quantum_ec"))

    # SCAR Feature 1: Alignment-Guided Search
    print("\n" + "=" * 80)
    print("SCAR FEATURE 1: Alignment-Guided Semantic Search")
    print("=" * 80)

    queries = [
        "What is the error threshold for surface codes?",
        "What are the practical challenges?",
        "How does syndrome extraction work?",
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 80)

        # Standard search (baseline)
        standard_results = base_compressor.search_semantic(
            query=query, file_id="quantum_ec", top_k=3
        )

        print("\n   Baseline Search (cosine similarity only):")
        for i, node_id in enumerate(standard_results, 1):
            node = base_compressor.chunks[node_id]
            summary = base_compressor._generate_summary(node.text, 60)
            print(f"   {i}. {node_id}: {summary}")

        # SCAR alignment-guided search
        scar_results = scar.search_with_alignment(
            query=query,
            file_id="quantum_ec",
            top_k=3,
            alignment_weight=0.5,  # 50% alignment, 50% similarity
        )

        print("\n   SCAR Search (alignment + similarity):")
        for i, (node_id, score) in enumerate(scar_results, 1):
            node = base_compressor.chunks[node_id]
            summary = base_compressor._generate_summary(node.text, 60)
            print(f"   {i}. [Score: {score:.3f}] {node_id}: {summary}")

    # SCAR Feature 2: Adaptive Modulation
    print("\n" + "=" * 80)
    print("SCAR FEATURE 2: Adaptive Fidelity Modulation")
    print("(High alignment → Full detail, Low alignment → Summary only)")
    print("=" * 80)

    query = "What is the error threshold for surface codes?"
    print(f"\n🎯 Query: {query}\n")

    result = scar.adaptive_modulate(
        query=query,
        file_id="quantum_ec",
        top_k=3,
        alignment_threshold=0.7,  # High-fidelity threshold
    )
    print(result)

    # SCAR Feature 3: Learnable Compression
    print("\n" + "=" * 80)
    print("SCAR FEATURE 3: Learnable Embedding Compression")
    print("=" * 80)

    # Get embeddings from a few nodes
    sample_node_ids = list(base_compressor.chunks.keys())[:5]
    sample_embeddings = np.array([base_compressor.chunks[nid].embedding for nid in sample_node_ids])

    print("\n📦 Embedding Compression Demo:")
    print(f"   Input shape: {sample_embeddings.shape}")
    print(f"   Input dimension: {sample_embeddings.shape[1]}")

    # Compress using SCAR
    compressed_embeddings = scar.compress_embeddings(sample_embeddings)

    print(f"\n   Compressed shape: {compressed_embeddings.shape}")
    print(f"   Compressed dimension: {compressed_embeddings.shape[1]}")
    print(
        f"   Compression ratio: {sample_embeddings.shape[1] / compressed_embeddings.shape[1]:.1f}×"
    )
    print(
        f"\n   Memory savings: {(1 - compressed_embeddings.nbytes / sample_embeddings.nbytes) * 100:.1f}%"
    )

    # SCAR Stats
    print("\n" + "=" * 80)
    print("SCAR Configuration Summary")
    print("=" * 80)

    stats = scar.get_compression_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 80)
    print("Comparison: SCAR vs Baseline")
    print("=" * 80)

    comparison_query = "How many physical qubits are needed?"

    print(f"\n🔍 Query: {comparison_query}\n")

    # Baseline
    print("   [Baseline] Standard Retrieval:")
    baseline_nodes = base_compressor.search_semantic(
        query=comparison_query, file_id="quantum_ec", top_k=2
    )
    baseline_content = base_compressor.modulate_region(
        node_ids=baseline_nodes, fidelity_level=base_compressor.FidelityLevel.STRUCTURE
    )
    print(baseline_content)

    # SCAR
    print("\n   [SCAR] Alignment-Guided Retrieval:")
    scar_content = scar.adaptive_modulate(
        query=comparison_query, file_id="quantum_ec", top_k=2, alignment_threshold=0.6
    )
    print(scar_content)

    print("\n" + "=" * 80)
    print("✓ Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. SCAR's learnable compression reduces embedding size by 4× (384D → 96D)")
    print("2. Semantic alignment guidance improves retrieval relevance")
    print("3. Adaptive fidelity automatically adjusts detail level based on alignment")
    print("4. All concepts from SCAR paper (arXiv:2511.14063v1) adapted to text compression")


if __name__ == "__main__":
    import numpy as np

    main()
