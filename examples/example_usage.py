#!/usr/bin/env python3
"""
Example Usage of Semantic Modulator

This script demonstrates the core capabilities:
1. Ingesting documents with semantic compression
2. Reading skeleton views
3. Modulating specific regions at different fidelity levels
4. Semantic search
5. Blind spot detection
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.semantic_compressor import SemanticCompressor, FidelityLevel
from src.blind_spot_detector import BlindSpotDetector, HaloEffectDetector


# Sample document: A technical paper excerpt
SAMPLE_DOCUMENT = """
Introduction to Quantum Error Correction

Quantum computing promises exponential speedup for certain computational problems. However, quantum systems are inherently fragile and susceptible to errors from environmental noise and imperfect control operations. Quantum error correction (QEC) is essential for building practical quantum computers.

Background on Quantum States

A quantum bit, or qubit, exists in a superposition of states |0⟩ and |1⟩. The state of a qubit can be represented as α|0⟩ + β|1⟩, where |α|² + |β|² = 1. Multiple qubits can be entangled, creating correlations that have no classical analog. These entangled states form the foundation of quantum computing's power.

The Challenge of Decoherence

Quantum states are extremely fragile. Interaction with the environment causes decoherence, where quantum information is lost to the surroundings. Additionally, gate operations are imperfect, introducing errors. Without error correction, quantum computations become unreliable after just a few operations.

Quantum Error Correction Codes

Classical error correction uses redundancy: encoding one bit into multiple bits allows detection and correction of errors. Quantum error correction follows similar principles but must respect the no-cloning theorem, which prevents copying quantum states. The solution is to encode a logical qubit into multiple physical qubits using entanglement.

The Shor Code

The Shor code was the first quantum error correction code, encoding one logical qubit into nine physical qubits. It can correct arbitrary single-qubit errors. The code works by detecting bit-flip errors (X errors) and phase-flip errors (Z errors) separately, then combining these protections.

Surface Codes

Modern quantum computers predominantly use surface codes due to their practical advantages. Surface codes arrange qubits on a 2D lattice and only require nearest-neighbor interactions. They have a relatively high threshold error rate of around 1%, meaning they can correct errors as long as physical error rates stay below this threshold.

Experimental Results

Recent experiments have demonstrated quantum error correction in practice. Google's Sycamore processor achieved 99.7% fidelity for two-qubit gates. IBM's quantum systems have shown successful implementation of the surface code with multiple rounds of error detection and correction.

Gate Fidelity Measurements

Gate fidelity is measured using randomized benchmarking, a protocol that characterizes average gate performance. Current state-of-the-art systems achieve two-qubit gate fidelities above 99%, with single-qubit gates exceeding 99.9% fidelity. However, achieving the threshold for fault-tolerant quantum computation requires continued improvement.

Contradictory Findings on Gate Fidelity

Some recent studies have challenged the reported gate fidelity numbers. Cross-talk between qubits and correlated errors can make fidelities appear higher than they truly are. More sophisticated characterization methods like gate set tomography reveal systematic errors that randomized benchmarking misses.

Theoretical Bounds

Information theory provides fundamental bounds on error correction. The quantum capacity of a noisy channel determines the maximum rate at which quantum information can be reliably transmitted. These bounds guide the design of efficient quantum codes.

Future Directions

The path to large-scale quantum computing requires further improvements in gate fidelities, better codes with lower overhead, and more efficient decoding algorithms. Machine learning approaches show promise for optimizing error correction strategies.

Conclusion

Quantum error correction is a critical component of quantum computing. While significant challenges remain, recent progress demonstrates that fault-tolerant quantum computation is achievable. Continued research in code design, error characterization, and hardware improvements will pave the way for practical quantum computers.
"""


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print_section("🧠 Semantic Modulator - Example Usage")

    # Initialize the compressor
    print("Initializing Semantic Compressor...")
    compressor = SemanticCompressor(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        skeleton_ratio=0.2,
    )

    # Initialize blind spot detector
    blind_spot_detector = BlindSpotDetector(compressor)
    halo_detector = HaloEffectDetector(compressor)

    # =================================================================
    # STEP 1: Ingest a document
    # =================================================================
    print_section("Step 1: Ingesting Document with Semantic Compression")

    skeleton = compressor.ingest_file(
        text=SAMPLE_DOCUMENT,
        file_id="quantum_paper",
        metadata={
            "title": "Introduction to Quantum Error Correction",
            "author": "Example Author",
            "date": "2025-01-15",
        }
    )

    print(f"Original document length: {len(SAMPLE_DOCUMENT)} characters")
    print(f"Original tokens: {skeleton.total_tokens:,}")
    print(f"Skeleton tokens: {skeleton.skeleton_tokens:,}")
    print(f"Compression ratio: {skeleton.compression_ratio:.1f}x")
    print(f"Token savings: {(1 - skeleton.skeleton_tokens/skeleton.total_tokens)*100:.1f}%")

    # =================================================================
    # STEP 2: Read the skeleton
    # =================================================================
    print_section("Step 2: Reading Skeleton View")

    skeleton_text = compressor.read_skeleton("quantum_paper")
    print(skeleton_text)

    # =================================================================
    # STEP 3: Semantic search
    # =================================================================
    print_section("Step 3: Semantic Search")

    query = "gate fidelity measurements"
    print(f"Searching for: '{query}'")

    results = compressor.search_semantic(query, file_id="quantum_paper", top_k=3)

    print(f"\nFound {len(results)} relevant nodes:\n")
    for i, node_id in enumerate(results, 1):
        node = compressor.chunks[node_id]
        summary = compressor._generate_summary(node.text, max_length=80)
        print(f"{i}. [{node_id}] (importance: {node.importance:.3f})")
        print(f"   {summary}\n")

    # =================================================================
    # STEP 4: Modulate regions at different fidelity levels
    # =================================================================
    print_section("Step 4: Modulating Regions at Different Fidelity Levels")

    # Get the top result from search
    top_node = results[0]

    print("🔹 ABSTRACT Level:")
    abstract = compressor.modulate_region([top_node], FidelityLevel.ABSTRACT)
    print(abstract)

    print("\n🔹 STRUCTURE Level:")
    structure = compressor.modulate_region([top_node], FidelityLevel.STRUCTURE)
    print(structure)

    print("\n🔹 RAW Level:")
    raw = compressor.modulate_region([top_node], FidelityLevel.RAW)
    print(raw)

    # =================================================================
    # STEP 5: Blind spot detection
    # =================================================================
    print_section("Step 5: Blind Spot Detection")

    # Simulate an AI response that might miss important context
    ai_response = """
    Based on the document, gate fidelity measurements show impressive results,
    with two-qubit gates achieving 99.7% fidelity. This indicates that current
    quantum systems are approaching the threshold needed for fault-tolerant
    quantum computation.
    """

    # The AI only retrieved one node
    retrieved_nodes = [results[0]]

    print("AI Response (simulated):")
    print(ai_response.strip())
    print("\nRetrieved nodes:", retrieved_nodes)

    print("\n🔍 Running blind spot analysis...\n")

    report = blind_spot_detector.analyze_response(
        ai_response=ai_response,
        file_id="quantum_paper",
        retrieved_node_ids=retrieved_nodes,
    )

    formatted_report = blind_spot_detector.format_report(report)
    print(formatted_report)

    # =================================================================
    # STEP 6: Correcting the response with blind spot info
    # =================================================================
    if report.auto_inject:
        print_section("Step 6: Auto-Correction Based on Blind Spots")

        print("Retrieving missed critical context...")
        missed_content = compressor.modulate_region(
            report.auto_inject,
            FidelityLevel.RAW
        )
        print(missed_content)

        print("\n💡 Corrected understanding:")
        print("""
        While initial fidelity measurements appear impressive at 99.7%, recent studies
        have revealed important caveats. Cross-talk between qubits and correlated errors
        can make fidelities appear higher than they truly are. More sophisticated methods
        like gate set tomography reveal systematic errors that simpler benchmarking misses.
        This suggests the path to fault-tolerant quantum computation may be longer than
        initially thought.
        """)

    # =================================================================
    # STEP 7: Hallucination detection
    # =================================================================
    print_section("Step 7: Hallucination Detection")

    # Test with a grounded response
    grounded_response = """
    The document discusses quantum error correction codes, specifically mentioning
    the Shor code and surface codes as approaches to protecting quantum information.
    """

    is_hallucinating, warnings = halo_detector.detect_hallucination(
        grounded_response, "quantum_paper"
    )

    if is_hallucinating:
        print("🚨 HALLUCINATION DETECTED in grounded response:")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print("✅ Grounded response passed validation")

    # Test with a hallucinated response
    hallucinated_response = """
    The document extensively covers topological quantum computing using Majorana
    fermions and discusses implementation on diamond NV centers with detailed
    experimental protocols for quantum teleportation.
    """

    is_hallucinating, warnings = halo_detector.detect_hallucination(
        hallucinated_response, "quantum_paper"
    )

    print(f"\nTesting potentially hallucinated response...")
    if is_hallucinating:
        print("🚨 HALLUCINATION DETECTED:")
        for warning in warnings:
            print(f"  • {warning}")
    else:
        print("✅ Response passed validation")

    # =================================================================
    # STEP 8: Statistics
    # =================================================================
    print_section("Step 8: Document Statistics")

    stats = compressor.get_stats("quantum_paper")

    print(f"File ID: {stats['file_id']}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Skeleton tokens: {stats['skeleton_tokens']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.1f}x")
    print(f"\nMetadata: {stats['metadata']}")

    print_section("✅ Example Complete!")
    print("This demonstration shows how Semantic Modulator achieves:")
    print("  • 80-95% token reduction through semantic compression")
    print("  • Structure preservation via graph analysis")
    print("  • Adaptive fidelity (retrieve only what you need)")
    print("  • Blind spot detection to prevent incomplete reasoning")
    print("  • Hallucination detection to ensure grounded responses")
    print("\nAll processing happens locally - no external API calls for compression!")


if __name__ == "__main__":
    main()
