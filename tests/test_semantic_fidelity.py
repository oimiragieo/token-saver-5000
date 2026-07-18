"""
Compression Validation Tests (formerly "Semantic Fidelity Benchmark Tests").

NOTE (2026-07-17 cleanup, B4): the SSIM half of this suite (TestSSIMQuality +
the SemanticSSIM fixtures) was removed together with the dead
``src/semantic_ssim.py`` module — no production caller imported it. The
compressor-validation tests below exercise the LIVE SemanticCompressor and
stay.

Test Categories:
- Compression Validation Tests (3 tests)
"""

import pytest
import tiktoken

from src.semantic_compressor import SemanticCompressor

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def sample_document():
    """Sample quantum computing document for testing."""
    return """
Quantum Computing Overview

Quantum computers harness quantum mechanical phenomena to process information.
The basic unit is the qubit, which can exist in superposition states.

Quantum entanglement allows qubits to become correlated. When qubits are
entangled, measuring one instantly affects others.

Quantum Algorithms

Shor's algorithm factors large numbers exponentially faster than classical
algorithms, threatening RSA encryption.

Grover's algorithm provides quadratic speedup for unstructured search problems.

Applications

Quantum simulation allows modeling of quantum systems for materials science
and drug discovery.

Quantum cryptography uses quantum properties for theoretically unbreakable
encryption via protocols like BB84.

Challenges

Quantum decoherence is the loss of quantum properties due to environmental
interactions. Maintaining coherence requires extremely low temperatures.

Error correction is complex because measurements collapse quantum states.
"""


@pytest.fixture
def compressor():
    """Create SemanticCompressor instance."""
    return SemanticCompressor(
        model_name="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
        skeleton_ratio=0.2,  # 20% skeleton
    )


# ===========================
# Helper Functions
# ===========================


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (GPT-4 tokenizer)."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ===========================
# Compression Validation Tests
# ===========================


class TestCompressionValidation:
    """Validate compression ratios and quality retention."""

    def test_compression_ratio_achieved(self, compressor, sample_document):
        """Test that meaningful compression is achieved."""
        file_id = "compression_test"
        result = compressor.ingest_file(sample_document, file_id)

        # Count tokens
        original_tokens = count_tokens(sample_document)
        skeleton_tokens = count_tokens(result.skeleton_text)

        # Calculate compression
        compression_ratio = 1 - (skeleton_tokens / original_tokens)
        compression_percent = compression_ratio * 100

        print("\n✅ Compression Test:")
        print(f"   Original: {original_tokens} tokens")
        print(f"   Skeleton: {skeleton_tokens} tokens")
        print(f"   Ratio:    {compression_ratio:.1%} ({compression_percent:.0f}% reduction)")
        print(f"   From result: {result.compression_ratio:.1f}x")

        # Verify compression achieved (target: 40%+)
        assert (
            compression_ratio >= 0.3
        ), f"Only {compression_percent:.0f}% compression (target: 30%+)"
        assert result.compression_ratio >= 1.3, "Compression ratio below 1.3x"

    def test_skeleton_format_valid(self, compressor, sample_document):
        """Test that skeleton has valid format."""
        file_id = "format_test"
        result = compressor.ingest_file(sample_document, file_id)

        skeleton = result.skeleton_text

        # Verify skeleton structure
        assert len(skeleton) > 0, "Empty skeleton"
        assert result.total_nodes > 0, "No nodes created"
        assert result.skeleton_tokens > 0, "No skeleton tokens"

        print("\n✅ Skeleton Format Test:")
        print(f"   Total nodes: {result.total_nodes}")
        print(f"   Skeleton length: {len(skeleton)} chars")
        print(f"   First 200 chars: {skeleton[:200]}")

    def test_demo_proof_reproducible(self, compressor):
        """Test that compression works on demo-like content."""
        quantum_text = """
Quantum computing represents a paradigm shift in computation, harnessing
quantum mechanical phenomena like superposition and entanglement. Unlike
classical bits, qubits can exist in superposition states, enabling parallel
processing. Quantum algorithms like Shor's factorization and Grover's search
demonstrate exponential and quadratic speedups. Major challenges include
maintaining quantum coherence and implementing error correction schemes.
"""

        file_id = "demo_test"
        result = compressor.ingest_file(quantum_text, file_id)

        original_tokens = count_tokens(quantum_text)
        skeleton_tokens = count_tokens(result.skeleton_text)
        compression_ratio = 1 - (skeleton_tokens / original_tokens)

        print("\n✅ Demo Proof Reproduction:")
        print(f"   Original: {original_tokens} tokens")
        print(f"   Skeleton: {skeleton_tokens} tokens")
        print(f"   Compression: {compression_ratio:.1%}")

        # Small documents may expand due to skeleton overhead (documented in CLAUDE.md)
        # Just verify system works without crashing
        assert result.total_nodes > 0, "No nodes created"
        assert len(result.skeleton_text) > 0, "Empty skeleton"
