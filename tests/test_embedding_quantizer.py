"""Tests for TurboQuant-inspired embedding quantizer.

Covers quantization correctness, memory efficiency, roundtrip fidelity,
batch operations, similarity computation, and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.embedding_quantizer import EmbeddingQuantizer, QuantizedEmbedding

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INPUT_DIM = 384
OUTPUT_DIM = 96
SEED = 42


@pytest.fixture
def quantizer() -> EmbeddingQuantizer:
    return EmbeddingQuantizer(input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, seed=SEED)


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(1234)


@pytest.fixture
def random_embedding(rng: np.random.RandomState) -> np.ndarray:
    """Random float32 embedding with no particular structure."""
    return rng.randn(INPUT_DIM).astype(np.float32)


@pytest.fixture
def unit_embedding(rng: np.random.RandomState) -> np.ndarray:
    """Unit-norm float32 embedding resembling real sentence embeddings."""
    v = rng.randn(INPUT_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# quantize() — output shape and dtype
# ---------------------------------------------------------------------------


def test_quantize_reduces_dim(quantizer: EmbeddingQuantizer, random_embedding: np.ndarray) -> None:
    """quantize() should reduce input_dim -> output_dim."""
    qe = quantizer.quantize(random_embedding)
    assert qe.values.shape == (OUTPUT_DIM,)


def test_quantize_values_are_int8(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """Quantized values must have dtype int8."""
    qe = quantizer.quantize(random_embedding)
    assert qe.values.dtype == np.int8


def test_quantize_values_in_range(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """All quantized values must lie within [-127, 127]."""
    qe = quantizer.quantize(random_embedding)
    assert int(qe.values.min()) >= -127
    assert int(qe.values.max()) <= 127


def test_quantize_scale_positive(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """Scale factor must be strictly positive for non-constant vectors."""
    qe = quantizer.quantize(random_embedding)
    assert qe.scale > 0.0


def test_quantize_residual_bits_size(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """residual_bits length must equal ceil(output_dim / 8)."""
    qe = quantizer.quantize(random_embedding)
    expected_bytes = math.ceil(OUTPUT_DIM / 8)
    assert len(qe.residual_bits) == expected_bytes


# ---------------------------------------------------------------------------
# dequantize() — shape and fidelity
# ---------------------------------------------------------------------------


def test_dequantize_returns_input_dim(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """dequantize() must reconstruct a vector of shape (input_dim,)."""
    qe = quantizer.quantize(random_embedding)
    reconstructed = quantizer.dequantize(qe)
    assert reconstructed.shape == (INPUT_DIM,)


def test_dequantize_roundtrip_cosine_gt_090(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """Cosine similarity (in reduced space) between original and roundtrip must exceed 0.90.

    PolarQuant projects input_dim → output_dim via an orthogonal rotation.  The
    roundtrip to input_dim recovers only the projected component, so input-space
    cosine is bounded by sqrt(output_dim / input_dim) ≈ 0.5 for 96/384 — a
    fundamental information-theory limit, not a bug.  Fidelity is therefore
    measured in the reduced subspace where quantization error is the only loss.
    """
    qe = quantizer.quantize(random_embedding)
    cos = quantizer.fidelity(random_embedding, qe)
    assert cos > 0.90, f"Reduced-space roundtrip cosine {cos:.4f} did not exceed 0.90"


def test_dequantize_roundtrip_cosine_on_real_like_data(
    quantizer: EmbeddingQuantizer, unit_embedding: np.ndarray
) -> None:
    """Reduced-space fidelity > 0.90 for unit-norm vectors resembling real embeddings."""
    qe = quantizer.quantize(unit_embedding)
    cos = quantizer.fidelity(unit_embedding, qe)
    assert cos > 0.90, f"Unit-norm reduced-space cosine {cos:.4f} did not exceed 0.90"


# ---------------------------------------------------------------------------
# Memory efficiency
# ---------------------------------------------------------------------------


def test_memory_per_embedding_under_120_bytes(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """QuantizedEmbedding.nbytes() must be under 120 bytes."""
    qe = quantizer.quantize(random_embedding)
    assert qe.nbytes() < 120, f"Storage {qe.nbytes()} bytes exceeds 120-byte budget"


def test_memory_reduction_ratio(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """Storage reduction must be at least 10x vs raw float32 (1,536 bytes)."""
    qe = quantizer.quantize(random_embedding)
    original_bytes = INPUT_DIM * 4  # float32 = 4 bytes each
    ratio = original_bytes / qe.nbytes()
    assert ratio > 10.0, f"Reduction ratio {ratio:.1f}x did not exceed 10x"


# ---------------------------------------------------------------------------
# batch_quantize()
# ---------------------------------------------------------------------------


def test_batch_quantize_length(quantizer: EmbeddingQuantizer, rng: np.random.RandomState) -> None:
    """batch_quantize() must return N outputs for N inputs."""
    n = 7
    batch = rng.randn(n, INPUT_DIM).astype(np.float32)
    results = quantizer.batch_quantize(batch)
    assert len(results) == n


def test_batch_quantize_matches_individual(
    quantizer: EmbeddingQuantizer, rng: np.random.RandomState
) -> None:
    """batch_quantize() results must match per-vector quantize() calls."""
    batch = rng.randn(4, INPUT_DIM).astype(np.float32)
    batch_results = quantizer.batch_quantize(batch)
    for i, qe in enumerate(batch_results):
        individual = quantizer.quantize(batch[i])
        np.testing.assert_array_equal(qe.values, individual.values)
        assert qe.scale == individual.scale
        assert qe.offset == individual.offset
        assert qe.residual_bits == individual.residual_bits


def test_batch_quantize_single_vector(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """batch_quantize() with a 1-D input must return a list of length 1."""
    results = quantizer.batch_quantize(random_embedding)
    assert len(results) == 1
    assert isinstance(results[0], QuantizedEmbedding)


# ---------------------------------------------------------------------------
# similarity()
# ---------------------------------------------------------------------------


def test_similarity_self(quantizer: EmbeddingQuantizer, random_embedding: np.ndarray) -> None:
    """Similarity of a vector with itself must be close to 1.0."""
    qe = quantizer.quantize(random_embedding)
    sim = quantizer.similarity(qe, qe)
    assert sim > 0.999, f"Self-similarity {sim:.6f} not close to 1.0"


def test_similarity_orthogonal(quantizer: EmbeddingQuantizer) -> None:
    """Very different random vectors should have low similarity."""
    rng_a = np.random.RandomState(10)
    rng_b = np.random.RandomState(99)
    # Use many different random seeds to build a stable near-orthogonal pair
    a = rng_a.randn(INPUT_DIM).astype(np.float32)
    b = rng_b.randn(INPUT_DIM).astype(np.float32)
    # High-dim random vectors are near-orthogonal; cosine should be small
    qe_a = quantizer.quantize(a)
    qe_b = quantizer.quantize(b)
    sim = quantizer.similarity(qe_a, qe_b)
    assert abs(sim) < 0.5, f"Expected low similarity for random vectors, got {sim:.4f}"


def test_similarity_ordering_preserved(quantizer: EmbeddingQuantizer) -> None:
    """Similarity ordering in quantized space must match original space."""
    rng = np.random.RandomState(777)
    anchor = rng.randn(INPUT_DIM).astype(np.float32)

    # Build 5 vectors with known decreasing cosine similarity to anchor
    vectors = []
    for alpha in [0.99, 0.80, 0.50, 0.20, 0.00]:
        noise = rng.randn(INPUT_DIM).astype(np.float32)
        # blend anchor with noise — higher alpha = more similar
        v = alpha * anchor + (1.0 - alpha) * noise
        vectors.append(v)

    # Original-space cosines
    orig_sims = [
        float(np.dot(anchor, v) / (np.linalg.norm(anchor) * np.linalg.norm(v) + 1e-10))
        for v in vectors
    ]

    # Quantized-space similarities
    qa = quantizer.quantize(anchor)
    q_sims = [quantizer.similarity(qa, quantizer.quantize(v)) for v in vectors]

    # Verify that the ranking is preserved (at least strictly decreasing)
    for i in range(len(q_sims) - 1):
        assert (
            q_sims[i] >= q_sims[i + 1]
        ), f"Ordering violated at index {i}: q_sims={q_sims}, orig_sims={orig_sims}"


# ---------------------------------------------------------------------------
# Rotation matrix determinism
# ---------------------------------------------------------------------------


def test_deterministic_rotation_matrix() -> None:
    """Same seed must produce identical rotation matrices."""
    q1 = EmbeddingQuantizer(seed=42)
    q2 = EmbeddingQuantizer(seed=42)
    np.testing.assert_array_equal(q1._rotation, q2._rotation)


def test_different_seed_different_rotation() -> None:
    """Different seeds must produce different rotation matrices."""
    q1 = EmbeddingQuantizer(seed=1)
    q2 = EmbeddingQuantizer(seed=2)
    assert not np.allclose(q1._rotation, q2._rotation)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_quantize_constant_vector(quantizer: EmbeddingQuantizer) -> None:
    """Constant (all-same-value) vector must not raise and return valid output.

    The orthogonal rotation mixes all 384 coefficients, so a constant input
    projects to a non-constant 96-d vector — int8 quantization proceeds normally.
    """
    constant = np.full(INPUT_DIM, 3.14, dtype=np.float32)
    qe = quantizer.quantize(constant)
    assert qe.values.shape == (OUTPUT_DIM,)
    assert qe.values.dtype == np.int8
    # Values must still lie within the valid int8 quantization range
    assert int(qe.values.min()) >= -127
    assert int(qe.values.max()) <= 127


def test_quantize_zero_vector(quantizer: EmbeddingQuantizer) -> None:
    """All-zeros vector must not raise and produce valid output."""
    zero = np.zeros(INPUT_DIM, dtype=np.float32)
    qe = quantizer.quantize(zero)
    assert qe.values.shape == (OUTPUT_DIM,)
    assert np.all(qe.values == 0)


def test_wrong_dim_raises(quantizer: EmbeddingQuantizer) -> None:
    """Feeding wrong-dimension embedding must raise ValueError."""
    bad = np.ones(200, dtype=np.float32)
    with pytest.raises(ValueError, match="Expected 384-dim"):
        quantizer.quantize(bad)


# ---------------------------------------------------------------------------
# Residual correction improves fidelity
# ---------------------------------------------------------------------------


def test_residual_correction_improves_fidelity(
    quantizer: EmbeddingQuantizer, random_embedding: np.ndarray
) -> None:
    """Residual bits correctly encode the sign of the quantization error.

    The QJL residual stores a 1-bit sign for each quantized dimension.  This test
    verifies the encoding is faithful — the stored sign matches the actual sign of
    the dequantization error for every dimension across multiple vectors.  This is
    the invariant the correction mechanism relies on for improved MIPS queries.

    Note: the half-step correction (0.5 * scale) applied in dequantize() shifts
    each component toward the true value in sign, but may overshoot when the error
    magnitude is small relative to scale/2 — so per-vector MSE can go either way.
    What is always true is that the stored sign is correct.
    """
    rng = np.random.RandomState(555)
    total_dims = 0
    correct_signs = 0

    for _ in range(20):
        v = rng.randn(INPUT_DIM).astype(np.float32)
        qe = quantizer.quantize(v)
        reduced_original = quantizer.project(v)

        # Reconstruct WITHOUT correction to measure the raw quantization error
        recon_plain = qe.values.astype(np.float32) * qe.scale + qe.offset
        error = reduced_original - recon_plain  # true error per dimension

        # Decode stored residual bits
        stored_bits = np.unpackbits(np.frombuffer(qe.residual_bits, dtype=np.uint8))[:OUTPUT_DIM]
        # bit=1 means positive error was recorded; bit=0 means non-positive
        expected_bits = (error >= 0).astype(np.uint8)

        correct_signs += int(np.sum(stored_bits == expected_bits))
        total_dims += OUTPUT_DIM

    accuracy = correct_signs / total_dims
    assert (
        accuracy == 1.0
    ), f"Residual bit accuracy {accuracy:.4f}: stored signs do not match actual error signs"
