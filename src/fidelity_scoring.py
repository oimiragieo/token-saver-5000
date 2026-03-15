"""
Compression fidelity scoring.

Based on SCOPE and the Empirical Study on Prompt Compression (ICLR 2025) —
measures compression quality by computing embedding similarity between
original and compressed text, giving clients a fidelity score (0-1).
"""

from typing import Callable, List

import numpy as np


def compute_fidelity_score(
    original_text: str,
    compressed_text: str,
    encode_fn: Callable[[List[str]], np.ndarray],
) -> float:
    """Compute fidelity score between original and compressed text.

    Uses cosine similarity of mean embeddings as a proxy for semantic
    preservation. Score of 1.0 = perfect preservation, 0.0 = total loss.

    Args:
        original_text: The original uncompressed text
        compressed_text: The compressed output text
        encode_fn: Function that encodes texts to embedding vectors

    Returns:
        Fidelity score in [0, 1]
    """
    if not original_text and not compressed_text:
        return 1.0  # both empty = perfect preservation

    if not original_text or not compressed_text:
        return 0.0

    embeddings = encode_fn([original_text, compressed_text])

    emb_orig = embeddings[0]
    emb_comp = embeddings[1]

    norm_orig = np.linalg.norm(emb_orig)
    norm_comp = np.linalg.norm(emb_comp)

    if norm_orig == 0 or norm_comp == 0:
        return 0.0

    similarity = float(np.dot(emb_orig, emb_comp) / (norm_orig * norm_comp))
    return max(0.0, min(1.0, similarity))
