"""
Semantic chunking for context ingestion.

Based on SCOPE (ACL 2025) — replaces fixed-size chunking with
embedding-based boundary detection that keeps semantically coherent
units together.
"""

from typing import Callable, List

import numpy as np


def detect_semantic_boundaries(
    sentences: List[str],
    encode_fn: Callable[[List[str]], np.ndarray],
    threshold: float = 0.5,
) -> List[int]:
    """Detect semantic boundaries between consecutive sentences.

    A boundary is placed where cosine similarity between consecutive
    sentence embeddings drops below the threshold.

    Args:
        sentences: List of sentences
        encode_fn: Function that encodes texts to embeddings
        threshold: Similarity threshold below which a boundary is placed

    Returns:
        List of boundary indices (where new chunks start)
    """
    if len(sentences) <= 1:
        return []

    embeddings = encode_fn(sentences)
    boundaries = []

    for i in range(1, len(sentences)):
        a = embeddings[i - 1]
        b = embeddings[i]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            sim = 0.0
        else:
            sim = float(np.dot(a, b) / (norm_a * norm_b))

        if sim < threshold:
            boundaries.append(i)

    return boundaries


def chunk_by_semantics(
    sentences: List[str],
    encode_fn: Callable[[List[str]], np.ndarray],
    threshold: float = 0.5,
    max_chunk_size: int = 50,
) -> List[List[str]]:
    """Group semantically similar sentences into chunks.

    Args:
        sentences: List of sentences
        encode_fn: Function that encodes texts to embeddings
        threshold: Similarity threshold for boundary detection
        max_chunk_size: Maximum sentences per chunk

    Returns:
        List of chunks (each chunk is a list of sentences)
    """
    if not sentences:
        return []

    boundaries = detect_semantic_boundaries(sentences, encode_fn, threshold)
    boundary_set = set(boundaries)

    chunks = []
    current_chunk = []

    for i, sentence in enumerate(sentences):
        if i in boundary_set and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []

        current_chunk.append(sentence)

        if len(current_chunk) >= max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
