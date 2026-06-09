"""
Semantic chunking for context ingestion.

Based on SCOPE (ACL 2025) — replaces fixed-size chunking with
embedding-based boundary detection that keeps semantically coherent
units together.
"""

from typing import Callable, List, Optional

import numpy as np


def _word_token_count(text: str) -> int:
    """Cheap word-count token proxy (whitespace split)."""
    return len(text.split())


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
    max_chunk_size: int = 512,
    token_count_fn: Optional[Callable[[str], int]] = None,
) -> List[List[str]]:
    """Group semantically similar sentences into chunks.

    A boundary is opened whenever consecutive-unit similarity drops below
    ``threshold`` OR the accumulated chunk would exceed ``max_chunk_size``
    TOKENS. The token-budget cap is load-bearing: when the embedding model
    rates a long run of paragraphs as mutually similar (no boundary fires),
    the chunk MUST still be split so it does not exceed the budget — otherwise
    a coherent single-topic doc collapses into one giant node, which destroys
    downstream skeleton/retrieval fidelity (A1 calibration, 2026-06-08).

    Args:
        sentences: List of sentences / paragraphs.
        encode_fn: Function that encodes texts to embeddings.
        threshold: Similarity threshold for boundary detection.
        max_chunk_size: Maximum TOKENS per chunk (not sentence count). The
            running token total uses ``token_count_fn`` (default: whitespace
            word count).
        token_count_fn: Optional token counter; defaults to a word-count proxy.

    Returns:
        List of chunks (each chunk is a list of sentences).
    """
    if not sentences:
        return []

    counter = token_count_fn or _word_token_count

    boundaries = detect_semantic_boundaries(sentences, encode_fn, threshold)
    boundary_set = set(boundaries)

    chunks: List[List[str]] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sent_tokens = counter(sentence)

        # Open a new chunk on a semantic boundary.
        if i in boundary_set and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        # Token-budget cap: if appending this unit would overflow the budget
        # AND the chunk already holds at least one unit, flush first. A single
        # unit larger than the budget is kept whole (the caller's
        # _split_oversized_section handles further sub-splitting downstream).
        if current_chunk and current_tokens + sent_tokens > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sent_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
