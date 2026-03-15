"""
Query-adaptive compression ratios.

Based on KVzip (2025) and LazyLLM (ICLR 2025) — dynamically adjusts
compression ratio per section based on query relevance. Sections matching
the query get lighter compression; irrelevant sections get compressed harder.
"""

from typing import Dict, List, Optional

import numpy as np


def compute_section_ratios(
    sections: List[dict],
    query_embedding: Optional[np.ndarray],
    base_ratio: float = 0.3,
    min_ratio: float = 0.05,
    max_ratio: float = 1.0,
) -> List[float]:
    """Compute per-section compression ratios based on query relevance.

    Relevant sections get higher ratios (keep more), irrelevant get lower.
    The average ratio is kept close to base_ratio.

    Args:
        sections: List of dicts with "embedding" key (np.ndarray)
        query_embedding: Query embedding vector (None = uniform ratios)
        base_ratio: Target average compression ratio
        min_ratio: Minimum ratio for any section
        max_ratio: Maximum ratio for any section

    Returns:
        List of per-section ratios
    """
    if not sections:
        return []

    if len(sections) == 1:
        return [max(min_ratio, min(max_ratio, base_ratio))]

    if query_embedding is None:
        return [base_ratio] * len(sections)

    # Compute relevance scores
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return [base_ratio] * len(sections)

    relevance_scores = []
    for section in sections:
        emb = section["embedding"]
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            relevance_scores.append(0.0)
        else:
            sim = float(np.dot(emb, query_embedding) / (emb_norm * query_norm))
            relevance_scores.append(max(0.0, sim))

    # Normalize scores to distribute budget
    total_relevance = sum(relevance_scores)
    if total_relevance == 0:
        return [base_ratio] * len(sections)

    # Scale ratios: more relevant = higher ratio
    n = len(sections)
    total_budget = base_ratio * n
    ratios = []
    for score in relevance_scores:
        # Proportional allocation
        ratio = (score / total_relevance) * total_budget
        ratio = max(min_ratio, min(max_ratio, ratio))
        ratios.append(ratio)

    return ratios
