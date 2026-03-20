"""
Attention-guided pruning for context compression.

Based on AttentionRAG (arXiv:2503.10720, 2025) — scores nodes by semantic
relevance to a query embedding, enabling 6.3x compression with better
quality than blind ratio-based pruning.
"""

import math
from typing import Dict, List

import numpy as np


def score_nodes_by_relevance(
    node_embeddings: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
) -> Dict[str, float]:
    """Score each node by cosine similarity to query embedding.

    Args:
        node_embeddings: Map of node_id -> embedding vector
        query_embedding: Query embedding vector

    Returns:
        Map of node_id -> relevance score in [0, 1]
    """
    if not node_embeddings:
        return {}

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return {nid: 0.0 for nid in node_embeddings}

    scores = {}
    for node_id, emb in node_embeddings.items():
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            scores[node_id] = 0.0
        else:
            sim = float(np.dot(emb, query_embedding) / (emb_norm * query_norm))
            scores[node_id] = max(0.0, sim)  # clamp negatives

    return scores


def prune_by_relevance(
    node_embeddings: Dict[str, np.ndarray],
    query_embedding: np.ndarray,
    keep_ratio: float = 0.5,
) -> List[str]:
    """Prune nodes by relevance, keeping top-k by cosine similarity.

    Args:
        node_embeddings: Map of node_id -> embedding vector
        query_embedding: Query embedding vector
        keep_ratio: Fraction of nodes to keep (0.0-1.0)

    Returns:
        List of kept node_ids, ordered by relevance (highest first)
    """
    if not node_embeddings:
        return []

    scores = score_nodes_by_relevance(node_embeddings, query_embedding)
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    k = max(1, math.ceil(len(sorted_nodes) * keep_ratio))
    return [node_id for node_id, _ in sorted_nodes[:k]]
