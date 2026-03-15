"""
Intra-document redundancy detection and collapse.

Based on R-KV (arXiv:2505.24133, 2025) — extends cross-document duplicate
detection to find and collapse near-redundant passages WITHIN a single
document, reducing token waste from repetitive content.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


def find_intra_duplicates(
    nodes: Dict[str, dict],
    threshold: float = 0.9,
) -> List[Dict]:
    """Find near-duplicate node pairs within the same document.

    Args:
        nodes: Map of node_id -> {"text": str, "embedding": np.ndarray}
        threshold: Cosine similarity threshold for duplicate detection

    Returns:
        List of duplicate pairs with similarity scores
    """
    if len(nodes) < 2:
        return []

    node_list = list(nodes.items())
    duplicates = []

    for i in range(len(node_list)):
        nid_a, data_a = node_list[i]
        emb_a = data_a["embedding"]
        norm_a = np.linalg.norm(emb_a)
        if norm_a == 0:
            continue

        for j in range(i + 1, len(node_list)):
            nid_b, data_b = node_list[j]
            emb_b = data_b["embedding"]
            norm_b = np.linalg.norm(emb_b)
            if norm_b == 0:
                continue

            sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            if sim >= threshold:
                duplicates.append({
                    "node_a": nid_a,
                    "node_b": nid_b,
                    "similarity": round(sim, 4),
                })

    return duplicates


def collapse_redundant_nodes(
    nodes: Dict[str, dict],
    threshold: float = 0.9,
) -> Dict[str, dict]:
    """Collapse near-duplicate nodes into representative nodes with counts.

    The representative node (first encountered) gets an occurrence_count
    field showing how many nodes it represents.

    Args:
        nodes: Map of node_id -> {"text": str, "embedding": np.ndarray}
        threshold: Cosine similarity threshold

    Returns:
        Collapsed node map with fewer entries
    """
    if len(nodes) < 2:
        return dict(nodes)

    duplicates = find_intra_duplicates(nodes, threshold)

    # Build union-find for grouping
    parent = {nid: nid for nid in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for dupe in duplicates:
        union(dupe["node_a"], dupe["node_b"])

    # Group by representative
    groups: Dict[str, List[str]] = {}
    for nid in nodes:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    # Build collapsed output
    collapsed = {}
    for representative, members in groups.items():
        node_data = dict(nodes[representative])
        node_data["occurrence_count"] = len(members)
        if len(members) > 1:
            node_data["collapsed_from"] = members
        collapsed[representative] = node_data

    return collapsed
