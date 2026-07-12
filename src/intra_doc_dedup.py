"""
Intra-document redundancy detection and collapse.

Based on R-KV (arXiv:2505.24133, 2025) — extends cross-document duplicate
detection to find and collapse near-redundant passages WITHIN a single
document, reducing token waste from repetitive content.
"""

import os
from typing import Dict, List

import numpy as np

# #212 (edge-case hardening, adversarial-repetition case): the naive
# all-pairs scan below is O(N^2) with NO ceiling, unlike the sibling
# similarity-edge builders in semantic_compressor.py / code_compressor.py
# (``_MAX_GRAPH_CHUNKS``, default 2500). This function is invoked
# UNCONDITIONALLY on every ingest with >2 chunks (see
# ``SemanticCompressor.ingest_file`` step 2b), so a large adversarial or
# highly-repetitive document (thousands of near-identical chunks) could hang
# for minutes. Bound participation the same way: nodes beyond the cap are
# simply not compared (they remain un-collapsed, not lost -- consistent with
# the "beyond that index still exist but unconnected" contract used
# elsewhere). Overridable via MAX_DEDUP_NODES env var.
_MAX_DEDUP_NODES: int = int(os.environ.get("MAX_DEDUP_NODES", "1000"))


def find_intra_duplicates(
    nodes: Dict[str, dict],
    threshold: float = 0.9,
    max_nodes: int = _MAX_DEDUP_NODES,
) -> List[Dict]:
    """Find near-duplicate node pairs within the same document.

    Args:
        nodes: Map of node_id -> {"text": str, "embedding": np.ndarray}
        threshold: Cosine similarity threshold for duplicate detection
        max_nodes: Hard ceiling on how many nodes participate in the O(N^2)
            pairwise scan (bounds worst-case time/memory on a pathologically
            large or repetitive document). Nodes beyond this index are
            skipped, not dropped from the caller's node map.

    Returns:
        List of duplicate pairs with similarity scores
    """
    if len(nodes) < 2:
        return []

    node_list = list(nodes.items())[:max_nodes]

    # Precompute norms once (O(N)) instead of recomputing norm_b on every
    # (i, j) pair (the old code called np.linalg.norm(emb_b) inside the
    # inner loop -- O(N^2) redundant work for a value that only depends on j).
    norms = [float(np.linalg.norm(data["embedding"])) for _, data in node_list]

    duplicates = []

    for i in range(len(node_list)):
        nid_a, data_a = node_list[i]
        emb_a = data_a["embedding"]
        norm_a = norms[i]
        if norm_a == 0:
            continue

        for j in range(i + 1, len(node_list)):
            nid_b, data_b = node_list[j]
            norm_b = norms[j]
            if norm_b == 0:
                continue
            emb_b = data_b["embedding"]

            sim = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
            if sim >= threshold:
                duplicates.append(
                    {
                        "node_a": nid_a,
                        "node_b": nid_b,
                        "similarity": round(sim, 4),
                    }
                )

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
