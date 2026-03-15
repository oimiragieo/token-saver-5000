"""
Multi-level skeleton output.

Based on Squeezed Attention (ACL 2025) and Hierarchical RAG — returns
3 fidelity tiers in one call so clients can pick the depth they need:
- Headline (top 10% nodes): Ultra-compressed overview
- Summary (top 30% nodes): Balanced compression
- Full (100% nodes): Complete content
"""

import math
from typing import Dict, List


def generate_multi_level_skeleton(
    nodes: List[dict],
    headline_ratio: float = 0.10,
    summary_ratio: float = 0.30,
) -> Dict[str, dict]:
    """Generate 3-tier skeleton from importance-ranked nodes.

    Args:
        nodes: List of dicts with node_id, text, importance
        headline_ratio: Fraction of nodes for headline tier (default: 10%)
        summary_ratio: Fraction of nodes for summary tier (default: 30%)

    Returns:
        Dict with headline, summary, full tiers, each containing
        nodes list and concatenated text.
    """
    if not nodes:
        empty = {"nodes": [], "text": "", "node_count": 0}
        return {"headline": dict(empty), "summary": dict(empty), "full": dict(empty)}

    # Sort by importance descending
    sorted_nodes = sorted(nodes, key=lambda n: n.get("importance", 0), reverse=True)

    n = len(sorted_nodes)
    headline_k = max(1, math.ceil(n * headline_ratio))
    summary_k = max(headline_k, math.ceil(n * summary_ratio))

    headline_nodes = sorted_nodes[:headline_k]
    summary_nodes = sorted_nodes[:summary_k]
    full_nodes = sorted_nodes

    def build_tier(tier_nodes):
        text = " ".join(n["text"] for n in tier_nodes)
        return {
            "nodes": [n["node_id"] for n in tier_nodes],
            "text": text,
            "node_count": len(tier_nodes),
        }

    return {
        "headline": build_tier(headline_nodes),
        "summary": build_tier(summary_nodes),
        "full": build_tier(full_nodes),
    }
