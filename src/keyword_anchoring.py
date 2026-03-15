"""
Keyword anchoring for compression.

Based on SCOPE (ACL 2025) — allows users to specify must-retain keywords
that are guaranteed to survive compression, regardless of importance scores.
"""

from typing import List


def apply_keyword_anchoring(
    nodes: List[dict],
    anchored_keywords: List[str],
    keep_ratio: float = 0.5,
) -> List[dict]:
    """Apply keyword anchoring to node selection.

    Nodes containing anchored keywords are always kept. Remaining budget
    is filled by highest-importance non-anchored nodes.

    Args:
        nodes: List of dicts with node_id, text, importance
        anchored_keywords: Keywords that must appear in output
        keep_ratio: Fraction of nodes to keep (0.0-1.0)

    Returns:
        List of kept nodes (anchored first, then by importance)
    """
    if not nodes:
        return []

    k = max(1, int(len(nodes) * keep_ratio))

    # Partition into anchored and non-anchored
    anchored = []
    non_anchored = []
    keywords_lower = [kw.lower() for kw in anchored_keywords]

    for node in nodes:
        text_lower = node["text"].lower()
        if any(kw in text_lower for kw in keywords_lower):
            anchored.append(node)
        else:
            non_anchored.append(node)

    # Sort non-anchored by importance
    non_anchored.sort(key=lambda n: n.get("importance", 0), reverse=True)

    # Fill remaining slots
    remaining_slots = max(0, k - len(anchored))
    result = anchored + non_anchored[:remaining_slots]

    return result
