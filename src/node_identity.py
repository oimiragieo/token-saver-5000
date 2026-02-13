"""
Node identity helpers shared across server and handlers.

Node IDs in this project currently use two patterns:
- Text nodes: "{file_id}_n{index}" (example: "paper_n12")
- Code nodes: "{file_id}::{symbol}" (example: "main.py::parse_args")
"""

from typing import Iterable, Set


def extract_file_id_from_node(node_id: str) -> str:
    """
    Extract file_id from a node ID, handling both text and code patterns.

    Args:
        node_id: Node identifier in either text or code format.

    Returns:
        The extracted file identifier.
    """
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    if "_n" in node_id:
        return node_id.rsplit("_n", 1)[0]
    return node_id


def collect_file_ids(node_ids: Iterable[str]) -> Set[str]:
    """
    Collect unique file IDs from an iterable of node IDs.

    Args:
        node_ids: Iterable of node identifiers.

    Returns:
        Set of unique file IDs.
    """
    return {extract_file_id_from_node(node_id) for node_id in node_ids}
