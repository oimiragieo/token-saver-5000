"""
Semantic SSIM (Structural Similarity Index for Semantic Content)

Inspired by FPQE paper (arXiv:2511.15695v1):
"Structure preservation metrics (SSIM) better predict performance than
compression-focused metrics (MSE/PSNR)."

This module implements SSIM for semantic graphs, measuring how well
a compressed skeleton preserves the original document's semantic structure.

Key Insight from FPQE:
- Visual SSIM has 3 components: luminance, contrast, structure
- Semantic SSIM adapts these for text/graphs:
  * Luminance → Average importance (information density)
  * Contrast → Variance in importance (dynamic range)
  * Structure → Graph connectivity (relational structure)
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSSIM:
    """
    Semantic Structural Similarity Index

    Measures how well a compressed skeleton preserves semantic structure
    compared to the original document graph.

    Based on SSIM formula:
    SSIM(x,y) = l(x,y)^α · c(x,y)^β · s(x,y)^γ

    Where:
    - l = luminance (mean importance)
    - c = contrast (variance in importance)
    - s = structure (connectivity preservation)
    - α, β, γ = component weights (default: 0.33, 0.33, 0.34)
    """

    def __init__(
        self,
        alpha: float = 0.33,
        beta: float = 0.33,
        gamma: float = 0.34,
        c1: float = 0.01,
        c2: float = 0.01,
    ):
        """
        Initialize Semantic SSIM calculator

        Args:
            alpha: Weight for luminance component
            beta: Weight for contrast component
            gamma: Weight for structure component
            c1, c2: Stability constants (prevent division by zero)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2

    def calculate_luminance(
        self, original_importance: np.ndarray, compressed_importance: np.ndarray
    ) -> float:
        """
        Luminance: Average importance comparison

        Measures if the compressed skeleton maintains similar
        "information density" to the original.

        Formula:
        l(x,y) = (2·μ_x·μ_y + c1) / (μ_x² + μ_y² + c1)

        Returns:
            float: Luminance similarity [0, 1]
        """
        mu_orig = np.mean(original_importance)
        mu_comp = np.mean(compressed_importance)

        numerator = 2 * mu_orig * mu_comp + self.c1
        denominator = mu_orig**2 + mu_comp**2 + self.c1

        return numerator / denominator

    def calculate_contrast(
        self, original_importance: np.ndarray, compressed_importance: np.ndarray
    ) -> float:
        """
        Contrast: Variance in importance comparison

        Measures if the compressed skeleton maintains similar
        "dynamic range" of importance scores.

        Formula:
        c(x,y) = (2·σ_x·σ_y + c2) / (σ_x² + σ_y² + c2)

        Returns:
            float: Contrast similarity [0, 1]
        """
        sigma_orig = np.std(original_importance)
        sigma_comp = np.std(compressed_importance)

        numerator = 2 * sigma_orig * sigma_comp + self.c2
        denominator = sigma_orig**2 + sigma_comp**2 + self.c2

        return numerator / denominator

    def calculate_structure(
        self, graph: nx.Graph, original_nodes: List[str], compressed_nodes: List[str]
    ) -> float:
        """
        Structure: Graph connectivity preservation

        Measures what fraction of the original graph's relationships
        are preserved in the compressed skeleton.

        Three sub-metrics:
        1. Edge preservation: Fraction of edges retained
        2. Community preservation: Clustering coefficient similarity
        3. Centrality preservation: PageRank correlation

        Returns:
            float: Structure similarity [0, 1]
        """
        # Sub-metric 1: Edge preservation
        total_edges = graph.number_of_edges()
        if total_edges == 0:
            edge_preservation = 1.0
        else:
            preserved_edges = sum(
                1 for u, v in graph.edges() if u in compressed_nodes and v in compressed_nodes
            )
            edge_preservation = preserved_edges / total_edges

        # Sub-metric 2: Community structure preservation
        try:
            # Original clustering coefficient
            original_clustering = nx.average_clustering(graph)

            # Compressed subgraph clustering
            compressed_subgraph = graph.subgraph(compressed_nodes)
            compressed_clustering = nx.average_clustering(compressed_subgraph)

            # Similarity
            if original_clustering == 0 and compressed_clustering == 0:
                clustering_similarity = 1.0
            else:
                clustering_similarity = min(
                    compressed_clustering / (original_clustering + 1e-10), 1.0
                )
        except Exception:
            clustering_similarity = 0.5  # Fallback

        # Sub-metric 3: Centrality preservation (PageRank correlation)
        try:
            original_pagerank = nx.pagerank(graph)
            compressed_pagerank = nx.pagerank(compressed_subgraph)

            # Get PageRank values for compressed nodes
            original_pr = [original_pagerank.get(n, 0) for n in compressed_nodes]
            compressed_pr = [compressed_pagerank.get(n, 0) for n in compressed_nodes]

            # Pearson correlation
            if len(original_pr) > 1:
                correlation = np.corrcoef(original_pr, compressed_pr)[0, 1]
                centrality_preservation = max(correlation, 0)  # Clip negative correlations
            else:
                centrality_preservation = 1.0
        except Exception:
            centrality_preservation = 0.5  # Fallback

        # Combine sub-metrics
        structure = (
            0.5 * edge_preservation + 0.25 * clustering_similarity + 0.25 * centrality_preservation
        )

        return structure

    def calculate_ssim(
        self,
        graph: nx.Graph,
        original_nodes: List[str],
        compressed_nodes: List[str],
        importance_key: str = "importance",
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Semantic SSIM

        Args:
            graph: Original semantic graph
            original_nodes: All nodes in original document
            compressed_nodes: Nodes retained in skeleton
            importance_key: Node attribute containing importance scores

        Returns:
            ssim: Overall SSIM score [0, 1]
            components: Dict with individual component scores
        """
        if len(compressed_nodes) == 0:
            return 0.0, {
                "luminance": 0.0,
                "contrast": 0.0,
                "structure": 0.0,
                "ssim": 0.0,
            }

        # Extract importance scores
        original_importance = np.array(
            [graph.nodes[n].get(importance_key, 0.5) for n in original_nodes]
        )

        compressed_importance = np.array(
            [graph.nodes[n].get(importance_key, 0.5) for n in compressed_nodes]
        )

        # Calculate components
        luminance = self.calculate_luminance(original_importance, compressed_importance)

        contrast = self.calculate_contrast(original_importance, compressed_importance)

        structure = self.calculate_structure(graph, original_nodes, compressed_nodes)

        # Combine using weighted geometric mean (like visual SSIM)
        ssim = (luminance**self.alpha) * (contrast**self.beta) * (structure**self.gamma)

        # Ensure [0, 1] range
        ssim = min(max(ssim, 0.0), 1.0)

        components = {
            "luminance": luminance,
            "contrast": contrast,
            "structure": structure,
            "ssim": ssim,
        }

        return ssim, components

    def calculate_embedding_ssim(
        self,
        original_embeddings: np.ndarray,
        compressed_embeddings: np.ndarray,
        original_importance: Optional[np.ndarray] = None,
        compressed_importance: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Alternative SSIM calculation using embeddings directly

        Useful when you have embeddings but not a full graph.

        Args:
            original_embeddings: (N, D) array of original node embeddings
            compressed_embeddings: (M, D) array of compressed node embeddings
            original_importance: Optional importance scores
            compressed_importance: Optional importance scores

        Returns:
            ssim: Overall SSIM score [0, 1]
            components: Dict with individual component scores
        """
        # If no importance provided, use uniform
        if original_importance is None:
            original_importance = np.ones(len(original_embeddings))
        if compressed_importance is None:
            compressed_importance = np.ones(len(compressed_embeddings))

        # Luminance and Contrast
        luminance = self.calculate_luminance(original_importance, compressed_importance)

        contrast = self.calculate_contrast(original_importance, compressed_importance)

        # Structure: Use embedding similarity instead of graph
        # Calculate pairwise similarity in original space
        orig_sim = cosine_similarity(original_embeddings)
        orig_connectivity = (orig_sim > 0.7).sum() / (len(original_embeddings) ** 2)

        # Calculate pairwise similarity in compressed space
        comp_sim = cosine_similarity(compressed_embeddings)
        comp_connectivity = (comp_sim > 0.7).sum() / (len(compressed_embeddings) ** 2)

        # Structure preservation
        structure = min(comp_connectivity / (orig_connectivity + 1e-10), 1.0)

        # Combine
        ssim = (luminance**self.alpha) * (contrast**self.beta) * (structure**self.gamma)

        ssim = min(max(ssim, 0.0), 1.0)

        components = {
            "luminance": luminance,
            "contrast": contrast,
            "structure": structure,
            "ssim": ssim,
        }

        return ssim, components


def interpret_ssim_score(ssim: float) -> str:
    """
    Interpret SSIM score with actionable guidance

    Based on FPQE paper findings:
    - SSIM > 0.9: Excellent preservation
    - SSIM 0.7-0.9: Good preservation
    - SSIM 0.5-0.7: Acceptable (some structure lost)
    - SSIM < 0.5: Poor preservation (consider lower compression)

    Args:
        ssim: SSIM score [0, 1]

    Returns:
        Interpretation string
    """
    if ssim >= 0.9:
        return "✅ Excellent - Structure fully preserved"
    elif ssim >= 0.7:
        return "✅ Good - Minor structural degradation"
    elif ssim >= 0.5:
        return "⚠️  Acceptable - Noticeable structure loss"
    else:
        return "❌ Poor - Significant structure loss, reduce compression"


# Example usage
if __name__ == "__main__":
    """Demonstrate Semantic SSIM on a toy example"""

    # Create a simple semantic graph
    graph = nx.Graph()

    # Add 10 nodes with varying importance
    nodes = [f"node_{i}" for i in range(10)]
    for i, node in enumerate(nodes):
        importance = 0.1 + (i / 10) * 0.9  # Linearly increasing importance
        graph.add_node(node, importance=importance)

    # Add edges (create some structure)
    edges = [
        ("node_0", "node_1"),
        ("node_1", "node_2"),
        ("node_2", "node_3"),
        ("node_0", "node_5"),
        ("node_5", "node_7"),
        ("node_7", "node_9"),
    ]
    for u, v in edges:
        graph.add_edge(u, v)

    # Original: All nodes
    original_nodes = nodes

    # Scenario 1: Keep top 50% by importance
    compressed_nodes_50 = nodes[5:]  # Top 50%

    # Scenario 2: Keep top 30% by importance
    compressed_nodes_30 = nodes[7:]  # Top 30%

    # Calculate SSIM
    ssim_calculator = SemanticSSIM()

    ssim_50, components_50 = ssim_calculator.calculate_ssim(
        graph, original_nodes, compressed_nodes_50
    )

    ssim_30, components_30 = ssim_calculator.calculate_ssim(
        graph, original_nodes, compressed_nodes_30
    )

    print("=" * 70)
    print("Semantic SSIM Demonstration")
    print("=" * 70)
    print(f"\nOriginal graph: {len(nodes)} nodes, {len(edges)} edges")

    print("\n--- Scenario 1: 50% Compression ---")
    print(f"Retained nodes: {len(compressed_nodes_50)}")
    print(f"Luminance: {components_50['luminance']:.3f}")
    print(f"Contrast:  {components_50['contrast']:.3f}")
    print(f"Structure: {components_50['structure']:.3f}")
    print(f"SSIM:      {ssim_50:.3f}")
    print(f"Quality:   {interpret_ssim_score(ssim_50)}")

    print("\n--- Scenario 2: 70% Compression ---")
    print(f"Retained nodes: {len(compressed_nodes_30)}")
    print(f"Luminance: {components_30['luminance']:.3f}")
    print(f"Contrast:  {components_30['contrast']:.3f}")
    print(f"Structure: {components_30['structure']:.3f}")
    print(f"SSIM:      {ssim_30:.3f}")
    print(f"Quality:   {interpret_ssim_score(ssim_30)}")

    print("\n💡 Insight: Higher compression → Lower SSIM (as expected)")
    print("   FPQE paper shows SSIM predicts downstream task performance!")
