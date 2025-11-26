"""
Graph Visualization Module for Token Saver 5000

Provides multiple visualization formats for semantic compression graphs:
- ASCII text rendering for terminal display
- JSON export for programmatic access
- HTML interactive visualization via pyvis
- GraphML export for graph analysis tools

Key Features:
- Highlights top-importance nodes (PageRank scores)
- Shows edge weights (semantic similarity)
- Explains compression decisions (why nodes were kept/dropped)
- Supports customizable layouts and filtering
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for graph visualization."""

    max_nodes: int = 50  # Limit nodes for readability
    min_importance: float = 0.0  # Filter low-importance nodes
    show_edge_weights: bool = True
    highlight_skeleton: bool = True  # Highlight nodes in skeleton
    layout: str = "spring"  # spring, circular, hierarchical


class GraphVisualizer:
    """
    Visualizes semantic compression graphs in multiple formats.

    Supports ASCII rendering, JSON/GraphML export, and interactive HTML.
    """

    def __init__(self, compressor):
        """
        Initialize the graph visualizer.

        Args:
            compressor: SemanticCompressor instance with graphs and chunks
        """
        self.compressor = compressor
        self.config = VisualizationConfig()

    def render_ascii(self, file_id: str, config: Optional[VisualizationConfig] = None) -> str:
        """
        Render semantic graph as ASCII text for terminal display.

        Args:
            file_id: Document ID to visualize
            config: Optional visualization configuration

        Returns:
            ASCII representation of the graph

        Example output:
            ```
            Semantic Graph: quantum_paper (10 nodes, 15 edges)

            Top Nodes by Importance:
            ★ quantum_paper_n0 [0.18] "Quantum computing uses qubits..."
            ★ quantum_paper_n3 [0.14] "Superposition enables parallel..."
              quantum_paper_n1 [0.09] "Entanglement creates correlations..."

            Edge connections:
            n0 --[0.82]--> n3
            n0 --[0.75]--> n1
            n3 --[0.79]--> n5
            ```
        """
        config = config or self.config

        # Get graph and validate
        if file_id not in self.compressor.graphs:
            raise ValueError(f"No graph found for file_id: {file_id}")

        graph = self.compressor.graphs[file_id]
        chunks = self.compressor.chunks

        # Filter nodes by importance
        nodes_with_importance = [
            (node_id, chunks[node_id].importance)
            for node_id in graph.nodes
            if node_id in chunks and chunks[node_id].importance >= config.min_importance
        ]
        nodes_with_importance.sort(key=lambda x: x[1], reverse=True)

        # Limit nodes for readability
        top_nodes = nodes_with_importance[: config.max_nodes]

        # Build ASCII output
        lines = []
        lines.append(
            f"Semantic Graph: {file_id} ({len(graph.nodes)} nodes, {len(graph.edges)} edges)\n"
        )

        lines.append("Top Nodes by Importance:")
        for node_id, importance in top_nodes:
            chunk = chunks[node_id]
            preview = chunk.text[:50] + "..." if len(chunk.text) > 50 else chunk.text
            preview = preview.replace("\n", " ")  # Single line

            # Highlight skeleton nodes with star
            marker = "★" if config.highlight_skeleton else " "
            lines.append(f'{marker} {node_id} [{importance:.2f}] "{preview}"')

        # Show edge connections for top nodes
        lines.append("\nEdge Connections (top nodes):")
        top_node_ids = {node_id for node_id, _ in top_nodes[:10]}  # Top 10 for edges

        edge_count = 0
        for u, v, data in graph.edges(data=True):
            if u in top_node_ids or v in top_node_ids:
                weight = data.get("weight", 0.0)
                u_short = u.split("_")[-1]  # n0, n1, etc.
                v_short = v.split("_")[-1]

                if config.show_edge_weights:
                    lines.append(f"  {u_short} --[{weight:.2f}]--> {v_short}")
                else:
                    lines.append(f"  {u_short} --> {v_short}")

                edge_count += 1
                if edge_count >= 20:  # Limit edges for readability
                    lines.append(f"  ... ({len(graph.edges) - 20} more edges)")
                    break

        return "\n".join(lines)

    def export_json(self, file_id: str, config: Optional[VisualizationConfig] = None) -> str:
        """
        Export semantic graph as JSON for programmatic access.

        Args:
            file_id: Document ID to export
            config: Optional visualization configuration

        Returns:
            JSON string representation of the graph

        Format:
            ```json
            {
              "file_id": "quantum_paper",
              "nodes": [
                {
                  "id": "quantum_paper_n0",
                  "text": "Quantum computing...",
                  "importance": 0.18,
                  "tokens": 45,
                  "position": 0
                }
              ],
              "edges": [
                {"source": "quantum_paper_n0", "target": "quantum_paper_n3", "weight": 0.82}
              ],
              "stats": {
                "total_nodes": 10,
                "total_edges": 15,
                "avg_importance": 0.10
              }
            }
            ```
        """
        config = config or self.config

        # Get graph and validate
        if file_id not in self.compressor.graphs:
            raise ValueError(f"No graph found for file_id: {file_id}")

        graph = self.compressor.graphs[file_id]
        chunks = self.compressor.chunks

        # Build nodes list
        nodes = []
        for node_id in graph.nodes:
            if node_id not in chunks:
                continue

            chunk = chunks[node_id]
            if chunk.importance < config.min_importance:
                continue

            nodes.append(
                {
                    "id": node_id,
                    "text": chunk.text,
                    "importance": float(chunk.importance),
                    "tokens": chunk.metadata.get("tokens", 0),
                    "position": chunk.metadata.get("position", 0),
                    "entities": chunk.metadata.get("entities", []),
                }
            )

        # Sort by importance
        nodes.sort(key=lambda n: n["importance"], reverse=True)

        # Limit nodes
        nodes = nodes[: config.max_nodes]
        node_ids = {n["id"] for n in nodes}

        # Build edges list (only for included nodes)
        edges = []
        for u, v, data in graph.edges(data=True):
            if u in node_ids and v in node_ids:
                edges.append(
                    {
                        "source": u,
                        "target": v,
                        "weight": float(data.get("weight", 0.0)),
                    }
                )

        # Calculate stats
        importance_values = [n["importance"] for n in nodes]
        avg_importance = (
            sum(importance_values) / len(importance_values) if importance_values else 0.0
        )

        result = {
            "file_id": file_id,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "avg_importance": round(avg_importance, 4),
                "max_importance": max(importance_values) if importance_values else 0.0,
                "min_importance": min(importance_values) if importance_values else 0.0,
            },
        }

        return json.dumps(result, indent=2)

    def export_graphml(self, file_id: str, output_path: str) -> str:
        """
        Export semantic graph as GraphML for analysis tools (Gephi, Cytoscape).

        Args:
            file_id: Document ID to export
            output_path: Path to save GraphML file

        Returns:
            Success message with file path

        GraphML format is a standard XML-based graph format supported by:
        - Gephi (graph visualization and exploration)
        - Cytoscape (network analysis)
        - igraph, NetworkX (programmatic analysis)
        """
        # Get graph and validate
        if file_id not in self.compressor.graphs:
            raise ValueError(f"No graph found for file_id: {file_id}")

        graph = self.compressor.graphs[file_id]
        chunks = self.compressor.chunks

        # Create annotated graph with node data
        annotated_graph = nx.Graph()

        for node_id in graph.nodes:
            if node_id in chunks:
                chunk = chunks[node_id]
                annotated_graph.add_node(
                    node_id,
                    text=chunk.text[:200],  # Truncate for file size
                    importance=float(chunk.importance),
                    tokens=chunk.metadata.get("tokens", 0),
                    position=chunk.metadata.get("position", 0),
                )

        # Copy edges
        for u, v, data in graph.edges(data=True):
            annotated_graph.add_edge(u, v, weight=float(data.get("weight", 0.0)))

        # Write GraphML
        nx.write_graphml(annotated_graph, output_path)

        return f"Exported graph to {output_path} ({len(annotated_graph.nodes)} nodes, {len(annotated_graph.edges)} edges)"

    def visualize_html(
        self, file_id: str, output_path: str, config: Optional[VisualizationConfig] = None
    ) -> str:
        """
        Generate interactive HTML visualization using pyvis.

        Args:
            file_id: Document ID to visualize
            output_path: Path to save HTML file
            config: Optional visualization configuration

        Returns:
            Success message with file path

        Creates an interactive HTML page with:
        - Draggable nodes
        - Zoom and pan
        - Node tooltips with text preview
        - Color-coded by importance
        - Edge thickness by similarity
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for HTML visualization. Install with: pip install pyvis"
            )

        config = config or self.config

        # Get graph and validate
        if file_id not in self.compressor.graphs:
            raise ValueError(f"No graph found for file_id: {file_id}")

        graph = self.compressor.graphs[file_id]
        chunks = self.compressor.chunks

        # Create pyvis network
        net = Network(
            height="750px", width="100%", notebook=False, heading=f"Semantic Graph: {file_id}"
        )
        net.barnes_hut()  # Better layout algorithm

        # Filter and add nodes
        nodes_with_importance = [
            (node_id, chunks[node_id].importance)
            for node_id in graph.nodes
            if node_id in chunks and chunks[node_id].importance >= config.min_importance
        ]
        nodes_with_importance.sort(key=lambda x: x[1], reverse=True)
        top_nodes = nodes_with_importance[: config.max_nodes]
        node_ids = {node_id for node_id, _ in top_nodes}

        for node_id, importance in top_nodes:
            chunk = chunks[node_id]
            preview = chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text

            # Color by importance (green = high, red = low)
            importance_normalized = min(importance * 5, 1.0)  # Scale up for visibility
            color = f"rgba({int(255 * (1 - importance_normalized))}, {int(255 * importance_normalized)}, 0, 0.8)"

            # Size by importance
            size = 10 + (importance * 100)

            net.add_node(
                node_id,
                label=node_id.split("_")[-1],  # n0, n1, etc.
                title=f"{node_id}\nImportance: {importance:.3f}\n\n{preview}",
                color=color,
                size=size,
            )

        # Add edges
        for u, v, data in graph.edges(data=True):
            if u in node_ids and v in node_ids:
                weight = data.get("weight", 0.0)
                # Edge thickness by weight
                width = 1 + (weight * 5)
                net.add_edge(u, v, value=width, title=f"Similarity: {weight:.2f}")

        # Save HTML
        net.save_graph(output_path)

        return f"Generated interactive visualization: {output_path} ({len(node_ids)} nodes)"

    def explain_compression_decision(self, file_id: str, node_id: str) -> str:
        """
        Explain why a specific node was kept or dropped in compression.

        Args:
            file_id: Document ID
            node_id: Specific node to explain

        Returns:
            Human-readable explanation of compression decision

        Example:
            ```
            Node: quantum_paper_n3
            Status: ✓ KEPT (in skeleton)

            Reasons:
            - High importance score (0.14, top 20%)
            - Highly connected (5 edges above threshold)
            - Central position in semantic cluster
            - Contains key entities: [superposition, entanglement]

            Connected to:
            - quantum_paper_n0 (0.82 similarity) - central concept
            - quantum_paper_n5 (0.79 similarity) - supporting detail
            ```
        """
        # Get graph and validate
        if file_id not in self.compressor.graphs:
            raise ValueError(f"No graph found for file_id: {file_id}")

        graph = self.compressor.graphs[file_id]
        chunks = self.compressor.chunks

        if node_id not in chunks:
            raise ValueError(f"Node {node_id} not found in chunks")

        chunk = chunks[node_id]

        # Determine if in skeleton
        all_importance = sorted(
            [chunks[nid].importance for nid in graph.nodes if nid in chunks],
            reverse=True,
        )
        skeleton_count = max(1, int(len(all_importance) * self.compressor.skeleton_ratio))
        skeleton_threshold = (
            all_importance[skeleton_count - 1] if skeleton_count <= len(all_importance) else 0.0
        )

        in_skeleton = chunk.importance >= skeleton_threshold

        # Build explanation
        lines = []
        lines.append(f"Node: {node_id}")
        status_marker = "✓ KEPT (in skeleton)" if in_skeleton else "✗ DROPPED (below threshold)"
        lines.append(f"Status: {status_marker}\n")

        lines.append("Reasons:")

        # Importance ranking
        rank = sum(1 for imp in all_importance if imp > chunk.importance) + 1
        percentile = (1 - (rank / len(all_importance))) * 100
        lines.append(
            f"- Importance score: {chunk.importance:.3f} (rank {rank}/{len(all_importance)}, top {percentile:.0f}%)"
        )

        # Connectivity
        edges = list(graph.edges(node_id, data=True))
        high_weight_edges = [
            e for e in edges if e[2].get("weight", 0) > self.compressor.similarity_threshold
        ]
        lines.append(f"- Connectivity: {len(high_weight_edges)} strong edges ({len(edges)} total)")

        # Entities
        entities = chunk.metadata.get("entities", [])
        if entities:
            lines.append(f"- Key entities: {entities}")

        # Token count
        tokens = chunk.metadata.get("tokens", 0)
        lines.append(f"- Token count: {tokens}")

        # Connected nodes
        if edges:
            lines.append("\nTop Connected Nodes:")
            sorted_edges = sorted(edges, key=lambda e: e[2].get("weight", 0), reverse=True)
            for u, v, data in sorted_edges[:5]:
                other = v if u == node_id else u
                weight = data.get("weight", 0.0)
                other_importance = chunks[other].importance if other in chunks else 0.0
                lines.append(
                    f"  - {other} (similarity: {weight:.2f}, importance: {other_importance:.3f})"
                )

        return "\n".join(lines)
