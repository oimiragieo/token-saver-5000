"""
Adaptive Rate Allocator

Inspired by JSCCM paper (arXiv:2511.15699v1):
- Dynamically adjust skeleton ratio based on document complexity
- Use Gumbel-Softmax for differentiable cutoff position selection
- Adapt to "channel conditions" (context window availability)

This replaces fixed skeleton_ratio with learned adaptive allocation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Tuple, Dict


class AdaptiveRateAllocator(nn.Module):
    """
    Adaptively determine skeleton ratio based on:
    1. Document semantic complexity
    2. Available context window ("channel SNR")
    3. Query requirements

    Inspired by JSCCM's rate allocator (Section IV-C)
    """

    def __init__(
        self,
        num_rate_levels: int = 5,
        temperature: float = 1.5,
    ):
        """
        Args:
            num_rate_levels: Number of discrete skeleton ratios to choose from
            temperature: Gumbel-Softmax temperature for smooth approximation
        """
        super().__init__()

        self.num_rate_levels = num_rate_levels
        self.temperature = temperature

        # Possible skeleton ratios (like JSCCM's rate levels)
        # [0.10, 0.15, 0.20, 0.25, 0.30]
        self.rate_levels = nn.Parameter(
            torch.linspace(0.10, 0.30, num_rate_levels), requires_grad=False
        )

        # MLP to predict optimal rate level
        # Input: [complexity_score, context_window_availability, ...]
        self.rate_predictor = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_rate_levels),  # Logits for each rate level
        )

    def calculate_complexity_score(self, graph: nx.Graph) -> float:
        """
        Calculate semantic complexity of document

        Higher complexity → more structure → need higher skeleton ratio

        Metrics:
        - Graph density
        - Clustering coefficient
        - Entropy of importance distribution
        """
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()

        if n_nodes <= 1:
            return 0.0

        # Graph density
        max_edges = n_nodes * (n_nodes - 1) / 2
        density = n_edges / max_edges if max_edges > 0 else 0

        # Average clustering coefficient
        clustering = nx.average_clustering(graph) if n_nodes > 2 else 0

        # PageRank entropy (measure of importance distribution)
        pagerank = nx.pagerank(graph)
        pr_values = np.array(list(pagerank.values()))
        pr_values = pr_values / pr_values.sum()  # Normalize
        entropy = -np.sum(pr_values * np.log(pr_values + 1e-10))
        entropy_normalized = entropy / np.log(n_nodes)  # Normalize by max entropy

        # Combine metrics (higher = more complex)
        complexity = 0.3 * density + 0.3 * clustering + 0.4 * entropy_normalized

        return complexity

    def gumbel_softmax_rate_selection(
        self, logits: torch.Tensor, hard: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """
        Gumbel-Softmax for differentiable rate level selection

        Like JSCCM paper Eq. (17), but for rate levels instead of constellation points

        Args:
            logits: [num_rate_levels] unnormalized log probabilities
            hard: If True, return one-hot in forward, soft in backward (STE)

        Returns:
            soft_selection: [num_rate_levels] soft probabilities
            selected_level: Integer index of selected level
        """
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Add noise to logits
        noisy_logits = (logits + gumbel_noise) / self.temperature

        # Soft selection (differentiable)
        soft_selection = F.softmax(noisy_logits, dim=-1)

        if hard:
            # Hard selection in forward pass (argmax)
            # Soft selection in backward pass (for gradients)
            selected_level = torch.argmax(soft_selection).item()

            # One-hot encoding
            hard_selection = torch.zeros_like(soft_selection)
            hard_selection[selected_level] = 1.0

            # Straight-through estimator
            # Forward: hard, Backward: soft
            selection = hard_selection - soft_selection.detach() + soft_selection
        else:
            selected_level = torch.argmax(soft_selection).item()
            selection = soft_selection

        return selection, selected_level

    def forward(
        self,
        graph: nx.Graph,
        available_context_tokens: int,
        max_context_tokens: int = 100000,
        query_priority: float = 0.5,
    ) -> Tuple[float, Dict]:
        """
        Determine optimal skeleton ratio

        Args:
            graph: Semantic graph of document
            available_context_tokens: How many tokens left in context window
            max_context_tokens: Maximum context window size
            query_priority: Priority of current query (0-1)

        Returns:
            skeleton_ratio: Selected ratio (0.1 - 0.3)
            diagnostics: Debug information
        """
        # 1. Calculate complexity
        complexity = self.calculate_complexity_score(graph)

        # 2. Calculate "channel SNR" (context window availability)
        context_availability = available_context_tokens / max_context_tokens

        # 3. Create feature vector
        features = torch.tensor(
            [complexity, context_availability, query_priority], dtype=torch.float32
        )

        # 4. Predict rate level
        logits = self.rate_predictor(features)

        # 5. Select rate using Gumbel-Softmax
        selection, selected_level = self.gumbel_softmax_rate_selection(logits, hard=True)

        # 6. Get skeleton ratio
        skeleton_ratio = self.rate_levels[selected_level].item()

        diagnostics = {
            "complexity": complexity,
            "context_availability": context_availability,
            "selected_level": selected_level,
            "skeleton_ratio": skeleton_ratio,
            "logits": logits.detach().numpy(),
            "selection_probs": F.softmax(logits, dim=-1).detach().numpy(),
        }

        return skeleton_ratio, diagnostics


class ContextWindowAdapter:
    """
    Adapt compression based on 'channel conditions'

    Analogous to JSCCM's Channel Adapter (Section IV-C, Fig. 5b)

    In JSCCM: Adapt to wireless channel SNR
    In our system: Adapt to context window availability
    """

    def __init__(self, compressor):
        self.compressor = compressor
        self.rate_allocator = AdaptiveRateAllocator()

    def adapt_to_context_window(
        self,
        file_id: str,
        available_tokens: int,
        max_tokens: int = 100000,
        query_priority: float = 0.5,
    ) -> str:
        """
        Generate skeleton adapted to context window availability

        Low availability (like low SNR) → More compression (lower skeleton ratio)
        High availability (like high SNR) → Less compression (higher skeleton ratio)

        Args:
            file_id: Document ID
            available_tokens: Remaining context window tokens
            max_tokens: Maximum context window size
            query_priority: How important is this query (0-1)

        Returns:
            Adapted skeleton text
        """
        graph = self.compressor.graphs.get(file_id)
        if not graph:
            raise ValueError(f"File {file_id} not found")

        # Determine optimal skeleton ratio using adaptive rate allocator
        skeleton_ratio, diagnostics = self.rate_allocator(
            graph=graph,
            available_context_tokens=available_tokens,
            max_context_tokens=max_tokens,
            query_priority=query_priority,
        )

        # Generate skeleton with adapted ratio
        # Temporarily override compressor's skeleton_ratio
        original_ratio = self.compressor.skeleton_ratio
        self.compressor.skeleton_ratio = skeleton_ratio

        skeleton_text = self.compressor.read_skeleton(file_id)

        # Restore original ratio
        self.compressor.skeleton_ratio = original_ratio

        # Add diagnostics to skeleton
        header = f"""
🔧 CONTEXT WINDOW ADAPTATION
Available tokens: {available_tokens:,} / {max_tokens:,} ({diagnostics['context_availability']:.1%})
Document complexity: {diagnostics['complexity']:.3f}
Selected skeleton ratio: {skeleton_ratio:.1%} (level {diagnostics['selected_level']})
Reason: {"High complexity + low availability → moderate compression" if skeleton_ratio > 0.2 else "Low complexity or high availability"}

---
"""

        return header + skeleton_text


class MultiLevelSemanticEncoder:
    """
    Two-branch architecture like JSCCM (Fig. 3)

    Main branch: Global semantic structure (high importance, always included)
    Auxiliary branch: Local details (lower importance, conditionally included)

    Like JSCCM's parallel JSCC encoders
    """

    def __init__(self, compressor):
        self.compressor = compressor

    def encode_multilevel(self, file_id: str, available_tokens: int):
        """
        Generate multi-level encoding

        Returns:
            {
                'main': High-importance anchor concepts (always include),
                'auxiliary': Medium-importance nodes (include if space allows),
                'detail': Low-importance nodes (only if plenty of space)
            }
        """
        graph = self.compressor.graphs[file_id]
        file_nodes = [
            (nid, self.compressor.chunks[nid]) for nid in graph.nodes() if nid.startswith(file_id)
        ]

        # Sort by importance
        file_nodes.sort(key=lambda x: x[1].importance, reverse=True)

        total_nodes = len(file_nodes)

        # Like JSCCM: Main branch gets 80%, Auxiliary gets 20%
        # But we split into 3 levels instead of 2
        main_count = int(total_nodes * 0.15)  # Top 15% - MUST include
        auxiliary_count = int(total_nodes * 0.25)  # Next 25% - include if space
        # Rest are details - only if plenty of space

        main_nodes = [nid for nid, _ in file_nodes[:main_count]]
        auxiliary_nodes = [nid for nid, _ in file_nodes[main_count : main_count + auxiliary_count]]
        detail_nodes = [nid for nid, _ in file_nodes[main_count + auxiliary_count :]]

        return {
            "main": main_nodes,
            "auxiliary": auxiliary_nodes,
            "detail": detail_nodes,
            "available_tokens": available_tokens,
        }

    def generate_adaptive_skeleton(self, file_id: str, available_tokens: int) -> str:
        """
        Generate skeleton by progressively adding levels based on available space

        Like JSCCM's rate allocation strategy (Section IV-C)
        """
        levels = self.encode_multilevel(file_id, available_tokens)

        # Always include main branch
        included_nodes = levels["main"]

        # Include auxiliary if we have space
        # Rough estimate: each node ~50 tokens
        main_token_estimate = len(levels["main"]) * 50

        if available_tokens - main_token_estimate > 2000:
            # Include auxiliary branch
            included_nodes.extend(levels["auxiliary"])

            auxiliary_token_estimate = len(levels["auxiliary"]) * 50

            if available_tokens - main_token_estimate - auxiliary_token_estimate > 5000:
                # Include details too
                included_nodes.extend(levels["detail"])

        # Generate skeleton with selected nodes
        skeleton_lines = []
        skeleton_lines.append(f"=== MULTI-LEVEL SEMANTIC SKELETON: {file_id} ===")
        skeleton_lines.append(f"Context budget: {available_tokens:,} tokens")
        skeleton_lines.append(
            f"Included: {len(included_nodes)} / {len(levels['main']) + len(levels['auxiliary']) + len(levels['detail'])} nodes"
        )
        skeleton_lines.append("")

        for node_id in included_nodes:
            node = self.compressor.chunks[node_id]

            # Mark level
            if node_id in levels["main"]:
                level = "MAIN"
                marker = "⭐⭐"
            elif node_id in levels["auxiliary"]:
                level = "AUX"
                marker = "⭐"
            else:
                level = "DETAIL"
                marker = "📦"

            summary = self.compressor._generate_summary(node.text, max_length=100)
            skeleton_lines.append(
                f"[{node_id}] {marker} {level} (importance: {node.importance:.3f})"
            )
            skeleton_lines.append(f"  {summary}\n")

        return "\n".join(skeleton_lines)


# Example usage
if __name__ == "__main__":
    """
    Demonstrate adaptive rate allocation inspired by JSCCM
    """
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.semantic_compressor import SemanticCompressor

    # Initialize compressor
    compressor = SemanticCompressor()

    # Initialize context window adapter
    adapter = ContextWindowAdapter(compressor)

    # Simulate document ingestion
    sample_doc = """
    Quantum computing represents a paradigm shift in computation.

    Unlike classical computers that use bits (0 or 1), quantum computers use qubits.
    Qubits can exist in superposition states, enabling parallel computation.

    However, quantum systems are fragile. Decoherence destroys quantum information.
    Error correction is essential for practical quantum computers.

    Various approaches exist: surface codes, topological codes, etc.
    Each has different trade-offs in terms of overhead and error thresholds.
    """

    compressor.ingest_file(sample_doc, "quantum_doc")

    print("=" * 70)
    print("SCENARIO 1: Plenty of context window (high 'SNR')")
    print("=" * 70)
    skeleton_high = adapter.adapt_to_context_window(
        file_id="quantum_doc",
        available_tokens=50000,  # Lots of space
        max_tokens=100000,
    )
    print(skeleton_high)

    print("\n" + "=" * 70)
    print("SCENARIO 2: Limited context window (low 'SNR')")
    print("=" * 70)
    skeleton_low = adapter.adapt_to_context_window(
        file_id="quantum_doc", available_tokens=5000, max_tokens=100000  # Limited space
    )
    print(skeleton_low)

    print("\n" + "=" * 70)
    print("SCENARIO 3: Multi-level encoding")
    print("=" * 70)
    multilevel = MultiLevelSemanticEncoder(compressor)
    skeleton_multilevel = multilevel.generate_adaptive_skeleton(
        file_id="quantum_doc", available_tokens=10000
    )
    print(skeleton_multilevel)
