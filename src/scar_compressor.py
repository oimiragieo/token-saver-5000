"""
SCAR-Inspired Learnable Semantic Compression

Adapts concepts from "Semantic Context Matters: Improving Conditioning for
Autoregressive Models" (arXiv:2511.14063v1) to text/document compression.

Key adaptations:
1. Compressed Semantic Prefilling (Section 3.2) → Learnable embedding compression
2. Semantic Alignment Guidance (Section 3.3) → Query-document alignment
3. Adaptive fidelity based on semantic similarity
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_UNTRAINED_WARNING_EMITTED = False
logger = logging.getLogger(__name__)


@dataclass
class CompressionStats:
    """Statistics from SCAR compression"""

    original_dim: int
    compressed_dim: int
    compression_ratio: float
    reconstruction_loss: float
    semantic_preservation: float  # 0-1, higher is better


class LearnableSemanticCompressor(nn.Module):
    """
    Learnable compression module inspired by SCAR's Pk(·) from Equation 3.

    SCAR compresses vision features (1024 → 256 tokens) while preserving semantics.
    We compress text embeddings (384D → 96D or similar) while preserving semantic information.

    Architecture (from SCAR Section 3.2):
    - Parallel downsampling: Convolution + Spatial reduction
    - Reconstruction objective: L2 loss for semantic preservation (Eq. 4)
    - At inference: Discard upsampling module
    """

    def __init__(
        self,
        input_dim: int = 384,  # sentence-transformers dimension
        compressed_dim: int = 96,  # 4× compression
        hidden_dim: int = 256,
    ):
        """
        Initialize learnable compressor.

        Args:
            input_dim: Input embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
            compressed_dim: Compressed embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.compression_ratio = input_dim / compressed_dim

        # Compression pathway (inspired by SCAR's Pk - Eq. 3)
        # Parallel branches: Conv-style downsampling + Residual downsampling
        self.compress_branch1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, compressed_dim),
        )

        self.compress_branch2 = nn.Sequential(
            nn.Linear(input_dim, compressed_dim),
        )

        # Reconstruction pathway (for training only, like SCAR's Uk - Eq. 4)
        # Discarded at inference
        self.reconstruct = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self, embeddings: torch.Tensor, return_reconstruction: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compress embeddings using parallel branches.

        SCAR Equation 3: Fc = Ck(Fs) + Rk(Fs)

        Args:
            embeddings: Input embeddings [batch_size, input_dim] or [num_nodes, input_dim]
            return_reconstruction: If True, return reconstructed embeddings for loss

        Returns:
            compressed: Compressed embeddings [batch_size, compressed_dim]
            reconstructed: (Optional) Reconstructed embeddings for loss computation
        """
        # Parallel compression (Eq. 3)
        branch1 = self.compress_branch1(embeddings)  # Ck(Fs)
        branch2 = self.compress_branch2(embeddings)  # Rk(Fs)
        compressed = branch1 + branch2  # Fc = Ck(Fs) + Rk(Fs)

        if return_reconstruction:
            # Reconstruct for semantic preservation loss (Eq. 4)
            reconstructed = self.reconstruct(compressed)  # Uk(Fc)
            return compressed, reconstructed

        return compressed, None

    def compute_preservation_loss(
        self, original: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """
        Semantic preservation loss from SCAR Equation 4:
        L_pres = ||Fs - Uk(Fc)||^2

        This ensures the compressor preserves critical semantic information.

        Args:
            original: Original embeddings [N, input_dim]
            reconstructed: Reconstructed embeddings [N, input_dim]

        Returns:
            L2 reconstruction loss
        """
        return F.mse_loss(reconstructed, original)

    def compress_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compress a batch of embeddings (numpy interface for integration).

        Args:
            embeddings: Input embeddings [N, input_dim] as numpy array

        Returns:
            Compressed embeddings [N, compressed_dim] as numpy array
        """
        self.eval()
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(embeddings).float()
            compressed, _ = self.forward(embeddings_tensor, return_reconstruction=False)
            return compressed.numpy()


class SemanticAlignmentModule(nn.Module):
    """
    Semantic Alignment Guidance inspired by SCAR Section 3.3.

    SCAR aligns AR model's hidden states with target image semantics (Eq. 8):
    L_align = ||Hs - Pt||^2

    We adapt this for text: align retrieved document nodes with query semantics.
    This provides dense, in-context guidance for better retrieval.
    """

    def __init__(self, embedding_dim: int = 384):
        """
        Initialize semantic alignment module.

        Args:
            embedding_dim: Dimension of embeddings to align
        """
        super().__init__()
        self.embedding_dim = embedding_dim

        # Optional: Learnable alignment projection (can improve alignment quality)
        self.alignment_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(
        self,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        use_projection: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute semantic alignment between source and target.

        SCAR Equation 8: L_align = ||Hs - Pt||^2

        Args:
            source_embedding: Source document/node embedding [embedding_dim]
            target_embedding: Target query embedding [embedding_dim]
            use_projection: Apply learnable projection before alignment

        Returns:
            alignment_loss: L2 alignment loss
            metrics: Dict with alignment quality metrics
        """
        # Optional projection (helps adapt embeddings for alignment)
        if use_projection:
            source_projected = self.alignment_projection(source_embedding)
        else:
            source_projected = source_embedding

        # Compute L2 alignment loss (Eq. 8)
        alignment_loss = F.mse_loss(source_projected, target_embedding)

        # Compute alignment metrics
        with torch.no_grad():
            # Cosine similarity (before and after projection)
            cosine_sim_original = F.cosine_similarity(
                source_embedding.unsqueeze(0), target_embedding.unsqueeze(0)
            ).item()

            if use_projection:
                cosine_sim_projected = F.cosine_similarity(
                    source_projected.unsqueeze(0), target_embedding.unsqueeze(0)
                ).item()
            else:
                cosine_sim_projected = cosine_sim_original

            # L2 distance
            l2_distance = torch.norm(source_projected - target_embedding).item()

        metrics = {
            "cosine_similarity_original": cosine_sim_original,
            "cosine_similarity_projected": cosine_sim_projected,
            "l2_distance": l2_distance,
            "alignment_loss": alignment_loss.item(),
        }

        return alignment_loss, metrics

    def compute_alignment_score(
        self,
        source_embeddings: np.ndarray,
        query_embedding: np.ndarray,
        use_projection: bool = False,
    ) -> np.ndarray:
        """
        Compute alignment scores for multiple source nodes against a query.

        Higher score = better alignment = more relevant to query

        Args:
            source_embeddings: Multiple document node embeddings [N, embedding_dim]
            query_embedding: Single query embedding [embedding_dim]
            use_projection: Apply learnable projection

        Returns:
            alignment_scores: Scores for each source [N], higher is better
        """
        self.eval()
        with torch.no_grad():
            source_tensor = torch.from_numpy(source_embeddings).float()
            query_tensor = torch.from_numpy(query_embedding).float()

            # Expand query to match batch size
            query_batch = query_tensor.unsqueeze(0).expand(source_tensor.shape[0], -1)

            if use_projection:
                source_projected = self.alignment_projection(source_tensor)
            else:
                source_projected = source_tensor

            # Compute cosine similarity as alignment score
            # (inverted L2 loss would also work, but cosine is more interpretable)
            cosine_scores = F.cosine_similarity(source_projected, query_batch)

            return cosine_scores.numpy()


class SCAREnhancedCompressor:
    """
    Integration of SCAR concepts with existing SemanticCompressor.

    Enhancements:
    1. Learnable embedding compression (SCAR Section 3.2)
    2. Semantic alignment guidance for retrieval (SCAR Section 3.3)
    3. Adaptive fidelity based on alignment scores
    """

    def __init__(
        self,
        base_compressor,
        use_learnable_compression: bool = True,
        use_alignment_guidance: bool = True,
        compression_ratio: float = 4.0,
    ):
        """
        Initialize SCAR-enhanced compressor.

        Args:
            base_compressor: Existing SemanticCompressor instance
            use_learnable_compression: Enable SCAR's learnable compression
            use_alignment_guidance: Enable SCAR's alignment guidance
            compression_ratio: How much to compress embeddings (default 4× like SCAR)
        """
        self.base_compressor = base_compressor
        self.use_learnable_compression = use_learnable_compression
        self.use_alignment_guidance = use_alignment_guidance

        # Get embedding dimension from base compressor's model
        embedding_dim = base_compressor.model.get_sentence_embedding_dimension()
        compressed_dim = int(embedding_dim / compression_ratio)

        # Initialize SCAR modules
        if use_learnable_compression:
            self.learnable_compressor = LearnableSemanticCompressor(
                input_dim=embedding_dim,
                compressed_dim=compressed_dim,
            )
            print(
                f"[SCAR] Learnable Compression: {embedding_dim}D -> {compressed_dim}D ({compression_ratio}x compression)"
            )
            # v0.8.0 AUDIT WARNING: Random weights used - not trained!
            # The LearnableSemanticCompressor uses PyTorch's default random weight initialization.
            # For production use, you would need to:
            # 1. Train the compressor on domain-specific data using training_utils.py
            # 2. Save trained weights with torch.save(self.learnable_compressor.state_dict(), 'scar_weights.pt')
            # 3. Load trained weights with self.learnable_compressor.load_state_dict(torch.load('scar_weights.pt'))
            # Without training, the compression is essentially a random projection,
            # which may not preserve semantic information optimally.
            global _UNTRAINED_WARNING_EMITTED
            if not _UNTRAINED_WARNING_EMITTED:
                logger.warning(
                    "[SCAR] Using UNTRAINED random weights for learnable compression. "
                    "For optimal semantic preservation, train the compressor first. "
                    "See training_utils.py for training infrastructure."
                )
                _UNTRAINED_WARNING_EMITTED = True

        if use_alignment_guidance:
            self.alignment_module = SemanticAlignmentModule(embedding_dim=embedding_dim)
            print("[SCAR] Semantic Alignment: Enabled")

    def compress_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compress embeddings using SCAR's learnable compression.

        Args:
            embeddings: Original embeddings [N, D]

        Returns:
            Compressed embeddings [N, D/k] where k is compression ratio
        """
        if not self.use_learnable_compression:
            return embeddings

        return self.learnable_compressor.compress_batch(embeddings)

    def search_with_alignment(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
        alignment_weight: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Enhanced semantic search using SCAR's alignment guidance.

        Combines:
        1. Standard cosine similarity (baseline)
        2. Semantic alignment score (SCAR's contribution)

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results
            alignment_weight: Weight for alignment score (0-1)
                             0 = pure cosine similarity
                             1 = pure alignment score

        Returns:
            List of (node_id, combined_score) tuples
        """
        if not 0.0 <= alignment_weight <= 1.0:
            raise ValueError(
                f"alignment_weight must be between 0.0 and 1.0, got {alignment_weight}"
            )

        # Get query embedding
        query_embedding = self.base_compressor.model.encode([query])[0]

        # Get candidate nodes
        candidates = []
        for node_id, node in self.base_compressor.chunks.items():
            if file_id and not node_id.startswith(file_id):
                continue

            # Standard cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity

            cosine_sim = cosine_similarity([query_embedding], [node.embedding])[0][0]

            # SCAR alignment score
            if self.use_alignment_guidance:
                alignment_score = self.alignment_module.compute_alignment_score(
                    source_embeddings=node.embedding.reshape(1, -1),
                    query_embedding=query_embedding,
                    use_projection=True,
                )[0]
            else:
                alignment_score = cosine_sim

            # Combine scores and normalize to [0, 1] for deterministic downstream behavior.
            # Raw cosine/alignment values are in [-1, 1], so we map via (x + 1) / 2.
            combined_raw = (1 - alignment_weight) * cosine_sim + alignment_weight * alignment_score
            combined_score = (combined_raw + 1.0) / 2.0
            combined_score = float(np.clip(combined_score, 0.0, 1.0))

            candidates.append((node_id, combined_score))

        # Sort by combined score
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def adaptive_modulate(
        self, query: str, file_id: str, top_k: int = 3, alignment_threshold: float = 0.7
    ) -> str:
        """
        Adaptive modulation with SCAR-inspired fidelity selection.

        High alignment → Return RAW (full detail)
        Medium alignment → Return STRUCTURE (moderate detail)
        Low alignment → Return ABSTRACT (summary only)

        This adapts SCAR's concept of adaptive fidelity based on semantic relevance.

        Args:
            query: User query
            file_id: Document to search
            top_k: Number of nodes to retrieve
            alignment_threshold: Threshold for high-fidelity retrieval

        Returns:
            Modulated content with adaptive fidelity
        """
        # Search with alignment
        results = self.search_with_alignment(
            query=query,
            file_id=file_id,
            top_k=top_k,
            alignment_weight=0.5,  # Balance between similarity and alignment
        )

        # Adaptive fidelity selection
        from .semantic_compressor import FidelityLevel

        output_lines = []
        output_lines.append("=== SCAR ADAPTIVE MODULATION ===")
        output_lines.append(f"Query: {query}")
        output_lines.append(f"Retrieved {len(results)} nodes with adaptive fidelity\n")

        for node_id, score in results:
            # Determine fidelity based on alignment score
            if score >= alignment_threshold:
                fidelity = FidelityLevel.RAW
                marker = "[TOP]"
            elif score >= 0.5:
                fidelity = FidelityLevel.STRUCTURE
                marker = "[HIGH]"
            else:
                fidelity = FidelityLevel.ABSTRACT
                marker = "[DOC]"

            # Retrieve at determined fidelity
            content = self.base_compressor.modulate_region(
                node_ids=[node_id], fidelity_level=fidelity
            )

            output_lines.append(
                f"{marker} Alignment Score: {score:.3f} → Fidelity: {fidelity.name}"
            )
            output_lines.append(content)
            output_lines.append("")

        return "\n".join(output_lines)

    def get_compression_stats(self) -> Dict:
        """Get statistics about SCAR enhancements."""
        stats = {
            "learnable_compression_enabled": self.use_learnable_compression,
            "alignment_guidance_enabled": self.use_alignment_guidance,
        }

        if self.use_learnable_compression:
            stats["compression_ratio"] = self.learnable_compressor.compression_ratio
            stats["input_dim"] = self.learnable_compressor.input_dim
            stats["compressed_dim"] = self.learnable_compressor.compressed_dim

        return stats


# Example usage
if __name__ == "__main__":
    """
    Demonstrate SCAR-inspired enhancements
    """
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.semantic_compressor import SemanticCompressor

    print("=" * 70)
    print("SCAR-INSPIRED SEMANTIC COMPRESSION DEMO")
    print("Adapting arXiv:2511.14063v1 concepts to text compression")
    print("=" * 70)

    # Initialize base compressor
    base_compressor = SemanticCompressor()

    # Initialize SCAR enhancements
    scar = SCAREnhancedCompressor(
        base_compressor=base_compressor,
        use_learnable_compression=True,
        use_alignment_guidance=True,
        compression_ratio=4.0,  # 4× compression like SCAR paper
    )

    # Ingest sample document
    sample_doc = """
    Quantum Error Correction in Surface Codes

    Surface codes are a leading approach to quantum error correction.
    They use a 2D lattice of qubits with stabilizer measurements.

    The error threshold for surface codes is approximately 1%.
    This makes them practical for near-term quantum computers.

    However, the overhead is significant. A logical qubit requires
    hundreds of physical qubits for fault-tolerant operation.

    Recent advances in code concatenation have improved efficiency.
    Researchers are exploring color codes and other topological variants.

    The challenge lies in balancing error suppression with qubit overhead.
    Different applications require different trade-offs.
    """

    base_compressor.ingest_file(sample_doc, "quantum_ec")

    print("\n" + "=" * 70)
    print("TEST 1: SCAR Alignment-Guided Search")
    print("=" * 70)

    query = "What is the error threshold for surface codes?"
    results = scar.search_with_alignment(
        query=query, file_id="quantum_ec", top_k=3, alignment_weight=0.5
    )

    print(f"\nQuery: {query}")
    print("\nTop results with alignment scores:")
    for node_id, score in results:
        node = base_compressor.chunks[node_id]
        summary = base_compressor._generate_summary(node.text, 60)
        print(f"  {node_id}: {score:.3f} - {summary}")

    print("\n" + "=" * 70)
    print("TEST 2: SCAR Adaptive Modulation")
    print("=" * 70)

    result = scar.adaptive_modulate(
        query=query, file_id="quantum_ec", top_k=3, alignment_threshold=0.7
    )
    print(result)

    print("\n" + "=" * 70)
    print("TEST 3: Learnable Compression")
    print("=" * 70)

    # Get some embeddings
    test_texts = [
        "Surface codes use stabilizer measurements",
        "The error threshold is approximately 1%",
        "Overhead requires hundreds of qubits",
    ]
    embeddings = base_compressor.model.encode(test_texts)

    print(f"Original embedding shape: {embeddings.shape}")

    # Compress using SCAR
    compressed = scar.compress_embeddings(embeddings)
    print(f"Compressed embedding shape: {compressed.shape}")
    print(f"Compression ratio: {embeddings.shape[1] / compressed.shape[1]:.1f}×")

    print("\n" + "=" * 70)
    print("SCAR Stats")
    print("=" * 70)
    stats = scar.get_compression_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
