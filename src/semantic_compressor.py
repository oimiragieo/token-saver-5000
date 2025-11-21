"""
Fidelity-Preserving Semantic Compressor

Implements the core encoding/decoding logic inspired by:
- Paper 1: JSCCM (Joint Semantic-Channel Coding) - Rate adaptation
- Paper 2: FPQE (Fidelity-Preserving Quantization) - Structure preservation
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken


class FidelityLevel(Enum):
    """
    Semantic fidelity levels for adaptive transmission

    Inspired by JSCCM's multi-rate allocation strategy.
    5 levels provide fine-grained control over token budget vs. information fidelity.
    """

    ABSTRACT = 1  # 1-sentence summary (~10 tokens)
    OUTLINE = 2  # Summary + section markers (~30 tokens)
    STRUCTURE = 3  # Headers + key entities (~50 tokens)
    DETAILED = 4  # Summary + entities + key excerpts (~100 tokens)
    RAW = 5  # Full original text (variable, typically 200-500 tokens)


@dataclass
class SemanticNode:
    """Represents a chunk in the semantic graph"""

    node_id: str
    text: str
    embedding: np.ndarray
    importance: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SkeletonResponse:
    """The compressed skeleton view of a document"""

    file_id: str
    total_nodes: int
    total_tokens: int
    skeleton_tokens: int
    compression_ratio: float
    skeleton_text: str
    node_map: Dict[str, str]  # node_id -> short description


class SemanticCompressor:
    """
    Core compressor implementing adaptive semantic fidelity.

    Architecture:
    1. Encoder: Text -> Semantic Graph (preserves structure)
    2. Rate Allocator: Determines importance via PageRank
    3. Modulator: Serves content at requested fidelity levels
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75,
        skeleton_ratio: float = 0.2,
    ):
        """
        Initialize the semantic compressor.

        Args:
            model_name: Local embedding model (lightweight recommended)
            similarity_threshold: Minimum similarity to create graph edges
            skeleton_ratio: Fraction of nodes to include in skeleton (top N%)
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.skeleton_ratio = skeleton_ratio

        # Storage
        self.graphs: Dict[str, nx.Graph] = {}
        self.chunks: Dict[str, SemanticNode] = {}
        self.file_metadata: Dict[str, Dict] = {}

        # Token counter
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def _chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """
        Intelligent text chunking that preserves semantic boundaries.

        Prioritizes:
        1. Paragraph boundaries (\n\n)
        2. Sentence boundaries (. ! ?)
        3. Fixed size fallback
        """
        # Split by double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is small enough, try to combine
            para_tokens = self._count_tokens(para)
            current_tokens = self._count_tokens(current_chunk)

            if current_tokens + para_tokens <= max_chunk_size:
                current_chunk += "\n\n" + para if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk)

                # If single paragraph is too large, split by sentences
                if para_tokens > max_chunk_size:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current_chunk = ""
                    for sent in sentences:
                        if self._count_tokens(current_chunk + " " + sent) <= max_chunk_size:
                            current_chunk += " " + sent if current_chunk else sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sent
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _extract_key_entities(self, text: str, max_entities: int = 5) -> List[str]:
        """
        Simple entity extraction (can be enhanced with NER).
        Currently uses capitalized words as proxy for entities.
        """
        # Find capitalized phrases (simple heuristic)
        words = text.split()
        entities = []

        for i, word in enumerate(words):
            # Look for capitalized words that aren't sentence starts
            if word[0].isupper() and i > 0 and words[i - 1][-1] not in ".!?":
                entities.append(word)

        # Return unique entities
        return list(set(entities))[:max_entities]

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        Generate a simple extractive summary.
        Takes first sentence or first max_length characters.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if sentences:
            summary = sentences[0]
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        return text[:max_length] + "..."

    def ingest_file(
        self, text: str, file_id: str, metadata: Optional[Dict] = None
    ) -> SkeletonResponse:
        """
        Step 1: Fidelity-Preserving Encoding

        Converts raw text into a semantic graph where:
        - Nodes = semantic chunks
        - Edges = similarity relationships (preserves global structure)
        - Weights = PageRank scores (importance)

        Args:
            text: Raw document text
            file_id: Unique identifier for this document
            metadata: Optional metadata (author, date, etc.)

        Returns:
            SkeletonResponse with compressed view
        """
        print(f"\n🔬 Ingesting file: {file_id}")

        # Count original tokens
        total_tokens = self._count_tokens(text)
        print(f"  Original tokens: {total_tokens}")

        # 1. Chunk the text semantically
        raw_chunks = self._chunk_text(text)
        print(f"  Created {len(raw_chunks)} semantic chunks")

        # 2. Generate embeddings
        print("  Generating embeddings...")
        embeddings = self.model.encode(raw_chunks, show_progress_bar=False)

        # 3. Build similarity graph (preserves global structure)
        print("  Building semantic graph...")
        graph = nx.Graph()
        similarity_matrix = cosine_similarity(embeddings)

        for i, chunk in enumerate(raw_chunks):
            # Create unique node ID
            node_id = f"{file_id}_n{i}"

            # Create semantic node
            node = SemanticNode(
                node_id=node_id,
                text=chunk,
                embedding=embeddings[i],
                metadata={
                    "position": i,
                    "tokens": self._count_tokens(chunk),
                    "entities": self._extract_key_entities(chunk),
                },
            )

            self.chunks[node_id] = node
            graph.add_node(node_id, **node.metadata)

            # Create edges based on semantic similarity
            for j in range(i + 1, len(raw_chunks)):
                similarity = similarity_matrix[i][j]
                if similarity > self.similarity_threshold:
                    edge_id = f"{file_id}_n{j}"
                    graph.add_edge(node_id, edge_id, weight=float(similarity))

        # 4. Calculate importance via PageRank (rate allocation)
        print("  Calculating importance scores (PageRank)...")
        if len(graph.nodes) > 0:
            pagerank = nx.pagerank(graph)

            # Update importance scores
            for node_id, score in pagerank.items():
                if node_id in self.chunks:
                    self.chunks[node_id].importance = score

        # Store graph
        self.graphs[file_id] = graph
        self.file_metadata[file_id] = metadata or {}

        # 5. Generate skeleton
        skeleton_response = self._generate_skeleton(file_id)

        print(f"  ✅ Compression: {total_tokens} -> {skeleton_response.skeleton_tokens} tokens")
        print(f"  📊 Ratio: {skeleton_response.compression_ratio:.1f}x")

        return skeleton_response

    def _generate_skeleton(self, file_id: str) -> SkeletonResponse:
        """
        Step 2: Rate Allocation (JSCCM)

        Generates a low-bandwidth skeleton view by:
        1. Ranking nodes by importance (PageRank)
        2. Keeping top N% as "anchor concepts"
        3. Hiding others as references
        """
        graph = self.graphs.get(file_id)
        if not graph:
            raise ValueError(f"File {file_id} not found")

        # Get all nodes for this file
        file_nodes = [(nid, self.chunks[nid]) for nid in graph.nodes() if nid.startswith(file_id)]

        # Sort by importance
        file_nodes.sort(key=lambda x: x[1].importance, reverse=True)

        # Determine skeleton nodes (top N%)
        num_skeleton = max(1, int(len(file_nodes) * self.skeleton_ratio))
        skeleton_nodes = set(nid for nid, _ in file_nodes[:num_skeleton])

        # Build skeleton text
        skeleton_lines = []
        skeleton_lines.append(f"=== SEMANTIC SKELETON: {file_id} ===")
        skeleton_lines.append(f"Total nodes: {len(file_nodes)} | Skeleton nodes: {num_skeleton}")
        skeleton_lines.append(f"Compression: {self.skeleton_ratio:.0%} of content shown\n")

        node_map = {}
        total_tokens = 0
        skeleton_tokens = 0

        for node_id, node in file_nodes:
            total_tokens += node.metadata["tokens"]

            if node_id in skeleton_nodes:
                # High-importance: Show summary + entities
                summary = self._generate_summary(node.text, max_length=150)
                entities = ", ".join(node.metadata["entities"][:3])

                line = f"[{node_id}] ⭐ ANCHOR (importance: {node.importance:.3f})\n"
                line += f"  Summary: {summary}\n"
                if entities:
                    line += f"  Key entities: {entities}\n"

                skeleton_lines.append(line)
                node_map[node_id] = f"ANCHOR: {summary[:50]}..."
                skeleton_tokens += self._count_tokens(line)
            else:
                # Low-importance: Just reference
                summary = self._generate_summary(node.text, max_length=50)
                line = f"[{node_id}] 📦 Detail hidden (use modulate_region to expand)\n"

                skeleton_lines.append(line)
                node_map[node_id] = f"Hidden: {summary[:30]}..."
                skeleton_tokens += self._count_tokens(line)

        skeleton_text = "\n".join(skeleton_lines)
        compression_ratio = total_tokens / max(skeleton_tokens, 1)

        return SkeletonResponse(
            file_id=file_id,
            total_nodes=len(file_nodes),
            total_tokens=total_tokens,
            skeleton_tokens=skeleton_tokens,
            compression_ratio=compression_ratio,
            skeleton_text=skeleton_text,
            node_map=node_map,
        )

    def read_skeleton(self, file_id: str) -> str:
        """
        MCP Tool: read_skeleton

        Returns the compressed skeleton view of a document.
        ~80-95% token savings vs raw text.
        """
        skeleton = self._generate_skeleton(file_id)
        return skeleton.skeleton_text

    def modulate_region(
        self, node_ids: List[str], fidelity_level: FidelityLevel = FidelityLevel.RAW
    ) -> str:
        """
        Step 3: The Modulator (Adaptive Fidelity)

        Returns content at requested fidelity level:
        - ABSTRACT: 1-sentence summary (~10 tokens)
        - OUTLINE: Summary + section markers (~30 tokens)
        - STRUCTURE: Headers + key entities (~50 tokens)
        - DETAILED: Summary + entities + key excerpts (~100 tokens)
        - RAW: Full original text (variable, typically 200-500 tokens)

        Inspired by JSCCM's adaptive modulation strategy.

        Args:
            node_ids: List of node IDs to retrieve
            fidelity_level: Desired level of detail

        Returns:
            Formatted content string
        """
        output_lines = []
        output_lines.append(f"=== MODULATED CONTENT (Fidelity: {fidelity_level.name}) ===\n")

        for node_id in node_ids:
            if node_id not in self.chunks:
                output_lines.append(f"[{node_id}] ⚠️  Node not found\n")
                continue

            node = self.chunks[node_id]

            if fidelity_level == FidelityLevel.ABSTRACT:
                # Level 1: Just a summary (~10 tokens)
                summary = self._generate_summary(node.text, max_length=100)
                output_lines.append(f"[{node_id}] Abstract:\n  {summary}\n")

            elif fidelity_level == FidelityLevel.OUTLINE:
                # Level 2: Summary + position context (~30 tokens)
                summary = self._generate_summary(node.text, max_length=120)
                position = node.metadata.get("position", "?")
                entities = ", ".join(node.metadata["entities"][:2])  # Top 2 entities

                output_lines.append(f"[{node_id}] Outline:")
                output_lines.append(f"  Position: Section {position}")
                output_lines.append(f"  Summary: {summary}")
                if entities:
                    output_lines.append(f"  Key terms: {entities}")
                output_lines.append("")

            elif fidelity_level == FidelityLevel.STRUCTURE:
                # Level 3: Summary + entities + metadata (~50 tokens)
                summary = self._generate_summary(node.text, max_length=150)
                entities = ", ".join(node.metadata["entities"])

                output_lines.append(f"[{node_id}] Structure:")
                output_lines.append(f"  Summary: {summary}")
                output_lines.append(f"  Entities: {entities}")
                output_lines.append(f"  Tokens: {node.metadata['tokens']}")
                output_lines.append(f"  Importance: {node.importance:.3f}\n")

            elif fidelity_level == FidelityLevel.DETAILED:
                # Level 4: Summary + entities + key excerpts (~100 tokens)
                summary = self._generate_summary(node.text, max_length=200)
                entities = ", ".join(node.metadata["entities"])

                # Extract first 2-3 sentences as excerpt
                sentences = re.split(r"(?<=[.!?])\s+", node.text)
                excerpt = " ".join(sentences[: min(3, len(sentences))])
                if len(excerpt) > 300:
                    excerpt = excerpt[:300] + "..."

                output_lines.append(f"[{node_id}] Detailed:")
                output_lines.append(f"  Summary: {summary}")
                output_lines.append(f"  Entities: {entities}")
                output_lines.append(f"  Key excerpt:\n    {excerpt}")
                output_lines.append(
                    f"  Metadata: {node.metadata['tokens']} tokens, importance {node.importance:.3f}\n"
                )

            else:  # FidelityLevel.RAW
                # Level 5: Full content (variable tokens)
                output_lines.append(f"[{node_id}] Full Content:")
                output_lines.append("--- BEGIN ---")
                output_lines.append(node.text)
                output_lines.append("--- END ---")
                output_lines.append(
                    f"Metadata: {node.metadata['tokens']} tokens, importance {node.importance:.3f}\n"
                )

        return "\n".join(output_lines)

    def search_semantic(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[str]:
        """
        Semantic search using vector similarity.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of node IDs ranked by relevance
        """
        # Embed the query
        query_embedding = self.model.encode([query])[0]

        # Get candidate nodes
        candidates = []
        for node_id, node in self.chunks.items():
            if file_id and not node_id.startswith(file_id):
                continue

            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]

            candidates.append((node_id, similarity))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [node_id for node_id, _ in candidates[:top_k]]

    def get_stats(self, file_id: Optional[str] = None) -> Dict:
        """Get statistics about stored documents"""
        if file_id:
            if file_id not in self.graphs:
                return {"error": f"File {file_id} not found"}

            graph = self.graphs[file_id]
            nodes = [nid for nid in graph.nodes() if nid.startswith(file_id)]

            total_tokens = sum(self.chunks[nid].metadata["tokens"] for nid in nodes)

            skeleton = self._generate_skeleton(file_id)

            return {
                "file_id": file_id,
                "total_nodes": len(nodes),
                "total_edges": graph.number_of_edges(),
                "total_tokens": total_tokens,
                "skeleton_tokens": skeleton.skeleton_tokens,
                "compression_ratio": skeleton.compression_ratio,
                "metadata": self.file_metadata.get(file_id, {}),
            }
        else:
            # Global stats
            return {
                "total_files": len(self.graphs),
                "total_nodes": len(self.chunks),
                "files": list(self.graphs.keys()),
            }
