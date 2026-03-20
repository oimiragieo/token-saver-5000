"""
Multi-Modal Semantic Compressor

[WARN] EXPERIMENTAL FEATURE - Not used in core MCP server (v0.4.3)

Status: Documented feature with working examples (examples/multimodal_example.py)
Coverage: 0% (no tests yet)
Production Ready: NO - requires tests and integration
Dependencies: Pillow, optional CLIP for image embeddings

TODO for production:
- [ ] Add comprehensive tests (target: 80%+ coverage)
- [ ] Integrate with MCP server tools
- [ ] Benchmark image compression quality
- [ ] Document multimodal API in server.py

Handles text, code, AND images in a unified semantic graph.

Uses:
- Text: Sentence-Transformers (all-MiniLM-L6-v2)
- Code: CodeBERT or similar
- Images: CLIP for vision-language alignment

All modalities are projected into a shared embedding space for
cross-modal search and retrieval.
"""

import base64
import logging
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import networkx as nx

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Content modality types"""

    TEXT = "text"
    CODE = "code"
    IMAGE = "image"


@dataclass
class MultiModalNode:
    """Node that can contain any modality"""

    node_id: str
    modality: ModalityType
    content: Union[str, bytes]  # Text/code string or image bytes
    embedding: np.ndarray
    importance: float = 0.0
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MultiModalCompressor:
    """
    Unified compressor for text, code, and images.

    Features:
    - Processes multiple content types
    - Creates unified semantic graph
    - Cross-modal search (e.g., "find images related to this code")
    - Mixed-modality retrieval
    """

    def __init__(
        self,
        use_clip_for_images: bool = True,
        use_codebert_for_code: bool = False,  # Fallback to general model if False
    ):
        """
        Initialize multi-modal compressor.

        Args:
            use_clip_for_images: Use CLIP for image embeddings
            use_codebert_for_code: Use CodeBERT for code (otherwise use general model)
        """
        # Use EmbeddingManager for shared model caching across all modalities
        embedding_manager = EmbeddingManager()

        # Text encoder
        self.text_encoder = embedding_manager.get_text_embedder()

        # Code encoder (optional CodeBERT)
        if use_codebert_for_code:
            self.code_encoder = embedding_manager.get_code_embedder()
        else:
            self.code_encoder = self.text_encoder

        # Image encoder (optional CLIP)
        self.image_encoder = None
        if use_clip_for_images:
            try:
                self.image_encoder = embedding_manager.get_image_embedder()
                print("[OK] CLIP loaded for image processing")
            except Exception as e:
                print(f"Warning: CLIP not available ({e})")
                print("Image support disabled. Install: pip install clip-ViT-B-32")

        # Storage
        self.nodes: Dict[str, MultiModalNode] = {}
        self.graphs: Dict[str, nx.Graph] = {}

        # Embedding dimension (should be same across modalities for cross-modal search)
        self.embedding_dim = self.text_encoder.get_sentence_embedding_dimension()

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        return self.text_encoder.encode([text], show_progress_bar=True)[0]

    def _encode_code(self, code: str) -> np.ndarray:
        """Encode code to embedding"""
        return self.code_encoder.encode([code], show_progress_bar=True)[0]

    def _encode_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Encode image to embedding using CLIP.

        Args:
            image_data: Image bytes (PNG, JPEG, etc.)

        Returns:
            Image embedding or None if CLIP not available
        """
        if self.image_encoder is None:
            logger.warning("Image encoder not available")
            return None

        try:
            from PIL import Image

            image = Image.open(BytesIO(image_data))

            # CLIP expects PIL images
            embedding = self.image_encoder.encode([image], show_progress_bar=True)[0]
            return embedding
        except Exception as e:
            logger.warning("Error encoding image: %s", e)
            return None

    def ingest_mixed_content(
        self,
        content_items: List[Dict] | str,
        project_id: str | List[Dict],
        similarity_threshold: float = 0.70,
    ) -> Dict:
        """
        Ingest mixed content (text, code, images) into unified graph.

        Args:
            content_items: List of dicts with:
                - 'type': 'text' | 'code' | 'image'
                - 'content': str (text/code) or bytes (image)
                - 'metadata': Optional metadata
            project_id: Unique project identifier
            similarity_threshold: Threshold for cross-modal connections

        Returns:
            Ingestion statistics

        Example:
            content_items = [
                {'type': 'text', 'content': 'README content...', 'metadata': {'file': 'README.md'}},
                {'type': 'code', 'content': 'def train()...', 'metadata': {'file': 'train.py'}},
                {'type': 'image', 'content': b'...png bytes...', 'metadata': {'file': 'diagram.png'}},
            ]
        """
        if isinstance(content_items, str) and isinstance(project_id, list):
            content_items, project_id = project_id, content_items

        if not isinstance(content_items, list) or not isinstance(project_id, str):
            raise ValueError("ingest_mixed_content expects (content_items, project_id)")

        logger.info("Ingesting mixed content for project: %s", project_id)

        nodes = []
        embeddings = []

        # Process each content item
        for i, item in enumerate(content_items):
            content_type = item.get("type", "text")
            content = item["content"]
            metadata = item.get("metadata", {})

            # Determine modality
            if content_type == "text":
                modality = ModalityType.TEXT
                embedding = self._encode_text(content)
            elif content_type == "code":
                modality = ModalityType.CODE
                embedding = self._encode_code(content)
            elif content_type == "image":
                modality = ModalityType.IMAGE
                if "content" not in item and item.get("path"):
                    with open(item["path"], "rb") as image_file:
                        content = image_file.read()
                embedding = self._encode_image(content)
                if embedding is None:
                    logger.warning(
                        "Skipping image %s for project %s because image encoding failed",
                        i,
                        project_id,
                    )
                    continue
            else:
                logger.warning("Unknown content type '%s', treating as text", content_type)
                modality = ModalityType.TEXT
                embedding = self._encode_text(str(content))

            node = MultiModalNode(
                node_id=f"{project_id}_n{i}",
                modality=modality,
                content=content,
                embedding=embedding,
                metadata=metadata,
            )

            nodes.append(node)
            embeddings.append(embedding)
            self.nodes[node.node_id] = node

        # Build cross-modal semantic graph
        graph = nx.Graph()

        from sklearn.metrics.pairwise import cosine_similarity

        embeddings_array = np.array(embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)

        for i, node_i in enumerate(nodes):
            graph.add_node(node_i.node_id, modality=node_i.modality.value)

            for j in range(i + 1, len(nodes)):
                node_j = nodes[j]
                similarity = similarity_matrix[i][j]

                if similarity > similarity_threshold:
                    graph.add_edge(
                        node_i.node_id,
                        node_j.node_id,
                        weight=float(similarity),
                        connection_type=f"{node_i.modality.value}-{node_j.modality.value}",
                    )

        # Calculate importance
        if len(graph.nodes) > 0:
            pagerank = nx.pagerank(graph)
            for node_id, score in pagerank.items():
                if node_id in self.nodes:
                    self.nodes[node_id].importance = score

        self.graphs[project_id] = graph

        # Statistics
        edge_types = {}
        for u, v, data in graph.edges(data=True):
            conn_type = data.get("connection_type", "unknown")
            edge_types[conn_type] = edge_types.get(conn_type, 0) + 1

        stats = {
            "project_id": project_id,
            "total_nodes": len(nodes),
            "nodes_by_modality": {
                "text": sum(1 for n in nodes if n.modality == ModalityType.TEXT),
                "code": sum(1 for n in nodes if n.modality == ModalityType.CODE),
                "image": sum(1 for n in nodes if n.modality == ModalityType.IMAGE),
            },
            "graph_edges": graph.number_of_edges(),
            "cross_modal_connections": edge_types,
        }

        logger.info(
            "Created multimodal graph for %s with %s nodes and %s edges",
            project_id,
            stats["total_nodes"],
            stats["graph_edges"],
        )

        return stats

    def search_cross_modal(
        self,
        query: Union[str, bytes],
        query_type: str = "text",
        project_id: Optional[str] = None,
        top_k: int = 5,
        filter_modality: Optional[str] = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Cross-modal semantic search.

        Args:
            query: Search query (text string or image bytes)
            query_type: 'text', 'code', or 'image'
            project_id: Optional project to search within
            top_k: Number of results
            filter_modality: Optional filter to specific modality

        Returns:
            List of (node_id, score, modality) tuples

        Examples:
            # Text query, find related images
            search_cross_modal("neural network architecture", query_type='text', filter_modality='image')

            # Image query, find related code
            search_cross_modal(image_bytes, query_type='image', filter_modality='code')

            # Code query, find related documentation
            search_cross_modal("def train_model():", query_type='code', filter_modality='text')
        """
        # Encode query
        if query_type == "text":
            query_embedding = self._encode_text(query)
        elif query_type == "code":
            query_embedding = self._encode_code(query)
        elif query_type == "image":
            query_embedding = self._encode_image(query)
            if query_embedding is None:
                return []
        else:
            print(f"Unknown query type '{query_type}', treating as text")
            query_embedding = self._encode_text(str(query))

        # Search all nodes
        candidates = []
        for node_id, node in self.nodes.items():
            if project_id and not node_id.startswith(project_id):
                continue

            if filter_modality and node.modality.value != filter_modality:
                continue

            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]

            candidates.append((node_id, float(similarity), node.modality.value))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def get_node_content(self, node_id: str) -> Dict:
        """
        Retrieve node content with metadata.

        Returns:
            Dict with node information
        """
        node = self.nodes.get(node_id)
        if not node:
            return {"error": f"Node {node_id} not found"}

        result = {
            "node_id": node_id,
            "modality": node.modality.value,
            "importance": node.importance,
            "metadata": node.metadata,
        }

        if node.modality == ModalityType.IMAGE:
            # For images, provide base64 encoding
            result["content"] = base64.b64encode(node.content).decode("utf-8")
            result["content_type"] = "base64"
        else:
            # For text/code, provide as string
            result["content"] = node.content
            result["content_type"] = "text"

        return result

    def generate_multimodal_summary(self, project_id: str) -> str:
        """
        Generate summary of multi-modal project.

        Shows:
        - Content overview by modality
        - Top important items per modality
        - Cross-modal connections
        """
        graph = self.graphs.get(project_id)
        if not graph:
            return f"Project {project_id} not found"

        # Get nodes by modality
        project_nodes = [
            (nid, self.nodes[nid]) for nid in graph.nodes() if nid.startswith(project_id)
        ]

        text_nodes = [(nid, n) for nid, n in project_nodes if n.modality == ModalityType.TEXT]
        code_nodes = [(nid, n) for nid, n in project_nodes if n.modality == ModalityType.CODE]
        image_nodes = [(nid, n) for nid, n in project_nodes if n.modality == ModalityType.IMAGE]

        lines = []
        lines.append(f"=== MULTI-MODAL PROJECT: {project_id} ===")
        lines.append(f"Total items: {len(project_nodes)}")
        lines.append("")

        # Text summary
        if text_nodes:
            lines.append(f"[TEXT] TEXT DOCUMENTS ({len(text_nodes)}):")
            text_nodes.sort(key=lambda x: x[1].importance, reverse=True)
            for nid, node in text_nodes[:3]:
                preview = node.content[:80].replace("\n", " ")
                lines.append(f"  {nid}: {preview}... (importance: {node.importance:.3f})")
            if len(text_nodes) > 3:
                lines.append(f"  ... and {len(text_nodes) - 3} more")
            lines.append("")

        # Code summary
        if code_nodes:
            lines.append(f"[CODE] CODE FILES ({len(code_nodes)}):")
            code_nodes.sort(key=lambda x: x[1].importance, reverse=True)
            for nid, node in code_nodes[:3]:
                file_name = node.metadata.get("file", "unknown")
                preview = node.content[:60].replace("\n", " ")
                lines.append(
                    f"  {nid} ({file_name}): {preview}... (importance: {node.importance:.3f})"
                )
            if len(code_nodes) > 3:
                lines.append(f"  ... and {len(code_nodes) - 3} more")
            lines.append("")

        # Image summary
        if image_nodes:
            lines.append(f"[IMAGE] IMAGES ({len(image_nodes)}):")
            image_nodes.sort(key=lambda x: x[1].importance, reverse=True)
            for nid, node in image_nodes[:3]:
                file_name = node.metadata.get("file", "unknown")
                size_kb = len(node.content) / 1024
                lines.append(
                    f"  {nid} ({file_name}): {size_kb:.1f}KB (importance: {node.importance:.3f})"
                )
            if len(image_nodes) > 3:
                lines.append(f"  ... and {len(image_nodes) - 3} more")
            lines.append("")

        # Cross-modal connections
        cross_modal = []
        for u, v, data in graph.edges(data=True):
            conn_type = data.get("connection_type", "")
            if "-" in conn_type and conn_type.split("-")[0] != conn_type.split("-")[1]:
                cross_modal.append((u, v, data["weight"]))

        if cross_modal:
            cross_modal.sort(key=lambda x: x[2], reverse=True)
            lines.append("[LINKS] TOP CROSS-MODAL CONNECTIONS:")
            for u, v, weight in cross_modal[:5]:
                u_mod = self.nodes[u].modality.value
                v_mod = self.nodes[v].modality.value
                lines.append(f"  {u} ({u_mod}) <-> {v} ({v_mod}): {weight:.3f}")

        return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit("Run multimodal examples from tests or notebooks instead.")
