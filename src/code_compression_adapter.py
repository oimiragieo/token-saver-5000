"""
Code Compression Adapter - Unified Interface for Text and Code Compression

This adapter provides transparent routing between SemanticCompressor (text) and
CodeSemanticCompressor (code) based on file type detection. It presents a unified
interface to handlers while leveraging AST-aware chunking for code files.

Key Features:
- Lazy loading of CodeBERT model (~400MB) on first code file
- Automatic file type detection based on extension
- Graceful fallback to text model if code model fails
- Unified node ID namespace for persistence compatibility

v0.9.0 - Programmer UX Improvements
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

from .semantic_compressor import SemanticCompressor, SemanticNode, FidelityLevel
from .code_compressor import CodeSemanticCompressor, CodeChunk

logger = logging.getLogger(__name__)


# Code file extensions that should use CodeSemanticCompressor
CODE_EXTENSIONS = {
    ".py",
    ".pyw",  # Python
    ".js",
    ".jsx",
    ".mjs",
    ".cjs",  # JavaScript
    ".ts",
    ".tsx",
    ".mts",
    ".cts",  # TypeScript
    ".java",  # Java
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".h",
    ".hpp",
    ".hxx",  # C/C++
    ".go",  # Go
    ".rs",  # Rust
    ".rb",  # Ruby
    ".php",  # PHP
    ".swift",  # Swift
    ".kt",
    ".kts",  # Kotlin
    ".scala",  # Scala
    ".cs",  # C#
}


@dataclass
class AdapterNode:
    """
    Unified node representation that wraps both SemanticNode and CodeChunk.

    Provides consistent interface for handlers regardless of source compressor.
    """

    node_id: str
    text: str  # 'text' for SemanticNode, 'code' for CodeChunk
    embedding: np.ndarray
    importance: float
    metadata: Dict[str, Any]
    source_type: str  # 'text' or 'code'

    # Code-specific fields (None for text nodes)
    chunk_type: Optional[str] = None  # 'function', 'class', 'import', etc.
    name: Optional[str] = None
    docstring: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    dependencies: Optional[List[str]] = None


class CodeCompressionAdapter:
    """
    Unified compressor adapter that transparently routes text vs code files.

    This adapter provides:
    1. Automatic file type detection based on extension
    2. Lazy loading of CodeBERT model (~400MB) to minimize baseline RAM
    3. Graceful fallback to text model if code model unavailable
    4. Unified interface compatible with existing handlers
    5. Consistent persistence format across both compressor types

    Environment Variables:
        PRELOAD_CODE_MODEL: Set to "true" to pre-load CodeBERT at startup

    Usage:
        adapter = CodeCompressionAdapter()
        # Routes automatically based on file_path extension
        skeleton = await adapter.ingest_file_async(text, file_id, file_path="main.py")
    """

    def __init__(
        self,
        text_model: str = "all-MiniLM-L6-v2",
        code_model: str = "microsoft/codebert-base",
        preload_code_model: bool = False,
        similarity_threshold: float = 0.75,
        skeleton_ratio: float = 0.2,
    ):
        """
        Initialize the code compression adapter.

        Args:
            text_model: Model for text compression (default: all-MiniLM-L6-v2)
            code_model: Model for code compression (default: microsoft/codebert-base)
            preload_code_model: If True, load code model immediately instead of lazily
            similarity_threshold: Minimum similarity for graph edges
            skeleton_ratio: Ratio of nodes to include in skeleton
        """
        # Text compressor - always loaded
        self._text_compressor = SemanticCompressor(
            model_name=text_model,
            similarity_threshold=similarity_threshold,
            skeleton_ratio=skeleton_ratio,
        )

        # Code compressor - lazy loaded
        self._code_model_name = code_model
        self._code_similarity_threshold = 0.70  # Slightly lower for code
        self._code_compressor: Optional[CodeSemanticCompressor] = None
        self._code_model_available: Optional[bool] = None  # None = not tried, True/False = result
        self._code_model_error: Optional[str] = None

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="code_compress_")

        # Track which file_ids use code compression
        self._code_file_ids: set = set()

        # Expose text compressor attributes that other modules access directly
        # (graph_visualizer, context_window_adapter, etc.)
        self.skeleton_ratio = skeleton_ratio
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Proxy properties and methods — CodeCompressionAdapter wraps
    # SemanticCompressor but many modules (graph_visualizer, context
    # window adapter, compression handlers) access compressor attributes
    # directly.  These proxies delegate to the underlying text compressor
    # so the adapter is a drop-in replacement everywhere.
    # ------------------------------------------------------------------

    @property
    def graphs(self):
        return self._text_compressor.graphs

    @property
    def chunks(self):
        return self._text_compressor.chunks

    @property
    def file_metadata(self):
        return self._text_compressor.file_metadata

    @property
    def model(self):
        return self._text_compressor.model

    def read_skeleton(self, file_id, **kwargs):
        return self._text_compressor.read_skeleton(file_id, **kwargs)

    def compress(self, text, **kwargs):
        return self._text_compressor.compress(text, **kwargs)

    def ingest_mixed_content(self, *args, **kwargs):
        return self._text_compressor.ingest_mixed_content(*args, **kwargs)

    def find_duplicates(self, *args, **kwargs):
        return self._text_compressor.find_duplicates(*args, **kwargs)

    async def diff_reingest_async(self, *args, **kwargs):
        return await self._text_compressor.diff_reingest_async(*args, **kwargs)

    @property
    def _temporal_graph(self):
        return getattr(self._text_compressor, "_temporal_graph", None)

        # Check for prewarm environment variable
        env_preload = os.environ.get("PRELOAD_CODE_MODEL", "").lower() == "true"
        if preload_code_model or env_preload:
            logger.info("Pre-warming CodeBERT model (~400MB, one-time)...")
            self._load_code_compressor()

        logger.info(
            "code_compression_adapter_initialized: text=%s code=%s preload=%s",
            text_model,
            code_model,
            preload_code_model or env_preload,
        )

    # =========================================================================
    # Property Proxies - Required for handler compatibility
    # =========================================================================

    @property
    def graphs(self) -> Dict[str, nx.Graph]:
        """Unified graph storage from both compressors."""
        result = dict(self._text_compressor.graphs)
        if self._code_compressor:
            result.update(self._code_compressor.graphs)
        return result

    @property
    def chunks(self) -> Dict[str, Union[SemanticNode, CodeChunk]]:
        """Unified chunk storage from both compressors."""
        result = dict(self._text_compressor.chunks)
        if self._code_compressor:
            result.update(self._code_compressor.chunks)
        return result

    @property
    def file_metadata(self) -> Dict[str, Dict]:
        """Unified metadata storage from both compressors."""
        result = dict(self._text_compressor.file_metadata)
        if self._code_compressor:
            result.update(self._code_compressor.file_metadata)
        return result

    @property
    def model(self):
        """Access to embedding model (delegates to text compressor)."""
        return self._text_compressor.model

    # =========================================================================
    # Code Model Management
    # =========================================================================

    def _load_code_compressor(self) -> Optional[CodeSemanticCompressor]:
        """
        Lazy-load CodeSemanticCompressor on first code file.

        Returns:
            CodeSemanticCompressor instance, or None if load failed
        """
        if self._code_compressor is not None:
            return self._code_compressor

        if self._code_model_available is False:
            # Already tried and failed
            return None

        try:
            logger.info("Loading CodeBERT model (~400MB, one-time)...")
            self._code_compressor = CodeSemanticCompressor(
                model_name=self._code_model_name,
                similarity_threshold=self._code_similarity_threshold,
            )
            self._code_model_available = True
            logger.info("CodeBERT model loaded successfully")
            return self._code_compressor
        except Exception as e:
            self._code_model_available = False
            self._code_model_error = str(e)
            logger.warning(
                f"CodeBERT load failed: {e}. Code files will use text model as fallback."
            )
            return None

    def _is_code_file(self, filepath: str) -> bool:
        """Detect if file is code based on extension."""
        if not filepath:
            return False
        ext = os.path.splitext(filepath)[1].lower()
        return ext in CODE_EXTENSIONS

    def is_code_model_available(self) -> bool:
        """Check if code model is available (loaded or loadable)."""
        if self._code_model_available is None:
            # Not tried yet - try to load
            return self._load_code_compressor() is not None
        return self._code_model_available

    def get_code_model_status(self) -> Dict[str, Any]:
        """Get detailed status of code model for diagnostics."""
        return {
            "available": self._code_model_available,
            "loaded": self._code_compressor is not None,
            "model_name": self._code_model_name,
            "error": self._code_model_error,
            "code_files_ingested": len(self._code_file_ids),
        }

    # =========================================================================
    # Core Compression Methods
    # =========================================================================

    async def ingest_file_async(
        self,
        text: str,
        file_id: str,
        metadata: Optional[Dict] = None,
        file_path: Optional[str] = None,
        chunking_strategy: str = "auto",
    ):
        """
        Ingest content with automatic routing based on file type.

        Args:
            text: Content to ingest
            file_id: Unique identifier for the document
            metadata: Optional metadata dict
            file_path: Optional file path for type detection

        Returns:
            SkeletonResponse from text compressor or stats dict from code compressor
        """
        # Determine if this is a code file
        is_code = file_path and self._is_code_file(file_path)

        if is_code:
            # Try to use code compressor
            code_compressor = self._load_code_compressor()

            if code_compressor:
                logger.info("ingesting_code_file: %s (path=%s, model=CodeBERT)", file_id, file_path)

                # Run code ingestion in thread pool (it's synchronous)
                loop = asyncio.get_running_loop()
                stats = await loop.run_in_executor(
                    self._executor,
                    lambda: code_compressor.ingest_code_file(
                        code=text,
                        file_id=file_id,
                        filepath=file_path,
                        metadata=metadata,
                    ),
                )

                self._code_file_ids.add(file_id)

                # Convert to SkeletonResponse-like format for consistency
                return self._convert_code_stats_to_skeleton(stats, file_id)
            else:
                # Fallback to text model
                logger.warning(
                    f"Using text model for {file_path} (CodeBERT unavailable: {self._code_model_error})"
                )

        # Use text compressor (default path)
        logger.info("ingesting_text_file: %s (path=%s, model=MiniLM)", file_id, file_path)
        return await self._text_compressor.ingest_file_async(
            text, file_id, metadata, chunking_strategy=chunking_strategy
        )

    def _convert_code_stats_to_skeleton(self, stats: Dict, file_id: str):
        """
        Convert CodeSemanticCompressor stats to SkeletonResponse format.

        This ensures handlers get a consistent response format.
        """
        # Import here to avoid circular dependency
        from .semantic_compressor import SkeletonResponse

        # Build skeleton text from code chunks
        skeleton_lines = [f"## Code Structure: {file_id}", ""]

        # Build node_map from code chunks (P1-1 fix)
        node_map: Dict[str, str] = {}

        if self._code_compressor and file_id in self._code_compressor.graphs:
            graph = self._code_compressor.graphs[file_id]

            # Group by type
            imports = []
            classes = []
            functions = []
            blocks = []

            for node_id in graph.nodes():
                if node_id in self._code_compressor.chunks:
                    chunk = self._code_compressor.chunks[node_id]
                    # Build node_map entry
                    if hasattr(chunk, "name") and chunk.name:
                        desc = f"{chunk.chunk_type}: {chunk.name}"
                    else:
                        desc = chunk.chunk_type
                    node_map[node_id] = desc[:50]

                    if chunk.chunk_type == "import":
                        imports.append(chunk)
                    elif chunk.chunk_type == "class":
                        classes.append(chunk)
                    elif chunk.chunk_type == "function":
                        functions.append(chunk)
                    else:
                        blocks.append(chunk)

            if imports:
                skeleton_lines.append("### Imports")
                for imp in imports:
                    skeleton_lines.append(f"- {imp.name}")
                skeleton_lines.append("")

            if classes:
                skeleton_lines.append("### Classes")
                for cls in classes:
                    doc = f": {cls.docstring[:50]}..." if cls.docstring else ""
                    skeleton_lines.append(f"- `{cls.name}` (L{cls.start_line}-{cls.end_line}){doc}")
                skeleton_lines.append("")

            if functions:
                skeleton_lines.append("### Functions")
                for func in functions:
                    doc = f": {func.docstring[:50]}..." if func.docstring else ""
                    skeleton_lines.append(
                        f"- `{func.name}` (L{func.start_line}-{func.end_line}){doc}"
                    )
                skeleton_lines.append("")

            if blocks:
                skeleton_lines.append(f"### Code Blocks: {len(blocks)}")

        skeleton_text = "\n".join(skeleton_lines)

        # P0-1 fix: Use correct SkeletonResponse field names
        # Note: skeleton_tokens uses word count as approximation (not actual tokens).
        # For code files, compression_ratio is approximate since code tokenization differs.
        return SkeletonResponse(
            file_id=file_id,
            total_nodes=stats.get("total_chunks", 0),
            total_tokens=stats.get("total_tokens", 0),
            skeleton_tokens=len(skeleton_text.split()),  # Word count approximation
            compression_ratio=stats.get("compression_ratio", 1.0),
            skeleton_text=skeleton_text,
            node_map=node_map,
        )

    def _generate_skeleton(self, file_id: str, **kwargs):
        """Generate skeleton for a document (routes to appropriate compressor)."""
        if file_id in self._code_file_ids and self._code_compressor:
            return self._generate_code_skeleton(file_id)
        return self._text_compressor._generate_skeleton(file_id, **kwargs)

    def _generate_code_skeleton(self, file_id: str):
        """Generate skeleton from code compressor."""

        if not self._code_compressor or file_id not in self._code_compressor.graphs:
            raise ValueError(f"Code file {file_id} not found")

        stats = {"total_chunks": 0, "total_tokens": 0}
        if file_id in self._code_compressor.graphs:
            # Use delimiter-aware check: code nodes use :: separator
            stats["total_chunks"] = len(
                [
                    n
                    for n in self._code_compressor.graphs[file_id].nodes()
                    if n.startswith(f"{file_id}::")
                ]
            )

        return self._convert_code_stats_to_skeleton(stats, file_id)

    def modulate_region(
        self,
        node_ids: List[str],
        fidelity: FidelityLevel = FidelityLevel.STRUCTURE,
    ) -> str:
        """
        Modulate content fidelity for specified nodes.

        Routes to appropriate compressor based on node source.
        """
        # Determine which compressor owns these nodes
        text_nodes = []
        code_nodes = []

        for node_id in node_ids:
            if node_id in self._text_compressor.chunks:
                text_nodes.append(node_id)
            elif self._code_compressor and node_id in self._code_compressor.chunks:
                code_nodes.append(node_id)

        results = []

        if text_nodes:
            results.append(self._text_compressor.modulate_region(text_nodes, fidelity))

        if code_nodes:
            results.append(self._modulate_code_region(code_nodes, fidelity))

        return "\n\n".join(results)

    def _modulate_code_region(
        self,
        node_ids: List[str],
        fidelity: FidelityLevel,
    ) -> str:
        """Modulate code nodes with fidelity level.

        Output format varies by fidelity level:
        - RAW: Full code with Markdown fences (```), docstrings, line numbers
        - DETAILED: Headers, docstrings, and first 5 lines of code with Markdown fences
        - STRUCTURE: Inline code (`) with docstring preview
        - ABSTRACT/OUTLINE: Simple bullet points with name and type

        Note: Output is intentionally Markdown-formatted for downstream MCP tool consumption.
        Code is displayed in Markdown code fences for proper rendering.
        """
        if not self._code_compressor:
            return "Error: Code compressor not available"

        output_lines = []

        for node_id in node_ids:
            if node_id not in self._code_compressor.chunks:
                continue

            chunk = self._code_compressor.chunks[node_id]

            if fidelity == FidelityLevel.RAW:
                # Full code
                output_lines.append(f"### {chunk.name} ({chunk.chunk_type})")
                if chunk.docstring:
                    output_lines.append(f"Docstring: {chunk.docstring}")
                output_lines.append(f"Lines: {chunk.start_line}-{chunk.end_line}")
                output_lines.append("```")
                output_lines.append(chunk.code)
                output_lines.append("```")
            elif fidelity == FidelityLevel.STRUCTURE:
                # Structure only
                output_lines.append(
                    f"- `{chunk.name}` ({chunk.chunk_type}, L{chunk.start_line}-{chunk.end_line})"
                )
                if chunk.docstring:
                    output_lines.append(f"  Doc: {chunk.docstring[:100]}...")
            elif fidelity == FidelityLevel.DETAILED:
                # P0-2 fix: BALANCED doesn't exist, use DETAILED for rich output
                # Detailed - signature + docstring + preview
                output_lines.append(f"### {chunk.name} ({chunk.chunk_type})")
                if chunk.docstring:
                    output_lines.append(f"Doc: {chunk.docstring}")
                # First few lines of code
                code_lines = chunk.code.split("\n")[:5]
                output_lines.append("```")
                output_lines.append("\n".join(code_lines))
                if len(chunk.code.split("\n")) > 5:
                    output_lines.append("...")
                output_lines.append("```")
            else:  # ABSTRACT or SKELETON
                # Just name and type
                output_lines.append(f"- {chunk.name} ({chunk.chunk_type})")

            output_lines.append("")

        return "\n".join(output_lines)

    def search_semantic(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[str]:
        """
        Semantic search across all documents (legacy method for compatibility).

        Returns:
            List of node IDs ranked by relevance
        """
        return [node_id for node_id, _ in self.search_semantic_with_scores(query, file_id, top_k)]

    def search_semantic_with_scores(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Semantic search with similarity scores.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of (node_id, similarity_score) tuples ranked by relevance

        Note:
            P1-2 fix: Uses appropriate embedding model per domain.
            Text chunks use text model embedding, code chunks use code model embedding.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        candidates = []

        # Search text compressor chunks with text model embedding
        text_query_embedding = self._text_compressor.model.encode([query])[0]
        for node_id, node in self._text_compressor.chunks.items():
            # P2-2 fix: Use explicit delimiter to avoid prefix overlap
            if file_id and not node_id.startswith(f"{file_id}_"):
                continue
            similarity = cosine_similarity([text_query_embedding], [node.embedding])[0][0]
            candidates.append((node_id, float(similarity)))

        # Search code compressor chunks with code model embedding (if available)
        if self._code_compressor:
            # P1-2 fix: Use code model for code chunk queries
            code_query_embedding = self._code_compressor.model.encode([query])[0]
            for node_id, chunk in self._code_compressor.chunks.items():
                # P2-2 fix: Code chunks use :: separator, check appropriately
                if file_id:
                    # Code node IDs use :: separator (e.g., "file.py::function_name")
                    if not (
                        node_id.startswith(f"{file_id}::") or node_id.startswith(f"{file_id}_")
                    ):
                        continue
                if chunk.embedding is not None:
                    similarity = cosine_similarity([code_query_embedding], [chunk.embedding])[0][0]
                    candidates.append((node_id, float(similarity)))

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """Generate summary for text (delegates to text compressor)."""
        return self._text_compressor._generate_summary(text, max_length)

    def get_stats(self, file_id: Optional[str] = None) -> Dict:
        """Get statistics for documents."""
        if file_id:
            # Route to appropriate compressor
            if file_id in self._code_file_ids and self._code_compressor:
                return self._get_code_stats(file_id)
            return self._text_compressor.get_stats(file_id)

        # Aggregate stats
        text_stats = self._text_compressor.get_stats()

        result = {
            "total_documents": text_stats.get("total_documents", 0),
            "total_nodes": text_stats.get("total_nodes", 0),
            "total_tokens": text_stats.get("total_tokens", 0),
            "text_documents": text_stats.get("total_documents", 0),
            "code_documents": len(self._code_file_ids),
            "code_model_available": self._code_model_available,
        }

        if self._code_compressor:
            result["code_documents"] = len(self._code_file_ids)
            result["total_documents"] += len(self._code_file_ids)
            for fid in self._code_file_ids:
                if fid in self._code_compressor.graphs:
                    # Use delimiter-aware check: code nodes use :: separator
                    result["total_nodes"] += len(
                        [
                            n
                            for n in self._code_compressor.graphs[fid].nodes()
                            if n.startswith(f"{fid}::")
                        ]
                    )

        return result

    def _get_code_stats(self, file_id: str) -> Dict:
        """Get statistics for a code file."""
        if not self._code_compressor or file_id not in self._code_compressor.graphs:
            raise ValueError(f"Code file {file_id} not found")

        graph = self._code_compressor.graphs[file_id]
        # Use delimiter-aware check: code nodes use :: separator
        nodes = [nid for nid in graph.nodes() if nid.startswith(f"{file_id}::")]

        total_lines = 0
        chunk_types = {}

        for nid in nodes:
            if nid in self._code_compressor.chunks:
                chunk = self._code_compressor.chunks[nid]
                total_lines += chunk.end_line - chunk.start_line
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        return {
            "file_id": file_id,
            "type": "code",
            "total_nodes": len(nodes),
            "total_edges": graph.number_of_edges(),
            "total_lines": total_lines,
            "chunk_types": chunk_types,
            "language": self._code_compressor.file_metadata.get(file_id, {}).get(
                "language", "unknown"
            ),
            "metadata": self._code_compressor.file_metadata.get(file_id, {}),
        }

    def delete_document(self, file_id: str) -> bool:
        """Delete a document from the appropriate compressor."""
        deleted = False

        # Try text compressor
        if file_id in self._text_compressor.graphs:
            # Delete chunks - use delimiter-aware check to avoid prefix overlap
            # Text node IDs use format: {file_id}_n{i}
            chunks_to_delete = [
                k for k in self._text_compressor.chunks.keys() if k.startswith(f"{file_id}_")
            ]
            for chunk_id in chunks_to_delete:
                del self._text_compressor.chunks[chunk_id]
            # Delete graph
            del self._text_compressor.graphs[file_id]
            # Delete metadata
            if file_id in self._text_compressor.file_metadata:
                del self._text_compressor.file_metadata[file_id]
            deleted = True

        # Try code compressor
        if self._code_compressor and file_id in self._code_compressor.graphs:
            # Delete chunks - use delimiter-aware check to avoid prefix overlap
            # Code node IDs use format: {file_id}::{name}
            chunks_to_delete = [
                k for k in self._code_compressor.chunks.keys() if k.startswith(f"{file_id}::")
            ]
            for chunk_id in chunks_to_delete:
                del self._code_compressor.chunks[chunk_id]
            del self._code_compressor.graphs[file_id]
            if file_id in self._code_compressor.file_metadata:
                del self._code_compressor.file_metadata[file_id]
            self._code_file_ids.discard(file_id)
            deleted = True

        return deleted

    def cleanup(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=False)
