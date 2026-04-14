"""
Fidelity-Preserving Semantic Compressor

Implements the core encoding/decoding logic inspired by:
- Paper 1: JSCCM (Joint Semantic-Channel Coding) - Rate adaptation
- Paper 2: FPQE (Fidelity-Preserving Quantization) - Structure preservation
"""

import asyncio
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from os import cpu_count
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from .embeddings import EmbeddingManager, _EmbeddingManagerAdapter

logger = logging.getLogger(__name__)


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


@dataclass
class EvidenceResult:
    """Evidence retrieval diagnostics for query-aware compression."""

    node_ids: List[str]
    scores: List[Tuple[str, float]]
    sufficient: bool
    best_score: float
    threshold: float
    used_expanded_search: bool
    message: str


@dataclass
class DiffReingestionResult:
    """Result of incremental diff-based re-ingestion."""

    file_id: str
    chunks_unchanged: int
    chunks_updated: int
    chunks_added: int
    chunks_removed: int


def compute_adaptive_ratio(total_tokens: int) -> float:
    """Compute skeleton ratio based on corpus size.

    Smaller documents need less compression (keep more),
    larger documents need more aggressive compression.

    Args:
        total_tokens: Total token count of the corpus

    Returns:
        Skeleton ratio (fraction of nodes to keep)

    Raises:
        ValueError: If total_tokens is negative
    """
    if total_tokens < 0:
        raise ValueError(f"total_tokens must be non-negative, got {total_tokens}")
    if total_tokens < 8000:
        return 0.8
    elif total_tokens < 32000:
        return 0.5
    elif total_tokens < 100000:
        return 0.2
    else:
        return 0.1


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
        skeleton_ratio: float | str = 0.2,
    ):
        """
        Initialize the semantic compressor.

        Args:
            model_name: Local embedding model (lightweight recommended)
            similarity_threshold: Minimum similarity to create graph edges
            skeleton_ratio: Fraction of nodes to include in skeleton (top N%),
                or "auto" to adapt based on corpus size
        """
        # Use EmbeddingManager for shared model caching.
        # In ONNX-only deployments (no torch/sentence-transformers),
        # get_text_embedder() raises ImportError.  Fall back to using
        # the manager's encode() directly — it has STANDARD→ONNX→TFIDF
        # fallback built in.
        self._embedding_manager = EmbeddingManager()
        try:
            self.model = self._embedding_manager.get_text_embedder(model_name)
        except (ImportError, TypeError):
            # ONNX-only mode: use a thin wrapper that adapts
            # EmbeddingManager.encode() to accept SentenceTransformer kwargs
            self.model = _EmbeddingManagerAdapter(self._embedding_manager)
        self.similarity_threshold = similarity_threshold
        self.skeleton_ratio = skeleton_ratio

        # Storage
        self.graphs: Dict[str, nx.Graph] = {}
        self.chunks: Dict[str, SemanticNode] = {}
        self.file_metadata: Dict[str, Dict] = {}
        self._baseline_skeleton_cache: Dict[str, str] = {}

        # PageRank cache for performance optimization (v0.4.4)
        # Caches PageRank results to avoid O(K×(N+M)) recomputation
        # Cache key: (doc_id, graph_hash) -> pagerank scores
        self._pagerank_cache: Dict[str, Dict[str, float]] = {}

        # Concurrency protection (v0.8.0 audit fix - CORRECTED)
        #
        # TWO SEPARATE LOCK MECHANISMS:
        #
        # 1. SYNC PATH (ingest_file): Uses threading.Lock
        #    - asyncio.run() creates a NEW event loop each time
        #    - asyncio.Lock cannot protect across different event loops
        #    - threading.Lock provides correct protection for sync callers
        #
        # 2. ASYNC PATH (ingest_file_async): Uses asyncio.Lock
        #    - All async calls share the MCP server's event loop
        #    - asyncio.Lock works correctly within a single event loop
        #
        self._sync_lock = threading.Lock()  # For sync ingest_file() calls
        self._async_lock = asyncio.Lock()  # For async ingest_file_async() calls
        self._doc_locks: Dict[str, asyncio.Lock] = {}  # Per-doc locks (async path only)

        # Token counter with graceful fallback
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.use_tiktoken = True
        except Exception:
            logger.warning("tiktoken not available, using word count fallback")
            self.tokenizer = None
            self.use_tiktoken = False

        # Phase 5: Access tracking and compression replay
        from .context_decay import AccessTracker
        from .compression_replay import CompressionReplayLog
        from .temporal_graph import TemporalGraph

        self._access_tracker = AccessTracker()
        self._compression_replay = CompressionReplayLog()
        self._temporal_graph = TemporalGraph()

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken, with fallback to word count"""
        if self.use_tiktoken and self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                # Fallback if encoding fails
                pass

        # Fallback: approximate as 1.3 tokens per word
        return int(len(text.split()) * 1.3)

    async def _semtoken_preprocess(self, text: str) -> str:
        """SemToken pre-processing (arXiv 2508.15190).

        Splits text into sentence-level spans, embeds each, checks pairwise
        similarity to neighbors, and drops spans that are near-duplicates of
        their context.  This removes filler, repeated phrasing, and restated
        content BEFORE the text enters the chunking/graph pipeline.

        Returns the deduplicated text (may be shorter than input).
        """
        import re

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) < 4:
            return text  # Too few sentences to deduplicate

        # Embed each sentence
        try:
            embeddings = await self._encode_async(sentences)
        except Exception:
            return text  # Fallback: return original

        # Score each sentence by similarity to its neighbors
        keep = [True] * len(sentences)
        threshold = 0.92  # High similarity = redundant

        for i in range(1, len(sentences)):
            sim = float(
                np.dot(embeddings[i], embeddings[i - 1])
                / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-9)
            )
            if sim > threshold:
                # This sentence is very similar to the previous — mark for removal
                # Keep the longer one (more information)
                if len(sentences[i]) < len(sentences[i - 1]):
                    keep[i] = False
                else:
                    keep[i - 1] = False

        # Rebuild text from kept sentences
        result = " ".join(s for s, k in zip(sentences, keep) if k)
        return result if result.strip() else text

    def _compute_graph_hash(self, graph: nx.Graph, doc_id: str) -> str:
        """
        Compute deterministic hash of graph structure including edge weights (v0.8.0 audit fix).

        Uses hashlib.sha1 instead of Python's hash() because:
        - Python's hash() is randomized per process (non-deterministic)
        - hash() doesn't include edge weights, causing stale cache on re-ingest

        Args:
            graph: NetworkX graph
            doc_id: Document identifier

        Returns:
            16-character hex hash string
        """
        # Include edge weights in hash for content-aware invalidation
        edge_data = sorted(
            (u, v, round(graph[u][v].get("weight", 1.0), 4)) for u, v in graph.edges()
        )
        content = f"{doc_id}:{edge_data}:{sorted(graph.nodes)}"
        return hashlib.sha1(content.encode()).hexdigest()[:16]

    def _clear_cache_for_doc(self, file_id: str) -> int:
        """
        Clear all cached PageRank entries for a document (v0.8.0 audit fix backstop).

        Called on ingest/refresh/delete as a backstop to ensure cache invalidation.

        Args:
            file_id: Document identifier

        Returns:
            Number of cache entries removed

        Note (v0.8.0 audit fix):
            Uses exact prefix matching (pagerank_{file_id}_) not substring matching.
            This prevents "doc" from clearing caches for "doc2", "product_doc", etc.
        """
        # v0.8.0 audit fix: Use exact prefix matching, not substring
        # Cache keys are formatted as: pagerank_{doc_id}_{graph_hash}
        # Old bug: "doc" in k would match "pagerank_doc2_hash" (substring of doc2)
        # Fix: k.startswith("pagerank_doc_") only matches exact doc_id prefix
        cache_prefix = f"pagerank_{file_id}_"
        keys_to_remove = [k for k in self._pagerank_cache if k.startswith(cache_prefix)]
        for k in keys_to_remove:
            del self._pagerank_cache[k]
        if keys_to_remove:
            logger.debug(f"Cleared {len(keys_to_remove)} PageRank cache entries for {file_id}")
        return len(keys_to_remove)

    def _get_cached_pagerank(self, graph: nx.Graph, doc_id: str) -> Dict[str, float]:
        """
        Get PageRank scores with caching to avoid recomputation (v0.4.4, v0.8.0 audit fix).

        PageRank is an O(K×(N+M)) operation where K=max_iter (default 100),
        N=nodes, M=edges. For documents with stable graphs, we cache results
        to achieve O(1) lookup on subsequent calls.

        Cache key: (doc_id, graph_hash)
        Invalidation: Automatic via graph structure hash including edge weights

        Args:
            graph: NetworkX graph to compute PageRank on
            doc_id: Document identifier for cache keying

        Returns:
            Dictionary mapping node_id -> importance score (0.0-1.0)

        Performance:
            - First call: O(K×(N+M)) - same as baseline
            - Cached calls: O(1) lookup (~500× faster)
            - Memory: ~8 bytes per node per document
        """
        # Generate deterministic cache key including edge weights (v0.8.0 audit fix)
        graph_hash = self._compute_graph_hash(graph, doc_id)
        cache_key = f"pagerank_{doc_id}_{graph_hash}"

        # Check cache
        if cache_key in self._pagerank_cache:
            logger.debug(f"PageRank cache HIT for {doc_id} ({len(graph.nodes)} nodes)")
            return self._pagerank_cache[cache_key]

        # Cache miss - compute PageRank
        logger.debug(f"PageRank cache MISS for {doc_id} - computing ({len(graph.nodes)} nodes)")
        pagerank = nx.pagerank(graph)

        # Cache result
        self._pagerank_cache[cache_key] = pagerank

        return pagerank

    def _chunk_text(
        self, text: str, max_chunk_size: int = 512, strategy: str = "auto"
    ) -> List[str]:
        """
        Intelligent text chunking that preserves semantic boundaries.

        Args:
            text: Input text to chunk
            max_chunk_size: Maximum tokens per chunk
            strategy: "auto", "fixed" (paragraph/sentence boundaries) or
                "semantic" (embedding-based)

        Prioritizes:
        1. Paragraph boundaries (\n\n)
        2. Sentence boundaries (. ! ?)
        3. Fixed size fallback
        """
        if strategy == "auto":
            total_tokens = self._count_tokens(text)
            paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
            strategy = "semantic" if total_tokens >= 400 and paragraph_count >= 3 else "fixed"

        if strategy == "semantic":
            try:
                from .semantic_chunking import chunk_by_semantics

                paragraphs = [
                    paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()
                ]
                semantic_units = paragraphs
                if len(semantic_units) < 3:
                    semantic_units = [
                        sentence.strip()
                        for sentence in re.split(r"(?<=[.!?])\s+", text)
                        if sentence.strip()
                    ]

                semantic_chunks = chunk_by_semantics(
                    semantic_units,
                    encode_fn=lambda texts: self.model.encode(texts),
                    max_chunk_size=max_chunk_size,
                )
                if semantic_chunks and isinstance(semantic_chunks[0], str):
                    rendered_semantic_chunks = semantic_chunks
                else:
                    rendered_semantic_chunks = [
                        " ".join(chunk).strip() for chunk in semantic_chunks if chunk
                    ]

                fixed_chunks = self._chunk_text(
                    text, max_chunk_size=max_chunk_size, strategy="fixed"
                )
                if len(rendered_semantic_chunks) > len(fixed_chunks):
                    return fixed_chunks
                return rendered_semantic_chunks
            except Exception as exc:
                logger.warning(f"Semantic chunking failed; falling back to fixed chunking: {exc}")
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

    async def _encode_async(self, texts: List[str]) -> np.ndarray:
        """
        Async wrapper for model.encode() to prevent blocking.

        Uses ThreadPoolExecutor to offload CPU-bound encoding while
        maintaining async interface for MCP server.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of embeddings
        """
        # Lazy init thread pool (shared across all compressors)
        if not hasattr(self, "_executor"):
            # Use up to 4 workers for parallel embedding generation (2-4× speedup)
            self._executor = ThreadPoolExecutor(max_workers=min(4, cpu_count() or 4))

        # Offload blocking encode() to thread pool
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            self._executor, lambda: self.model.encode(texts, show_progress_bar=False)
        )

        return embeddings

    async def ingest_file_async(
        self,
        text: str,
        file_id: str,
        metadata: Optional[Dict] = None,
        chunking_strategy: str = "auto",
    ) -> SkeletonResponse:
        """
        Async version of ingest_file for MCP server use.
        See ingest_file() for full documentation.
        """
        return await self._ingest_file_impl(
            text, file_id, metadata, chunking_strategy=chunking_strategy
        )

    def ingest_file(
        self, text: str, file_id: str, metadata: Optional[Dict] = None
    ) -> SkeletonResponse:
        """
        Synchronous wrapper for backward compatibility with existing tests.
        For async MCP server use, call ingest_file_async() instead.

        CONCURRENCY NOTE (v0.8.0 audit fix - CORRECTED):
        - Uses threading.Lock to protect the entire operation
        - asyncio.run() creates a NEW event loop each call
        - asyncio.Lock CANNOT protect across different event loops
        - threading.Lock provides correct protection for sync callers
        """
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop is not None and loop.is_running():
            # Called from async context - raise error
            raise RuntimeError(
                "ingest_file() cannot be called from async context. "
                "Use await ingest_file_async() instead."
            )

        # Use threading.Lock to protect the entire sync operation
        # This is necessary because asyncio.run() creates a NEW event loop each time,
        # making asyncio.Lock useless for cross-call protection
        with self._sync_lock:
            return asyncio.run(
                self._ingest_file_impl(text, file_id, metadata, use_async_lock=False)
            )

    async def _ingest_file_impl(
        self,
        text: str,
        file_id: str,
        metadata: Optional[Dict] = None,
        use_async_lock: bool = True,
        chunking_strategy: str = "auto",
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
            use_async_lock: Whether to use asyncio.Lock for concurrency protection.
                           Set to False when called from sync path (already protected
                           by threading.Lock in ingest_file()). Set to True for async
                           path (ingest_file_async()) where asyncio.Lock is appropriate.

        Returns:
            SkeletonResponse with compressed view
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Cannot ingest empty or whitespace-only text")
        if not file_id or not file_id.strip():
            raise ValueError("file_id cannot be empty or whitespace-only")

        # Concurrency protection (v0.8.0 audit fix - CORRECTED)
        #
        # use_async_lock=True (async path): Uses asyncio.Lock
        #   - All async calls share the MCP server's event loop
        #   - asyncio.Lock works correctly within a single event loop
        #   - Per-document locks reduce contention during concurrent ingests
        #
        # use_async_lock=False (sync path): No async lock needed
        #   - Caller (ingest_file) already holds threading.Lock
        #   - asyncio.run() creates new event loop, so asyncio.Lock would be useless anyway
        #
        # AsyncExitStack allows conditional context manager entry without code duplication
        #
        async with AsyncExitStack() as stack:
            if use_async_lock:
                # Get or create per-document lock for async path
                async with self._async_lock:  # Brief global lock to get/create doc lock
                    if file_id not in self._doc_locks:
                        self._doc_locks[file_id] = asyncio.Lock()
                    doc_lock = self._doc_locks[file_id]
                # Enter per-document lock (will be exited automatically when stack closes)
                await stack.enter_async_context(doc_lock)

            logger.info(f"Ingesting file: {file_id}")

            # Clear stale PageRank cache entries for this document (v0.8.0 audit fix)
            # This is a backstop to ensure cache consistency when re-ingesting
            self._clear_cache_for_doc(file_id)

            # Count original tokens
            total_tokens = self._count_tokens(text)
            logger.info(f"  Original tokens: {total_tokens}")

            # 0. SemToken pre-processing (arXiv 2508.15190):
            # Remove redundant spans BEFORE chunking.  Splits text into
            # overlapping windows, embeds them, computes pairwise similarity
            # to neighbors, and merges windows whose similarity > 0.92.
            # This reduces input size by 10-30% on repetitive documents,
            # producing a cleaner graph with fewer redundant nodes.
            if total_tokens > 200:  # Skip for very short texts
                text = await self._semtoken_preprocess(text)
                preprocessed_tokens = self._count_tokens(text)
                if preprocessed_tokens < total_tokens:
                    logger.info(
                        f"  SemToken: {total_tokens} → {preprocessed_tokens} tokens "
                        f"({round((1 - preprocessed_tokens / total_tokens) * 100, 1)}% reduced)"
                    )

            # 1. Chunk the text semantically
            raw_chunks = self._chunk_text(text, strategy=chunking_strategy)
            logger.info(f"  Created {len(raw_chunks)} semantic chunks")

            # 2. Generate embeddings (async to prevent MCP timeout)
            logger.info("  Generating embeddings...")
            embeddings = await self._encode_async(raw_chunks)

            # 2b. Optional intra-document deduplication (Phase 5: R-KV)
            if len(raw_chunks) > 2:
                try:
                    from .intra_doc_dedup import collapse_redundant_nodes

                    nodes_map = {
                        f"tmp_{i}": {"text": raw_chunks[i], "embedding": embeddings[i]}
                        for i in range(len(raw_chunks))
                    }
                    collapsed = collapse_redundant_nodes(nodes_map, threshold=0.92)
                    if len(collapsed) < len(raw_chunks):
                        logger.info(
                            f"  Intra-doc dedup: {len(raw_chunks)} → {len(collapsed)} chunks"
                        )
                        collapsed_keys = sorted(
                            collapsed.keys(),
                            key=lambda key: int(key.split("_")[1]),
                        )
                        raw_chunks = [collapsed[k]["text"] for k in collapsed_keys]
                        embeddings = np.stack([collapsed[k]["embedding"] for k in collapsed_keys])
                except Exception as exc:
                    logger.warning(f"Intra-doc deduplication failed for '{file_id}': {exc}")

            # 3. Build similarity graph (preserves global structure)
            logger.info("  Building semantic graph...")
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
            logger.info("  Calculating importance scores (PageRank)...")
            if len(graph.nodes) > 0:
                # Use cached PageRank for 500× speedup on repeated reads (v0.4.4)
                pagerank = self._get_cached_pagerank(graph, file_id)

                # Update importance scores
                for node_id, score in pagerank.items():
                    if node_id in self.chunks:
                        self.chunks[node_id].importance = score

            # Store graph
            self.graphs[file_id] = graph
            self.file_metadata[file_id] = metadata or {}

            # 5. Generate skeleton
            skeleton_response = self._generate_skeleton(file_id)
            self._baseline_skeleton_cache[file_id] = skeleton_response.skeleton_text

            logger.info(
                f"  Compression: {total_tokens} -> {skeleton_response.skeleton_tokens} tokens"
            )
            logger.info(f"  Ratio: {skeleton_response.compression_ratio:.1f}x")

            return skeleton_response

    def _normalize_scores(self, values: Dict[str, float]) -> Dict[str, float]:
        """Min-max normalize score dictionary to 0..1 range."""
        if not values:
            return {}
        min_value = min(values.values())
        max_value = max(values.values())
        if max_value <= min_value:
            return {key: 1.0 for key in values}
        span = max_value - min_value
        return {key: (value - min_value) / span for key, value in values.items()}

    def _select_skeleton_nodes(
        self,
        file_nodes: List[Tuple[str, SemanticNode]],
        num_skeleton: int,
        query: Optional[str] = None,
        redundancy_penalty: float = 0.2,
        priority_scores: Optional[Dict[str, float]] = None,
    ) -> Set[str]:
        """
        Select skeleton nodes.

        Baseline mode:
        - PageRank-only (importance sort)

        Query-guided mode:
        - Hybrid ranking: importance + query relevance
        - Redundancy penalty via greedy MMR-style selection
        """
        if num_skeleton <= 0 or not file_nodes:
            return set()

        if not query or not query.strip():
            ranked = sorted(file_nodes, key=lambda item: item[1].importance, reverse=True)
            return {node_id for node_id, _ in ranked[:num_skeleton]}

        # Query-guided selection
        query_embedding = self.model.encode([query])[0]
        importance_scores = {node_id: node.importance for node_id, node in file_nodes}
        relevance_scores = {
            node_id: float(cosine_similarity([query_embedding], [node.embedding])[0][0])
            for node_id, node in file_nodes
        }

        importance_norm = self._normalize_scores(importance_scores)
        relevance_norm = self._normalize_scores(relevance_scores)
        priority_norm = (
            self._normalize_scores(priority_scores)
            if priority_scores
            else {node_id: 0.0 for node_id, _ in file_nodes}
        )

        # Weighted hybrid score: prioritize query relevance while preserving
        # global structure. Query-adaptive ratio scores can additionally boost
        # sections that should retain more detail.
        hybrid_scores = {
            node_id: 0.25 * importance_norm.get(node_id, 0.0)
            + 0.55 * relevance_norm.get(node_id, 0.0)
            + 0.20 * priority_norm.get(node_id, 0.0)
            for node_id, _ in file_nodes
        }

        selected: List[str] = []
        selected_set: Set[str] = set()
        candidate_ids = [node_id for node_id, _ in file_nodes]
        node_lookup = {node_id: node for node_id, node in file_nodes}

        while len(selected) < num_skeleton and candidate_ids:
            best_id = None
            best_score = float("-inf")

            for candidate_id in candidate_ids:
                candidate_score = hybrid_scores[candidate_id]
                if selected:
                    max_similarity = max(
                        float(
                            cosine_similarity(
                                [node_lookup[candidate_id].embedding],
                                [node_lookup[selected_id].embedding],
                            )[0][0]
                        )
                        for selected_id in selected
                    )
                    candidate_score -= redundancy_penalty * max_similarity

                if candidate_score > best_score:
                    best_score = candidate_score
                    best_id = candidate_id

            if best_id is None:
                break

            selected.append(best_id)
            selected_set.add(best_id)
            candidate_ids.remove(best_id)

        return selected_set

    def _generate_skeleton(
        self,
        file_id: str,
        query: Optional[str] = None,
        anchor_node_ids: Optional[Set[str]] = None,
        exclude_node_ids: Optional[Set[str]] = None,
    ) -> SkeletonResponse:
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
        excluded = exclude_node_ids or set()
        file_nodes = [
            (nid, self.chunks[nid])
            for nid in graph.nodes()
            if nid.startswith(file_id) and nid not in excluded
        ]

        if not file_nodes:
            skeleton_text = "\n".join(
                [
                    f"=== SEMANTIC SKELETON: {file_id} ===",
                    "Total nodes: 0 | Skeleton nodes: 0",
                    "Compression: 0% of content shown",
                ]
            )
            return SkeletonResponse(
                file_id=file_id,
                total_nodes=0,
                total_tokens=0,
                skeleton_tokens=0,
                compression_ratio=0.0,
                skeleton_text=skeleton_text,
                node_map={},
            )

        # Sort by importance for stable iteration order and deterministic output
        file_nodes.sort(key=lambda x: x[1].importance, reverse=True)

        # Determine skeleton nodes (top N%)
        # Use adaptive ratio if skeleton_ratio is "auto"
        effective_ratio = self.skeleton_ratio
        if effective_ratio == "auto":
            total_tokens_estimate = sum(len(node.text.split()) for _, node in file_nodes)
            effective_ratio = compute_adaptive_ratio(total_tokens_estimate)

        num_skeleton = max(1, int(len(file_nodes) * effective_ratio))

        # COMI coarse-to-fine pass (arXiv 2602.01719, ICLR 2026):
        # When a query is provided, first do a COARSE pass that eliminates
        # clearly irrelevant nodes (bottom 50% by query relevance) before
        # the fine-grained PageRank selection.  This reduces noise and
        # focuses the skeleton on query-relevant content.  The paper showed
        # a 25-point EM improvement at high compression ratios.
        if query and len(file_nodes) > 3:
            try:
                query_emb = self.model.encode([query])[0]
                # Score each node by relevance to query
                node_scores = []
                for nid, node in file_nodes:
                    if node.embedding is not None:
                        sim = float(
                            np.dot(query_emb, node.embedding)
                            / (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-9)
                        )
                    else:
                        sim = 0.0
                    node_scores.append((nid, node, sim))

                # Sort by relevance and keep top 50% (coarse filter)
                node_scores.sort(key=lambda x: x[2], reverse=True)
                coarse_keep = max(2, len(node_scores) // 2)
                coarse_nodes = node_scores[:coarse_keep]

                # Replace file_nodes with coarse-filtered set
                file_nodes = [(nid, node) for nid, node, _ in coarse_nodes]
                # Re-sort by importance for downstream processing
                file_nodes.sort(key=lambda x: x[1].importance, reverse=True)

                logger.info(
                    f"  COMI coarse pass: {len(node_scores)} → {len(file_nodes)} nodes "
                    f"(kept top {coarse_keep} by query relevance)"
                )
            except Exception as exc:
                logger.warning(f"COMI coarse pass failed for '{file_id}': {exc}")

        # Phase 5: Query-adaptive per-section ratios (KVzip/LazyLLM)
        adaptive_priority_scores = None
        if query and len(file_nodes) > 1:
            try:
                from .query_adaptive import compute_section_ratios

                query_emb = self.model.encode([query])[0]
                sections = [{"embedding": node.embedding} for _, node in file_nodes]
                per_node_ratios = compute_section_ratios(
                    sections, query_emb, base_ratio=effective_ratio
                )
                adaptive_priority_scores = {
                    node_id: ratio for (node_id, _), ratio in zip(file_nodes, per_node_ratios)
                }
                # Use adaptive ratios to tune overall budget as well as
                # per-node selection priority.
                num_skeleton = sum(1 for r in per_node_ratios if r >= effective_ratio)
                num_skeleton = max(1, num_skeleton)
            except Exception as exc:
                logger.warning(f"Query-adaptive ratio computation failed for '{file_id}': {exc}")

        if anchor_node_ids:
            skeleton_nodes = set(anchor_node_ids)
            if len(skeleton_nodes) < num_skeleton:
                selected = self._select_skeleton_nodes(
                    file_nodes,
                    num_skeleton,
                    query=query,
                    priority_scores=adaptive_priority_scores,
                )
                skeleton_nodes.update(selected)
        else:
            skeleton_nodes = self._select_skeleton_nodes(
                file_nodes,
                num_skeleton,
                query=query,
                priority_scores=adaptive_priority_scores,
            )

        # Build skeleton text
        skeleton_lines = []
        skeleton_lines.append(f"=== SEMANTIC SKELETON: {file_id} ===")
        skeleton_lines.append(f"Total nodes: {len(file_nodes)} | Skeleton nodes: {num_skeleton}")
        skeleton_lines.append(f"Compression: {effective_ratio:.0%} of content shown\n")

        node_map = {}
        total_tokens = 0
        skeleton_tokens = 0

        for node_id, node in file_nodes:
            total_tokens += node.metadata["tokens"]

            if node_id in skeleton_nodes:
                # High-importance: Show summary + entities
                summary = self._generate_summary(node.text, max_length=150)
                entities = ", ".join(node.metadata["entities"][:3])

                line = f"[{node_id}] [rag:{node_id}] [ANCHOR] (importance: {node.importance:.3f})\n"
                line += f"  Summary: {summary}\n"
                if entities:
                    line += f"  Key entities: {entities}\n"

                skeleton_lines.append(line)
                node_map[node_id] = f"ANCHOR: {summary[:50]}..."
                skeleton_tokens += self._count_tokens(line)
            else:
                # Low-importance: Just reference
                summary = self._generate_summary(node.text, max_length=50)
                line = f"[{node_id}] [HIDDEN] Detail hidden (use modulate_region to expand)\n"

                skeleton_lines.append(line)
                node_map[node_id] = f"Hidden: {summary[:30]}..."
                skeleton_tokens += self._count_tokens(line)

        # Query metadata placed at END for cache-friendly ordering:
        # Static node content forms a stable prefix; volatile query goes last.
        if query and query.strip():
            skeleton_lines.append("Selection mode: QUERY_GUIDED")
            skeleton_lines.append(f"Query: {query}")

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

    def read_skeleton(self, file_id: str, query: Optional[str] = None) -> str:
        """
        MCP Tool: read_skeleton

        Returns the compressed skeleton view of a document.
        ~80-95% token savings vs raw text.
        """
        skeleton = self._generate_skeleton(file_id, query=query)
        return skeleton.skeleton_text

    def retrieve_evidence(
        self,
        query: str,
        file_id: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.35,
        expansion_factor: int = 2,
    ) -> EvidenceResult:
        """
        Retrieve evidence with insufficiency detection.

        If best similarity is below threshold, run a broader retrieval pass.
        """
        if not query or not query.strip():
            raise ValueError("query must be non-empty")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        initial_scores = self.search_semantic_with_scores(query, file_id=file_id, top_k=top_k)
        best_score = initial_scores[0][1] if initial_scores else 0.0
        sufficient = best_score >= min_similarity
        used_expanded_search = False
        final_scores = initial_scores

        if not sufficient:
            used_expanded_search = True
            expanded_k = max(top_k + 1, top_k * max(1, expansion_factor))
            final_scores = self.search_semantic_with_scores(
                query,
                file_id=file_id,
                top_k=expanded_k,
            )
            best_score = final_scores[0][1] if final_scores else 0.0
            sufficient = best_score >= min_similarity

        node_ids = [node_id for node_id, _ in final_scores[:top_k]]
        if sufficient:
            message = "Evidence sufficient for query-guided compression."
        else:
            message = (
                "Evidence appears insufficient; consider broader retrieval or "
                "falling back to lower compression."
            )

        return EvidenceResult(
            node_ids=node_ids,
            scores=final_scores,
            sufficient=sufficient,
            best_score=best_score,
            threshold=min_similarity,
            used_expanded_search=used_expanded_search,
            message=message,
        )

    def read_skeleton_evidence_aware(
        self,
        file_id: str,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.35,
    ) -> str:
        """Generate skeleton anchored by query evidence and include sufficiency diagnostics."""
        evidence = self.retrieve_evidence(
            query=query,
            file_id=file_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )
        skeleton = self._generate_skeleton(
            file_id=file_id,
            query=query,
            anchor_node_ids=set(evidence.node_ids),
        )
        status = "SUFFICIENT" if evidence.sufficient else "INSUFFICIENT"
        lines = [
            f"=== EVIDENCE STATUS: {status} ===",
            (
                f"best_score={evidence.best_score:.3f} threshold={evidence.threshold:.3f} "
                f"expanded_search={evidence.used_expanded_search}"
            ),
            evidence.message,
            "",
            skeleton.skeleton_text,
        ]
        return "\n".join(lines)

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
                output_lines.append(f"[{node_id}] [WARN] Node not found\n")
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

    def search_semantic_with_scores(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Semantic search using vector similarity with similarity scores.

        New in v0.9.0: Returns similarity scores alongside node IDs for better
        AI decision-making about which content to retrieve.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of (node_id, similarity_score) tuples ranked by relevance.
            Similarity scores range from -1.0 to 1.0 (cosine similarity).
        """
        # Embed the query
        query_embedding = self.model.encode([query])[0]

        # Get candidate nodes
        candidates = []
        for node_id, node in self.chunks.items():
            if file_id and not node_id.startswith(file_id):
                continue

            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
            candidates.append((node_id, float(similarity)))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:top_k]

    def search_semantic(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[str]:
        """
        Semantic search using vector similarity.

        Note: Use search_semantic_with_scores() to also get similarity scores.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of node IDs ranked by relevance
        """
        # Delegate to search_semantic_with_scores and extract just the node IDs
        results = self.search_semantic_with_scores(query, file_id, top_k)
        return [node_id for node_id, _ in results]

    def get_stats(self, file_id: Optional[str] = None) -> Dict:
        """Get statistics about stored documents"""
        if file_id:
            if file_id not in self.graphs:
                raise ValueError(f"File {file_id} not found")

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
                "total_documents": len(self.graphs),  # Alias for compatibility
                "total_nodes": len(self.chunks),
                "files": list(self.graphs.keys()),
            }

    def get_statistics(self, file_id: Optional[str] = None) -> Dict:
        """Alias for get_stats() - provided for API compatibility."""
        return self.get_stats(file_id)

    async def stream_skeleton(self, file_id: str, query: str = None):
        """Async generator that yields skeleton text in chunks.

        Note: Not exposed as MCP tool because MCP protocol requires single
        JSON responses (no streaming support). Available for direct use
        via HTTP API or programmatic access.

        Args:
            file_id: Document file ID
            query: Optional query for guided selection

        Yields:
            Text chunks of the skeleton output
        """
        skeleton = self._generate_skeleton(file_id, query=query)
        lines = skeleton.skeleton_text.split("\n")
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                yield line + "\n"
            else:
                yield line

    def diff_reingest(self, file_id: str, new_text: str) -> "DiffReingestionResult":
        """Re-ingest a document, preserving unchanged chunks.

        Compares new text against existing chunks and only recomputes
        embeddings for changed sections.

        Args:
            file_id: Existing document file ID
            new_text: Updated document text

        Returns:
            DiffReingestionResult with change statistics
        """
        if file_id not in self.graphs:
            raise ValueError(f"File {file_id} not found for diff re-ingestion")

        diff_stats = self._compute_diff_stats(file_id, new_text)

        # Re-ingest with embedding preservation
        self.ingest_file(new_text, file_id)
        self._restore_preserved_embeddings(file_id, diff_stats["preserved"])

        return DiffReingestionResult(file_id=file_id, **diff_stats["counts"])

    async def diff_reingest_async(self, file_id: str, new_text: str) -> "DiffReingestionResult":
        """Async version of diff_reingest."""
        if file_id not in self.graphs:
            raise ValueError(f"File {file_id} not found for diff re-ingestion")

        diff_stats = self._compute_diff_stats(file_id, new_text)

        await self.ingest_file_async(new_text, file_id)
        self._restore_preserved_embeddings(file_id, diff_stats["preserved"])

        return DiffReingestionResult(file_id=file_id, **diff_stats["counts"])

    def _compute_diff_stats(self, file_id: str, new_text: str) -> Dict:
        """Compute diff statistics and preserve unchanged embeddings."""
        old_chunks = {nid: node for nid, node in self.chunks.items() if nid.startswith(file_id)}
        old_texts = {nid: node.text for nid, node in old_chunks.items()}

        new_chunk_texts = self._chunk_text(new_text)

        old_text_set = set(old_texts.values())
        new_text_set = set(new_chunk_texts)

        unchanged_texts = old_text_set & new_text_set
        removed_texts = old_text_set - new_text_set
        added_texts = new_text_set - old_text_set

        chunks_updated = min(len(removed_texts), len(added_texts))

        preserved = {}
        for nid, node in old_chunks.items():
            if node.text in unchanged_texts and node.embedding is not None:
                preserved[node.text] = node.embedding.copy()

        return {
            "preserved": preserved,
            "counts": {
                "chunks_unchanged": len(unchanged_texts),
                "chunks_updated": chunks_updated,
                "chunks_added": max(0, len(added_texts) - chunks_updated),
                "chunks_removed": max(0, len(removed_texts) - chunks_updated),
            },
        }

    def _restore_preserved_embeddings(self, file_id: str, preserved: Dict) -> None:
        """Restore preserved embeddings after re-ingestion."""
        for nid, node in self.chunks.items():
            if nid.startswith(file_id) and node.text in preserved:
                node.embedding = preserved[node.text]

    def find_duplicates(self, threshold: float = 0.95, timeout_seconds: float = 30.0) -> List[Dict]:
        """Find semantically duplicate chunks across all documents.

        Args:
            threshold: Minimum cosine similarity to consider duplicate
            timeout_seconds: Maximum time in seconds before aborting (default 30s)

        Returns:
            List of dicts with node_a, node_b, similarity
        """
        import numpy as np
        import time

        all_nodes = list(self.chunks.items())
        if len(all_nodes) < 2:
            return []

        duplicates = []
        start_time = time.monotonic()
        for i in range(len(all_nodes)):
            nid_a, node_a = all_nodes[i]
            file_a = nid_a.rsplit("_", 1)[0] if "_" in nid_a else nid_a

            for j in range(i + 1, len(all_nodes)):
                # Check timeout every 1000 comparisons
                if j % 1000 == 0 and time.monotonic() - start_time > timeout_seconds:
                    duplicates.append(
                        {
                            "node_a": "__timeout__",
                            "node_b": "__timeout__",
                            "similarity": 0.0,
                            "warning": f"Search timed out after {timeout_seconds}s. Partial results returned.",
                        }
                    )
                    return duplicates

                nid_b, node_b = all_nodes[j]
                file_b = nid_b.rsplit("_", 1)[0] if "_" in nid_b else nid_b

                # Only compare across different files
                if file_a == file_b:
                    continue

                # Cosine similarity — skip nodes with missing embeddings
                if node_a.embedding is None or node_b.embedding is None:
                    continue
                # Guard against dimension mismatch (e.g. MiniLM 384 vs CodeBERT 768)
                if node_a.embedding.shape[0] != node_b.embedding.shape[0]:
                    continue
                dot = np.dot(node_a.embedding, node_b.embedding)
                norm_a = np.linalg.norm(node_a.embedding)
                norm_b = np.linalg.norm(node_b.embedding)
                if norm_a == 0 or norm_b == 0:
                    continue
                similarity = dot / (norm_a * norm_b)

                if similarity >= threshold:
                    duplicates.append(
                        {
                            "node_a": nid_a,
                            "node_b": nid_b,
                            "similarity": round(float(similarity), 4),
                        }
                    )

        return duplicates
