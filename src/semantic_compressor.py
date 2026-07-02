"""
Fidelity-Preserving Semantic Compressor

Implements the core encoding/decoding logic inspired by:
- Paper 1: JSCCM (Joint Semantic-Channel Coding) - Rate adaptation
- Paper 2: FPQE (Fidelity-Preserving Quantization) - Structure preservation
"""

import asyncio
import hashlib
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from dataclasses import dataclass
from os import cpu_count
from typing import Dict, List, Literal, Optional, Set, Tuple
from enum import Enum

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from .embeddings import EmbeddingManager, _EmbeddingManagerAdapter
from .bm25_utils import bm25_scores as _bm25_score_texts
from .constants import _RRF_K, F11_RANKER_PATH, DEFAULT_TEXT_MODEL
from .node_identity import extract_file_id_from_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory-safety constants for graph construction (OOM fix)
# ---------------------------------------------------------------------------

# Process embeddings in row-blocks of this size when building the similarity
# graph. Peak similarity memory is O(block × N) rather than O(N²).
# Overridable via SIMILARITY_BLOCK_SIZE env var.
_SIMILARITY_BLOCK_SIZE: int = int(os.environ.get("SIMILARITY_BLOCK_SIZE", "256"))

# Hard ceiling on the number of chunks that participate in the dense edge-
# building loop. Nodes above this index are still created (with uniform
# PageRank) but no similarity edges are computed for them, bounding both
# peak memory and O(N²) edge-build time for pathologically large documents.
# Overridable via MAX_GRAPH_CHUNKS env var.
_MAX_GRAPH_CHUNKS: int = int(os.environ.get("MAX_GRAPH_CHUNKS", "2500"))

# NOTE (audit re-fix): the former _RRF_SUFFICIENCY_THRESHOLD constant was
# removed. retrieve_evidence() once gated sufficiency on the RRF fusion score
# under F11 Path C, but RRF encodes rank POSITION not relevance MAGNITUDE — the
# rank-1 node of any non-empty doc has RRF >= 1/(k+1) ≈ 0.0164, so the bar was
# cleared unconditionally (sufficient=True for every query, incl. irrelevant
# ones). Sufficiency now thresholds on the dense COSINE magnitude of the
# top-ranked candidate against min_similarity for BOTH paths; RRF remains the
# Path C ranking/ordering method. See SemanticCompressor._max_dense_cosine.


def _node_belongs_to_file(node_id: str, file_id: str) -> bool:
    """Boundary-safe membership test for ``node_id`` against ``file_id``.

    Audit P1-5: a bare ``node_id.startswith(file_id)`` collides across file_ids
    that share a prefix — ``"foobar_n0".startswith("foo")`` is True, so document
    ``"foo"`` leaked ``"foobar"`` nodes into skeletons, search, stats and diff
    re-ingestion. The fix compares the EXTRACTED file_id (boundary-aware for both
    text ``"{file_id}_n{i}"`` and code ``"{file_id}::{symbol}"`` node formats)
    against the target, which makes the membership test collision-proof.
    """
    return extract_file_id_from_node(node_id) == file_id


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
        model_name: str = DEFAULT_TEXT_MODEL,
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
        # Retain the requested model id so model-aware tuning (e.g. the
        # semantic-chunking boundary threshold, A1 calibration) can adapt to the
        # active encoder's similarity distribution.
        self.model_name = model_name
        try:
            self.model = self._embedding_manager.get_text_embedder(model_name)
        except (ImportError, TypeError):
            # ONNX-only mode: use a thin wrapper that adapts
            # EmbeddingManager.encode() to accept SentenceTransformer kwargs
            self.model = _EmbeddingManagerAdapter(self._embedding_manager)
        self.similarity_threshold = similarity_threshold
        self.skeleton_ratio = skeleton_ratio

        # B1 (modernization roadmap 2026-06-08): COMI/MIG redundancy weight for
        # the query-guided skeleton selector. Sourced from the canonical COMI
        # MIGConfig default (arXiv 2602.01719, lambda_redundancy=0.5) so there is
        # a SINGLE source of truth. When a query is present, ``_generate_skeleton``
        # threads this into the VECTORIZED ``_select_skeleton_nodes`` redundancy
        # term (preserving A2's numpy matmul path) instead of the legacy fixed
        # 0.2. Callers can override per-session via ``set_lambda_redundancy`` or
        # per-call by passing ``redundancy_penalty`` to ``_select_skeleton_nodes``.
        from .token_refiner import MIGConfig

        self.lambda_redundancy: float = MIGConfig().lambda_redundancy

        # Storage
        self.graphs: Dict[str, nx.Graph] = {}
        self.chunks: Dict[str, SemanticNode] = {}
        self.file_metadata: Dict[str, Dict] = {}
        self._baseline_skeleton_cache: Dict[str, str] = {}
        # Audit P2-4: cache the baseline skeleton's token counts at ingest so
        # get_stats() can report skeleton_tokens / compression_ratio WITHOUT
        # re-invoking _generate_skeleton (which had MIG-mutation + cost side
        # effects). Keyed by file_id → {"skeleton_tokens": int, "ratio": float}.
        self._baseline_skeleton_stats: Dict[str, Dict[str, float]] = {}

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

    # F11: compiled once at class level to avoid re-compiling per call
    _HEADING_FIRST_LINE_RE = re.compile(r"^(#{1,6}) (.+?)[ \t]*$", re.MULTILINE)

    def _split_oversized_section(self, section: str, max_chunk_size: int) -> List[str]:
        """
        Sub-split a heading section that exceeds max_chunk_size × 1.5.

        Prepends the heading text to each child chunk so the heading's semantic
        signal is preserved in every child embedding (F11 fix).
        """
        heading_match = self._HEADING_FIRST_LINE_RE.match(section.strip())
        heading_prefix = (heading_match.group(0).rstrip() + "\n\n") if heading_match else ""
        body = section[len(heading_prefix) :].strip() if heading_prefix else section
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        chunks: List[str] = []
        current = heading_prefix
        for para in paragraphs:
            sep = "\n\n" if current.strip() else ""
            candidate = current + sep + para
            if self._count_tokens(candidate) <= max_chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # Start fresh sub-chunk with heading prefix
                current = heading_prefix + para

        if current:
            chunks.append(current)
        return chunks if chunks else [section]

    def _extract_heading_metadata(self, text: str) -> dict:
        """
        Return heading metadata for a chunk that begins with a markdown heading.

        Used to enrich SemanticNode.metadata so heading signal is queryable
        without re-parsing during selection (F11).
        """
        stripped = text.strip()
        match = self._HEADING_FIRST_LINE_RE.match(stripped)
        if not match:
            return {}
        return {
            "heading_level": len(match.group(1)),
            "heading_text": match.group(2).strip(),
            "chunking_strategy_resolved": "markdown_section_v1",
        }

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

                # A1 calibration (2026-06-08): use a MODEL-AWARE boundary
                # threshold. bge-small-en-v1.5 has a ~+0.25 higher baseline
                # inter-paragraph cosine than all-MiniLM-L6-v2; the legacy fixed
                # 0.5 boundary would never fire under bge (distinct topics score
                # >0.5), collapsing multi-topic docs into a single chunk. The
                # per-model threshold keeps chunk granularity comparable across
                # encoders. See constants.get_semantic_chunk_boundary_threshold.
                from .constants import get_semantic_chunk_boundary_threshold

                boundary_threshold = get_semantic_chunk_boundary_threshold(
                    getattr(self, "model_name", None)
                )

                semantic_chunks = chunk_by_semantics(
                    semantic_units,
                    encode_fn=lambda texts: self.model.encode(texts),
                    threshold=boundary_threshold,
                    max_chunk_size=max_chunk_size,
                    token_count_fn=self._count_tokens,
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
        # F4-followup / F11-fix: for structured markdown, split strictly on H2/H3 boundaries
        # so that each major section becomes its own node, regardless of the
        # min-token floor.  This prevents 3-4 sections being bundled into one
        # node (e.g. a 1592-token doc with 15 H2 sections yielding only 4 nodes).
        #
        # F11 root cause: the old gate required BOTH ≥3 headings AND ≥3 list items.
        # Structured handoff docs have many H2s but zero bullet lists, so they fell
        # through to paragraph-based chunking.  Multiple sections merged across heading
        # boundaries, diluting heading embeddings and causing query-guided selection to
        # miss exact heading matches (cosine similarity < 0.4 instead of ≥ 0.6).
        #
        # Fix: use heading-density alone as the gate (≥2 H2/H3 headings, OR ≥3 headings
        # of any level).  Also sub-split oversized sections with the heading prefix
        # prepended to each child chunk so the heading signal survives in the embedding.
        _H2H3_RE = re.compile(r"(?=^#{2,3} )", re.MULTILINE)
        _HEADING_DETECT_RE = re.compile(r"^#{1,3} ", re.MULTILINE)
        _H2H3_ONLY_RE = re.compile(r"^#{2,3} ", re.MULTILINE)
        _is_structured = (
            len(_H2H3_ONLY_RE.findall(text)) >= 2 or len(_HEADING_DETECT_RE.findall(text)) >= 3
        )
        _MAX_SECTION_FACTOR = 1.5
        if _is_structured:
            # Split at H2/H3 heading boundaries, discarding empty pieces
            heading_chunks = [c.strip() for c in _H2H3_RE.split(text) if c.strip()]
            if len(heading_chunks) > 1:
                result: list = []
                for chunk in heading_chunks:
                    if self._count_tokens(chunk) > max_chunk_size * _MAX_SECTION_FACTOR:
                        result.extend(self._split_oversized_section(chunk, max_chunk_size))
                    else:
                        result.append(chunk)
                return result

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

        # Return unique entities. dict.fromkeys preserves first-occurrence order
        # (deterministic across processes); list(set(...)) was PYTHONHASHSEED-dependent,
        # so WHICH entities survived the [:max_entities] truncation varied run-to-run
        # and the "Key entities:" skeleton line was non-deterministic. See
        # tests/test_output_determinism.py.
        return list(dict.fromkeys(entities))[:max_entities]

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

    @staticmethod
    def _build_similarity_edges(
        embeddings: np.ndarray,
        node_ids: List[str],
        similarity_threshold: float,
        block_size: int = _SIMILARITY_BLOCK_SIZE,
        max_chunks: int = _MAX_GRAPH_CHUNKS,
    ) -> List[Tuple[str, str, float]]:
        """
        Build the upper-triangle similarity edge list without materialising
        the full N×N cosine-similarity matrix.

        Embeddings are assumed L2-normalised (cosine sim == dot product).
        Processes embeddings in row-blocks of ``block_size`` so peak
        similarity buffer is O(block_size × min(N, max_chunks)), not O(N²).

        Only the first ``max_chunks`` rows participate in edge building;
        nodes beyond that index still exist but remain unconnected (uniform
        PageRank fallback).

        Returns:
            List of (node_id_i, node_id_j, weight) triples where weight is
            the cosine similarity value (float).
        """
        n = len(node_ids)
        edge_count = min(n, max_chunks)
        edges: List[Tuple[str, str, float]] = []

        edge_embeddings = embeddings[:edge_count]  # rows that *receive* edges

        for block_start in range(0, edge_count, block_size):
            block_end = min(block_start + block_size, edge_count)
            block = embeddings[block_start:block_end]  # shape: (bs, dim)
            # sim_block[r, c] = cosine_similarity(block_start+r, c)
            sim_block = block @ edge_embeddings.T  # shape: (bs, edge_count)

            for r in range(block_end - block_start):
                i = block_start + r
                for j in range(i + 1, edge_count):
                    sim = float(sim_block[r, j])
                    if sim > similarity_threshold:
                        edges.append((node_ids[i], node_ids[j], sim))

        return edges

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
            #
            # F11 fix: Skip SemToken for structured markdown docs.
            # SemToken merges similar adjacent windows — on structured docs this
            # collapses H2/H3 sections into a single block before _chunk_text()
            # can split on heading boundaries, defeating the heading-aware chunking.
            # Detect structured docs by counting H2/H3 headings in the ORIGINAL text.
            import re as _re_semtoken  # noqa: PLC0415

            _H2H3_COUNT = len(_re_semtoken.findall(r"^#{2,3} ", text, _re_semtoken.MULTILINE))
            _skip_semtoken = _H2H3_COUNT >= 2
            if (
                total_tokens > 200 and not _skip_semtoken
            ):  # Skip for very short texts and structured docs
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

            # 2a. Guard against non-finite embeddings (NaN/Inf) — poisoned vectors
            # corrupt every downstream cosine similarity and PageRank score.
            if len(embeddings) and not np.isfinite(np.asarray(embeddings)).all():
                raise ValueError(
                    "non-finite embedding (NaN/Inf) detected — refusing to ingest corrupted vectors"
                )

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
            # Memory-safety: never materialise the full N×N similarity matrix.
            # We build the graph in two passes:
            #   Pass A — create all SemanticNode objects (O(N) memory).
            #   Pass B — add edges in row-blocks of _SIMILARITY_BLOCK_SIZE so
            #            peak similarity buffer is O(block×N), not O(N²).
            # Hard ceiling _MAX_GRAPH_CHUNKS bounds both peak memory and the
            # O(N²) edge-build time for pathologically large documents.  Nodes
            # above the ceiling still exist (and get uniform PageRank) but are
            # not connected via dense similarity edges.
            logger.info("  Building semantic graph...")
            n_chunks = len(raw_chunks)
            if n_chunks > _MAX_GRAPH_CHUNKS:
                logger.warning(
                    f"  Document '{file_id}' has {n_chunks} chunks which exceeds "
                    f"_MAX_GRAPH_CHUNKS={_MAX_GRAPH_CHUNKS}. Similarity edges will "
                    f"only be built for the first {_MAX_GRAPH_CHUNKS} chunks to "
                    f"bound peak memory. Remaining nodes receive uniform PageRank."
                )
            edge_chunk_count = min(n_chunks, _MAX_GRAPH_CHUNKS)

            graph = nx.Graph()

            # --- Pass A: create all nodes (no similarity computation yet) ---
            for i, chunk in enumerate(raw_chunks):
                node_id = f"{file_id}_n{i}"
                _node_meta: dict = {
                    "position": i,
                    "tokens": self._count_tokens(chunk),
                    "entities": self._extract_key_entities(chunk),
                }
                _node_meta.update(self._extract_heading_metadata(chunk))
                node = SemanticNode(
                    node_id=node_id,
                    text=chunk,
                    embedding=embeddings[i],
                    metadata=_node_meta,
                )
                self.chunks[node_id] = node
                graph.add_node(node_id, **node.metadata)

            # --- Pass B: block-wise edge building (O(block × N) peak memory) ---
            # Delegate to the static helper so the logic is independently
            # unit-testable (see TestGraphBuildingBlockWise in the test suite).
            node_ids = [f"{file_id}_n{i}" for i in range(n_chunks)]
            for src, dst, weight in self._build_similarity_edges(
                embeddings=embeddings,
                node_ids=node_ids,
                similarity_threshold=self.similarity_threshold,
                block_size=_SIMILARITY_BLOCK_SIZE,
                max_chunks=edge_chunk_count,
            ):
                graph.add_edge(src, dst, weight=weight)

            # 4. Calculate importance via PageRank (rate allocation)
            logger.info("  Calculating importance scores (PageRank)...")
            if len(graph.nodes) > 0:
                # Use cached PageRank for 500× speedup on repeated reads (v0.4.4)
                pagerank = self._get_cached_pagerank(graph, file_id)

                # Update importance scores
                for node_id, score in pagerank.items():
                    if node_id in self.chunks:
                        self.chunks[node_id].importance = score

                # F3: Content-based importance boosts (applied after PageRank, then re-normalised)
                # Rationale: PageRank measures graph connectivity, not semantic signal strength.
                # Structured audit docs have verdict/CRITICAL nodes that are peripheral in the
                # graph (few neighbours) but maximally important to the reader.
                _boosted: Dict[str, float] = {}
                for node_id, node in self.chunks.items():
                    if not _node_belongs_to_file(node_id, file_id):
                        continue
                    t = node.text
                    boost = 0.0
                    # Heading-level boosts
                    if re.search(r"^# ", t, re.MULTILINE):
                        boost += 0.5  # H1 — document title / top-level heading
                    if re.search(r"^## ", t, re.MULTILINE):
                        boost += 0.3  # H2 — major section
                    # Severity / priority markers
                    if re.search(r"\b(CRITICAL|HIGH|P0|BLOCKER)\b", t, re.IGNORECASE):
                        boost += 0.4
                    # Verdict / conclusion patterns
                    if re.search(
                        r"\b(verdict|conclusion|summary|finding|result|status)s?\b",
                        t,
                        re.IGNORECASE,
                    ):
                        boost += 0.3
                    # Ordered list items (numbered findings)
                    if re.search(r"^\d+\.", t, re.MULTILINE):
                        boost += 0.2
                    if boost > 0:
                        _boosted[node_id] = node.importance + boost

                if _boosted:
                    # Re-normalise only the boosted nodes' scores across all file nodes
                    all_scores = {
                        nid: (_boosted.get(nid, n.importance))
                        for nid, n in self.chunks.items()
                        if _node_belongs_to_file(nid, file_id)
                    }
                    normalised = self._normalize_scores(all_scores)
                    for node_id, score in normalised.items():
                        if node_id in self.chunks:
                            self.chunks[node_id].importance = score

            # Store graph
            self.graphs[file_id] = graph
            self.file_metadata[file_id] = metadata or {}

            # 5. Generate skeleton
            skeleton_response = self._generate_skeleton(file_id)
            self._baseline_skeleton_cache[file_id] = skeleton_response.skeleton_text
            # Audit P2-4: cache numeric stats so get_stats() need not re-generate.
            self._baseline_skeleton_stats[file_id] = {
                "skeleton_tokens": skeleton_response.skeleton_tokens,
                "ratio": skeleton_response.compression_ratio,
            }

            logger.info(
                f"  Compression: {total_tokens} -> {skeleton_response.skeleton_tokens} tokens"
            )
            logger.info(f"  Ratio: {skeleton_response.compression_ratio:.1f}x")

            return skeleton_response

    def set_lambda_redundancy(self, lambda_redundancy: float) -> None:
        """Set the COMI/MIG redundancy weight for query-guided skeleton selection.

        B1 (modernization roadmap 2026-06-08): higher values penalise
        near-duplicate nodes more aggressively, surfacing more diverse evidence
        at the same skeleton size. ``0.5`` is the COMI default
        (arXiv 2602.01719); ``0.0`` disables redundancy-aware diversification.
        Driven by ``CompressionPreset.lambda_redundancy`` when a preset is
        applied. Only affects the query-present selection path.

        Args:
            lambda_redundancy: Redundancy weight in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``lambda_redundancy`` is outside ``[0.0, 1.0]``.
        """
        if not 0.0 <= lambda_redundancy <= 1.0:
            raise ValueError(f"lambda_redundancy must be in [0.0, 1.0], got {lambda_redundancy}")
        self.lambda_redundancy = float(lambda_redundancy)

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
        importance_override: Optional[Dict[str, float]] = None,
    ) -> Set[str]:
        """
        Select skeleton nodes.

        Baseline mode:
        - PageRank-only (importance sort)

        Query-guided mode:
        - Hybrid ranking: importance + query relevance
        - Redundancy penalty via greedy MMR-style selection

        Args:
            importance_override: Optional per-node importance scores that take
                precedence over the persisted ``node.importance`` for THIS call
                only. Used to thread query-local MIG scores without mutating the
                shared, long-lived ``SemanticNode.importance`` (audit P1-4).

        Note:
            Public contract is the selected SET. The ordered greedy pick sequence
            is computed by ``_select_skeleton_nodes_ordered`` (used by the
            output-equivalence regression test); this method just returns it as a
            set so existing call sites are unaffected.
        """
        return set(
            self._select_skeleton_nodes_ordered(
                file_nodes,
                num_skeleton,
                query=query,
                redundancy_penalty=redundancy_penalty,
                priority_scores=priority_scores,
                importance_override=importance_override,
            )
        )

    def _select_skeleton_nodes_ordered(
        self,
        file_nodes: List[Tuple[str, SemanticNode]],
        num_skeleton: int,
        query: Optional[str] = None,
        redundancy_penalty: float = 0.2,
        priority_scores: Optional[Dict[str, float]] = None,
        importance_override: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Vectorized greedy MMR skeleton selection (roadmap A2).

        Returns the selected node ids in PICK ORDER. Output-equivalent to the
        prior per-pair ``sklearn.cosine_similarity`` Python loop:

        - Relevance is one L2-normalised matrix–vector product
          (``E_norm @ q_norm``) instead of N per-node cosine calls.
        - Redundancy uses a running ``max_sim`` vector updated by a single
          ``E_norm @ E_norm[picked]`` column per pick, instead of an O(picks)
          inner ``max`` over per-pair cosine calls each iteration.

        Equivalence notes (load-bearing):
        - Dot products accumulate in float64 to stay maximally stable; the
          original sklearn path also upcasts list inputs internally, so this is
          the closest reproduction (residual diff is float32 BLAS summation-order
          noise ≈1e-8, far below any realistic selection margin).
        - Tie-break matches the original ``candidate_score > best_score`` strict
          comparison over insertion-ordered candidates: ``np.argmax`` returns the
          FIRST maximal index, so the earliest candidate wins a tie exactly as the
          Python loop did.
        """
        if num_skeleton <= 0 or not file_nodes:
            return []

        override = importance_override or {}

        def _imp(node_id: str, node: "SemanticNode") -> float:
            return override.get(node_id, node.importance)

        if not query or not query.strip():
            ranked = sorted(file_nodes, key=lambda item: _imp(item[0], item[1]), reverse=True)
            return [node_id for node_id, _ in ranked[:num_skeleton]]

        node_ids = [node_id for node_id, _ in file_nodes]

        # Query-guided selection
        query_embedding = self.model.encode([query])[0]
        # A degenerate (NaN/Inf) query embedding would make q_unit (and the whole
        # relevance_vec) NaN — a NaN norm is truthy, so the ``q_norm_val != 0.0``
        # guard below does NOT catch it — silently corrupting MMR ranking. Fall
        # back to importance-only ordering (the no-query path) instead. (#134)
        if not np.isfinite(query_embedding).all():
            ranked = sorted(file_nodes, key=lambda item: _imp(item[0], item[1]), reverse=True)
            return [node_id for node_id, _ in ranked[:num_skeleton]]

        # L2-normalise the node-embedding matrix ONCE (float64 accumulation,
        # contiguous). cosine(a, b) == dot(a/||a||, b/||b||); normalising up front
        # means relevance and redundancy are pure dot products against unit rows.
        embeddings = np.ascontiguousarray(
            np.stack([np.asarray(node.embedding, dtype=np.float64) for _, node in file_nodes])
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Guard zero-norm rows (degenerate embeddings) — leave them as zero so
        # their cosine is 0.0, matching sklearn's handling of zero vectors.
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        e_norm = embeddings / safe_norms  # (N, dim), unit rows

        q = np.asarray(query_embedding, dtype=np.float64)
        q_norm_val = np.linalg.norm(q)
        q_unit = q / q_norm_val if q_norm_val != 0.0 else q

        relevance_vec = e_norm @ q_unit  # (N,) cosine(query, node_i)

        importance_scores = {node_id: _imp(node_id, node) for node_id, node in file_nodes}
        relevance_scores = {node_ids[i]: float(relevance_vec[i]) for i in range(len(node_ids))}

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
        hybrid_vec = np.array(
            [
                0.25 * importance_norm.get(nid, 0.0)
                + 0.55 * relevance_norm.get(nid, 0.0)
                + 0.20 * priority_norm.get(nid, 0.0)
                for nid in node_ids
            ],
            dtype=np.float64,
        )

        n = len(node_ids)
        target = min(num_skeleton, n)
        selected_order: List[str] = []
        chosen_mask = np.zeros(n, dtype=bool)
        # Running max cosine similarity of each node vs the already-selected set.
        # -inf sentinel means "no selection yet" so the first pick uses hybrid only.
        max_sim = np.full(n, -np.inf, dtype=np.float64)

        while len(selected_order) < target:
            if not selected_order:
                scores = hybrid_vec.copy()
            else:
                scores = hybrid_vec - redundancy_penalty * max_sim

            # Mask already-selected candidates so they cannot be re-picked. The
            # original loop removed them from candidate_ids; -inf is equivalent
            # and preserves argmax's first-max tie-break over remaining nodes.
            scores = np.where(chosen_mask, -np.inf, scores)
            best_idx = int(np.argmax(scores))

            # Mirror the original guard: if no candidate beats -inf, stop.
            if not np.isfinite(scores[best_idx]):
                break

            selected_order.append(node_ids[best_idx])
            chosen_mask[best_idx] = True

            # Update the running redundancy vector with one matmul column:
            # cosine(node_i, newly_selected) for all i.
            sims_to_new = e_norm @ e_norm[best_idx]
            max_sim = np.maximum(max_sim, sims_to_new)

        return selected_order

    def _generate_skeleton(
        self,
        file_id: str,
        query: Optional[str] = None,
        anchor_node_ids: Optional[Set[str]] = None,
        exclude_node_ids: Optional[Set[str]] = None,
        selection_strategy: Literal["mig", "pagerank", "auto"] = "auto",
    ) -> SkeletonResponse:
        """
        Step 2: Rate Allocation (JSCCM)

        Generates a low-bandwidth skeleton view by:
        1. Ranking nodes by importance (PageRank)
        2. Keeping top N% as "anchor concepts"
        3. Hiding others as references

        Args:
            file_id: Document identifier previously ingested.
            query: Optional retrieval query for guided selection.
            anchor_node_ids: Force-include specific node IDs in skeleton.
            exclude_node_ids: Force-exclude specific node IDs from skeleton.
            selection_strategy: Node-selection algorithm.
                - ``"auto"`` (default): current behaviour — COMI coarse filter
                  (when query provided) followed by PageRank-guided selection.
                - ``"mig"``: force MIG (Marginal Information Gain) path via
                  ``MIGScorer`` for node importance re-ranking instead of the
                  PageRank-only ranking step.  COMI coarse filter is still
                  applied when a query is present.
                - ``"pagerank"``: skip the COMI coarse filter entirely, relying
                  only on PageRank importance scores for selection.
        """
        graph = self.graphs.get(file_id)
        if not graph:
            raise ValueError(f"File {file_id} not found")

        # Get all nodes for this file
        excluded = exclude_node_ids or set()
        file_nodes = [
            (nid, self.chunks[nid])
            for nid in graph.nodes()
            if _node_belongs_to_file(nid, file_id) and nid not in excluded
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
        # When a query is provided AND strategy is not "pagerank", first do a
        # COARSE pass that eliminates clearly irrelevant nodes (bottom 50% by
        # query relevance) before the fine-grained PageRank selection.  This
        # reduces noise and focuses the skeleton on query-relevant content.
        # The paper showed a 25-point EM improvement at high compression ratios.
        # ``selection_strategy="pagerank"`` skips this filter entirely.
        if query and len(file_nodes) > 3 and selection_strategy != "pagerank":
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

                # Never let the COMI coarse filter drop an explicitly-anchored
                # node. Anchors are merged into ``skeleton_nodes`` below, but that
                # set only controls [ANCHOR]/[HIDDEN] *labelling* — the render
                # loop iterates ``file_nodes``. An anchor dropped here would be
                # absent from the output entirely, silently violating the
                # "always keep this region" anchor / evidence-aware contract.
                # Union dropped anchors back in. (audit 2026-06-24)
                if anchor_node_ids:
                    _anchor_set = set(anchor_node_ids)
                    _kept_ids = {nid for nid, _, _ in coarse_nodes}
                    for _scored in node_scores:
                        if _scored[0] in _anchor_set and _scored[0] not in _kept_ids:
                            coarse_nodes.append(_scored)
                            _kept_ids.add(_scored[0])

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

        # MIG (Marginal Information Gain) node re-ranking:
        # When ``selection_strategy="mig"`` and a query is present, re-rank
        # nodes using token-level MIG scores aggregated per node. These scores
        # are passed to ``_select_skeleton_nodes`` as a query-local
        # ``importance_override`` so the greedy MMR selection operates on
        # MIG-weighted scores rather than graph centrality.
        #
        # Audit P1-4: this MUST NOT write ``node.importance``. The compressor is
        # a long-lived singleton; mutating the shared PageRank importance in
        # place corrupts every subsequent query on the same document (and the
        # side-effecting get_stats path). We build a per-call dict instead.
        mig_importance_override: Optional[Dict[str, float]] = None
        if selection_strategy == "mig" and query and query.strip():
            try:
                from .token_refiner import MIGConfig, MIGScorer

                mig_scorer = MIGScorer(config=MIGConfig())
                mig_importance_override = {}
                for nid, node in file_nodes:
                    tokens = node.text.split()
                    if tokens:
                        scored = mig_scorer.score_tokens_mig(tokens, query)
                        # Aggregate: mean MIG score across tokens → query-local score
                        mig_importance_override[nid] = float(
                            sum(s for _, s in scored) / len(scored)
                        )
                    else:
                        # nodes with no tokens keep their PageRank importance
                        mig_importance_override[nid] = node.importance

                # Re-sort by the query-local MIG override for stable downstream
                # order WITHOUT touching node.importance.
                file_nodes.sort(
                    key=lambda x: mig_importance_override.get(x[0], x[1].importance),
                    reverse=True,
                )
                logger.info(
                    f"  MIG re-ranking applied for '{file_id}' " f"({len(file_nodes)} nodes scored)"
                )
            except Exception as exc:
                logger.warning(f"MIG re-ranking failed for '{file_id}': {exc}")
                mig_importance_override = None

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

        # B1 (modernization roadmap 2026-06-08): unify on COMI/MIG as the
        # production redundancy-aware selector. When a query is present, route the
        # redundancy weight through ``MIGScorer``'s COMI config
        # (``MIGConfig.lambda_redundancy``, default 0.5) instead of the legacy
        # fixed 0.2 MMR term. The weight feeds the VECTORIZED
        # ``_select_skeleton_nodes`` numpy path (``hybrid - lambda * max_sim``) —
        # A2's matmul selector is preserved, no per-pair cosine loop is
        # reintroduced. The no-query PageRank-only path keeps the engine default
        # (0.2); its short-circuit ignores the redundancy term anyway.
        if query and query.strip():
            from .token_refiner import MIGConfig, MIGScorer

            # Instantiate the COMI scorer with this compressor's lambda so the
            # routing is explicit and a single config drives the weight.
            _mig_scorer = MIGScorer(config=MIGConfig(lambda_redundancy=self.lambda_redundancy))
            effective_redundancy_penalty = _mig_scorer.config.lambda_redundancy
        else:
            effective_redundancy_penalty = 0.2

        if anchor_node_ids:
            skeleton_nodes = set(anchor_node_ids)
            if len(skeleton_nodes) < num_skeleton:
                selected = self._select_skeleton_nodes(
                    file_nodes,
                    num_skeleton,
                    query=query,
                    redundancy_penalty=effective_redundancy_penalty,
                    priority_scores=adaptive_priority_scores,
                    importance_override=mig_importance_override,
                )
                skeleton_nodes.update(selected)
        else:
            skeleton_nodes = self._select_skeleton_nodes(
                file_nodes,
                num_skeleton,
                query=query,
                redundancy_penalty=effective_redundancy_penalty,
                priority_scores=adaptive_priority_scores,
                importance_override=mig_importance_override,
            )

        # Build skeleton text
        skeleton_lines = []
        skeleton_lines.append(f"=== SEMANTIC SKELETON: {file_id} ===")
        skeleton_lines.append("Skeleton-Version: 2")
        skeleton_lines.append(f"Total nodes: {len(file_nodes)} | Skeleton nodes: {num_skeleton}")
        skeleton_lines.append(f"Compression: {effective_ratio:.0%} of content shown")
        # Explain hidden-region drill-down ONCE here (Skeleton-Version 2) instead of
        # repeating the phrase on every [HIDDEN] node — the per-node repetition was a
        # hard ratio ceiling (~15-20x). Consumers can branch on Skeleton-Version.
        skeleton_lines.append("Hidden regions expand via modulate_region(node_id).\n")

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
                # Low-importance: keep the [node_id] (drill-down addressability) and
                # the [HIDDEN] marker, plus the already-computed short summary so a
                # non-drill-down reader is not left empty-handed. The verbose
                # "Detail hidden (use modulate_region to expand)" phrase is HOISTED to
                # the skeleton header once (Skeleton-Version 2) rather than repeated
                # per node — repetition capped the ratio. Live web consumers (gc_lookup,
                # KB search) prefer raw_text over this text, so dropping the repeated
                # phrase is wire-safe; the [HIDDEN] marker + header pointer remain.
                summary = self._generate_summary(node.text, max_length=50)
                summary = summary.strip() if summary else ""
                if summary:
                    line = f"[{node_id}] [HIDDEN] - {summary}\n"
                else:
                    line = f"[{node_id}] [HIDDEN]\n"

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

    def read_skeleton(
        self,
        file_id: str,
        query: Optional[str] = None,
        selection_strategy: Literal["mig", "pagerank", "auto"] = "auto",
    ) -> str:
        """
        MCP Tool: read_skeleton

        Returns the compressed skeleton view of a document.
        ~80-95% token savings vs raw text.

        Args:
            file_id: Document identifier previously ingested.
            query: Optional retrieval query for query-guided skeleton.
            selection_strategy: Node-selection algorithm (``"auto"``,
                ``"mig"``, or ``"pagerank"``).  Defaults to ``"auto"``
                which preserves pre-v1.11.0 behaviour.
        """
        skeleton = self._generate_skeleton(
            file_id, query=query, selection_strategy=selection_strategy
        )
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

        # Audit re-fix (supersedes the P1-3 RRF-threshold branch): sufficiency
        # must reflect relevance MAGNITUDE, not rank-fusion POSITION. The prior
        # fix gated on the RRF fusion score under Path C, but RRF encodes only
        # rank position — the rank-1 node of ANY non-empty doc has RRF
        # >= 1/(k+1) ≈ 0.0164, so 0.0164 >= 0.015 was unconditionally True and
        # `sufficient` was True for EVERY query, including irrelevant ones.
        #
        # Correct gate: threshold sufficiency on the DENSE COSINE similarity of
        # the top-ranked candidate against the cosine bar (min_similarity),
        # REGARDLESS of whether the RANKING method is cosine (Path A) or RRF
        # (Path C). RRF stays as the ordering method under Path C; only the
        # SUFFICIENCY decision uses cosine magnitude.
        effective_threshold = min_similarity

        initial_scores = self.search_semantic_with_scores(query, file_id=file_id, top_k=top_k)
        best_cosine = self._max_dense_cosine(query, file_id=file_id)
        sufficient = best_cosine >= effective_threshold
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
            # Broadening top_k cannot change the document-wide max cosine, but the
            # re-query keeps the contract (more candidates surfaced) and a fresh
            # cosine read guards against any candidate-set drift between calls.
            best_cosine = self._max_dense_cosine(query, file_id=file_id)
            sufficient = best_cosine >= effective_threshold

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
            # best_score reports the cosine magnitude the sufficiency decision is
            # made on — NOT the (Path C) RRF fusion score. final_scores still
            # carries the ranking-method scores for callers that inspect ordering.
            sufficient=sufficient,
            best_score=best_cosine,
            threshold=effective_threshold,
            used_expanded_search=used_expanded_search,
            message=message,
        )

    def _max_dense_cosine(self, query: str, file_id: Optional[str] = None) -> float:
        """Return the max dense cosine similarity of ``query`` vs candidate nodes.

        This is the relevance-MAGNITUDE signal the sufficiency gate thresholds on
        (see ``retrieve_evidence``). It mirrors the dense-ranking branch of
        ``search_semantic_with_scores`` but returns only the top cosine value, so
        the sufficiency decision is independent of the active ranking method
        (cosine Path A or RRF Path C).

        Args:
            query: Search query.
            file_id: Optional file to scope the candidate set.

        Returns:
            Max cosine similarity in [-1.0, 1.0], or 0.0 when no candidates exist.
        """
        candidate_nodes = [
            node
            for node_id, node in self.chunks.items()
            if not (file_id and not _node_belongs_to_file(node_id, file_id))
        ]
        if not candidate_nodes:
            return 0.0

        query_embedding = self.model.encode([query])[0]
        # Guard a degenerate (NaN/Inf) query embedding so this returns 0.0
        # relevance instead of propagating NaN into the sufficiency gate. (#134)
        if not np.isfinite(query_embedding).all():
            return 0.0
        best = -1.0
        for node in candidate_nodes:
            similarity = float(cosine_similarity([query_embedding], [node.embedding])[0][0])
            if similarity > best:
                best = similarity
        return best

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

    # ------------------------------------------------------------------
    # F11 Path C — BM25 + Reciprocal Rank Fusion helpers
    # ------------------------------------------------------------------

    def _bm25_scores_for_nodes(
        self,
        query: str,
        candidate_nodes: List[Tuple[str, "SemanticNode"]],
    ) -> List[Tuple[str, float]]:
        """Compute BM25Okapi scores for a file_id-filtered candidate set.

        IMPORTANT (council patch P1): receives the SAME file_id-filtered
        candidate_nodes as the dense ranker — NEVER scores self.chunks globally.
        Scoring the entire corpus would pollute IDF with documents unrelated to
        the current file_id, degrading both BM25 precision and RRF fusion quality.

        Uses raw node text (node.text) per the Phase 7c-3 lesson: compressed
        skeleton text contains [HIDDEN] placeholders that produce near-meaningless
        BM25 IDF vectors. Raw text is the correct scoring surface.

        Args:
            query: Search query string.
            candidate_nodes: List of (node_id, SemanticNode) for the file_id scope.

        Returns:
            List of (node_id, bm25_score) sorted descending.
            Nodes with score=0 are excluded (zero-score nodes are noise in RRF).
        """
        if not candidate_nodes:
            return []

        node_ids = [nid for nid, _ in candidate_nodes]
        # Use raw text (not compressed skeleton) — Phase 7c-3 lesson.
        texts = [node.text for _, node in candidate_nodes]

        raw_scores = _bm25_score_texts(query, texts)

        scored = [
            (node_id, float(score)) for node_id, score in zip(node_ids, raw_scores) if score > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _rrf_fuse(
        dense_ranked: List[Tuple[str, float]],
        bm25_ranked: List[Tuple[str, float]],
        k: int = _RRF_K,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion (Cormack, Clarke, Buettcher — SIGIR 2009).

        RRF(d) = Σ_r  1 / (k + rank_r(d))

        Score-agnostic: cosine and BM25 magnitudes are ignored; only rank
        position matters. k=60 is the SOTA empirically validated default
        (arxiv 2210.11934; BigDataBoutique 2026; original Cormack 2009 paper).

        Degrades gracefully to dense-only when bm25_ranked is empty — identical
        to current Path A behavior, zero regression risk on sparse queries.

        Args:
            dense_ranked: (node_id, cosine_score) sorted descending.
            bm25_ranked:  (node_id, bm25_score) sorted descending.
            k: RRF constant (default 60, tunable via _RRF_K env var).
            top_k: Number of top results to return.

        Returns:
            (node_id, rrf_score) sorted descending. rrf_score is NOT a cosine
            value — callers must check score_type in the handler response to
            distinguish Path A (cosine) from Path C (rrf).
        """
        rrf_scores: Dict[str, float] = {}
        for rank, (node_id, _) in enumerate(dense_ranked, start=1):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)
        for rank, (node_id, _) in enumerate(bm25_ranked, start=1):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    def search_semantic_with_scores(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Ranker dispatch — respects F11_RANKER_PATH env var.

        Path "a" (default): dense cosine only (backward compatible).
        Path "c": BM25 + Reciprocal Rank Fusion hybrid (F11 Path C plan).

        New in v0.9.0: Returns similarity scores alongside node IDs.
        New in v1.34.35 (F11 Path C): RRF hybrid when F11_RANKER_PATH=c.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of (node_id, score) tuples ranked by relevance.
            Under Path A: score is cosine similarity in [-1.0, 1.0].
            Under Path C: score is RRF score (float, NOT bounded to [0,1]).
                          Callers must NOT assume score is cosine similarity.
        """
        # --- Build file_id-filtered candidate list (shared by both paths) ---
        candidate_nodes: List[Tuple[str, SemanticNode]] = []
        for node_id, node in self.chunks.items():
            if file_id and not _node_belongs_to_file(node_id, file_id):
                continue
            candidate_nodes.append((node_id, node))

        if not candidate_nodes:
            return []

        # --- Dense (cosine) ranking — always computed ---
        query_embedding = self.model.encode([query])[0]
        dense_ranked: List[Tuple[str, float]] = []
        for node_id, node in candidate_nodes:
            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
            dense_ranked.append((node_id, float(similarity)))
        dense_ranked.sort(key=lambda x: x[1], reverse=True)

        # --- Path dispatch ---
        if F11_RANKER_PATH != "c":
            # Path A (default): dense-only, backward compatible
            return dense_ranked[:top_k]

        # Path C: BM25 + RRF hybrid
        # Council patch P1: BM25 receives the SAME file_id-filtered candidate_nodes,
        # never self.chunks globally (cross-document IDF pollution prevention).
        bm25_ranked = self._bm25_scores_for_nodes(query, candidate_nodes)

        fused = self._rrf_fuse(dense_ranked, bm25_ranked, k=_RRF_K, top_k=top_k)
        return fused

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

    def _baseline_skeleton_metrics(self, file_id: str, total_tokens: int) -> Tuple[int, float]:
        """Return (skeleton_tokens, compression_ratio) without generating a skeleton.

        Serves from the numeric cache populated at ingest (audit P2-4). Falls
        back to counting the cached baseline skeleton TEXT if only that is
        present, and finally — when no baseline cache exists at all (cold cache
        after a persistence restore that rebuilt the graph but not the caches) —
        computes a real skeleton-token estimate directly FROM the graph nodes
        (audit P2-4 edge case). Never invokes ``_generate_skeleton``.
        """
        cached = self._baseline_skeleton_stats.get(file_id)
        if cached is not None:
            return int(cached["skeleton_tokens"]), float(cached["ratio"])

        baseline_text = self._baseline_skeleton_cache.get(file_id)
        if baseline_text is not None:
            skeleton_tokens = self._count_tokens(baseline_text)
            ratio = total_tokens / max(skeleton_tokens, 1)
            # Memoize so repeat get_stats calls stay O(1).
            self._baseline_skeleton_stats[file_id] = {
                "skeleton_tokens": skeleton_tokens,
                "ratio": ratio,
            }
            return skeleton_tokens, ratio

        # No baseline cache (e.g. document restored from persistence without
        # re-ingest). Audit P2-4 edge case: returning a flat (total_tokens, 1.0)
        # reported a misleading "no compression happened" ratio. Instead compute
        # the skeleton-token estimate from the graph's selected anchor nodes,
        # WITHOUT regenerating the skeleton (preserves the side-effect-free
        # property — no MIG mutation, no _generate_skeleton call).
        skeleton_tokens = self._estimate_skeleton_tokens_from_graph(file_id)
        if skeleton_tokens > 0:
            ratio = total_tokens / skeleton_tokens
            self._baseline_skeleton_stats[file_id] = {
                "skeleton_tokens": skeleton_tokens,
                "ratio": ratio,
            }
            return skeleton_tokens, ratio

        # Truly nothing to measure (no graph / empty doc) — neutral fallback.
        return total_tokens, 1.0

    def _estimate_skeleton_tokens_from_graph(self, file_id: str) -> int:
        """Estimate baseline skeleton token count from the graph nodes alone.

        Side-effect-free (audit P2-4): replicates ONLY the baseline (no-query)
        node-selection math from ``_generate_skeleton`` — top ``num_skeleton``
        nodes by importance, where ``num_skeleton = max(1, int(N * ratio))`` —
        and sums those anchor nodes' raw token counts. Does NOT build summary
        lines, run MIG/COMI, or mutate ``node.importance``. Returns 0 when no
        graph or no measurable nodes exist (caller falls back to a neutral
        ratio).
        """
        graph = self.graphs.get(file_id)
        if graph is None:
            return 0

        file_nodes = [
            (nid, self.chunks[nid])
            for nid in graph.nodes()
            if _node_belongs_to_file(nid, file_id) and nid in self.chunks
        ]
        if not file_nodes:
            return 0

        effective_ratio = self.skeleton_ratio
        if effective_ratio == "auto":
            total_tokens_estimate = sum(len(node.text.split()) for _, node in file_nodes)
            effective_ratio = compute_adaptive_ratio(total_tokens_estimate)

        num_skeleton = max(1, int(len(file_nodes) * effective_ratio))
        # Baseline selection = top-N by importance (mirrors _select_skeleton_nodes
        # with no query). Anchor nodes carry their full raw token count in the
        # skeleton; the rest collapse to short reference lines, so summing the
        # anchor nodes' tokens is the side-effect-free proxy for skeleton size.
        ranked = sorted(file_nodes, key=lambda item: item[1].importance, reverse=True)
        skeleton_tokens = sum(
            int(node.metadata.get("tokens", 0)) for _, node in ranked[:num_skeleton]
        )
        return skeleton_tokens

    def get_stats(self, file_id: Optional[str] = None) -> Dict:
        """Get statistics about stored documents.

        Audit P2-4: this is a READ-ONLY introspection call. It must NOT invoke
        ``_generate_skeleton`` — doing so had side effects (MIG mutation via the
        skeleton path, recomputation cost) and is wholly unnecessary for stats.
        Skeleton token counts are served from the baseline cache populated at
        ingest; if that cache is cold (e.g. a document restored from persistence
        without re-ingest), the counts are computed directly from the cached
        baseline skeleton text rather than regenerating it.
        """
        if file_id:
            if file_id not in self.graphs:
                raise ValueError(f"File {file_id} not found")

            graph = self.graphs[file_id]
            nodes = [nid for nid in graph.nodes() if _node_belongs_to_file(nid, file_id)]

            total_tokens = sum(self.chunks[nid].metadata["tokens"] for nid in nodes)

            skeleton_tokens, compression_ratio = self._baseline_skeleton_metrics(
                file_id, total_tokens
            )

            return {
                "file_id": file_id,
                "total_nodes": len(nodes),
                "total_edges": graph.number_of_edges(),
                "total_tokens": total_tokens,
                "skeleton_tokens": skeleton_tokens,
                "compression_ratio": compression_ratio,
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
        old_chunks = {
            nid: node for nid, node in self.chunks.items() if _node_belongs_to_file(nid, file_id)
        }
        old_texts = {nid: node.text for nid, node in old_chunks.items()}

        new_chunk_texts = self._chunk_text(new_text)

        old_text_set = set(old_texts.values())
        new_text_set = set(new_chunk_texts)

        unchanged_texts = old_text_set & new_text_set
        removed_texts = old_text_set - new_text_set
        added_texts = new_text_set - old_text_set

        chunks_updated = min(len(removed_texts), len(added_texts))

        preserved = {}
        for node in old_chunks.values():
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
            if _node_belongs_to_file(nid, file_id) and node.text in preserved:
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
            # Use the canonical node-id parser, not rsplit("_", 1): a code node
            # is "file_id::symbol" where the symbol may contain underscores, so
            # rsplit mis-extracts the file id and two functions in the SAME file
            # get compared as if cross-file (false-positive duplicates). (#134)
            file_a = extract_file_id_from_node(nid_a)

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
                file_b = extract_file_id_from_node(nid_b)

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
