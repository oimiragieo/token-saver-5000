"""Fidelity-Preserving Semantic Compressor (facade + core graph/chunking)."""

import asyncio
import hashlib
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from os import cpu_count
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from .constants import DEFAULT_TEXT_MODEL, F11_RANKER_PATH, RERANK_ENABLED, RERANK_POOL_SIZE
from .embeddings import EmbeddingManager, _EmbeddingManagerAdapter
from .node_identity import extract_file_id_from_node
from .reranker_gate import RerankConfig, rerank_candidates
from .semantic_compressor_types import (
    DiffReingestionResult,
    EvidenceResult,
    FidelityLevel,
    SemanticNode,
    SkeletonResponse,
    compute_adaptive_ratio,
    _gate_should_fuse_g,
    _gate_query_has_lexical_shape,
    _MAX_GRAPH_CHUNKS,
    _SIMILARITY_BLOCK_SIZE,
    _node_belongs_to_file,
    _SENTENCE_SPLIT_RE,
    _strip_admonition_markers,
)
from .semantic_compressor_ingest import SemanticCompressorIngestMixin
from .semantic_compressor_retrieval import SemanticCompressorRetrievalMixin

logger = logging.getLogger(__name__)


class SemanticCompressor(SemanticCompressorIngestMixin, SemanticCompressorRetrievalMixin):
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

        # Per-document skeleton_ratio override (2026-07-06, knob-honesty fix,
        # architecture plan Move 5). This compressor instance is a long-lived
        # singleton shared across concurrent MCP requests (production wiring —
        # ``server_factory_service.code_adapter_config``). The REST engine
        # (``api/app/services/compression.py``) safely mutates
        # ``compressor.skeleton_ratio`` per call because each request thread
        # owns its OWN compressor there; doing the same on the shared MCP
        # singleton would race concurrently in-flight calls on DIFFERENT
        # documents. Keying the override by file_id avoids that race: each
        # document's requested ratio is independent and read back by
        # ``_resolve_skeleton_ratio`` instead of the shared default.
        self._file_skeleton_ratio_overrides: Dict[str, float | str] = {}

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

    def _prepare_raw_chunks(self, text: str, structured_kind, strategy: str = "auto") -> List[str]:
        """Choose raw chunks for ingest.

        For a detected JSON array, JSONL, or CSV document, chunk on RECORD
        boundaries (records grouped by size become individually rankable nodes) so
        structured data survives compression instead of collapsing to one hidden
        mega-node (#190 + #279 + #280). All split paths are non-destructive
        (``group_records_by_size`` keeps every record; CSV bails to the text
        chunker on any embedded-newline / multiline-quote quirk rather than
        corrupt a row). All other content uses the semantic/fixed text chunker.
        """
        if structured_kind == "json_array":
            from .structured_content import group_records_by_size, split_json_records

            records = split_json_records(text)
            if records:
                return group_records_by_size(records, 512, self._count_tokens)
        elif structured_kind == "jsonl":
            from .structured_content import group_records_by_size, split_jsonl_records

            records = split_jsonl_records(text)
            if records:
                return group_records_by_size(records, 512, self._count_tokens)
        elif structured_kind == "csv":
            from .structured_content import group_records_by_size, split_csv_records

            split = split_csv_records(text)
            if split:
                header, rows = split
                # Header is its OWN node (NOT prepended to each group). This keeps
                # every row-group strictly within the embedding window and never
                # duplicates the header — closing both the wide-header/oversized
                # budget blowout (codex #280 HIGH-2) and the false-positive header
                # duplication (droid MED-4). The header node labels the columns
                # once for the table; row values stay searchable in the row groups.
                groups = group_records_by_size(rows, 512, self._count_tokens)
                return [header, *groups]
        return self._chunk_text(text, strategy=strategy)

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
                    # #212 (edge-case hardening): use the CJK-aware splitter, not the
                    # ASCII-only ``[.!?]`` regex -- a dense CJK document with no
                    # paragraph breaks and no ASCII terminators would otherwise return
                    # ONE "sentence" (the whole text), so semantic chunking could never
                    # subdivide it. ``_SENTENCE_SPLIT_RE`` also splits on full-width
                    # ``。！？`` and is byte-identical to the old regex for pure-ASCII
                    # text (no behavior change for existing corpora).
                    semantic_units = [
                        sentence.strip()
                        for sentence in _SENTENCE_SPLIT_RE.split(text)
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

                # If single paragraph is too large, split by sentences.
                # #212 (edge-case hardening): CJK-aware splitter -- see the semantic-
                # chunking branch above for why the ASCII-only ``[.!?]`` regex silently
                # collapsed a dense, punctuation-free (or full-width-punctuated) CJK
                # paragraph into ONE oversized "sentence"/mega-chunk instead of
                # subdividing it.
                if para_tokens > max_chunk_size:
                    sentences = _SENTENCE_SPLIT_RE.split(para)
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
        Uses capitalized words as a proxy for entities, minus common English words
        that are capitalized only because they follow punctuation — e.g. after a
        colon "Section 2: This ..." must NOT surface "This" as an entity (Task 4:
        dogfood found "Key entities: This" garbage). Also strips surrounding
        punctuation so entities read clean.
        """
        # Capitalized-but-not-an-entity words (sentence openers / determiners /
        # pronouns / aux verbs). Local frozenset — entity extraction runs once per
        # chunk at ingest, so the rebuild cost is negligible.
        stopwords = frozenset(
            {
                "The",
                "This",
                "That",
                "These",
                "Those",
                "A",
                "An",
                "It",
                "Its",
                "He",
                "She",
                "They",
                "We",
                "You",
                "In",
                "On",
                "At",
                "For",
                "But",
                "And",
                "Or",
                "Nor",
                "If",
                "So",
                "As",
                "Of",
                "To",
                "From",
                "By",
                "With",
                "Then",
                "There",
                "Here",
                "When",
                "Where",
                "What",
                "Which",
                "Who",
                "Whom",
                "How",
                "Why",
                "Section",
                "Also",
                "However",
                "Their",
                "Our",
                "Your",
                "His",
                "Her",
                "My",
                "Not",
                "No",
                "Yes",
                "Is",
                "Are",
                "Was",
                "Were",
                "Will",
                "Would",
                "Can",
                "Could",
                "Should",
                "May",
                "Might",
                "Must",
                "Do",
                "Does",
                "Did",
                "Have",
                "Has",
                "Had",
                "Be",
                "Been",
                "Being",
                "Each",
                "Every",
                "Some",
                "Any",
                "All",
                "Both",
                "Because",
                # Python literals that surface capitalized when code is ingested as
                # text (dogfood 2026-07-13: `Key entities: None, ...` on a pasted .py).
                # Never a meaningful domain entity. `self`/`cls` are lowercase so the
                # isupper() gate already excludes them; `Exception` is kept — a real
                # class name, not a literal.
                "None",
                "True",
                "False",
                # Generic markdown section-heading / structural words (#360, dogfood
                # 2026-07-16: `## Overview` / `## Design` leaked "Overview"/"Design"
                # as Key entities). Unlike _generate_summary, this path does NOT strip
                # `#` heading markers, so a heading word survives the isupper() gate.
                # A bare structural heading is never a useful domain entity.
                "Overview",
                "Summary",
                "Introduction",
                "Conclusion",
                "Background",
                "Motivation",
                "Abstract",
                "Appendix",
                "Contents",
                "Prerequisites",
                "Requirements",
                "Installation",
                "Notes",
                "Example",
                "Examples",
                "References",
                # "Design" kept per CEO #360 (explicitly named). codex flagged
                # Architecture/Configuration/Reference/Well/Note/Usage as words that
                # CAN be legit single-word domain entities (the stoplist can't tell a
                # heading from mid-sentence use) — those were dropped from the list.
                "Design",
                # Common adverbs / connectives capitalized mid-sentence (#360:
                # "Only" leaked). Never domain entities.
                "Only",
                "Just",
                "Even",
                "Once",
                "Still",
                "Yet",
                "Again",
                "Perhaps",
                "Instead",
                "Otherwise",
                "Meanwhile",
                "Therefore",
                "Thus",
                "Hence",
                "Rather",
                "Furthermore",
                "Moreover",
                "Additionally",
                # ALL-CAPS emphasis of pure function words (#360: `must NOT fail`
                # leaked "NOT"). The exact-match check above only listed title-case
                # ("Not"), so the SHOUTED form slipped through. Kept TIGHT to
                # unambiguous connectives/negations — deliberately NOT adding 2-letter
                # forms like IT/US/OR whose upper-case IS a legit acronym
                # (Information Technology / United States) that must still surface.
                "NOT",
                "AND",
                "BUT",
                "NOR",
            }
        )
        # Strip well-formed markdown links so their URLs never leak into entities
        # (dogfood 2026-07-11: "Apache-2.0)](https://github.com/oimiragieo/tensor-grep"
        # surfaced as a "Key entity" from a `[text](url)` link). Mirrors
        # _generate_summary's link strip; a token-level guard below additionally
        # rejects any residual URL/path fragment (bare or split links the regex
        # can't collapse).
        cleaned_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
        words = cleaned_text.split()
        entities = []
        for i, word in enumerate(words):
            # Capitalized and NOT a sentence start (prev word didn't end a sentence).
            if word[0].isupper() and i > 0 and words[i - 1][-1] not in ".!?":
                # Strip surrounding punctuation AND markdown emphasis asterisks so
                # "JWT**" -> "JWT" (dogfood 2026-07-11 #287). Only '*' is added, NOT
                # '_': codex flagged that stripping trailing '_' would mangle legit
                # identifier conventions (Type_, id_), and '_'-wrapped emphasis
                # (_JWT_) is already excluded by the leading-'_' isupper check.
                # .strip() only touches the ends, so mid-word underscores
                # (gc_kb_query) and slashes (TCP/IP) are preserved.
                cleaned = word.strip(".,;:!?\"'()[]{}*")
                # Truncate at the first EMBEDDED paren/double-quote: a code fragment
                # like `SettlementError(f"failed` (dogfood 2026-07-12, fenced code)
                # recovers to the clean identifier `SettlementError` instead of
                # surfacing raw code syntax as an entity. `.strip()` only touches ends,
                # so a mid-word '('/')'/'"' survives — this cuts it. The apostrophe is
                # deliberately EXCLUDED (droid gate): a legit entity like O'Reilly must
                # not truncate to "O". The '(' still catches the code case. Legit
                # entities (Fly.io, TCP/IP, gc_kb_query) contain no parens/quotes.
                cleaned = re.split(r'[()"]', cleaned, maxsplit=1)[0]
                # An entity is a WORD, not a link. '://' catches URLs (codex
                # 2026-07-11: a bare '/' wrongly dropped legit tech entities like
                # TCP/IP, CI/CD, AC/DC); '](' catches a malformed/split markdown
                # link the regex missed; an http(s) prefix catches a scheme-only
                # residue.
                if (
                    "://" in cleaned
                    or "](" in cleaned
                    or cleaned.lower().startswith(("http:", "https:"))
                ):
                    continue
                if len(cleaned) >= 2 and cleaned not in stopwords:
                    entities.append(cleaned)

        # dict.fromkeys preserves first-occurrence order (deterministic across
        # PYTHONHASHSEED); list(set(...)) was hashseed-dependent, so the surviving
        # entities after [:max_entities] varied run-to-run. See test_output_determinism.py.
        return list(dict.fromkeys(entities))[:max_entities]

    def _generate_summary(self, text: str, max_length: int = 100) -> str:
        """
        Generate a simple extractive summary: the first substantive sentence with
        markdown noise stripped (Task 4 — dogfood found headings/links/backticks
        leaking into the "Summary:" line, e.g. a summary that was literally "## Section").
        """
        # Strip inline markdown so the summary reads as prose, not raw markup.
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
        cleaned = cleaned.replace("`", "")
        # Strip paired markdown emphasis so bold/italic markers don't leak into the
        # summary (dogfood 2026-07-12: `**token-bucket**` leaked verbatim). Asterisk
        # emphasis only — underscores are left alone so snake_case identifiers survive.
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)  # **bold** -> bold
        cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)  # *italic* -> italic
        # Strip mkdocs/python-markdown admonition markers ('!!! note', '??? tip "Title"')
        # BEFORE the split: the sentence splitter treats '!!!' as a sentence end, so a
        # leading admonition fragmented into a bare "!!!" summary (dogfood 2026-07-11).
        cleaned = _strip_admonition_markers(cleaned)
        # Strip markdown heading markers at the START OF EVERY LINE (not just the
        # candidate start): a chunk with multiple headings whose lines whitespace-
        # collapse into one candidate — e.g. non-Latin text the '.!?' splitter can't
        # segment on full-width '。' — otherwise leaks a mid-summary '##' (dogfood
        # 2026-07-12: CJK `部署手册 ## 概述 ...`). `#` mid-word (C#, F#) is NOT a
        # line-start heading, so it survives; only line-initial `#{1,6} ` is cut.
        cleaned = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", cleaned)
        # Strip markdown blockquote markers at the START OF EVERY LINE: a multi-line
        # blockquote (`> line one\n> line two`) whitespace-collapses into one candidate
        # and otherwise leaks the `>` markers mid-summary (dogfood 2026-07-14: our own
        # llms.txt `> Semantic compression API ... > PageRank-ranked ...`). Nested
        # blockquotes (`>> `) are stripped too. A `>` mid-line (e.g. `a > b`) is NOT
        # line-initial, so comparisons / HTML survive.
        cleaned = re.sub(r"(?m)^[ \t]*(?:>[ \t]?)+", "", cleaned)
        sentences = _SENTENCE_SPLIT_RE.split(cleaned)
        summary = ""
        for candidate in sentences:
            # Drop leading heading / list markers from the candidate sentence.
            candidate = re.sub(r"^\s*#{1,6}\s+", "", candidate)
            candidate = re.sub(r"^\s*[-*+]\s+", "", candidate)
            candidate = re.sub(r"^\s*\d+\.\s+", "", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()  # collapse internal whitespace
            # Require substantive content: skip punctuation-only fragments ('!!!', '---').
            # `[^\W_]` = any Unicode letter or digit (no '_'), so a non-Latin sentence
            # (CJK/Cyrillic/Arabic) IS substantive and gets the markdown-heading strip
            # above — the old ASCII-only `[A-Za-z0-9]` treated every non-Latin line as
            # non-substantive, so it fell to the `cleaned.strip()` fallback that retains
            # the leading '#'/'-' markers (dogfood 2026-07-12: `Summary: "# 部署手册"`).
            if candidate and re.search(r"[^\W_]", candidate):
                summary = candidate
                break
        if not summary:
            summary = cleaned.strip()
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary

    def _extractive_anchor_content(self, text: str, token_budget: int) -> str:
        """Budgeted extractive render for a KEPT anchor (world-class audit #1).

        A kept anchor previously rendered only a 1-sentence ``_generate_summary``
        (<=150 chars) -- an outline, not a compression: the actual content lived
        only behind ``modulate_region``. Instead keep the leading markdown-cleaned
        sentences of ``text`` up to ``token_budget`` tokens so the skeleton itself
        is faithful (LLMLingua-2 faithfulness lever). At least one sentence is
        always kept (a tiny budget still yields content); the 1-sentence summary
        stays the fallback for ``[HIDDEN]`` nodes. Deterministic (source sentence
        order preserved), no model load -- unit-tested via ``object.__new__``.
        """
        if not text or token_budget <= 0:
            return ""
        cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
        cleaned = cleaned.replace("`", "")
        # Strip paired markdown emphasis (see _generate_summary): asterisk bold/italic
        # otherwise leaks into a kept anchor's extractive render (dogfood 2026-07-12).
        cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cleaned)  # **bold** -> bold
        cleaned = re.sub(r"\*(.+?)\*", r"\1", cleaned)  # *italic* -> italic
        cleaned = _strip_admonition_markers(cleaned)
        # Strip line-initial markdown heading markers (see _generate_summary): a
        # mid-content '##' otherwise survives when multiple heading lines collapse
        # into one candidate (dogfood 2026-07-12 CJK). 'C#'/'F#' are preserved.
        cleaned = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", cleaned)
        kept: list[str] = []
        used = 0
        for candidate in _SENTENCE_SPLIT_RE.split(cleaned):
            candidate = re.sub(r"^\s*#{1,6}\s+", "", candidate)
            candidate = re.sub(r"^\s*[-*+]\s+", "", candidate)
            candidate = re.sub(r"^\s*\d+\.\s+", "", candidate)
            candidate = re.sub(r"\s+", " ", candidate).strip()
            # Skip punctuation-only fragments ('!!!', '---') -- not substantive content.
            # `[^\W_]` matches any Unicode letter/digit (no '_'), so a non-Latin
            # sentence still counts as substantive (dogfood 2026-07-12: CJK/Cyrillic).
            if not candidate or not re.search(r"[^\W_]", candidate):
                continue
            t = self._count_tokens(candidate)
            if kept and used + t > token_budget:
                break  # keep >=1 sentence even if it alone exceeds the budget
            kept.append(candidate)
            used += t
            if used >= token_budget:
                break
        return " ".join(kept)

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
