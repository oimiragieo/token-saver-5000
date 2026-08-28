"""Ingestion and skeleton generation mixin for SemanticCompressor."""

import asyncio
import re
from contextlib import AsyncExitStack
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import networkx as nx

from .semantic_compressor_types import (
    SemanticNode,
    SkeletonResponse,
    compute_adaptive_ratio,
    _MAX_GRAPH_CHUNKS,
    _SIMILARITY_BLOCK_SIZE,
    _node_belongs_to_file,
    logger,
)


class SemanticCompressorIngestMixin:
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
            # #190: detect structured data (JSON/table) on the ORIGINAL text.
            # SemToken would merge "similar" records and the text chunker collapses
            # a raw JSON array into one hidden mega-node — skip SemToken for
            # structured content and chunk on record boundaries (see below).
            from .constants import STRUCTURED_CHUNKING_ENABLED  # noqa: PLC0415
            from .structured_content import detect_structured_content  # noqa: PLC0415

            _structured_kind = (
                detect_structured_content(text) if STRUCTURED_CHUNKING_ENABLED else None
            )
            # json_array, jsonl, and csv have a record-level chunker path
            # (#190 + #279 + #280): skip SemToken so it cannot merge "similar"
            # records into one block before record-splitting runs. json_object
            # still falls through to _chunk_text, so it must NOT skip SemToken.
            _skip_semtoken = _H2H3_COUNT >= 2 or _structured_kind in (
                "json_array",
                "jsonl",
                "csv",
            )
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

            # 1. Chunk the text — record-level for a structured JSON array (#190,
            # so records survive as rankable nodes), else the semantic/fixed chunker.
            raw_chunks = self._prepare_raw_chunks(
                text, _structured_kind, strategy=chunking_strategy
            )
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

            # 2b. Optional intra-document deduplication (Phase 5: R-KV). Skipped for
            # record-chunked structured data (json_array/jsonl/csv — #190 + #279 +
            # #280): identical or near-identical records (repeated log lines, CSV
            # rows) must NOT be collapsed — that would silently drop records (codex
            # P1 = data loss).
            if len(raw_chunks) > 2 and _structured_kind not in ("json_array", "jsonl", "csv"):
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

    def set_file_skeleton_ratio(self, file_id: str, ratio: float | str | None) -> None:
        """Record (or clear) a per-document ``skeleton_ratio`` override.

        Read back by ``_resolve_skeleton_ratio`` instead of the shared
        ``self.skeleton_ratio`` default. This is the mechanism that makes the
        ``ingest_context`` MCP tool's ``skeleton_ratio`` schema parameter a
        real, race-safe knob on the shared singleton compressor — see the
        ``_file_skeleton_ratio_overrides`` note in ``__init__``.

        Args:
            file_id: Document identifier the override applies to.
            ratio: A float in (0.0, 1.0], the string ``"auto"``, or ``None`` to
                clear any existing override for this document (falls back to
                the instance default).
        """
        if ratio is None:
            self._file_skeleton_ratio_overrides.pop(file_id, None)
            return
        self._file_skeleton_ratio_overrides[file_id] = ratio

    def _resolve_skeleton_ratio(self, file_id: str) -> float | str:
        """Return this document's skeleton_ratio override, or the instance default."""
        return self._file_skeleton_ratio_overrides.get(file_id, self.skeleton_ratio)

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
        # Use adaptive ratio if skeleton_ratio is "auto". Per-document override
        # (2026-07-06 knob-honesty fix) takes precedence over the shared
        # instance default — see _resolve_skeleton_ratio.
        effective_ratio = self._resolve_skeleton_ratio(file_id)
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
                #
                # Collapse-bug clamp (2026-07-06, architecture plan Move 5,
                # plan item MF6): counting nodes whose PER-NODE ratio clears
                # the (uniform) effective_ratio bar is a live bug when query
                # relevance concentrates on a small subset of sections —
                # compute_section_ratios keeps the ratio AVERAGE at
                # effective_ratio, but a highly concentrated relevance
                # distribution pushes most per-node ratios toward min_ratio
                # (0.05), so the count-based budget can collapse to as few as
                # 1 node regardless of document size. That silently discards
                # the proportional budget the caller actually requested via
                # skeleton_ratio/fidelity. Clamp to the PROPORTIONAL floor
                # already computed above (num_skeleton = max(1, int(N *
                # effective_ratio))) — the query-adaptive budget may GROW that
                # floor (more relevant sections get more detail) but must
                # never shrink below it.
                num_skeleton_query_adaptive = sum(
                    1 for r in per_node_ratios if r >= effective_ratio
                )
                num_skeleton = max(num_skeleton, num_skeleton_query_adaptive)
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
        # "Skeleton nodes" is the ACTUAL anchor count rendered, not the target
        # `num_skeleton`. On the query_guided path `file_nodes` is narrowed to the
        # query-relevant subset but `num_skeleton` was sized off the original node
        # set, so the header showed impossible counts like "Total 19 | Skeleton 30"
        # (dogfood 2026-07-11 #287). Intersect the selected anchor ids with the
        # in-scope nodes so the count is always <= Total.
        # file_nodes are (node_id, node) tuples (see the render loop below).
        _file_node_ids = {item[0] for item in file_nodes}
        _num_rendered_skeleton = len(set(skeleton_nodes) & _file_node_ids)
        skeleton_lines = []
        skeleton_lines.append(f"=== SEMANTIC SKELETON: {file_id} ===")
        skeleton_lines.append("Skeleton-Version: 2")
        skeleton_lines.append(
            f"Total nodes: {len(file_nodes)} | Skeleton nodes: {_num_rendered_skeleton}"
        )
        skeleton_lines.append(f"Compression: {effective_ratio:.0%} of content shown")
        # Explain hidden-region drill-down ONCE here (Skeleton-Version 2) instead of
        # repeating the phrase on every [HIDDEN] node — the per-node repetition was a
        # hard ratio ceiling (~15-20x). Consumers can branch on Skeleton-Version.
        skeleton_lines.append("Hidden regions expand via modulate_region(node_id).\n")

        node_map = {}
        total_tokens = 0
        skeleton_tokens = 0

        # Render in ORIGINAL DOCUMENT ORDER, not importance-descending order
        # (world-class compression audit #2, 2026-07-07). Node SELECTION above
        # (skeleton_nodes / anchor_node_ids / the COMI+MMR machinery) is
        # UNCHANGED — it still picks which nodes survive by importance, query
        # relevance, and redundancy. Only the ITERATION ORDER for the render
        # loop below changes. Rendering in importance order destroys
        # narrative/legal/code document structure: a supporting section can
        # render AFTER the section that depends on it, or after a [HIDDEN]
        # marker whose detail hasn't been drilled into yet. Every node
        # created during ingest carries its original position in
        # ``metadata["position"]`` (see the Pass-A node-construction loop
        # above). Stable sort with a ``(position, node_id)`` tiebreak keeps
        # output fully deterministic (no PYTHONHASHSEED dependence, no
        # reliance on dict/set iteration order) even for nodes that are
        # missing the key (defensive ``.get(..., 0)`` fallback, consistent
        # with the existing ``persistence.py`` / ``graph_visualizer.py``
        # pattern for this same field).
        render_nodes = sorted(
            file_nodes,
            key=lambda item: (item[1].metadata.get("position") or 0, item[0]),
        )

        for node_id, node in render_nodes:
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
