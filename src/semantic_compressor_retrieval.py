"""Read, search, and evidence retrieval mixin for SemanticCompressor."""

import re
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .bm25_utils import bm25_scores as _bm25_score_texts
from . import constants
from .constants import _RRF_K
from .node_identity import extract_file_id_from_node
from .reranker_gate import RerankConfig, rerank_candidates
from .semantic_compressor_types import (
    DiffReingestionResult,
    EvidenceResult,
    FidelityLevel,
    SemanticNode,
    compute_adaptive_ratio,
    _gate_should_fuse_g,
    _node_belongs_to_file,
    logger,
)


class SemanticCompressorRetrievalMixin:
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

        initial_scores, initial_score_type = self.search_semantic_with_scores_typed(
            query, file_id=file_id, top_k=top_k
        )
        best_cosine = self._max_dense_cosine(query, file_id=file_id)
        sufficient = best_cosine >= effective_threshold
        used_expanded_search = False
        final_scores = initial_scores
        final_score_type = initial_score_type

        if not sufficient:
            used_expanded_search = True
            expanded_k = max(top_k + 1, top_k * max(1, expansion_factor))
            final_scores, final_score_type = self.search_semantic_with_scores_typed(
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
            score_type=final_score_type,
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

    def search_semantic_with_scores_typed(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> Tuple[List[Tuple[str, float]], str]:
        """Ranker dispatch that ALSO reports what it actually ran.

        Returns ``(ranked, score_type)`` where ``score_type`` is:
          - "cosine" — Path A (default), OR Path G when the gate stayed closed
            (dense-only fallback);
          - "rrf" — Path C (unconditional fusion), OR Path G when the gate
            opened and BM25+RRF fusion was applied.

        This is the single source of truth for the F11 dispatch. The public
        ``search_semantic_with_scores`` wraps it and discards the label (its
        return shape is UNCHANGED). ``retrieve_evidence`` uses it to surface the
        per-call label on ``EvidenceResult.score_type`` so the MCP handler can
        label the wire ``score_type`` from what ACTUALLY ran under Path G
        (blocker-3 fix, 2026-07-08 codex review) instead of a hardcoded
        ``== "c"`` check.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            (ranked (node_id, score) tuples, score_type). Under a "cosine"
            score_type the scores are cosine similarity in [-1.0, 1.0]; under
            "rrf" they are RRF scores (NOT bounded to [0, 1]).
        """
        # --- Build file_id-filtered candidate list (shared by all paths) ---
        candidate_nodes: List[Tuple[str, SemanticNode]] = []
        for node_id, node in self.chunks.items():
            if file_id and not _node_belongs_to_file(node_id, file_id):
                continue
            candidate_nodes.append((node_id, node))

        if not candidate_nodes:
            return [], "cosine"

        # --- Dense (cosine) ranking — always computed ---
        query_embedding = self.model.encode([query])[0]
        dense_ranked: List[Tuple[str, float]] = []
        for node_id, node in candidate_nodes:
            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
            dense_ranked.append((node_id, float(similarity)))
        dense_ranked.sort(key=lambda x: x[1], reverse=True)

        # --- Retrieval depth: pull a larger candidate POOL only when the
        # optional cross-encoder rerank is on (#187), so it has candidates to
        # promote; otherwise exactly top_k -> byte-identical to the pre-rerank
        # path (RERANK_ENABLED defaults OFF). ---
        rerank_on = constants.RERANK_ENABLED
        pool_k = max(top_k, constants.RERANK_POOL_SIZE) if rerank_on else top_k

        # --- Path dispatch (produces `ranked` + `score_type`) ---
        if constants.F11_RANKER_PATH == "c":
            # Path C: BM25 + RRF hybrid — unconditional fusion (HOLD as of
            # 2026-07-08; see docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md).
            # Council patch P1: BM25 receives the SAME file_id-filtered
            # candidate_nodes, never self.chunks globally (cross-document IDF
            # pollution prevention).
            bm25_ranked = self._bm25_scores_for_nodes(query, candidate_nodes)
            ranked = self._rrf_fuse(dense_ranked, bm25_ranked, k=_RRF_K, top_k=pool_k)
            score_type = "rrf"
        elif constants.F11_RANKER_PATH == "g" and _gate_should_fuse_g(query):
            # Path G: gated fusion (design memo idea #1, EXPERIMENTAL) — fuse
            # only when the query has provable lexical signal; otherwise degrade
            # to Path A dense-only. The gate is query-shape only (#267) so it is
            # checked BEFORE computing BM25 — a closed NL query never pays for, or
            # can fail inside, BM25 scoring (codex 2026-07-10).
            bm25_ranked = self._bm25_scores_for_nodes(query, candidate_nodes)
            ranked = self._rrf_fuse(dense_ranked, bm25_ranked, k=_RRF_K, top_k=pool_k)
            score_type = "rrf"
        else:
            # Path A (default), or Path G with the gate closed: dense-only.
            ranked = dense_ranked[:pool_k]
            score_type = "cosine"

        # --- Optional recall-gated cross-encoder rerank (#187, default-OFF) ---
        # No-op unless RERANK_ENABLED; reorders the pool, then the top_k truncation
        # below applies to either the reranked or the original order.
        if rerank_on:
            ranked = self._rerank_pool(query, ranked)

        return ranked[:top_k], score_type

    def _get_rerank_scorer(self):
        """Lazily build + cache the ONNX cross-encoder scorer (#187, default-OFF)."""
        scorer = getattr(self, "_rerank_scorer", None)
        if scorer is None:
            from .cross_encoder_scorer import CrossEncoderScorer

            scorer = CrossEncoderScorer()
            self._rerank_scorer = scorer
        return scorer

    def _rerank_pool(self, query: str, ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Recall-gated cross-encoder reorder of the retrieved pool (#187).

        Delegates the pure gate/reorder to ``reranker_gate.rerank_candidates`` with
        the lazily-built ONNX cross-encoder scorer. ``text_of`` pulls each node's
        chunk text; ``score_of`` is the first-stage retrieval score (for the
        confidence-skip). FAIL-SAFE: any failure (model load, scoring) falls back
        to the input retrieval order — reranking must never break retrieval.
        """
        try:
            reordered, _did = rerank_candidates(
                query,
                ranked,
                text_of=lambda pair: (self.chunks[pair[0]].text if pair[0] in self.chunks else ""),
                score_of=lambda pair: pair[1],
                scorer=self._get_rerank_scorer(),
                config=RerankConfig(enabled=True, pool_size=constants.RERANK_POOL_SIZE),
            )
            return reordered
        except Exception:  # pragma: no cover — rerank must never break retrieval
            logger.warning("rerank_pool failed; using retrieval order", exc_info=True)
            return ranked

    def search_semantic_with_scores(
        self, query: str, file_id: Optional[str] = None, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Ranker dispatch — respects F11_RANKER_PATH env var.

        Path "a" (default): dense cosine only (backward compatible).
        Path "c": BM25 + Reciprocal Rank Fusion hybrid (F11 Path C plan; HOLD).
        Path "g": Gated fusion (EXPERIMENTAL) — fuses BM25+RRF only when the
            query has provable lexical signal, else Path A dense-only. See
            docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md section 2.

        Thin wrapper over ``search_semantic_with_scores_typed`` — return shape is
        UNCHANGED (list of tuples). Callers that need the score_type label call
        the typed method directly.

        New in v0.9.0: Returns similarity scores alongside node IDs.
        New in v1.34.35 (F11 Path C): RRF hybrid when F11_RANKER_PATH=c.
        New (F11 Path G, EXPERIMENTAL): gated hybrid when F11_RANKER_PATH=g.

        Args:
            query: Search query
            file_id: Optional file to search within
            top_k: Number of results to return

        Returns:
            List of (node_id, score) tuples ranked by relevance.
            Under Path A: score is cosine similarity in [-1.0, 1.0].
            Under Path C: score is RRF score (float, NOT bounded to [0,1]).
                          Callers must NOT assume score is cosine similarity.
            Under Path G: score is cosine similarity when the gate stayed
                          closed (Path A fallback), or RRF score when the
                          gate opened (same caveat as Path C).
        """
        ranked, _score_type = self.search_semantic_with_scores_typed(query, file_id, top_k)
        return ranked

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

        effective_ratio = self._resolve_skeleton_ratio(file_id)
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

        # #190: mirror the ingest chunking so a re-ingested structured doc diffs
        # against record-level chunks, not the text chunker — otherwise every record
        # reads as changed (codex P2). Flag OFF -> _diff_kind None -> identical.
        from .constants import STRUCTURED_CHUNKING_ENABLED  # noqa: PLC0415
        from .structured_content import detect_structured_content  # noqa: PLC0415

        _diff_kind = detect_structured_content(new_text) if STRUCTURED_CHUNKING_ENABLED else None
        new_chunk_texts = self._prepare_raw_chunks(new_text, _diff_kind)

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
