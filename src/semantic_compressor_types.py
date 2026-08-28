"""Types, constants, and helpers for semantic compression."""

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
from .bm25_utils import (
    bm25_scores as _bm25_score_texts,
    query_has_lexical_shape as _gate_query_has_lexical_shape,
)
from .constants import (
    _RRF_K,
    F11_RANKER_PATH,
    RERANK_ENABLED,
    RERANK_POOL_SIZE,
    DEFAULT_TEXT_MODEL,
)
from .node_identity import extract_file_id_from_node
from .reranker_gate import RerankConfig, rerank_candidates

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


def _gate_should_fuse_g(query: str) -> bool:
    """F11 Path G gate (design memo idea #1, EXPERIMENTAL --
    ``docs/audits/2026-07-08-f11-retrieval-fusion-ideas.md`` section 2):
    should this query's ranking fuse BM25+RRF, or fall back to Path A
    dense-only?

    Gate = query-shape ONLY: fuse only when the query itself carries a
    provable lexical signal -- a digit-bearing token, an identifier-shaped
    token, or a quoted phrase (``query_has_lexical_shape``). These are the
    classes BM25 wins or ties-at-rank-1 on.

    The former second predicate (``bm25_top1_is_discriminative``) was REMOVED
    2026-07-10 (#267): the #266 scaled significance corpus (15 fixtures / 104
    queries) proved it fires on 14/15 ``pure_paraphrase`` queries as a FALSE
    POSITIVE -- a single incidental stemmed-term match on a small candidate
    set reads as "discriminative" (high IDF), opens the gate, and RRF then
    drops the true answer on a zero-overlap NL query. It regressed
    ``pure_paraphrase`` top-1 67->20% to gain only +7pp on ``lexical_trap``.
    Query-shape alone has ZERO false positives on NL in the corpus and keeps
    the bare_numeric top-1 win (43->100%). See ``f11_fixture_harness --gac``
    + memory ``compression-ranker-flipped-to-g``. The gate now takes ONLY the
    query -- it is evaluated before BM25 is computed, so a closed NL query
    never pays for (or can fail inside) BM25 scoring (#267 / codex 2026-07-10).

    Pure function, no model dependency -- unit-testable in isolation (see
    ``tests/test_f11_gated_fusion.py``).
    """
    return _gate_query_has_lexical_shape(query)


def _strip_admonition_markers(text: str) -> str:
    """Remove leading mkdocs / python-markdown admonition markers.

    ``!!! note``, ``??? tip "Title"``, ``???+ warning`` otherwise fragment on the
    sentence splitter (which treats ``!!!`` as a sentence end) into a bare "!!!"
    summary (dogfood 2026-07-11: httpx's mkdocs docs surfaced ``Summary: "!!!"``).
    Strips the marker + admonition type + optional quoted title, leaving the
    admonition body as the first substantive text. Pure function, model-free.
    """
    return re.sub(
        r'(?m)^[ \t]*[!?]{3}\+?[ \t]+[\w-]+(?:[ \t]+"[^"]*")?[ \t]*',
        "",
        text,
    )


# Sentence splitter for summary / anchor extraction. The ASCII branch
# ``(?<=[.!?])\s+`` requires trailing whitespace so an internal dot ("Fly.io")
# never splits; the CJK branch ``(?<=[。！？])`` splits immediately after a
# full-width terminator because CJK/Japanese text has no inter-sentence space,
# so a multi-sentence CJK paragraph would otherwise collapse into ONE candidate
# and the summary would return the whole blob (dogfood 2026-07-12, #212). Latin
# text has no ``。！？`` so its behavior is byte-identical.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])")


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
    # Ranker score_type of ``scores`` -- "cosine" (Path A / Path G gate-closed)
    # or "rrf" (Path C / Path G gate-open). Lets the handler label the wire
    # `score_type` from what ACTUALLY ran, not a hardcoded path check
    # (blocker-3 fix, 2026-07-08 codex review). Defaults to "cosine" so any
    # older constructor path stays valid.
    score_type: str = "cosine"


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
