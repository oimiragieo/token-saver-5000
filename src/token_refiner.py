"""Token-level skeleton refinement inspired by LLMLingua.

Scores individual tokens by importance and removes low-value tokens
(articles, fillers, redundant connectors) while preserving semantic
anchors (numbers, code identifiers, quoted strings, URLs).

Extended with MIG (Marginal Information Gain) scoring based on
COMI arXiv 2602.01719. The MIG scorer selects tokens greedily:
each step picks the token with highest relevance minus lambda * max-redundancy
relative to already-selected tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Tokens that are always removable (low information density)
FILLER_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "basically",
        "actually",
        "essentially",
        "really",
        "very",
        "quite",
        "just",
        "simply",
        "merely",
        "rather",
        "somewhat",
        "perhaps",
        "this",
        "that",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "it",
        "its",
    }
)

# Patterns that indicate a token MUST be preserved
PRESERVE_PATTERNS = [
    re.compile(r"^\d[\d,._]*$"),  # Numbers: 42, 3.14, 1,000
    re.compile(r"^[a-z_][a-z0-9_]*_\w+", re.I),  # snake_case identifiers
    re.compile(r"^[a-z]+[A-Z]"),  # camelCase identifiers
    re.compile(r'^["\']'),  # Quoted strings
    re.compile(r"^https?://"),  # URLs
    re.compile(r"^/[\w/.]"),  # File paths
    re.compile(r"^[A-Z]{2,}$"),  # ACRONYMS
    re.compile(r"^\$\w"),  # Variables like $HOME
    re.compile(r"^[@#]\w"),  # Tags like @param, #TODO
]


@dataclass
class RefinedResult:
    """Result of a token refinement operation."""

    original_text: str
    refined_text: str
    original_tokens: int
    refined_tokens: int
    removed_count: int
    compression_ratio: float  # original / refined


@dataclass
class MIGConfig:
    """Configuration for Marginal Information Gain (MIG) token scoring.

    Attributes:
        lambda_redundancy: Weight for redundancy penalty (0.0 = no penalty,
            1.0 = full penalty).  Default 0.5 from COMI paper.
        min_corpus_tokens: Minimum token count to attempt TF-IDF vectorisation.
            Below this threshold the scorer falls back to heuristic scoring.
    """

    lambda_redundancy: float = 0.5
    min_corpus_tokens: int = 10


class MIGScorer:
    """Marginal Information Gain scorer (COMI, arXiv 2602.01719).

    Scores tokens by:
        MIG(t) = relevance(t, query) - lambda * max_redundancy(t, selected)

    Tokens matching PRESERVE_PATTERNS are always assigned score 1.0.
    Falls back to uniform scoring when sklearn is unavailable or the
    query is empty.

    Args:
        config: :class:`MIGConfig` tuning parameters.
        preserve_patterns: Patterns for hard-preserved tokens.
    """

    def __init__(
        self,
        config: Optional[MIGConfig] = None,
        preserve_patterns: list[re.Pattern] | None = None,
    ) -> None:
        self.config = config or MIGConfig()
        self.preserve_patterns = preserve_patterns or PRESERVE_PATTERNS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tokens_mig(
        self,
        tokens: list[str],
        query: str,
    ) -> list[tuple[str, float]]:
        """Score *tokens* by Marginal Information Gain relative to *query*.

        Args:
            tokens: List of token strings to score.
            query: Query string used to compute relevance.

        Returns:
            List of ``(token, score)`` pairs in original order.
            Scores are in ``[0.0, 1.0]``.
        """
        if not tokens:
            return []

        # Identify hard-preserved indices (pattern matches → score 1.0)
        forced: set[int] = set()
        for i, token in enumerate(tokens):
            stripped = token.strip(".,;:!?()[]{}\"'")
            for pattern in self.preserve_patterns:
                if stripped and pattern.match(stripped):
                    forced.add(i)
                    break

        # Fall back to heuristic when query is empty or too short
        if not query or not query.strip():
            return self._fallback_scores(tokens, forced)

        # Attempt TF-IDF-based MIG scoring
        try:
            return self._score_tfidf_mig(tokens, query, forced)
        except Exception:
            return self._fallback_scores(tokens, forced)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fallback_scores(self, tokens: list[str], forced: set[int]) -> list[tuple[str, float]]:
        """Uniform fallback: forced=1.0, others=0.5."""
        return [(t, 1.0 if i in forced else 0.5) for i, t in enumerate(tokens)]

    def _score_tfidf_mig(
        self, tokens: list[str], query: str, forced: set[int]
    ) -> list[tuple[str, float]]:
        """TF-IDF based MIG scoring using sklearn (no scipy dependency)."""
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        n = len(tokens)

        # Build corpus: each token is its own "document" + query document
        corpus = tokens + [query]
        if len(corpus) < 2:
            return self._fallback_scores(tokens, forced)

        vectorizer = TfidfVectorizer(max_features=500, lowercase=True)
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Can fail on empty vocabulary
            return self._fallback_scores(tokens, forced)

        # Dense arrays for simplicity (avoids scipy dependency)
        token_vecs = matrix[:n].toarray()  # shape (n, features)
        query_vec = matrix[n].toarray().flatten()  # shape (features,)
        query_norm = np.linalg.norm(query_vec)

        # Compute relevance: cosine(token_vec, query_vec)
        if query_norm == 0:
            relevances = np.zeros(n)
        else:
            norms = np.linalg.norm(token_vecs, axis=1)
            relevances = np.where(
                norms > 0,
                (token_vecs @ query_vec) / (norms * query_norm),
                0.0,
            )

        # Greedy MIG selection
        scores = np.full(n, -1.0)
        selected_vecs: list[np.ndarray] = []

        # First pass: force-assign preserved tokens
        for i in sorted(forced):
            scores[i] = 1.0
            selected_vecs.append(token_vecs[i])

        # Remaining tokens: greedy MIG
        unscored = [i for i in range(n) if scores[i] < 0]
        for _ in range(len(unscored)):
            best_idx = -1
            best_mig = -float("inf")

            for i in unscored:
                if scores[i] >= 0:
                    continue
                relevance = float(relevances[i])
                redundancy = 0.0
                if selected_vecs:
                    selected_matrix = np.array(selected_vecs)  # (k, features)
                    tok_norm = np.linalg.norm(token_vecs[i])
                    if tok_norm > 0:
                        sel_norms = np.linalg.norm(selected_matrix, axis=1)
                        dot_products = selected_matrix @ token_vecs[i]
                        sims = np.where(
                            sel_norms > 0,
                            dot_products / (sel_norms * tok_norm),
                            0.0,
                        )
                        redundancy = float(np.max(sims)) if len(sims) > 0 else 0.0

                mig = relevance - self.config.lambda_redundancy * redundancy
                if mig > best_mig:
                    best_mig = mig
                    best_idx = i

            if best_idx >= 0:
                # Map MIG score to [0.01, 1.0] range
                raw = max(0.01, min(1.0, best_mig + 0.5))
                scores[best_idx] = raw
                selected_vecs.append(token_vecs[best_idx])
            else:
                break

        # Any remaining unscored (shouldn't happen) get 0.5
        scores = np.where(scores < 0, 0.5, scores)

        return [(tokens[i], float(scores[i])) for i in range(n)]


class TokenRefiner:
    """Refines skeleton text by removing low-importance tokens.

    Supports two scoring modes:
    - ``'heuristic'`` (default): rule-based filler word removal.
    - ``'mig'``: Marginal Information Gain scoring (requires ``query``).
    """

    def __init__(
        self,
        filler_words: frozenset[str] = FILLER_WORDS,
        preserve_patterns: list[re.Pattern] | None = None,
        scoring_mode: str = "heuristic",
        mig_config: Optional[MIGConfig] = None,
    ) -> None:
        self.filler_words = filler_words
        self.preserve_patterns = preserve_patterns or PRESERVE_PATTERNS
        self.scoring_mode = scoring_mode
        self.mig_config = mig_config
        self._mig_scorer: Optional[MIGScorer] = None
        if scoring_mode == "mig":
            self._mig_scorer = MIGScorer(
                config=mig_config,
                preserve_patterns=self.preserve_patterns,
            )

    def score_token(self, token: str) -> float:
        """Score a single token from 0.0 (removable) to 1.0 (must keep).

        Args:
            token: The raw token string (may include surrounding punctuation).

        Returns:
            Score between 0.0 and 1.0. Scores <= 0.1 are candidates for removal.
        """
        stripped = token.strip(".,;:!?()[]{}\"'")
        if not stripped:
            return 0.5  # punctuation-only tokens: keep

        lower = stripped.lower()

        # Check preserve patterns first (highest priority)
        for pattern in self.preserve_patterns:
            if pattern.match(stripped):
                return 1.0

        # Filler words are candidates for removal
        if lower in self.filler_words:
            return 0.1

        # Short common words (2-3 chars) get low scores but aren't auto-removed
        if len(stripped) <= 2:
            return 0.3

        # Everything else is content - keep
        return 0.7

    def score_tokens(self, text: str) -> list[tuple[str, float]]:
        """Score all tokens in text.

        Args:
            text: Input text to score.

        Returns:
            List of (token, score) pairs in original order.
        """
        tokens = text.split()
        return [(t, self.score_token(t)) for t in tokens]

    def refine(
        self,
        text: str,
        target_ratio: float = 0.7,
        query: Optional[str] = None,
    ) -> RefinedResult:
        """Refine text by removing low-importance tokens.

        Args:
            text: Input text to refine.
            target_ratio: Fraction of tokens to keep (0.7 = keep 70%, remove 30%).
                Set to 1.0 to skip refinement.
            query: Optional query string.  When ``scoring_mode='mig'`` this
                activates MIG-based token scoring; ignored in heuristic mode.

        Returns:
            RefinedResult with original and refined text plus stats.
        """
        if not text or not text.strip() or target_ratio >= 1.0:
            token_count = len(text.split()) if text else 0
            return RefinedResult(
                original_text=text,
                refined_text=text,
                original_tokens=token_count,
                refined_tokens=token_count,
                removed_count=0,
                compression_ratio=1.0,
            )

        # Split into sentences to enforce boundary protection
        sentences = re.split(r"(?<=[.!?])\s+", text)
        refined_sentences = []
        total_original = 0
        total_removed = 0

        for sentence in sentences:
            tokens = sentence.split()
            if not tokens:
                continue
            total_original += len(tokens)

            if len(tokens) <= 3:
                # Very short sentences: keep entirely
                refined_sentences.append(sentence)
                continue

            # Choose scoring method
            if self.scoring_mode == "mig" and self._mig_scorer is not None and query:
                scored = self._mig_scorer.score_tokens_mig(tokens, query)
            else:
                scored = [(t, self.score_token(t)) for t in tokens]

            # Collect removal candidates: score <= 0.1 (or < 0.4 for MIG),
            # not at sentence boundaries
            threshold = 0.4 if self.scoring_mode == "mig" else 0.1
            removable_indices = [
                i
                for i, (_, score) in enumerate(scored)
                if score <= threshold and i != 0 and i != len(tokens) - 1
            ]

            # Calculate how many to remove to hit target ratio
            target_keep = max(2, int(len(tokens) * target_ratio))
            to_remove = min(len(removable_indices), len(tokens) - target_keep)

            if to_remove <= 0:
                refined_sentences.append(sentence)
                continue

            # Sort removable by score ascending (remove lowest-scored first)
            removable_indices_sorted = sorted(removable_indices, key=lambda i: scored[i][1])
            remove_set = set(removable_indices_sorted[:to_remove])
            kept = [t for i, t in enumerate(tokens) if i not in remove_set]
            total_removed += to_remove
            refined_sentences.append(" ".join(kept))

        refined_text = " ".join(refined_sentences)
        refined_tokens = total_original - total_removed

        return RefinedResult(
            original_text=text,
            refined_text=refined_text,
            original_tokens=total_original,
            refined_tokens=refined_tokens,
            removed_count=total_removed,
            compression_ratio=round(total_original / max(1, refined_tokens), 2),
        )
