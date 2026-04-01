"""Token-level skeleton refinement inspired by LLMLingua.

Scores individual tokens by importance and removes low-value tokens
(articles, fillers, redundant connectors) while preserving semantic
anchors (numbers, code identifiers, quoted strings, URLs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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


class TokenRefiner:
    """Refines skeleton text by removing low-importance tokens."""

    def __init__(
        self,
        filler_words: frozenset[str] = FILLER_WORDS,
        preserve_patterns: list[re.Pattern] | None = None,
    ) -> None:
        self.filler_words = filler_words
        self.preserve_patterns = preserve_patterns or PRESERVE_PATTERNS

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

    def refine(self, text: str, target_ratio: float = 0.7) -> RefinedResult:
        """Refine text by removing low-importance tokens.

        Args:
            text: Input text to refine.
            target_ratio: Fraction of tokens to keep (0.7 = keep 70%, remove 30%).
                Set to 1.0 to skip refinement.

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

            scored = [(t, self.score_token(t)) for t in tokens]

            # Collect removal candidates: score <= 0.1, not at sentence boundaries
            removable_indices = [
                i
                for i, (_, score) in enumerate(scored)
                if score <= 0.1 and i != 0 and i != len(tokens) - 1
            ]

            # Calculate how many to remove to hit target ratio
            target_keep = max(2, int(len(tokens) * target_ratio))
            to_remove = min(len(removable_indices), len(tokens) - target_keep)

            if to_remove <= 0:
                refined_sentences.append(sentence)
                continue

            # Remove the first `to_remove` lowest-scored candidates (already score <= 0.1)
            remove_set = set(removable_indices[:to_remove])
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
