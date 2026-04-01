"""PoC quality predictor for semantic compression.

Predicts compression quality (0.0–1.0) without running the full compression
pipeline by simulating skeleton selection and measuring:

* **Entity retention**: fraction of named entities preserved.
* **Coverage**: fraction of sentences with ≥1 word in compressed text.
* **Relevance**: optional query-sentence similarity (falls back to coverage).

Used by ``auto_select_profile()`` in ``compression_profiles.py`` to pick the
lightest profile that meets a caller-specified quality floor.
"""

from __future__ import annotations

import re
from typing import Optional

from src.constants import (
    COMPRESSION_PROFILES,
    QUALITY_COVERAGE_WEIGHT,
    QUALITY_ENTITY_WEIGHT,
    QUALITY_RELEVANCE_WEIGHT,
)


class QualityPredictor:
    """Lightweight, dependency-free quality predictor.

    All methods are synchronous and require no embedding models.
    """

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def extract_entities(self, text: str) -> list[str]:
        """Find capitalised words that do not start a sentence.

        Heuristic: a word is an entity if it starts with an uppercase letter
        AND it is not the first word of a sentence (i.e., it is not preceded
        by a sentence-ending punctuation followed by whitespace).

        Args:
            text: Input text.

        Returns:
            Deduplicated list of entity strings, preserving first-seen order.
        """
        if not text:
            return []

        # Tokenise by whitespace
        words = text.split()
        entities: list[str] = []
        seen: set[str] = set()

        # Positions that are sentence-starters: index 0, or index after sentence end
        sentence_starters: set[int] = {0}
        for i, word in enumerate(words[:-1]):
            stripped = word.rstrip()
            if stripped and stripped[-1] in ".!?":
                sentence_starters.add(i + 1)

        for i, word in enumerate(words):
            if i in sentence_starters:
                continue
            # Strip punctuation for matching
            clean = word.strip(".,;:!?()[]{}\"'")
            if clean and clean[0].isupper() and clean not in seen:
                entities.append(clean)
                seen.add(clean)

        return entities

    def compute_entity_retention(self, entities: list[str], compressed: str) -> float:
        """Fraction of *entities* found in *compressed*.

        Args:
            entities: Reference entity list (from :meth:`extract_entities`).
            compressed: Compressed text to check.

        Returns:
            Float in ``[0.0, 1.0]``.
        """
        if not entities:
            return 1.0
        found = sum(1 for e in entities if e in compressed)
        return found / len(entities)

    def compute_coverage(self, sentences: list[str], compressed: str) -> float:
        """Fraction of *sentences* that have ≥1 word present in *compressed*.

        A sentence is "covered" if at least one of its non-trivial words
        (length ≥ 3) appears in the compressed text.

        Args:
            sentences: Reference sentence list.
            compressed: Compressed text.

        Returns:
            Float in ``[0.0, 1.0]``.
        """
        if not sentences:
            return 1.0
        if not compressed:
            return 0.0

        compressed_words = set(compressed.lower().split())
        covered = 0
        for sentence in sentences:
            words = [w.strip(".,;:!?()[]{}\"'") for w in sentence.lower().split()]
            meaningful = [w for w in words if len(w) >= 3]
            if not meaningful:
                covered += 1  # trivially covered
                continue
            if any(w in compressed_words for w in meaningful):
                covered += 1

        return covered / len(sentences)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_compression(self, text: str, profile_name: str) -> str:
        """Simulate compression by keeping top sentences by importance score.

        Uses a simple heuristic: sentences scored by word count + entity density.
        The ``skeleton_ratio`` of the profile controls how many sentences to keep.

        Args:
            text: Original text.
            profile_name: Profile name from ``COMPRESSION_PROFILES``.

        Returns:
            Simulated compressed text (subset of original sentences).
        """
        if not text or not text.strip():
            return ""

        profile_spec = COMPRESSION_PROFILES.get(
            profile_name, COMPRESSION_PROFILES.get("balanced", {})
        )
        ratio = float(profile_spec.get("skeleton_ratio", 0.25))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return text

        # Score by (length_score + entity_density)
        def _score(sentence: str) -> float:
            words = sentence.split()
            if not words:
                return 0.0
            length_score = min(1.0, len(words) / 20.0)
            uppercase = sum(1 for w in words[1:] if w and w[0].isupper())
            entity_density = uppercase / len(words)
            return length_score + entity_density

        scored = sorted(enumerate(sentences), key=lambda x: _score(x[1]), reverse=True)
        keep_n = max(1, round(len(sentences) * ratio))
        kept_indices = sorted(i for i, _ in scored[:keep_n])
        return " ".join(sentences[i] for i in kept_indices)

    # ------------------------------------------------------------------
    # Quality prediction
    # ------------------------------------------------------------------

    def predict_quality(
        self,
        original: str,
        compressed: str,
        query: Optional[str] = None,
    ) -> float:
        """Predict quality of *compressed* relative to *original*.

        Weighted combination:
        - entity retention (weight: ``QUALITY_ENTITY_WEIGHT``)
        - coverage (weight: ``QUALITY_COVERAGE_WEIGHT``)
        - relevance (weight: ``QUALITY_RELEVANCE_WEIGHT``) — falls back to
          coverage when *query* is ``None``.

        Args:
            original: Full original text.
            compressed: Compressed text to evaluate.
            query: Optional query for relevance scoring.

        Returns:
            Quality score in ``[0.0, 1.0]``.
        """
        if not original or not original.strip():
            return 1.0

        entities = self.extract_entities(original)
        entity_score = self.compute_entity_retention(entities, compressed)

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", original) if s.strip()]
        coverage_score = self.compute_coverage(sentences, compressed)

        # Relevance: simple word-overlap between query and compressed
        if query and compressed:
            query_words = set(query.lower().split())
            comp_words = set(compressed.lower().split())
            if query_words:
                relevance_score = len(query_words & comp_words) / len(query_words)
            else:
                relevance_score = coverage_score
        else:
            relevance_score = coverage_score

        score = (
            QUALITY_ENTITY_WEIGHT * entity_score
            + QUALITY_COVERAGE_WEIGHT * coverage_score
            + QUALITY_RELEVANCE_WEIGHT * relevance_score
        )
        return max(0.0, min(1.0, score))

    def predict_all_profiles(self, text: str, query: Optional[str] = None) -> dict[str, float]:
        """Predict quality for every profile in ``COMPRESSION_PROFILES``.

        Args:
            text: Original text.
            query: Optional query string.

        Returns:
            Dict mapping profile name -> predicted quality score.
        """
        results: dict[str, float] = {}
        for profile_name in COMPRESSION_PROFILES:
            simulated = self.simulate_compression(text, profile_name)
            results[profile_name] = self.predict_quality(text, simulated, query)
        return results
