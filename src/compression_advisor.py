"""
Compression Size Preview and Estimation

This module provides size estimation for documents BEFORE actual compression.
Helps users understand expected token savings without expensive embedding operations.

**Why This Matters:**
- Compression takes time (~1-3 seconds for medium docs)
- Users want to know if compression is worth it before committing
- Provides realistic expectations based on document characteristics

**How It Works:**
1. Analyze document length, structure, repetition
2. Apply heuristic formulas based on proven compression ratios
3. Return estimated ranges (best/expected/worst case)

**Proven Baseline (demo_proof.py):**
- 485-token quantum paper → 61 tokens (7.9× compression, 87.4% reduction)
- Small docs (<100 tokens): 2-4× (may expand due to skeleton overhead)
- Medium docs (500 tokens): 5-10×
- Large docs (5000+ tokens): 15-20×

Version: 0.4.1
Author: Token Saver 5000 Team
"""

import re
from typing import Dict, Any, Optional
from dataclasses import dataclass

import tiktoken

from .metrics import compute_cost_savings


@dataclass
class CompressionEstimate:
    """Estimated compression results"""

    original_tokens: int
    estimated_compressed: int
    compression_ratio: float
    token_savings_pct: float
    confidence: str  # "high", "medium", "low"
    reasoning: str
    best_case: Dict[str, Any]
    worst_case: Dict[str, Any]
    cost_telemetry: Optional[Dict[str, Any]] = None


class CompressionAdvisor:
    """
    Estimates compression performance before actual compression.

    Uses heuristic analysis of document characteristics to predict
    token savings without expensive embedding operations.

    Example:
        >>> advisor = CompressionAdvisor()
        >>> text = "..." # 500-token document
        >>> estimate = advisor.estimate_compression(text)
        >>> print(f"Estimated: {estimate.compression_ratio:.1f}×")
        Estimated: 1.4×
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize compression advisor.

        Args:
            encoding_name: Tokenizer encoding (default: cl100k_base for GPT-4)
        """
        self.encoder = tiktoken.get_encoding(encoding_name)

    def estimate_compression(
        self, text: str, skeleton_ratio: float = 0.2, model: Optional[str] = None
    ) -> CompressionEstimate:
        """
        Estimate compression results for document.

        Args:
            text: The document text to analyze
            skeleton_ratio: Target skeleton ratio (default: 0.2 = 20%)

        Returns:
            CompressionEstimate with predictions and reasoning

        Example:
            >>> advisor = CompressionAdvisor()
            >>> text = "Quantum computing..." * 100  # ~500 tokens
            >>> est = advisor.estimate_compression(text)
            >>> print(est.reasoning)
            Medium document (500 tokens) with moderate redundancy...
        """
        # Step 1: Count tokens
        original_tokens = len(self.encoder.encode(text))

        # Step 2: Analyze document characteristics
        redundancy_score = self._estimate_redundancy(text)

        # Step 3: Apply size-based heuristics.
        #
        # Recalibrated 2026-06-12 (#92). The old bands here (7-9x at <500
        # tokens, 15-20x at 2K+) predicted ratios the adaptive engine NEVER
        # produces on those sizes — a live dogfood response carried
        # estimated_ratio=7.08 next to an ACTUAL compression_ratio of 1.17 on
        # the same document. The engine's real curve
        # (semantic_compressor.compute_adaptive_ratio) keeps ~80% of nodes
        # below 8K tokens (faithful-by-design), 50% to 32K, 20% to 100K, 10%
        # above. Outcomes within a band still swing hard with redundancy
        # (measured anchors: 1.17x on a dense ~500-token structured doc vs
        # 9.54x on a highly redundant ~2.3K-token benchmark doc), so these
        # estimates anchor on the CONSERVATIVE end and say so — a user who
        # beats the estimate is delighted; the reverse reads as a broken
        # promise.
        if original_tokens < 200:
            base_ratio = 1.0
            confidence = "high"
            reasoning = (
                f"Tiny document ({original_tokens} tokens) - the skeleton's fixed "
                "overhead (~50 tokens) typically cancels or exceeds any savings. "
                "Send it as-is; compression is not worthwhile below ~200 tokens."
            )
        elif original_tokens < 8_000:
            base_ratio = 1.2 + (redundancy_score * 0.8)  # ~1.2-2.0x conservative
            confidence = "low"
            reasoning = (
                f"Small/medium document ({original_tokens} tokens) - the adaptive "
                "engine keeps ~80% of nodes at this size (faithful-by-design), so "
                "expect modest savings: ~1.2x measured on a dense structured doc. "
                "Highly redundant prose can land far above this estimate "
                "(9.54x benchmark on a redundant doc of similar size), but the "
                "estimate anchors on the conservative end."
            )
        elif original_tokens < 32_000:
            base_ratio = 2.0 + (redundancy_score * 1.0)  # ~2-3x
            confidence = "medium"
            reasoning = (
                f"Medium-large document ({original_tokens:,} tokens) - the adaptive "
                "engine keeps ~50% of nodes at this size. Conservative estimate "
                f"{2.0 + (redundancy_score * 1.0):.1f}x; redundant content can exceed it."
            )
        elif original_tokens < 100_000:
            base_ratio = 4.0 + (redundancy_score * 2.0)  # ~4-6x
            confidence = "medium"
            reasoning = (
                f"Large document ({original_tokens:,} tokens) - the adaptive engine "
                "keeps ~20% of nodes at this size; compression starts paying "
                "substantially here."
            )
        else:
            base_ratio = 6.0 + (redundancy_score * 4.0)  # ~6-10x
            confidence = "medium"
            reasoning = (
                f"Very large document ({original_tokens:,} tokens) - the adaptive "
                "engine keeps ~10% of nodes; this is the strongest-compression "
                f"regime. Expected ~{6.0 + (redundancy_score * 4.0):.1f}x with "
                f"{skeleton_ratio * 100:.0f}% skeleton ratio."
            )

        # Step 4: Adjust for skeleton ratio
        # Skeleton ratio = % of nodes shown as anchors
        # Lower skeleton ratio = more compression
        # Boost the ratio for low skeleton ratios; CLAMP to a sane positive band so a
        # high skeleton_ratio (small docs get ~0.8 from compute_adaptive_ratio) can't
        # drive this to zero (ZeroDivisionError below) or negative -> a nonsensical
        # negative estimated_ratio. #142 dogfood: a 213-token doc returned
        # estimated_ratio -2.45 (ratio_adjustment -2.0 * base 1.2). A compression ratio
        # is always positive; a value below 1.0 is honest expansion (skeleton overhead).
        ratio_adjustment = max(0.5, min(2.0, 1.0 + (0.2 - skeleton_ratio) * 5))
        adjusted_ratio = base_ratio * ratio_adjustment

        # Step 5: Calculate estimates
        estimated_compressed = int(original_tokens / adjusted_ratio)

        # Ensure minimum size (skeleton has overhead)
        min_size = max(30, int(original_tokens * 0.05))  # At least 5% or 30 tokens
        estimated_compressed = max(estimated_compressed, min_size)

        # Calculate savings
        token_savings_pct = (1 - estimated_compressed / original_tokens) * 100

        # Step 6: Generate best/worst case scenarios
        best_ratio = adjusted_ratio * 1.3  # 30% better
        worst_ratio = adjusted_ratio * 0.7  # 30% worse

        best_case = {
            "compressed_tokens": max(min_size, int(original_tokens / best_ratio)),
            "ratio": best_ratio,
            "savings_pct": (1 - max(min_size, int(original_tokens / best_ratio)) / original_tokens)
            * 100,
            "scenario": "High semantic redundancy, clear structure, optimal chunking",
        }

        worst_case = {
            "compressed_tokens": max(min_size, int(original_tokens / worst_ratio)),
            "ratio": worst_ratio,
            "savings_pct": (1 - max(min_size, int(original_tokens / worst_ratio)) / original_tokens)
            * 100,
            "scenario": "Low redundancy, dense unique facts, suboptimal structure",
        }

        return CompressionEstimate(
            original_tokens=original_tokens,
            estimated_compressed=estimated_compressed,
            compression_ratio=adjusted_ratio,
            token_savings_pct=token_savings_pct,
            confidence=confidence,
            reasoning=reasoning,
            best_case=best_case,
            worst_case=worst_case,
            cost_telemetry=(
                compute_cost_savings(
                    original_tokens=original_tokens,
                    compressed_tokens=estimated_compressed,
                    model=model,
                ).to_dict()
                if model
                else None
            ),
        )

    def _average_word_length(self, text: str) -> float:
        """Calculate average word length (proxy for technical density)"""
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)

    def _estimate_redundancy(self, text: str) -> float:
        """
        Estimate semantic redundancy (0.0-1.0).

        Higher redundancy = better compression potential.
        Uses heuristics like repeated phrases, similar sentences.
        """
        # Heuristic 1: Repeated n-grams (phrases)
        words = text.lower().split()
        if len(words) < 10:
            return 0.0

        # Count 3-word phrases
        trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        repetition_rate = 1 - (unique_trigrams / max(total_trigrams, 1))

        # Heuristic 2: Sentence similarity (simple character overlap)
        sentences = re.split(r"[.!?]+", text)
        if len(sentences) < 3:
            return repetition_rate

        # Sample first 5 sentences vs last 5 sentences for thematic overlap
        start_text = " ".join(sentences[:5]).lower()
        end_text = " ".join(sentences[-5:]).lower()

        # Simple Jaccard similarity on words
        start_words = set(start_text.split())
        end_words = set(end_text.split())
        if not start_words or not end_words:
            return repetition_rate

        overlap = len(start_words & end_words) / len(start_words | end_words)

        # Combine metrics
        redundancy = (repetition_rate * 0.7) + (overlap * 0.3)
        return min(redundancy, 1.0)

    def _estimate_structure(self, text: str) -> float:
        """
        Estimate structural quality (0.0-1.0).

        Higher structure = better graph representation = better compression.
        Looks for headers, paragraphs, lists, logical flow.
        """
        score = 0.0

        # Heuristic 1: Headers and sections
        header_patterns = [
            r"^#{1,6}\s+",  # Markdown headers
            r"^[A-Z][A-Za-z\s]+:$",  # Title Case Headers:
            r"^\d+\.\s+[A-Z]",  # 1. Numbered sections
        ]
        header_count = sum(
            len(re.findall(pattern, text, re.MULTILINE)) for pattern in header_patterns
        )
        if header_count > 0:
            score += 0.3

        # Heuristic 2: Paragraphs (double newlines)
        paragraphs = text.split("\n\n")
        if len(paragraphs) >= 3:
            score += 0.2

        # Heuristic 3: Lists and bullet points
        list_markers = len(re.findall(r"^\s*[-*•]\s+", text, re.MULTILINE))
        if list_markers > 0:
            score += 0.2

        # Heuristic 4: Consistent sentence length (indicates organized writing)
        sentences = re.split(r"[.!?]+", text)
        if len(sentences) >= 5:
            sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
            if sentence_lengths:
                avg_len = sum(sentence_lengths) / len(sentence_lengths)
                variance = sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
                # Low variance = consistent writing style
                if variance < 100:  # Threshold for "consistent"
                    score += 0.3

        return min(score, 1.0)

    def format_estimate(self, estimate: CompressionEstimate) -> str:
        """
        Format estimate as human-readable string.

        Args:
            estimate: The CompressionEstimate to format

        Returns:
            Formatted multi-line string with all estimate details
        """
        lines = [
            "[COMPRESSION ESTIMATE]",
            "-" * 60,
            f"Original tokens:        {estimate.original_tokens:,}",
            f"Estimated compressed:   {estimate.estimated_compressed:,} tokens",
            f"Compression ratio:      {estimate.compression_ratio:.1f}×",
            f"Token savings:          {estimate.token_savings_pct:.1f}%",
            f"Confidence:             {estimate.confidence.upper()}",
            "",
            "[REASONING]",
            f"   {estimate.reasoning}",
            "",
            "[BEST CASE]",
            f"   Compressed: {estimate.best_case['compressed_tokens']:,} tokens "
            f"({estimate.best_case['ratio']:.1f}×, {estimate.best_case['savings_pct']:.1f}% savings)",
            f"   {estimate.best_case['scenario']}",
            "",
            "[WORST CASE]",
            f"   Compressed: {estimate.worst_case['compressed_tokens']:,} tokens "
            f"({estimate.worst_case['ratio']:.1f}×, {estimate.worst_case['savings_pct']:.1f}% savings)",
            f"   {estimate.worst_case['scenario']}",
            "-" * 60,
        ]

        # Add recommendation
        if estimate.original_tokens < 100:
            lines.append(
                "[WARN] Recommendation: Document may be too small for effective compression."
            )
            lines.append("   Consider ingesting larger documents or combining multiple small ones.")
        elif estimate.token_savings_pct > 80:
            lines.append("[OK] Recommendation: Excellent candidate for compression!")
            lines.append(
                f"   Expected to save ~{int(estimate.original_tokens - estimate.estimated_compressed):,} tokens."
            )
        else:
            lines.append("[OK] Recommendation: Good candidate for compression.")
            lines.append(
                f"   Expected to save ~{int(estimate.original_tokens - estimate.estimated_compressed):,} tokens."
            )

        return "\n".join(lines)


# Convenience function for backward compatibility
def estimate_compression_size(text: str, skeleton_ratio: float = 0.2) -> Dict[str, Any]:
    """
    Functional interface to compression estimation.

    Args:
        text: Document text to analyze
        skeleton_ratio: Target skeleton ratio (default: 0.2)

    Returns:
        Dictionary with estimate details

    Example:
        >>> result = estimate_compression_size("Long document text...")
        >>> print(result['compression_ratio'])
        7.5
    """
    advisor = CompressionAdvisor()
    est = advisor.estimate_compression(text, skeleton_ratio)

    return {
        "original_tokens": est.original_tokens,
        "estimated_compressed": est.estimated_compressed,
        "compression_ratio": est.compression_ratio,
        "token_savings_pct": est.token_savings_pct,
        "confidence": est.confidence,
        "reasoning": est.reasoning,
        "best_case": est.best_case,
        "worst_case": est.worst_case,
    }
