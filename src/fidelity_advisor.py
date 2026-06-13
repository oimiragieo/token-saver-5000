"""
Fidelity Level Recommendation System

This module provides intelligent recommendations for choosing the appropriate
fidelity level when retrieving content from compressed documents.

**Why This Matters:**
- Users often don't know which fidelity level to use
- Wrong fidelity = wasted tokens or missing critical detail
- This advisor uses heuristics to suggest the optimal level

**Use Cases:**
1. Quick summary → ABSTRACT (10 tokens/node)
2. Topic overview → OUTLINE (30 tokens/node)
3. Entity extraction → STRUCTURE (50 tokens/node)
4. Detailed Q&A → DETAILED (100 tokens/node)
5. Exact quotes → RAW (full text)

Version: 0.4.1
Author: Token Saver 5000 Team
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .semantic_compressor import FidelityLevel


class UseCase(str, Enum):
    """Common use cases for content retrieval"""

    QUICK_SUMMARY = "quick_summary"
    TOPIC_OVERVIEW = "topic_overview"
    ENTITY_EXTRACTION = "entity_extraction"
    QUESTION_ANSWERING = "question_answering"
    DETAILED_ANALYSIS = "detailed_analysis"
    EXACT_QUOTES = "exact_quotes"
    CODE_REVIEW = "code_review"
    FACT_VERIFICATION = "fact_verification"


@dataclass
class FidelityRecommendation:
    """Recommendation result with reasoning"""

    recommended_level: FidelityLevel
    confidence: float  # 0.0-1.0
    reasoning: str
    token_estimate: int
    alternatives: List[Dict[str, Any]]  # Alternative levels with trade-offs


class FidelityAdvisor:
    """
    Intelligent fidelity level recommendation system.

    Analyzes use case, document characteristics, and token budget to suggest
    the optimal fidelity level for content retrieval.

    Example:
        >>> advisor = FidelityAdvisor()
        >>> rec = advisor.recommend(
        ...     use_case=UseCase.QUESTION_ANSWERING,
        ...     num_nodes=5,
        ...     token_budget=300
        ... )
        >>> print(rec.recommended_level)
        FidelityLevel.DETAILED
        >>> print(rec.reasoning)
        Question answering requires detailed context...
    """

    # Token estimates per node for each fidelity level (empirical averages)
    TOKENS_PER_NODE = {
        FidelityLevel.ABSTRACT: 10,
        FidelityLevel.OUTLINE: 30,
        FidelityLevel.STRUCTURE: 50,
        FidelityLevel.DETAILED: 100,
        FidelityLevel.RAW: 200,  # Average estimate, varies widely
    }

    # Use case → recommended fidelity mapping
    USE_CASE_DEFAULTS = {
        UseCase.QUICK_SUMMARY: FidelityLevel.ABSTRACT,
        UseCase.TOPIC_OVERVIEW: FidelityLevel.OUTLINE,
        UseCase.ENTITY_EXTRACTION: FidelityLevel.STRUCTURE,
        UseCase.QUESTION_ANSWERING: FidelityLevel.DETAILED,
        UseCase.DETAILED_ANALYSIS: FidelityLevel.DETAILED,
        UseCase.EXACT_QUOTES: FidelityLevel.RAW,
        UseCase.CODE_REVIEW: FidelityLevel.RAW,
        UseCase.FACT_VERIFICATION: FidelityLevel.RAW,
    }

    def recommend(
        self,
        use_case: UseCase,
        num_nodes: int,
        token_budget: Optional[int] = None,
        query_complexity: str = "medium",
        model: Optional[str] = None,
    ) -> FidelityRecommendation:
        """
        Recommend optimal fidelity level.

        Args:
            use_case: What the user wants to do with the content
            num_nodes: Number of nodes being retrieved
            token_budget: Maximum tokens available (None = no limit)
            query_complexity: "simple", "medium", or "complex"

        Returns:
            FidelityRecommendation with suggested level and reasoning

        Example:
            >>> advisor = FidelityAdvisor()
            >>> rec = advisor.recommend(
            ...     use_case=UseCase.QUESTION_ANSWERING,
            ...     num_nodes=3,
            ...     token_budget=200
            ... )
        """
        # Step 1: Get default recommendation for use case
        default_level = self.USE_CASE_DEFAULTS.get(use_case, FidelityLevel.STRUCTURE)

        # Step 2: Calculate token requirements
        default_tokens = self.TOKENS_PER_NODE[default_level] * num_nodes

        # Step 3: Adjust for token budget constraints
        recommended_level = default_level
        reasoning_parts = []

        # Check if default fits budget
        if token_budget and default_tokens > token_budget:
            # Need to downgrade fidelity
            for level in [
                FidelityLevel.DETAILED,
                FidelityLevel.STRUCTURE,
                FidelityLevel.OUTLINE,
                FidelityLevel.ABSTRACT,
            ]:
                level_tokens = self.TOKENS_PER_NODE[level] * num_nodes
                if level_tokens <= token_budget:
                    recommended_level = level
                    reasoning_parts.append(
                        f"Downgraded from {default_level.name} to fit {token_budget} token budget"
                    )
                    break
        else:
            reasoning_parts.append(f"Use case '{use_case.value}' suggests {default_level.name}")

        # Step 4: Adjust for query complexity
        if query_complexity == "complex" and recommended_level != FidelityLevel.RAW:
            # Bump up one level for complex queries
            level_upgrade = self._upgrade_level(recommended_level)
            if level_upgrade:
                upgrade_tokens = self.TOKENS_PER_NODE[level_upgrade] * num_nodes
                if not token_budget or upgrade_tokens <= token_budget:
                    recommended_level = level_upgrade
                    reasoning_parts.append(
                        "Upgraded one level due to complex query requiring more context"
                    )

        if model:
            model_name = model.lower()
            if ("opus" in model_name or "gpt-5.4" in model_name) and recommended_level not in {
                FidelityLevel.ABSTRACT,
                FidelityLevel.RAW,
            }:
                downgraded = self._downgrade_level(recommended_level)
                if downgraded is not None:
                    recommended_level = downgraded
                    reasoning_parts.append(
                        f"Adjusted downward for higher-cost model '{model}' to preserve cost efficiency"
                    )
            elif "gemini" in model_name and query_complexity == "complex":
                upgraded = self._upgrade_level(recommended_level)
                if upgraded is not None:
                    recommended_level = upgraded
                    reasoning_parts.append(
                        f"Adjusted upward for large-context model '{model}' on a complex query"
                    )

        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(
            use_case, recommended_level, default_level, token_budget, num_nodes
        )

        # Step 6: Generate alternatives
        alternatives = self._generate_alternatives(
            recommended_level, num_nodes, token_budget, use_case
        )

        # Step 7: Build reasoning
        reasoning = ". ".join(reasoning_parts) + "."
        token_estimate = self.TOKENS_PER_NODE[recommended_level] * num_nodes

        return FidelityRecommendation(
            recommended_level=recommended_level,
            confidence=confidence,
            reasoning=reasoning,
            token_estimate=token_estimate,
            alternatives=alternatives,
        )

    def _upgrade_level(self, current: FidelityLevel) -> Optional[FidelityLevel]:
        """Upgrade to next higher fidelity level"""
        upgrades = {
            FidelityLevel.ABSTRACT: FidelityLevel.OUTLINE,
            FidelityLevel.OUTLINE: FidelityLevel.STRUCTURE,
            FidelityLevel.STRUCTURE: FidelityLevel.DETAILED,
            FidelityLevel.DETAILED: FidelityLevel.RAW,
        }
        return upgrades.get(current)

    def _downgrade_level(self, current: FidelityLevel) -> Optional[FidelityLevel]:
        """Downgrade to next lower fidelity level"""
        downgrades = {
            FidelityLevel.RAW: FidelityLevel.DETAILED,
            FidelityLevel.DETAILED: FidelityLevel.STRUCTURE,
            FidelityLevel.STRUCTURE: FidelityLevel.OUTLINE,
            FidelityLevel.OUTLINE: FidelityLevel.ABSTRACT,
        }
        return downgrades.get(current)

    def _calculate_confidence(
        self,
        use_case: UseCase,
        recommended: FidelityLevel,
        default: FidelityLevel,
        token_budget: Optional[int],
        num_nodes: int,
    ) -> float:
        """Calculate confidence score for recommendation"""
        confidence = 1.0

        # Reduce confidence if we had to deviate from default
        if recommended != default:
            confidence -= 0.2

        # Reduce confidence if budget is very tight
        if token_budget:
            recommended_tokens = self.TOKENS_PER_NODE[recommended] * num_nodes
            if recommended_tokens > token_budget * 0.9:
                confidence -= 0.1  # Using >90% of budget

        # Boost confidence for clear-cut use cases
        if use_case in [UseCase.QUICK_SUMMARY, UseCase.EXACT_QUOTES]:
            confidence += 0.1

        return max(0.5, min(1.0, confidence))

    def _generate_alternatives(
        self,
        recommended: FidelityLevel,
        num_nodes: int,
        token_budget: Optional[int],
        use_case: UseCase,
    ) -> List[Dict[str, Any]]:
        """Generate alternative fidelity options with trade-offs"""
        alternatives = []

        # Add lower fidelity option (save tokens)
        lower = self._downgrade_level(recommended)
        if lower:
            lower_tokens = self.TOKENS_PER_NODE[lower] * num_nodes
            savings = self.TOKENS_PER_NODE[recommended] * num_nodes - lower_tokens
            alternatives.append(
                {
                    # NAME, not enum int — modulate_region's fidelity_level
                    # takes the label (#92, 2026-06-12).
                    "level": lower.name,
                    "level_value": lower.value,
                    "tokens": lower_tokens,
                    "trade_off": f"Save {savings} tokens, but less detail",
                }
            )

        # Add higher fidelity option (more detail)
        higher = self._upgrade_level(recommended)
        if higher:
            higher_tokens = self.TOKENS_PER_NODE[higher] * num_nodes
            cost = higher_tokens - self.TOKENS_PER_NODE[recommended] * num_nodes

            # Only suggest if within budget (or no budget)
            if not token_budget or higher_tokens <= token_budget:
                alternatives.append(
                    {
                        "level": higher.name,
                        "level_value": higher.value,
                        "tokens": higher_tokens,
                        "trade_off": f"Use {cost} more tokens for richer context",
                    }
                )

        return alternatives

    def estimate_tokens(self, fidelity_level: FidelityLevel, num_nodes: int) -> int:
        """
        Estimate total tokens for given fidelity and node count.

        Args:
            fidelity_level: The fidelity level to use
            num_nodes: Number of nodes being retrieved

        Returns:
            Estimated total tokens

        Example:
            >>> advisor = FidelityAdvisor()
            >>> tokens = advisor.estimate_tokens(FidelityLevel.DETAILED, 5)
            >>> print(tokens)  # ~500
            500
        """
        return self.TOKENS_PER_NODE[fidelity_level] * num_nodes


# Convenience function for backward compatibility
def recommend_fidelity(
    use_case: str,
    num_nodes: int,
    token_budget: Optional[int] = None,
    query_complexity: str = "medium",
) -> Dict[str, Any]:
    """
    Functional interface to fidelity recommendation.

    Args:
        use_case: String name of use case (e.g., "question_answering")
        num_nodes: Number of nodes being retrieved
        token_budget: Optional token limit
        query_complexity: "simple", "medium", or "complex"

    Returns:
        Dictionary with recommendation details

    Example:
        >>> result = recommend_fidelity("question_answering", 3, 200)
        >>> print(result['recommended_level'])
        'DETAILED'
    """
    advisor = FidelityAdvisor()

    # Convert string to enum
    try:
        use_case_enum = UseCase(use_case)
    except ValueError:
        # Default to medium complexity use case
        use_case_enum = UseCase.QUESTION_ANSWERING

    rec = advisor.recommend(use_case_enum, num_nodes, token_budget, query_complexity)

    return {
        "recommended_level": rec.recommended_level.value,
        "confidence": rec.confidence,
        "reasoning": rec.reasoning,
        "token_estimate": rec.token_estimate,
        "alternatives": rec.alternatives,
    }
