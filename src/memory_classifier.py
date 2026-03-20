"""
Memory classification system for auto-categorizing insights.

Classifies compressed insights into categories (gotcha, issue,
decision, pattern, general) using keyword-based matching.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationResult:
    """Result of classifying an insight."""

    text: str
    category: str  # "gotcha", "issue", "decision", "pattern", "general"
    confidence: float  # 0.0 to 1.0


# Keyword patterns per category (case-insensitive)
_CATEGORY_PATTERNS = {
    "gotcha": [
        r"\bwatch\s*out\b",
        r"\bgotcha\b",
        r"\bcaveat\b",
        r"\bpitfall\b",
        r"\btrap\b",
        r"\bcareful\b",
        r"\bsilently\b",
        r"\bnever\b.*\buse\b",
        r"\bwarning\b",
        r"\bbeware\b",
        r"\bgotchas?\b",
        r"\bavoid\b",
        r"\bdon'?t\b.*\bforget\b",
        r"\bsubtl[ey]\b",
    ],
    "issue": [
        r"\bbug\b",
        r"\berror\b",
        r"\bfail(?:s|ed|ure|ing)?\b",
        r"\bcrash\b",
        r"\bbroken\b",
        r"\bregression\b",
        r"\btimeout\b",
        r"\bnull\s*pointer\b",
        r"\bexception\b",
        r"\bflaw\b",
        r"\bdefect\b",
    ],
    "decision": [
        r"\bdecid(?:e|ed)\b",
        r"\bchose\b",
        r"\bchoice\b",
        r"\binstead\s*of\b",
        r"\bopt(?:ed)?\s*for\b",
        r"\bwe\s*went\s*with\b",
        r"\breason(?:ing)?\s*(?:for|behind)\b",
        r"\btrade-?off\b",
        r"\bselect(?:ed|ion)\b",
    ],
    "pattern": [
        r"\bpattern\b",
        r"\bconsistently\b",
        r"\balways\b.*\buse\b",
        r"\bconvention\b",
        r"\bbest\s*practice\b",
        r"\bidiom\b",
        r"\brecurring\b",
        r"\bstandard\b.*\bapproach\b",
    ],
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in _CATEGORY_PATTERNS.items()
}


def classify_insight(text: str) -> ClassificationResult:
    """Classify a single insight text into a category.

    Args:
        text: The insight text to classify

    Returns:
        ClassificationResult with category and confidence
    """
    scores = {}
    for category, patterns in _COMPILED.items():
        matches = sum(1 for p in patterns if p.search(text))
        if matches > 0:
            scores[category] = matches / len(patterns)

    if not scores:
        return ClassificationResult(text=text, category="general", confidence=0.3)

    best_category = max(scores, key=scores.get)
    confidence = min(1.0, scores[best_category] * 3)  # Scale up, cap at 1.0

    return ClassificationResult(
        text=text,
        category=best_category,
        confidence=round(confidence, 2),
    )


def classify_insights(texts: List[str]) -> List[ClassificationResult]:
    """Classify multiple insights.

    Args:
        texts: List of insight texts

    Returns:
        List of ClassificationResult objects
    """
    return [classify_insight(t) for t in texts]
