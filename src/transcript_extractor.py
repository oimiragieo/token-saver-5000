"""
Transcript-to-memory extraction pipeline.

Accepts raw conversation transcripts (text or file), splits them into
candidate insights, classifies each via memory_classifier, and stores
results through MemoryAPI.  Reuses existing history_compaction for
optional pre-compression of long transcripts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .memory_api import MemoryAPI
from .memory_classifier import classify_insight, ClassificationResult

# ---------------------------------------------------------------------------
# Insight extraction heuristics
# ---------------------------------------------------------------------------

# Sentence-boundary splitter (handles '. ', '! ', '? ', newlines)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")

# Signal phrases that indicate an insight worth extracting
_SIGNAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "decision",
        re.compile(r"\b(?:decid(?:e|ed)|chose|choice|went\s+with|opt(?:ed)?\s+for)\b", re.I),
    ),
    (
        "lesson",
        re.compile(r"\b(?:lesson|learned|takeaway|insight|realize[ds]?|turns?\s+out)\b", re.I),
    ),
    (
        "pattern",
        re.compile(r"\b(?:always|never|convention|best\s+practice|pattern|recurring)\b", re.I),
    ),
    ("gotcha", re.compile(r"\b(?:gotcha|caveat|pitfall|watch\s+out|careful|beware|avoid)\b", re.I)),
    ("issue", re.compile(r"\b(?:bug|error|fail(?:ed|ure)?|broken|regression|crash)\b", re.I)),
    ("preference", re.compile(r"\b(?:prefer|like\s+to|rather|instead\s+of|don'?t\s+like)\b", re.I)),
    (
        "action",
        re.compile(
            r"\b(?:going\s+forward|from\s+now\s+on|next\s+time|we\s+should|must|need\s+to)\b",
            re.I,
        ),
    ),
]

_MIN_INSIGHT_LENGTH = 20  # Ignore very short fragments
_MAX_INSIGHT_LENGTH = 500  # Truncate overly long fragments


@dataclass
class ExtractionResult:
    """Result of extracting insights from a transcript."""

    insights: list[dict[str, Any]] = field(default_factory=list)
    total_sentences: int = 0
    extracted_count: int = 0
    stored_count: int = 0
    source: str = "transcript"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks."""
    raw = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in raw if s and s.strip()]


def _has_signal(sentence: str) -> bool:
    """Return True if the sentence contains at least one insight signal phrase."""
    return any(pat.search(sentence) for _, pat in _SIGNAL_PATTERNS)


def _strip_role_prefix(sentence: str) -> str:
    """Remove common role prefixes like '[user]', '[assistant]', 'Human:', etc."""
    return re.sub(
        r"^\s*\[?\s*(?:user|assistant|human|ai|system)\s*\]?\s*:?\s*", "", sentence, flags=re.I
    )


def extract_insights(
    text: str,
    *,
    mode: str = "all",
) -> list[tuple[str, ClassificationResult]]:
    """Extract candidate insights from a transcript.

    Args:
        text: Raw transcript text.
        mode: Extraction mode — ``"all"`` extracts every sentence that
              contains a signal phrase; ``"decisions"`` limits to decision
              signals; ``"patterns"`` limits to pattern/gotcha signals.

    Returns:
        List of ``(cleaned_text, ClassificationResult)`` tuples.
    """
    sentences = _split_sentences(text)
    allowed_signals: set[str] | None = None
    if mode == "decisions":
        allowed_signals = {"decision", "action"}
    elif mode == "patterns":
        allowed_signals = {"pattern", "gotcha", "preference"}

    results: list[tuple[str, ClassificationResult]] = []
    seen: set[str] = set()

    for sentence in sentences:
        cleaned = _strip_role_prefix(sentence)
        if len(cleaned) < _MIN_INSIGHT_LENGTH:
            continue
        if len(cleaned) > _MAX_INSIGHT_LENGTH:
            cleaned = cleaned[:_MAX_INSIGHT_LENGTH]

        # Check signal phrases
        if allowed_signals is not None:
            matched = any(
                pat.search(cleaned)
                for sig_name, pat in _SIGNAL_PATTERNS
                if sig_name in allowed_signals
            )
        else:
            matched = _has_signal(cleaned)

        if not matched:
            continue

        # Deduplicate by lowercased text
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)

        classification = classify_insight(cleaned)
        results.append((cleaned, classification))

    return results


def ingest_transcript(
    text: str,
    *,
    mode: str = "all",
    source: str = "transcript",
    workspace_id: str | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    memory_api: MemoryAPI | None = None,
) -> ExtractionResult:
    """End-to-end pipeline: extract insights from *text* and store them.

    Args:
        text: Raw conversation transcript.
        mode: ``"all"``, ``"decisions"``, or ``"patterns"``.
        source: Source tag for stored memories (default ``"transcript"``).
        workspace_id / user_id / agent_id / session_id: Scope fields.
        memory_api: Optional pre-existing MemoryAPI instance.

    Returns:
        ExtractionResult with stored insights and counts.
    """
    api = memory_api or MemoryAPI.get_api()
    sentences = _split_sentences(text)
    extracted = extract_insights(text, mode=mode)

    result = ExtractionResult(
        total_sentences=len(sentences),
        extracted_count=len(extracted),
        source=source,
    )

    for cleaned_text, classification in extracted:
        memory = api.add_memory(
            text=cleaned_text,
            category=classification.category,
            source=source,
            metadata={"confidence": classification.confidence, "extraction_mode": mode},
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        result.insights.append(memory)
        result.stored_count += 1

    return result
