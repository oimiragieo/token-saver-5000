"""
Token threshold monitoring for proactive compression suggestions.

Monitors context size and recommends compression actions
at configurable thresholds (40%, 60%, 75% of context limit).
"""

from dataclasses import dataclass, asdict


@dataclass
class ContextBudgetResult:
    """Result of a context budget check."""

    current_tokens: int
    context_limit: int
    usage_percent: float
    status: str  # "ok", "warning", "urgent", "critical"
    should_compress: bool
    recommendation: str

    def to_dict(self) -> dict:
        return asdict(self)


def check_context_budget(
    current_tokens: int,
    context_limit: int = 200_000,
) -> ContextBudgetResult:
    """Check context usage against thresholds and recommend action.

    Thresholds (percentage of context_limit):
    - < 40%: ok (no action needed)
    - 40-60%: warning (suggest compression)
    - 60-75%: urgent (strongly recommend compression)
    - >= 75%: critical (compress immediately)

    Args:
        current_tokens: Current token usage
        context_limit: Maximum context window size

    Returns:
        ContextBudgetResult with status and recommendation
    """
    usage_pct = (current_tokens / context_limit * 100) if context_limit > 0 else 100.0

    if usage_pct >= 75:
        status = "critical"
        should_compress = True
        recommendation = (
            f"Context is {usage_pct:.0f}% full ({current_tokens:,}/{context_limit:,} tokens). "
            "Compress immediately using aggressive preset or adapt_to_context_window."
        )
    elif usage_pct >= 60:
        status = "urgent"
        should_compress = True
        recommendation = (
            f"Context is {usage_pct:.0f}% full. "
            "Strongly recommend compressing large documents with ingest_context."
        )
    elif usage_pct >= 40:
        status = "warning"
        should_compress = True
        recommendation = (
            f"Context is {usage_pct:.0f}% full. "
            "Consider compressing documents to preserve headroom."
        )
    else:
        status = "ok"
        should_compress = False
        recommendation = f"Context usage is healthy at {usage_pct:.0f}%."

    return ContextBudgetResult(
        current_tokens=current_tokens,
        context_limit=context_limit,
        usage_percent=round(usage_pct, 1),
        status=status,
        should_compress=should_compress,
        recommendation=recommendation,
    )
