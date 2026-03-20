"""
Context window advisor.

Based on MCP Best Practices (2025) — analyzes all ingested documents and
recommends optimal model selection, pruning priorities, and compression
strategy based on total token budget.
"""

from typing import List

# Model context window sizes (tokens)
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-3.5-turbo": 16_385,
    "claude-3.5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "llama-3.1-405b": 128_000,
    "llama-3.1-70b": 128_000,
    "mistral-large": 128_000,
    "deepseek-v3": 128_000,
}


def advise_context(
    doc_stats: List[dict],
    safety_margin: float = 0.75,
) -> dict:
    """Analyze documents and recommend context strategy.

    Args:
        doc_stats: List of {"doc_id": str, "tokens": int, "importance": float}
        safety_margin: Use only this fraction of model window (default: 75%)

    Returns:
        Advisory dict with total_tokens, recommended_models, prune_first, strategy
    """
    if not doc_stats:
        return {
            "total_tokens": 0,
            "recommended_models": list(MODEL_CONTEXT_WINDOWS.keys()),
            "prune_first": [],
            "strategy": "No documents ingested. Ingest context to get recommendations.",
            "token_breakdown": [],
        }

    total_tokens = sum(d["tokens"] for d in doc_stats)

    # Find models that fit
    recommended = []
    for model, window in sorted(MODEL_CONTEXT_WINDOWS.items(), key=lambda x: x[1]):
        usable = int(window * safety_margin)
        if total_tokens <= usable:
            recommended.append(
                {
                    "model": model,
                    "context_window": window,
                    "usable_tokens": usable,
                    "utilization": round(total_tokens / usable * 100, 1),
                }
            )

    # Prune suggestions: lowest importance first
    sorted_docs = sorted(doc_stats, key=lambda d: d.get("importance", 0))
    prune_first = [
        {
            "doc_id": d["doc_id"],
            "tokens": d["tokens"],
            "importance": d.get("importance", 0),
            "savings_percent": (
                round(d["tokens"] / total_tokens * 100, 1) if total_tokens > 0 else 0
            ),
        }
        for d in sorted_docs
    ]

    # Strategy recommendation
    if total_tokens < 8_000:
        strategy = "Context fits easily. No compression needed for most models."
    elif total_tokens < 32_000:
        strategy = "Moderate context. Use balanced compression (0.5 ratio) for smaller models, or use as-is with 128K+ models."
    elif total_tokens < 128_000:
        strategy = "Large context. Recommend aggressive compression (0.2-0.3 ratio) or use Gemini/Claude with 200K+ windows."
    elif total_tokens < 500_000:
        strategy = "Very large context. Use multi-level skeleton with headline tier. Only Gemini 1.5 Pro can handle uncompressed."
    else:
        strategy = "Massive context. Mandatory compression required. Use headline skeleton + query-adaptive pruning."

    return {
        "total_tokens": total_tokens,
        "recommended_models": recommended,
        "prune_first": prune_first,
        "strategy": strategy,
        "token_breakdown": [
            {
                "doc_id": d["doc_id"],
                "tokens": d["tokens"],
                "percent": round(d["tokens"] / total_tokens * 100, 1),
            }
            for d in sorted(doc_stats, key=lambda d: d["tokens"], reverse=True)
        ],
    }
