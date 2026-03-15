"""
Compression replay log for tracking and optimizing compression strategies.

Based on ACON (2025) — tracks which compression settings produced the best
results per content type, enabling data-driven strategy optimization.
"""

import time
from typing import Dict, List, Optional


class CompressionReplayLog:
    """Records compression events and computes insights."""

    def __init__(self):
        self._log: List[dict] = []

    def record(
        self,
        doc_id: str,
        content_type: str,
        input_tokens: int,
        output_tokens: int,
        ratio: float,
        fidelity_score: float,
    ) -> None:
        """Record a compression event.

        Args:
            doc_id: Document identifier
            content_type: Type of content (code, prose, etc.)
            input_tokens: Input token count
            output_tokens: Output token count
            ratio: Compression ratio used
            fidelity_score: Measured fidelity score (0-1)
        """
        self._log.append({
            "doc_id": doc_id,
            "content_type": content_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "ratio": ratio,
            "fidelity_score": fidelity_score,
            "timestamp": time.time(),
        })

    def get_history(self, doc_id: str) -> List[dict]:
        """Get compression history for a specific document."""
        return [e for e in self._log if e["doc_id"] == doc_id]

    def get_insights(self) -> Dict[str, dict]:
        """Compute insights: best ratio per content type.

        Returns:
            Map of content_type -> {best_ratio, avg_fidelity, count}
        """
        if not self._log:
            return {}

        # Group by content type
        by_type: Dict[str, List[dict]] = {}
        for entry in self._log:
            ct = entry["content_type"]
            by_type.setdefault(ct, []).append(entry)

        insights = {}
        for content_type, entries in by_type.items():
            # Best ratio = the one with highest fidelity
            best_entry = max(entries, key=lambda e: e["fidelity_score"])
            avg_fidelity = sum(e["fidelity_score"] for e in entries) / len(entries)

            insights[content_type] = {
                "best_ratio": best_entry["ratio"],
                "best_fidelity": best_entry["fidelity_score"],
                "avg_fidelity": round(avg_fidelity, 4),
                "count": len(entries),
                "avg_input_tokens": int(sum(e["input_tokens"] for e in entries) / len(entries)),
            }

        return insights

    def recommend_ratio(
        self,
        content_type: str,
        min_fidelity: float = 0.8,
    ) -> Optional[float]:
        """Recommend compression ratio based on past performance.

        Args:
            content_type: Type of content to get recommendation for
            min_fidelity: Minimum acceptable fidelity score

        Returns:
            Recommended ratio, or None if no data available
        """
        entries = [e for e in self._log if e["content_type"] == content_type]
        if not entries:
            return None

        # Filter by min fidelity, then pick the most compressed (lowest ratio)
        qualifying = [e for e in entries if e["fidelity_score"] >= min_fidelity]
        if not qualifying:
            return None

        # Best = highest fidelity among qualifying
        best = max(qualifying, key=lambda e: e["fidelity_score"])
        return best["ratio"]
