"""
TOON production gate.

Auto-benchmarks TOON vs JSON serialization and enables TOON
only when benchmarks prove consistent token savings.
"""

from typing import Dict, List

MIN_BENCHMARKS = 3
MIN_SAVINGS_PERCENT = 10.0  # Require at least 10% savings to enable


class TOONGate:
    """Gate that controls whether TOON serialization is enabled."""

    def __init__(self):
        self._benchmarks: List[Dict] = []

    def record_benchmark(self, json_tokens: int, toon_tokens: int) -> None:
        """Record a TOON vs JSON benchmark result.

        Args:
            json_tokens: Token count for JSON serialization
            toon_tokens: Token count for TOON serialization
        """
        savings_pct = ((json_tokens - toon_tokens) / json_tokens * 100) if json_tokens > 0 else 0
        self._benchmarks.append(
            {
                "json_tokens": json_tokens,
                "toon_tokens": toon_tokens,
                "savings_percent": savings_pct,
            }
        )

    def is_enabled(self) -> bool:
        """Check if TOON should be enabled based on benchmarks."""
        if len(self._benchmarks) < MIN_BENCHMARKS:
            return False
        avg_savings = sum(b["savings_percent"] for b in self._benchmarks) / len(self._benchmarks)
        return avg_savings >= MIN_SAVINGS_PERCENT

    def recommended_format(self) -> str:
        """Get the recommended output format."""
        return "toon" if self.is_enabled() else "json"

    def get_stats(self) -> Dict:
        """Get benchmark statistics."""
        if not self._benchmarks:
            return {"benchmarks_run": 0, "avg_savings_percent": 0.0, "enabled": False}

        avg_savings = sum(b["savings_percent"] for b in self._benchmarks) / len(self._benchmarks)
        return {
            "benchmarks_run": len(self._benchmarks),
            "avg_savings_percent": round(avg_savings, 1),
            "enabled": self.is_enabled(),
        }
