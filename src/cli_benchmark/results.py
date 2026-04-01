"""Benchmark result dataclasses and output formatters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class CLIResult:
    """Unified result from a CLI invocation."""

    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_time_ms: float = 0.0
    tool_calls: int = 0
    num_turns: int = 0
    raw_response: str = ""
    raw_json: dict = field(default_factory=dict)
    is_dry_run: bool = False


@dataclass
class ComparisonResult:
    """A single baseline vs compressed comparison."""

    corpus_name: str
    provider: str
    mode: str  # "skill" or "mcp"
    baseline: CLIResult
    compressed: CLIResult
    compression_ratio: float = 0.0
    input_token_savings_pct: float = 0.0
    cost_savings_pct: float = 0.0

    def __post_init__(self):
        if self.baseline.input_tokens > 0:
            saved = self.baseline.input_tokens - self.compressed.input_tokens
            self.input_token_savings_pct = round(saved / self.baseline.input_tokens * 100, 1)
        if self.baseline.total_cost_usd > 0:
            saved = self.baseline.total_cost_usd - self.compressed.total_cost_usd
            self.cost_savings_pct = round(saved / self.baseline.total_cost_usd * 100, 1)


@dataclass
class BenchmarkReport:
    """Full benchmark run report."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    results: list[ComparisonResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, result: ComparisonResult) -> None:
        self.results.append(result)

    def to_json(self, path: Path) -> None:
        """Write report as JSON file."""
        data = asdict(self)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def to_table(self) -> str:
        """Render results as an ASCII table."""
        if not self.results:
            return "No results to display."
        header = (
            f"{'Corpus':<10} {'Provider':<10} {'Mode':<6} "
            f"{'Input(base)':>12} {'Input(comp)':>12} {'Savings':>8} "
            f"{'Cost(base)':>11} {'Cost(comp)':>11} {'Cost Svgs':>10} "
            f"{'Time(base)':>11} {'Time(comp)':>11}"
        )
        sep = "-" * len(header)
        rows = [sep, header, sep]
        for r in self.results:
            row = (
                f"{r.corpus_name:<10} {r.provider:<10} {r.mode:<6} "
                f"{r.baseline.input_tokens:>12,} {r.compressed.input_tokens:>12,} "
                f"{r.input_token_savings_pct:>7.1f}% "
                f"${r.baseline.total_cost_usd:>10.4f} ${r.compressed.total_cost_usd:>10.4f} "
                f"{r.cost_savings_pct:>9.1f}% "
                f"{r.baseline.wall_time_ms:>10.0f}ms {r.compressed.wall_time_ms:>10.0f}ms"
            )
            rows.append(row)
        rows.append(sep)

        # Cache breakdown (helps diagnose cost anomalies)
        has_cache = any(
            r.baseline.cache_creation_tokens > 0 or r.compressed.cache_creation_tokens > 0
            for r in self.results
        )
        if has_cache:
            rows.append("")
            rows.append("Cache breakdown (tokens):")
            for r in self.results:
                b = r.baseline
                c = r.compressed
                rows.append(
                    f"  {r.corpus_name:<10} baseline: "
                    f"create={b.cache_creation_tokens:,} read={b.cache_read_tokens:,} | "
                    f"compressed: create={c.cache_creation_tokens:,} read={c.cache_read_tokens:,}"
                )

        return "\n".join(rows)
