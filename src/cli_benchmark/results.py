"""Benchmark result dataclasses and output formatters."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
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
    # Document-level compression stats (from compressor output, not CLI token counts)
    document_original_tokens: int = 0
    document_compressed_tokens: int = 0
    document_savings_pct: float = 0.0

    def __post_init__(self):
        if self.baseline.input_tokens > 0:
            saved = self.baseline.input_tokens - self.compressed.input_tokens
            self.input_token_savings_pct = round(saved / self.baseline.input_tokens * 100, 1)
        if self.baseline.total_cost_usd > 0:
            saved = self.baseline.total_cost_usd - self.compressed.total_cost_usd
            self.cost_savings_pct = round(saved / self.baseline.total_cost_usd * 100, 1)
        if self.document_original_tokens > 0:
            doc_saved = self.document_original_tokens - self.document_compressed_tokens
            self.document_savings_pct = round(doc_saved / self.document_original_tokens * 100, 1)


def _aggregate_repeats(results: list[ComparisonResult]) -> dict:
    """Group results by (corpus, provider, mode) and compute stats.

    Returns a dict keyed by (corpus_name, provider, mode) where each value
    contains median, min, max, n, and median token counts for display.
    """
    groups: dict = defaultdict(list)
    for r in results:
        key = (r.corpus_name, r.provider, r.mode)
        groups[key].append(r)

    aggregated = {}
    for key, runs in groups.items():
        savings = [r.input_token_savings_pct for r in runs if r.baseline.input_tokens > 0]
        doc_savings = [r.document_savings_pct for r in runs if r.document_original_tokens > 0]
        if not savings:
            aggregated[key] = {
                "median": 0,
                "min": 0,
                "max": 0,
                "n": len(runs),
                "baseline_median": 0,
                "compressed_median": 0,
                "doc_savings_median": 0,
                "doc_savings_min": 0,
                "doc_savings_max": 0,
                "cost_base_median": 0.0,
                "cost_comp_median": 0.0,
                "time_base_median": 0.0,
                "time_comp_median": 0.0,
            }
            continue
        aggregated[key] = {
            "median": round(statistics.median(savings), 1),
            "min": round(min(savings), 1),
            "max": round(max(savings), 1),
            "n": len(savings),
            "baseline_median": round(statistics.median([r.baseline.input_tokens for r in runs]), 0),
            "compressed_median": round(
                statistics.median([r.compressed.input_tokens for r in runs]), 0
            ),
            "doc_savings_median": round(statistics.median(doc_savings), 1) if doc_savings else 0,
            "doc_savings_min": round(min(doc_savings), 1) if doc_savings else 0,
            "doc_savings_max": round(max(doc_savings), 1) if doc_savings else 0,
            "cost_base_median": statistics.median([r.baseline.total_cost_usd for r in runs]),
            "cost_comp_median": statistics.median([r.compressed.total_cost_usd for r in runs]),
            "time_base_median": statistics.median([r.baseline.wall_time_ms for r in runs]),
            "time_comp_median": statistics.median([r.compressed.wall_time_ms for r in runs]),
        }
    return aggregated


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
        """Render results as an ASCII table.

        When multiple runs exist for the same (corpus, provider, mode) key,
        displays median savings with min-max range instead of raw single values.
        Single-run entries are marked as "(1 run)" for clarity.
        """
        if not self.results:
            return "No results to display."

        aggregated = _aggregate_repeats(self.results)
        # Determine if any key has more than one run
        multi_run = any(v["n"] > 1 for v in aggregated.values())

        # Deduplicate to one row per (corpus, provider, mode) key
        seen: set = set()
        display_rows = []
        for r in self.results:
            key = (r.corpus_name, r.provider, r.mode)
            if key in seen:
                continue
            seen.add(key)
            display_rows.append((key, r, aggregated[key]))

        header = (
            f"{'Corpus':<10} {'Provider':<10} {'Mode':<6} "
            f"{'Input(base)':>12} {'Input(comp)':>12} {'Total API Svgs':>14} "
            f"{'Doc Compression':>16} "
            f"{'Cost(base)':>11} {'Cost(comp)':>11} {'Cost Svgs':>10} "
            f"{'Time(base)':>11} {'Time(comp)':>11}"
        )
        sep = "-" * len(header)
        rows = [sep, header, sep]

        for key, r, agg in display_rows:
            n = agg["n"]
            run_label = f"({n} runs)" if n > 1 else "(1 run)"

            if n > 1:
                # Median with range for multi-run keys
                savings_str = (
                    f"{agg['median']:>5.1f}% {run_label}" f" [{agg['min']:.1f}-{agg['max']:.1f}]"
                )
                doc_str = (
                    f"{agg['doc_savings_median']:>5.1f}%"
                    f" [{agg['doc_savings_min']:.1f}-{agg['doc_savings_max']:.1f}]"
                    if agg["doc_savings_median"] != 0
                    else "        n/a"
                )
                base_tok = int(agg["baseline_median"])
                comp_tok = int(agg["compressed_median"])
                cost_base = agg["cost_base_median"]
                cost_comp = agg["cost_comp_median"]
                time_base = agg["time_base_median"]
                time_comp = agg["time_comp_median"]
                cost_svgs = (
                    round((cost_base - cost_comp) / cost_base * 100, 1) if cost_base > 0 else 0.0
                )
            else:
                savings_str = f"{r.input_token_savings_pct:>7.1f}% {run_label}"
                doc_str = (
                    f"{r.document_savings_pct:>7.1f}%          "
                    if r.document_original_tokens > 0
                    else "        n/a"
                )
                base_tok = r.baseline.input_tokens
                comp_tok = r.compressed.input_tokens
                cost_base = r.baseline.total_cost_usd
                cost_comp = r.compressed.total_cost_usd
                time_base = r.baseline.wall_time_ms
                time_comp = r.compressed.wall_time_ms
                cost_svgs = r.cost_savings_pct

            row = (
                f"{r.corpus_name:<10} {r.provider:<10} {r.mode:<6} "
                f"{base_tok:>12,} {comp_tok:>12,} {savings_str:<14} "
                f"{doc_str:<16} "
                f"${cost_base:>10.4f} ${cost_comp:>10.4f} "
                f"{cost_svgs:>9.1f}% "
                f"{time_base:>10.0f}ms {time_comp:>10.0f}ms"
            )
            rows.append(row)

        rows.append(sep)

        if multi_run:
            rows.append(
                "* Multi-run results show median savings with [min-max] range across all repeats."
            )

        rows.append("")
        rows.append(
            "Methodology: 'Total API Savings' = reduction in total input tokens sent to the provider"
            " (system prompt + document)."
        )
        rows.append(
            "             'Doc Compression'   = reduction of the document itself"
            " (original vs compressed tokens, from semantic compressor)."
        )

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

    def to_summary_table(self) -> str:
        """Render results with median + range when repeats > 1.

        This is an alias for to_table() — the main table already uses median
        and range for multi-run entries. Provided as a named entry point for
        explicit summary requests.
        """
        return self.to_table()
