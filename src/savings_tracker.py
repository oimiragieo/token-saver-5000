"""Real-time token savings tracker for gotcontext.ai.

Tracks every compression operation, computes cost savings based on the
configured model's pricing, and provides cumulative ROI reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .cli_benchmark.pricing import get_model_rates


logger = logging.getLogger("semantic-modulator")


@dataclass
class SavingsEvent:
    """A single compression savings event."""

    timestamp: float
    tool_name: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    model: str
    cost_without_compression: float  # what it would have cost raw
    cost_with_compression: float  # what it actually costs
    dollars_saved: float  # the difference
    compression_ratio: float


@dataclass
class SavingsReport:
    """Cumulative savings report for a session."""

    session_id: str
    model: str
    # Lifetime totals
    total_operations: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0
    total_cost_without: float = 0.0
    total_cost_with: float = 0.0
    total_dollars_saved: float = 0.0
    # Averages
    avg_compression_ratio: float = 0.0
    avg_savings_pct: float = 0.0
    # ROI
    monthly_projected_savings: float = 0.0  # extrapolated from usage rate
    roi_vs_pro_plan: float = 0.0  # dollars_saved / $29 Pro price
    breakeven_operations: int = 0  # how many ops to pay for Pro
    # Time range
    first_event_time: float | None = None
    last_event_time: float | None = None
    # Per-tool breakdown
    by_tool: dict[str, dict[str, Any]] = field(default_factory=dict)


class SavingsTracker:
    """Tracks token savings across compression operations.

    Persists events to SessionJournal and provides real-time reports.
    Degrades gracefully if the journal fails to initialize or write.
    """

    PRO_PLAN_PRICE = 29.0  # $/month

    def __init__(self, session_id: str = "default", model: str = "claude-sonnet-4-6"):
        self._session_id = session_id
        self._model = model
        self._events: list[SavingsEvent] = []
        self._journal: Any = None
        self._init_journal()
        self._load_from_journal()

    def _init_journal(self) -> None:
        """Initialize the SessionJournal, failing gracefully on error."""
        try:
            from .session_journal import SessionJournal

            self._journal = SessionJournal(self._session_id)
        except Exception as exc:
            logger.warning(f"SavingsTracker: could not initialize SessionJournal: {exc}")
            self._journal = None

    def _load_from_journal(self) -> None:
        """Load prior savings events from the session journal."""
        if self._journal is None:
            return
        try:
            summary = self._journal.recover()
            for f in summary.ingested_files:
                orig = f.get("original_tokens")
                comp = f.get("compressed_tokens")
                if orig is None or comp is None:
                    continue
                orig = int(orig)
                comp = int(comp)
                saved = orig - comp
                if saved > 0:
                    rates = get_model_rates(self._model)
                    cost_without = orig * rates["input"] / 1_000_000
                    cost_with = comp * rates["input"] / 1_000_000
                    self._events.append(
                        SavingsEvent(
                            timestamp=f.get("timestamp", 0),
                            tool_name=f.get("tool_name", "ingest_context"),
                            original_tokens=orig,
                            compressed_tokens=comp,
                            tokens_saved=saved,
                            model=self._model,
                            cost_without_compression=round(cost_without, 6),
                            cost_with_compression=round(cost_with, 6),
                            dollars_saved=round(max(0.0, cost_without - cost_with), 6),
                            compression_ratio=round(orig / max(1, comp), 1),
                        )
                    )
        except Exception as exc:
            logger.warning(f"SavingsTracker: could not load events from journal: {exc}")

    def record(
        self,
        tool_name: str,
        original_tokens: int,
        compressed_tokens: int,
        model: str | None = None,
    ) -> SavingsEvent:
        """Record a compression savings event.

        Call this after any compression operation to track savings.
        """
        model = model or self._model
        saved = max(0, original_tokens - compressed_tokens)
        rates = get_model_rates(model)
        cost_without = original_tokens * rates["input"] / 1_000_000
        cost_with = compressed_tokens * rates["input"] / 1_000_000
        dollars_saved = max(0.0, cost_without - cost_with)
        ratio = (
            round(original_tokens / max(1, compressed_tokens), 1) if compressed_tokens > 0 else 0.0
        )

        event = SavingsEvent(
            timestamp=time.time(),
            tool_name=tool_name,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=saved,
            model=model,
            cost_without_compression=round(cost_without, 6),
            cost_with_compression=round(cost_with, 6),
            dollars_saved=round(dollars_saved, 6),
            compression_ratio=ratio,
        )
        self._events.append(event)

        # Persist to journal, degrading gracefully on failure
        if self._journal is not None:
            try:
                self._journal.write_event(
                    "savings",
                    {
                        "tool_name": tool_name,
                        "original_tokens": original_tokens,
                        "compressed_tokens": compressed_tokens,
                        "tokens_saved": saved,
                        "dollars_saved": round(dollars_saved, 6),
                        "model": model,
                        "compression_ratio": ratio,
                    },
                )
            except Exception as exc:
                logger.warning(f"SavingsTracker: failed to persist savings event: {exc}")

        return event

    def get_report(self) -> SavingsReport:
        """Generate a cumulative savings report."""
        report = SavingsReport(
            session_id=self._session_id,
            model=self._model,
        )

        if not self._events:
            return report

        report.total_operations = len(self._events)
        report.total_original_tokens = sum(e.original_tokens for e in self._events)
        report.total_compressed_tokens = sum(e.compressed_tokens for e in self._events)
        report.total_tokens_saved = sum(e.tokens_saved for e in self._events)
        report.total_cost_without = round(sum(e.cost_without_compression for e in self._events), 4)
        report.total_cost_with = round(sum(e.cost_with_compression for e in self._events), 4)
        report.total_dollars_saved = round(sum(e.dollars_saved for e in self._events), 4)

        if report.total_original_tokens > 0:
            report.avg_savings_pct = round(
                report.total_tokens_saved / report.total_original_tokens * 100, 1
            )
        if report.total_compressed_tokens > 0:
            report.avg_compression_ratio = round(
                report.total_original_tokens / report.total_compressed_tokens, 1
            )

        report.first_event_time = self._events[0].timestamp
        report.last_event_time = self._events[-1].timestamp

        # Monthly projection based on usage rate
        time_span = (report.last_event_time or 0) - (report.first_event_time or 0)
        if time_span > 60:  # at least 1 minute of data
            ops_per_second = report.total_operations / time_span
            seconds_per_month = 30 * 24 * 3600
            monthly_ops = ops_per_second * seconds_per_month
            savings_per_op = report.total_dollars_saved / report.total_operations
            report.monthly_projected_savings = round(monthly_ops * savings_per_op, 2)
        else:
            # Not enough data for projection; use simple scaling
            report.monthly_projected_savings = round(report.total_dollars_saved * 100, 2)

        # ROI calculation
        if report.monthly_projected_savings > 0:
            report.roi_vs_pro_plan = round(
                report.monthly_projected_savings / self.PRO_PLAN_PRICE, 1
            )

        # Breakeven: how many operations to save $29
        if report.total_operations > 0:
            savings_per_op = report.total_dollars_saved / report.total_operations
            if savings_per_op > 0:
                report.breakeven_operations = int(self.PRO_PLAN_PRICE / savings_per_op)

        # Per-tool breakdown
        by_tool: dict[str, dict[str, Any]] = {}
        for e in self._events:
            if e.tool_name not in by_tool:
                by_tool[e.tool_name] = {
                    "operations": 0,
                    "tokens_saved": 0,
                    "dollars_saved": 0.0,
                }
            by_tool[e.tool_name]["operations"] += 1
            by_tool[e.tool_name]["tokens_saved"] += e.tokens_saved
            by_tool[e.tool_name]["dollars_saved"] = round(
                by_tool[e.tool_name]["dollars_saved"] + e.dollars_saved, 6
            )
        report.by_tool = by_tool

        return report

    def get_inline_summary(self, event: SavingsEvent) -> str:
        """Generate a one-line savings summary for embedding in tool responses.

        Example: "Saved 3,400 tokens ($0.051) | Session total: $2.34 saved (8.1x ROI)"
        """
        report = self.get_report()
        parts = [
            f"Saved {event.tokens_saved:,} tokens (${event.dollars_saved:.3f})",
        ]
        if report.total_dollars_saved > 0:
            parts.append(f"Session total: ${report.total_dollars_saved:.2f} saved")
        if report.roi_vs_pro_plan > 0:
            parts.append(f"{report.roi_vs_pro_plan}x ROI vs Pro plan")
        return " | ".join(parts)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value
