"""Cross-session savings dashboard for gotcontext.ai.

Aggregates savings data from all SessionJournal SQLite files to produce
cumulative reports: daily/weekly trends, per-tool breakdowns, cost estimates,
and export in JSON/CSV formats.

Entry point: ``token-saver-stats`` CLI command.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cli_benchmark.pricing import get_model_rates

_DEFAULT_STORAGE_DIR = Path(".semantic_modulator_data") / "sessions"
_SECONDS_PER_DAY = 86_400


@dataclass
class AggregatedReport:
    """Cross-session aggregated savings report."""

    total_sessions: int = 0
    total_events: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_tokens_saved: int = 0
    total_dollars_saved: float = 0.0
    avg_compression_ratio: float = 0.0
    avg_savings_pct: float = 0.0
    first_event_time: float | None = None
    last_event_time: float | None = None
    by_tool: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_day: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str = "claude-sonnet-4-6"


class SavingsDashboard:
    """Reads all session journals and aggregates savings metrics."""

    def __init__(
        self,
        storage_dir: Path | str | None = None,
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self._storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self._model = model

    def _iter_session_dbs(self) -> list[Path]:
        """Find all session .db files."""
        if not self._storage_dir.exists():
            return []
        return sorted(self._storage_dir.glob("*.db"))

    def _read_savings_events(self, db_path: Path) -> list[dict[str, Any]]:
        """Read savings events from a single session journal."""
        events: list[dict[str, Any]] = []
        try:
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            rows = conn.execute(
                "SELECT timestamp, data_json FROM events WHERE event_type = 'savings' "
                "ORDER BY id ASC"
            ).fetchall()
            for ts, data_json in rows:
                try:
                    data = json.loads(data_json)
                    data["timestamp"] = ts
                    events.append(data)
                except json.JSONDecodeError:
                    pass
            conn.close()
        except (sqlite3.Error, OSError):
            pass
        return events

    def aggregate(self, days: int | None = None) -> AggregatedReport:
        """Aggregate savings across all sessions.

        Args:
            days: If set, only include events from the last N days.
        """
        report = AggregatedReport(model=self._model)
        cutoff = (time.time() - days * _SECONDS_PER_DAY) if days else 0.0
        rates = get_model_rates(self._model)

        db_files = self._iter_session_dbs()
        report.total_sessions = len(db_files)

        all_events: list[dict[str, Any]] = []
        for db_path in db_files:
            all_events.extend(self._read_savings_events(db_path))

        if not all_events:
            return report

        for ev in all_events:
            ts = ev.get("timestamp", 0)
            if cutoff and ts < cutoff:
                continue

            orig = int(ev.get("original_tokens", 0))
            comp = int(ev.get("compressed_tokens", 0))
            saved = int(ev.get("tokens_saved", max(0, orig - comp)))
            tool = ev.get("tool_name", "unknown")
            dollars = float(ev.get("dollars_saved", 0.0))

            if dollars == 0.0 and saved > 0:
                dollars = saved * rates["input"] / 1_000_000

            report.total_events += 1
            report.total_original_tokens += orig
            report.total_compressed_tokens += comp
            report.total_tokens_saved += saved
            report.total_dollars_saved += dollars

            if report.first_event_time is None or ts < report.first_event_time:
                report.first_event_time = ts
            if report.last_event_time is None or ts > report.last_event_time:
                report.last_event_time = ts

            # By tool
            if tool not in report.by_tool:
                report.by_tool[tool] = {"operations": 0, "tokens_saved": 0, "dollars_saved": 0.0}
            report.by_tool[tool]["operations"] += 1
            report.by_tool[tool]["tokens_saved"] += saved
            report.by_tool[tool]["dollars_saved"] = round(
                report.by_tool[tool]["dollars_saved"] + dollars, 6
            )

            # By day
            day_key = time.strftime("%Y-%m-%d", time.localtime(ts))
            if day_key not in report.by_day:
                report.by_day[day_key] = {
                    "operations": 0,
                    "tokens_saved": 0,
                    "dollars_saved": 0.0,
                }
            report.by_day[day_key]["operations"] += 1
            report.by_day[day_key]["tokens_saved"] += saved
            report.by_day[day_key]["dollars_saved"] = round(
                report.by_day[day_key]["dollars_saved"] + dollars, 6
            )

        report.total_dollars_saved = round(report.total_dollars_saved, 4)

        if report.total_original_tokens > 0:
            report.avg_savings_pct = round(
                report.total_tokens_saved / report.total_original_tokens * 100, 1
            )
        if report.total_compressed_tokens > 0:
            report.avg_compression_ratio = round(
                report.total_original_tokens / report.total_compressed_tokens, 1
            )

        return report


def format_summary(report: AggregatedReport) -> str:
    """Human-readable summary."""
    lines = [
        "Token Saver — Savings Summary",
        "=" * 40,
        f"Sessions:      {report.total_sessions}",
        f"Operations:    {report.total_events:,}",
        f"Tokens saved:  {report.total_tokens_saved:,}",
        f"Dollars saved: ${report.total_dollars_saved:.4f}",
        f"Avg savings:   {report.avg_savings_pct}%",
        f"Avg ratio:     {report.avg_compression_ratio}x",
        f"Model:         {report.model}",
    ]
    if report.first_event_time:
        first = time.strftime("%Y-%m-%d %H:%M", time.localtime(report.first_event_time))
        last = time.strftime("%Y-%m-%d %H:%M", time.localtime(report.last_event_time or 0))
        lines.append(f"Period:        {first} → {last}")

    # ROI vs Pro plan
    if report.total_events > 0 and report.first_event_time and report.last_event_time:
        span = max(1.0, (report.last_event_time - report.first_event_time))
        monthly_proj = report.total_dollars_saved * (30 * _SECONDS_PER_DAY / span)
        if monthly_proj > 0:
            roi = monthly_proj / 29.0
            lines.extend(
                [
                    "",
                    f"Monthly proj:  ${monthly_proj:.2f}",
                    f"Pro plan ROI:  {roi:.1f}x (vs $29/mo)",
                ]
            )

    return "\n".join(lines)


def format_daily(report: AggregatedReport) -> str:
    """Day-by-day breakdown."""
    if not report.by_day:
        return "No daily data available."
    lines = ["Date         | Ops   | Tokens Saved | $ Saved", "-" * 50]
    for day in sorted(report.by_day.keys()):
        d = report.by_day[day]
        lines.append(
            f"{day}  | {d['operations']:>5} | {d['tokens_saved']:>12,} | ${d['dollars_saved']:.4f}"
        )
    return "\n".join(lines)


def format_by_tool(report: AggregatedReport) -> str:
    """Per-tool breakdown."""
    if not report.by_tool:
        return "No per-tool data available."
    lines = ["Tool                      | Ops   | Tokens Saved | $ Saved", "-" * 60]
    sorted_tools = sorted(report.by_tool.items(), key=lambda x: x[1]["tokens_saved"], reverse=True)
    for tool, d in sorted_tools:
        lines.append(
            f"{tool:<26}| {d['operations']:>5} | {d['tokens_saved']:>12,} | ${d['dollars_saved']:.4f}"
        )
    return "\n".join(lines)


def format_cost(report: AggregatedReport) -> str:
    """Cost savings with model pricing."""
    rates = get_model_rates(report.model)
    lines = [
        f"Model: {report.model}",
        f"Input rate: ${rates['input']}/MTok",
        "",
        f"Total input tokens (raw):        {report.total_original_tokens:>12,}",
        f"Total input tokens (compressed):  {report.total_compressed_tokens:>12,}",
        f"Tokens saved:                    {report.total_tokens_saved:>12,}",
        "",
        f"Cost without compression: ${report.total_original_tokens * rates['input'] / 1e6:.4f}",
        f"Cost with compression:    ${report.total_compressed_tokens * rates['input'] / 1e6:.4f}",
        f"Savings:                  ${report.total_dollars_saved:.4f}",
    ]
    return "\n".join(lines)


def report_to_dict(report: AggregatedReport) -> dict[str, Any]:
    """Convert report to a JSON-serializable dict."""
    return {
        "total_sessions": report.total_sessions,
        "total_events": report.total_events,
        "total_original_tokens": report.total_original_tokens,
        "total_compressed_tokens": report.total_compressed_tokens,
        "total_tokens_saved": report.total_tokens_saved,
        "total_dollars_saved": report.total_dollars_saved,
        "avg_compression_ratio": report.avg_compression_ratio,
        "avg_savings_pct": report.avg_savings_pct,
        "first_event_time": report.first_event_time,
        "last_event_time": report.last_event_time,
        "by_tool": report.by_tool,
        "by_day": report.by_day,
        "model": report.model,
    }


def report_to_csv(report: AggregatedReport) -> str:
    """Export daily data as CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["date", "operations", "tokens_saved", "dollars_saved"])
    for day in sorted(report.by_day.keys()):
        d = report.by_day[day]
        writer.writerow([day, d["operations"], d["tokens_saved"], d["dollars_saved"]])
    return buf.getvalue()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token Saver savings dashboard — view cross-session metrics."
    )
    parser.add_argument("--daily", action="store_true", help="Day-by-day breakdown")
    parser.add_argument("--weekly", action="store_true", help="Last 7 days only")
    parser.add_argument("--by-tool", action="store_true", help="Per-tool breakdown")
    parser.add_argument("--cost", action="store_true", help="Cost savings with model pricing")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--csv", action="store_true", help="CSV export (daily data)")
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model for cost calculations (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Only include events from the last N days",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=None,
        help="Override session journal storage directory",
    )
    return parser


def main() -> None:
    """Entry point for token-saver-stats CLI."""
    args = build_arg_parser().parse_args()
    dashboard = SavingsDashboard(storage_dir=args.storage_dir, model=args.model)
    days = args.days
    if args.weekly:
        days = 7

    report = dashboard.aggregate(days=days)

    if getattr(args, "json"):
        print(json.dumps(report_to_dict(report), indent=2))
        return

    if args.csv:
        print(report_to_csv(report), end="")
        return

    print(format_summary(report))

    if args.daily:
        print()
        print(format_daily(report))

    if args.by_tool:
        print()
        print(format_by_tool(report))

    if args.cost:
        print()
        print(format_cost(report))
