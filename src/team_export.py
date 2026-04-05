"""Team dashboard data export.

Aggregates savings data across sessions and exports in
JSON, CSV, and Prometheus formats for team dashboards.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TeamMemberStats:
    """Aggregated stats for one team member or session."""

    user_id: str
    sessions: int = 0
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_operations: int = 0

    @property
    def tokens_saved(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens

    @property
    def savings_pct(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        return self.tokens_saved / self.total_original_tokens * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "sessions": self.sessions,
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "tokens_saved": self.tokens_saved,
            "savings_pct": round(self.savings_pct, 1),
            "total_operations": self.total_operations,
        }


@dataclass
class TeamReport:
    """Aggregated team savings report."""

    members: List[TeamMemberStats] = field(default_factory=list)
    total_original_tokens: int = 0
    total_compressed_tokens: int = 0
    total_sessions: int = 0
    total_operations: int = 0

    @property
    def total_tokens_saved(self) -> int:
        return self.total_original_tokens - self.total_compressed_tokens

    @property
    def overall_savings_pct(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        return self.total_tokens_saved / self.total_original_tokens * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_original_tokens": self.total_original_tokens,
            "total_compressed_tokens": self.total_compressed_tokens,
            "total_tokens_saved": self.total_tokens_saved,
            "overall_savings_pct": round(self.overall_savings_pct, 1),
            "total_sessions": self.total_sessions,
            "total_operations": self.total_operations,
            "members": [m.to_dict() for m in self.members],
        }


class TeamExporter:
    """Aggregates and exports team savings data."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = Path(data_dir) if data_dir else Path(".semantic_modulator_data")
        self._members: Dict[str, TeamMemberStats] = {}

    def add_member_stats(
        self,
        user_id: str,
        sessions: int = 0,
        original_tokens: int = 0,
        compressed_tokens: int = 0,
        operations: int = 0,
    ) -> None:
        """Add or accumulate stats for a team member."""
        if user_id not in self._members:
            self._members[user_id] = TeamMemberStats(user_id=user_id)
        member = self._members[user_id]
        member.sessions += sessions
        member.total_original_tokens += original_tokens
        member.total_compressed_tokens += compressed_tokens
        member.total_operations += operations

    def build_report(self) -> TeamReport:
        """Build a team report from accumulated stats."""
        members = sorted(self._members.values(), key=lambda m: m.tokens_saved, reverse=True)
        report = TeamReport(
            members=members,
            total_original_tokens=sum(m.total_original_tokens for m in members),
            total_compressed_tokens=sum(m.total_compressed_tokens for m in members),
            total_sessions=sum(m.sessions for m in members),
            total_operations=sum(m.total_operations for m in members),
        )
        return report

    def export_json(self, report: Optional[TeamReport] = None) -> str:
        """Export report as JSON."""
        report = report or self.build_report()
        return json.dumps(report.to_dict(), indent=2)

    def export_csv(self, report: Optional[TeamReport] = None) -> str:
        """Export report as CSV."""
        report = report or self.build_report()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "user_id",
                "sessions",
                "original_tokens",
                "compressed_tokens",
                "tokens_saved",
                "savings_pct",
                "operations",
            ]
        )
        for member in report.members:
            writer.writerow(
                [
                    member.user_id,
                    member.sessions,
                    member.total_original_tokens,
                    member.total_compressed_tokens,
                    member.tokens_saved,
                    round(member.savings_pct, 1),
                    member.total_operations,
                ]
            )
        # Summary row
        writer.writerow(
            [
                "TOTAL",
                report.total_sessions,
                report.total_original_tokens,
                report.total_compressed_tokens,
                report.total_tokens_saved,
                round(report.overall_savings_pct, 1),
                report.total_operations,
            ]
        )
        return output.getvalue()

    def export_prometheus(self, report: Optional[TeamReport] = None) -> str:
        """Export report in Prometheus text exposition format."""
        report = report or self.build_report()
        lines = [
            "# HELP gotcontext_tokens_original_total Total original tokens before compression",
            "# TYPE gotcontext_tokens_original_total counter",
            f"gotcontext_tokens_original_total {report.total_original_tokens}",
            "",
            "# HELP gotcontext_tokens_compressed_total Total tokens after compression",
            "# TYPE gotcontext_tokens_compressed_total counter",
            f"gotcontext_tokens_compressed_total {report.total_compressed_tokens}",
            "",
            "# HELP gotcontext_tokens_saved_total Total tokens saved",
            "# TYPE gotcontext_tokens_saved_total counter",
            f"gotcontext_tokens_saved_total {report.total_tokens_saved}",
            "",
            "# HELP gotcontext_savings_ratio Overall savings ratio",
            "# TYPE gotcontext_savings_ratio gauge",
            f"gotcontext_savings_ratio {report.overall_savings_pct / 100:.4f}",
            "",
            "# HELP gotcontext_sessions_total Total sessions",
            "# TYPE gotcontext_sessions_total counter",
            f"gotcontext_sessions_total {report.total_sessions}",
            "",
            "# HELP gotcontext_operations_total Total compression operations",
            "# TYPE gotcontext_operations_total counter",
            f"gotcontext_operations_total {report.total_operations}",
            "",
            "# HELP gotcontext_member_tokens_saved Tokens saved per team member",
            "# TYPE gotcontext_member_tokens_saved counter",
        ]
        for member in report.members:
            lines.append(
                f'gotcontext_member_tokens_saved{{user_id="{member.user_id}"}} '
                f"{member.tokens_saved}"
            )
        lines.append("")
        return "\n".join(lines)
