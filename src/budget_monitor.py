"""Token budget monitoring with configurable limits and alerts.

Extends the existing context budget checker with per-session,
per-day, and per-month budget limits. Tracks cumulative usage
and projects end-of-period usage.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BudgetLimit:
    """A single budget limit with threshold alerts."""

    name: str  # "session", "daily", "monthly"
    max_tokens: int
    current_tokens: int = 0
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])

    @property
    def usage_pct(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return min(self.current_tokens / self.max_tokens * 100, 100.0)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def alert_level(self) -> str:
        """Return alert level based on usage vs thresholds."""
        ratio = self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0
        if ratio >= self.alert_thresholds[-1]:
            return "critical"
        if len(self.alert_thresholds) >= 2 and ratio >= self.alert_thresholds[-2]:
            return "warning"
        if len(self.alert_thresholds) >= 3 and ratio >= self.alert_thresholds[-3]:
            return "info"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "max_tokens": self.max_tokens,
            "current_tokens": self.current_tokens,
            "remaining_tokens": self.remaining_tokens,
            "usage_pct": round(self.usage_pct, 1),
            "alert_level": self.alert_level,
        }


@dataclass
class BudgetCheckResult:
    """Result of a budget check across all limits."""

    limits: List[BudgetLimit]
    overall_status: str = "ok"
    projected_daily_tokens: int = 0
    projected_monthly_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "limits": [lim.to_dict() for lim in self.limits],
            "projected_daily_tokens": self.projected_daily_tokens,
            "projected_monthly_tokens": self.projected_monthly_tokens,
        }


class TokenBudgetMonitor:
    """Monitors token usage against configurable budget limits."""

    def __init__(
        self,
        session_limit: int = 0,
        daily_limit: int = 0,
        monthly_limit: int = 0,
        alert_thresholds: Optional[List[float]] = None,
    ):
        thresholds = alert_thresholds or [0.5, 0.75, 0.9]
        self._limits: Dict[str, BudgetLimit] = {}

        if session_limit > 0:
            self._limits["session"] = BudgetLimit("session", session_limit, 0, thresholds)
        if daily_limit > 0:
            self._limits["daily"] = BudgetLimit("daily", daily_limit, 0, thresholds)
        if monthly_limit > 0:
            self._limits["monthly"] = BudgetLimit("monthly", monthly_limit, 0, thresholds)

        self._start_time = time.time()
        self._usage_history: List[Dict[str, Any]] = []

    def record_usage(self, tokens: int, tool_name: str = "") -> None:
        """Record token usage across all active limits."""
        for limit in self._limits.values():
            limit.current_tokens += tokens
        self._usage_history.append({"tokens": tokens, "tool": tool_name, "timestamp": time.time()})

    def check_budget(self) -> BudgetCheckResult:
        """Check all budget limits and return status."""
        limits = list(self._limits.values())
        statuses = [lim.alert_level for lim in limits]

        if "critical" in statuses:
            overall = "critical"
        elif "warning" in statuses:
            overall = "warning"
        elif "info" in statuses:
            overall = "info"
        else:
            overall = "ok"

        # Project usage
        elapsed = time.time() - self._start_time
        total_tokens = sum(lim.current_tokens for lim in limits[:1])  # use first limit
        if elapsed > 60:
            rate = total_tokens / (elapsed / 3600)  # tokens per hour
            projected_daily = int(rate * 8)  # 8 working hours
            projected_monthly = projected_daily * 22
        else:
            projected_daily = 0
            projected_monthly = 0

        return BudgetCheckResult(
            limits=limits,
            overall_status=overall,
            projected_daily_tokens=projected_daily,
            projected_monthly_tokens=projected_monthly,
        )

    def get_limit(self, name: str) -> Optional[BudgetLimit]:
        """Get a specific budget limit by name."""
        return self._limits.get(name)

    @property
    def active_limits(self) -> List[str]:
        return list(self._limits.keys())

    def reset(self, name: Optional[str] = None) -> None:
        """Reset token counters. If name given, reset only that limit."""
        if name and name in self._limits:
            self._limits[name].current_tokens = 0
        elif name is None:
            for limit in self._limits.values():
                limit.current_tokens = 0
            self._usage_history.clear()


def create_budget_monitor() -> TokenBudgetMonitor:
    """Create a budget monitor from environment variables."""
    return TokenBudgetMonitor(
        session_limit=int(os.getenv("TOKEN_BUDGET_SESSION", "0")),
        daily_limit=int(os.getenv("TOKEN_BUDGET_DAILY", "0")),
        monthly_limit=int(os.getenv("TOKEN_BUDGET_MONTHLY", "0")),
    )
