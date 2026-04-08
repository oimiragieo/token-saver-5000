"""
Knowledge lint — quality checks on stored memories and compiled articles.

Detects orphan memories, staleness, near-duplicates, contradictions, and
ACE bullet decay.  Surfaces results via MCP tool or CLI.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc

from .memory_api import MemoryAPI

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

# Contradiction signal pairs — lightweight heuristic
_NEGATION_PAIRS: list[tuple[re.Pattern[str], re.Pattern[str]]] = [
    (re.compile(r"\balways\b", re.I), re.compile(r"\bnever\b", re.I)),
    (re.compile(r"\bshould\b", re.I), re.compile(r"\bshould\s+not\b", re.I)),
    (re.compile(r"\bmust\b", re.I), re.compile(r"\bmust\s+not\b", re.I)),
    (re.compile(r"\buse\b", re.I), re.compile(r"\bavoid\b", re.I)),
    (re.compile(r"\benable\b", re.I), re.compile(r"\bdisable\b", re.I)),
]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _parse_iso(ts: str) -> datetime | None:
    """Best-effort ISO 8601 timestamp parse."""
    try:
        cleaned = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Finding types
# ---------------------------------------------------------------------------


@dataclass
class LintFinding:
    """A single lint finding."""

    severity: str  # "warning" | "error" | "info"
    check: str  # e.g. "stale", "duplicate", "contradiction"
    message: str
    memory_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "memory_ids": self.memory_ids,
            "details": self.details,
        }


@dataclass
class LintReport:
    """Aggregated lint report."""

    findings: list[LintFinding] = field(default_factory=list)
    total_memories: int = 0
    checks_run: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_memories": self.total_memories,
            "findings_count": len(self.findings),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "checks_run": self.checks_run,
            "findings": [f.to_dict() for f in self.findings],
        }


# ---------------------------------------------------------------------------
# Linter
# ---------------------------------------------------------------------------


class KnowledgeLinter:
    """Run quality checks on a collection of memory dicts."""

    def __init__(
        self,
        *,
        stale_days: int = 30,
        duplicate_threshold: float = 0.75,
    ):
        self._stale_days = stale_days
        self._dup_threshold = duplicate_threshold

    def lint(
        self,
        memories: list[dict[str, Any]],
        *,
        ace_bullets: list[dict[str, Any]] | None = None,
        compiled_article_titles: list[str] | None = None,
    ) -> LintReport:
        """Run all checks and return a LintReport.

        Args:
            memories: List of memory dicts from MemoryAPI.
            ace_bullets: Optional list of ACE bullet dicts for decay check.
            compiled_article_titles: Optional list of compiled article titles
                for orphan detection.
        """
        report = LintReport(total_memories=len(memories))

        report.findings.extend(self._check_stale(memories))
        report.checks_run.append("stale")

        report.findings.extend(self._check_duplicates(memories))
        report.checks_run.append("duplicates")

        report.findings.extend(self._check_contradictions(memories))
        report.checks_run.append("contradictions")

        if ace_bullets is not None:
            report.findings.extend(self._check_ace_decay(ace_bullets))
            report.checks_run.append("ace_decay")

        if compiled_article_titles is not None:
            report.findings.extend(self._check_orphans(memories, compiled_article_titles))
            report.checks_run.append("orphans")

        return report

    def lint_from_api(
        self,
        *,
        memory_api: MemoryAPI | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        ace_bullets: list[dict[str, Any]] | None = None,
        compiled_article_titles: list[str] | None = None,
    ) -> LintReport:
        """Convenience wrapper that pulls memories from MemoryAPI."""
        api = memory_api or MemoryAPI.get_api()
        memories = api.list_memories(
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return self.lint(
            memories,
            ace_bullets=ace_bullets,
            compiled_article_titles=compiled_article_titles,
        )

    # -- checks --------------------------------------------------------------

    def _check_stale(self, memories: list[dict[str, Any]]) -> list[LintFinding]:
        findings: list[LintFinding] = []
        now = datetime.now(UTC)
        for mem in memories:
            created = _parse_iso(str(mem.get("created_at", "")))
            if created is None:
                continue
            age_days = (now - created).days
            if age_days > self._stale_days:
                findings.append(
                    LintFinding(
                        severity="warning",
                        check="stale",
                        message=f"Memory '{mem.get('memory_id')}' is {age_days} days old",
                        memory_ids=[str(mem.get("memory_id", ""))],
                        details={"age_days": age_days},
                    )
                )
        return findings

    def _check_duplicates(self, memories: list[dict[str, Any]]) -> list[LintFinding]:
        findings: list[LintFinding] = []
        texts = [(str(m.get("memory_id", "")), str(m.get("text", ""))) for m in memories]
        seen_pairs: set[tuple[str, str]] = set()
        for i, (id_a, text_a) in enumerate(texts):
            for j in range(i + 1, len(texts)):
                id_b, text_b = texts[j]
                pair = (min(id_a, id_b), max(id_a, id_b))
                if pair in seen_pairs:
                    continue
                if _similarity(text_a, text_b) >= self._dup_threshold:
                    seen_pairs.add(pair)
                    findings.append(
                        LintFinding(
                            severity="warning",
                            check="duplicate",
                            message=f"Memories '{id_a}' and '{id_b}' are near-duplicates",
                            memory_ids=[id_a, id_b],
                            details={
                                "similarity": round(_similarity(text_a, text_b), 3),
                            },
                        )
                    )
        return findings

    def _check_contradictions(self, memories: list[dict[str, Any]]) -> list[LintFinding]:
        findings: list[LintFinding] = []
        for i, mem_a in enumerate(memories):
            text_a = str(mem_a.get("text", ""))
            cat_a = mem_a.get("category", "")
            for j in range(i + 1, len(memories)):
                mem_b = memories[j]
                text_b = str(mem_b.get("text", ""))
                cat_b = mem_b.get("category", "")
                # Only compare within same category
                if cat_a != cat_b:
                    continue
                for pos_re, neg_re in _NEGATION_PAIRS:
                    a_pos = bool(pos_re.search(text_a)) and bool(neg_re.search(text_b))
                    b_pos = bool(pos_re.search(text_b)) and bool(neg_re.search(text_a))
                    if a_pos or b_pos:
                        findings.append(
                            LintFinding(
                                severity="error",
                                check="contradiction",
                                message=(
                                    f"Possible contradiction between "
                                    f"'{mem_a.get('memory_id')}' and '{mem_b.get('memory_id')}'"
                                ),
                                memory_ids=[
                                    str(mem_a.get("memory_id", "")),
                                    str(mem_b.get("memory_id", "")),
                                ],
                                details={"text_a": text_a[:100], "text_b": text_b[:100]},
                            )
                        )
                        break  # One contradiction per pair
        return findings

    def _check_ace_decay(self, bullets: list[dict[str, Any]]) -> list[LintFinding]:
        findings: list[LintFinding] = []
        for bullet in bullets:
            success_rate = bullet.get("success_rate", 1.0)
            total_usage = bullet.get("total_usage", 0)
            if total_usage >= 5 and success_rate < 0.3:
                findings.append(
                    LintFinding(
                        severity="warning",
                        check="ace_decay",
                        message=(
                            f"ACE bullet '{bullet.get('bullet_id', '?')}' has decayed "
                            f"(success_rate={success_rate:.0%}, usage={total_usage})"
                        ),
                        details={
                            "bullet_id": bullet.get("bullet_id", ""),
                            "success_rate": success_rate,
                            "total_usage": total_usage,
                        },
                    )
                )
        return findings

    def _check_orphans(
        self,
        memories: list[dict[str, Any]],
        compiled_titles: list[str],
    ) -> list[LintFinding]:
        findings: list[LintFinding] = []
        compiled_cats = {t.lower().replace(" ", "_") for t in compiled_titles}
        for mem in memories:
            cat = str(mem.get("category", "general"))
            if cat not in compiled_cats:
                findings.append(
                    LintFinding(
                        severity="info",
                        check="orphan",
                        message=(
                            f"Memory '{mem.get('memory_id')}' (category '{cat}') "
                            f"has no compiled article"
                        ),
                        memory_ids=[str(mem.get("memory_id", ""))],
                    )
                )
        return findings
