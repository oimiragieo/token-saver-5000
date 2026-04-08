"""Tests for knowledge lint (src/knowledge_lint.py)."""

from datetime import datetime, timezone, timedelta

import pytest

from src.memory_api import MemoryAPI
from src.knowledge_lint import KnowledgeLinter, LintFinding, LintReport


@pytest.fixture(autouse=True)
def reset_memory():
    MemoryAPI.reset_singleton()
    yield
    MemoryAPI.reset_singleton()


def _utc_iso(days_ago: int = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat().replace("+00:00", "Z")


def _make_memory(
    memory_id: str,
    text: str,
    category: str = "general",
    created_at: str | None = None,
) -> dict:
    return {
        "memory_id": memory_id,
        "text": text,
        "category": category,
        "created_at": created_at or _utc_iso(0),
    }


# ---------------------------------------------------------------------------
# LintFinding / LintReport
# ---------------------------------------------------------------------------


class TestLintFinding:
    def test_to_dict(self):
        f = LintFinding(
            severity="warning",
            check="stale",
            message="Memory is old",
            memory_ids=["m1"],
        )
        d = f.to_dict()
        assert d["severity"] == "warning"
        assert d["check"] == "stale"
        assert "m1" in d["memory_ids"]


class TestLintReport:
    def test_empty_report(self):
        r = LintReport(total_memories=0)
        assert r.error_count == 0
        assert r.warning_count == 0
        d = r.to_dict()
        assert d["findings_count"] == 0

    def test_report_counts(self):
        r = LintReport(
            total_memories=3,
            findings=[
                LintFinding(severity="error", check="x", message="x"),
                LintFinding(severity="warning", check="y", message="y"),
                LintFinding(severity="warning", check="z", message="z"),
            ],
        )
        assert r.error_count == 1
        assert r.warning_count == 2


# ---------------------------------------------------------------------------
# Staleness checks
# ---------------------------------------------------------------------------


class TestStaleCheck:
    def test_fresh_memory_not_flagged(self):
        linter = KnowledgeLinter(stale_days=30)
        memories = [_make_memory("m1", "Fresh memory", created_at=_utc_iso(0))]
        report = linter.lint(memories)
        stale_findings = [f for f in report.findings if f.check == "stale"]
        assert len(stale_findings) == 0

    def test_old_memory_flagged(self):
        linter = KnowledgeLinter(stale_days=30)
        memories = [_make_memory("m1", "Old memory", created_at=_utc_iso(60))]
        report = linter.lint(memories)
        stale_findings = [f for f in report.findings if f.check == "stale"]
        assert len(stale_findings) == 1
        assert stale_findings[0].severity == "warning"

    def test_custom_stale_threshold(self):
        linter = KnowledgeLinter(stale_days=7)
        memories = [_make_memory("m1", "Week old", created_at=_utc_iso(10))]
        report = linter.lint(memories)
        stale_findings = [f for f in report.findings if f.check == "stale"]
        assert len(stale_findings) == 1

    def test_invalid_timestamp_skipped(self):
        linter = KnowledgeLinter()
        memories = [_make_memory("m1", "Bad date", created_at="not-a-date")]
        report = linter.lint(memories)
        stale_findings = [f for f in report.findings if f.check == "stale"]
        assert len(stale_findings) == 0


# ---------------------------------------------------------------------------
# Duplicate checks
# ---------------------------------------------------------------------------


class TestDuplicateCheck:
    def test_near_duplicates_detected(self):
        linter = KnowledgeLinter(duplicate_threshold=0.75)
        memories = [
            _make_memory("m1", "Always use black before committing code"),
            _make_memory("m2", "Always use black before committing your code"),
        ]
        report = linter.lint(memories)
        dup_findings = [f for f in report.findings if f.check == "duplicate"]
        assert len(dup_findings) == 1
        assert set(dup_findings[0].memory_ids) == {"m1", "m2"}

    def test_different_memories_not_flagged(self):
        linter = KnowledgeLinter(duplicate_threshold=0.75)
        memories = [
            _make_memory("m1", "Use PostgreSQL for the database"),
            _make_memory("m2", "Watch out for race conditions in async code"),
        ]
        report = linter.lint(memories)
        dup_findings = [f for f in report.findings if f.check == "duplicate"]
        assert len(dup_findings) == 0

    def test_empty_list(self):
        linter = KnowledgeLinter()
        report = linter.lint([])
        assert len(report.findings) == 0


# ---------------------------------------------------------------------------
# Contradiction checks
# ---------------------------------------------------------------------------


class TestContradictionCheck:
    def test_always_vs_never_detected(self):
        linter = KnowledgeLinter()
        memories = [
            _make_memory("m1", "Always use mocks in tests", category="pattern"),
            _make_memory("m2", "Never use mocks in integration tests", category="pattern"),
        ]
        report = linter.lint(memories)
        contradiction_findings = [f for f in report.findings if f.check == "contradiction"]
        assert len(contradiction_findings) >= 1
        assert contradiction_findings[0].severity == "error"

    def test_use_vs_avoid_detected(self):
        linter = KnowledgeLinter()
        memories = [
            _make_memory("m1", "Use global state for configuration", category="pattern"),
            _make_memory("m2", "Avoid global state in all modules", category="pattern"),
        ]
        report = linter.lint(memories)
        contradiction_findings = [f for f in report.findings if f.check == "contradiction"]
        assert len(contradiction_findings) >= 1

    def test_different_categories_not_compared(self):
        linter = KnowledgeLinter()
        memories = [
            _make_memory("m1", "Always use mocks in tests", category="pattern"),
            _make_memory("m2", "Never use mocks in production", category="gotcha"),
        ]
        report = linter.lint(memories)
        contradiction_findings = [f for f in report.findings if f.check == "contradiction"]
        assert len(contradiction_findings) == 0


# ---------------------------------------------------------------------------
# ACE bullet decay checks
# ---------------------------------------------------------------------------


class TestACEDecayCheck:
    def test_decayed_bullet_flagged(self):
        linter = KnowledgeLinter()
        bullets = [
            {"bullet_id": "b1", "success_rate": 0.1, "total_usage": 10},
        ]
        report = linter.lint([], ace_bullets=bullets)
        decay_findings = [f for f in report.findings if f.check == "ace_decay"]
        assert len(decay_findings) == 1

    def test_healthy_bullet_not_flagged(self):
        linter = KnowledgeLinter()
        bullets = [
            {"bullet_id": "b1", "success_rate": 0.8, "total_usage": 20},
        ]
        report = linter.lint([], ace_bullets=bullets)
        decay_findings = [f for f in report.findings if f.check == "ace_decay"]
        assert len(decay_findings) == 0

    def test_low_usage_not_flagged(self):
        linter = KnowledgeLinter()
        bullets = [
            {"bullet_id": "b1", "success_rate": 0.1, "total_usage": 2},
        ]
        report = linter.lint([], ace_bullets=bullets)
        decay_findings = [f for f in report.findings if f.check == "ace_decay"]
        assert len(decay_findings) == 0

    def test_no_ace_bullets_skips_check(self):
        linter = KnowledgeLinter()
        report = linter.lint([])
        assert "ace_decay" not in report.checks_run


# ---------------------------------------------------------------------------
# Orphan checks
# ---------------------------------------------------------------------------


class TestOrphanCheck:
    def test_orphan_detected(self):
        linter = KnowledgeLinter()
        memories = [_make_memory("m1", "Some insight", category="exotic_category")]
        report = linter.lint(memories, compiled_article_titles=["Pattern", "Decision"])
        orphan_findings = [f for f in report.findings if f.check == "orphan"]
        assert len(orphan_findings) == 1
        assert orphan_findings[0].severity == "info"

    def test_no_orphan_when_category_matches(self):
        linter = KnowledgeLinter()
        memories = [_make_memory("m1", "Some pattern", category="pattern")]
        report = linter.lint(memories, compiled_article_titles=["Pattern"])
        orphan_findings = [f for f in report.findings if f.check == "orphan"]
        assert len(orphan_findings) == 0

    def test_no_compiled_titles_skips_check(self):
        linter = KnowledgeLinter()
        report = linter.lint([_make_memory("m1", "test")])
        assert "orphans" not in report.checks_run


# ---------------------------------------------------------------------------
# Full lint from API
# ---------------------------------------------------------------------------


class TestLintFromAPI:
    def test_lint_from_api(self):
        api = MemoryAPI()
        api.add_memory(text="Fresh insight about testing", category="pattern")
        linter = KnowledgeLinter()
        report = linter.lint_from_api(memory_api=api)
        assert report.total_memories == 1
        assert "stale" in report.checks_run
        assert "duplicates" in report.checks_run
        assert "contradictions" in report.checks_run

    def test_lint_report_serializable(self):
        linter = KnowledgeLinter()
        memories = [
            _make_memory("m1", "Always use mocks", category="pattern"),
            _make_memory("m2", "Never use mocks in tests", category="pattern"),
        ]
        report = linter.lint(memories)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "findings" in d
        assert isinstance(d["findings"], list)
