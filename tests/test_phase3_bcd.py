"""Tests for ROI calculator, budget monitor, and team export (Phase 3B-3D)."""

from __future__ import annotations

import csv
import io
import json

import pytest

from src.budget_monitor import BudgetLimit, TokenBudgetMonitor, create_budget_monitor
from src.team_export import TeamExporter, TeamMemberStats

# ─── ROI Calculator handler tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_calculate_roi_defaults():
    """Default ROI calculation with claude-sonnet-4-6."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(await handle_calculate_roi({}, {}))
    assert result["status"] == "success"
    assert result["model"] == "claude-sonnet-4-6"
    assert result["tokens_per_day"] == 500_000
    assert result["team_size"] == 1
    assert result["compression_ratio"] == 0.85
    assert result["cost_without_monthly"] > 0
    assert result["cost_with_monthly"] < result["cost_without_monthly"]
    assert result["dollars_saved_monthly"] > 0
    assert result["roi_multiplier"] > 0
    assert "comparison" in result
    assert "available_models" in result


@pytest.mark.asyncio
async def test_calculate_roi_custom_params():
    """Custom model, tokens, team size."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(
        await handle_calculate_roi(
            {},
            {
                "model": "gpt-4o",
                "tokens_per_day": 1_000_000,
                "team_size": 10,
                "compression_ratio": 0.90,
            },
        )
    )
    assert result["model"] == "gpt-4o"
    assert result["team_size"] == 10
    assert result["compression_ratio"] == 0.90
    # 10 users * $49 = $490/mo plan cost (live Pro price)
    assert result["pro_plan_cost_monthly"] == 490.0
    assert result["monthly_tokens"] == 1_000_000 * 22 * 10


@pytest.mark.asyncio
async def test_calculate_roi_unknown_model_uses_default():
    """Unknown model falls back to default pricing."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(await handle_calculate_roi({}, {"model": "unknown-model-xyz"}))
    assert result["status"] == "success"
    assert result["cost_without_monthly"] > 0


@pytest.mark.asyncio
async def test_calculate_roi_zero_compression():
    """Zero compression ratio means no savings."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(await handle_calculate_roi({}, {"compression_ratio": 0.0}))
    assert result["dollars_saved_monthly"] == 0
    assert result["cost_with_monthly"] == result["cost_without_monthly"]


@pytest.mark.asyncio
async def test_calculate_roi_full_compression():
    """100% compression = all savings."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(await handle_calculate_roi({}, {"compression_ratio": 1.0}))
    assert result["cost_with_monthly"] == 0
    assert result["dollars_saved_monthly"] == result["cost_without_monthly"]


@pytest.mark.asyncio
async def test_calculate_roi_comparison_format():
    """Comparison text has expected structure."""
    from src.handlers.token_optimization_handlers import handle_calculate_roi

    result = json.loads(await handle_calculate_roi({}, {}))
    comp = result["comparison"]
    assert "Without gotcontext:" in comp
    assert "With gotcontext:" in comp
    assert "Pro plan cost:" in comp
    assert "Net savings:" in comp
    assert "ROI" in comp


# ─── Budget Monitor unit tests ───────────────────────────────────────────────


class TestBudgetLimit:
    def test_usage_pct(self):
        limit = BudgetLimit("test", 1000, 500)
        assert limit.usage_pct == 50.0

    def test_remaining_tokens(self):
        limit = BudgetLimit("test", 1000, 750)
        assert limit.remaining_tokens == 250

    def test_remaining_clamped_to_zero(self):
        limit = BudgetLimit("test", 1000, 1500)
        assert limit.remaining_tokens == 0

    def test_alert_level_ok(self):
        limit = BudgetLimit("test", 1000, 100)
        assert limit.alert_level == "ok"

    def test_alert_level_info(self):
        limit = BudgetLimit("test", 1000, 550)
        assert limit.alert_level == "info"

    def test_alert_level_warning(self):
        limit = BudgetLimit("test", 1000, 800)
        assert limit.alert_level == "warning"

    def test_alert_level_critical(self):
        limit = BudgetLimit("test", 1000, 950)
        assert limit.alert_level == "critical"

    def test_to_dict(self):
        limit = BudgetLimit("session", 1000, 500)
        d = limit.to_dict()
        assert d["name"] == "session"
        assert d["usage_pct"] == 50.0
        assert d["alert_level"] == "info"


class TestTokenBudgetMonitor:
    def test_no_limits_configured(self):
        monitor = TokenBudgetMonitor()
        assert monitor.active_limits == []
        result = monitor.check_budget()
        assert result.overall_status == "ok"

    def test_session_limit(self):
        monitor = TokenBudgetMonitor(session_limit=10000)
        assert "session" in monitor.active_limits
        monitor.record_usage(5000, "ingest_context")
        result = monitor.check_budget()
        assert result.limits[0].current_tokens == 5000

    def test_multiple_limits(self):
        monitor = TokenBudgetMonitor(session_limit=10000, daily_limit=50000)
        monitor.record_usage(8000)
        for limit in monitor.check_budget().limits:
            assert limit.current_tokens == 8000

    def test_critical_status_propagates(self):
        monitor = TokenBudgetMonitor(session_limit=1000)
        monitor.record_usage(950)
        result = monitor.check_budget()
        assert result.overall_status == "critical"

    def test_reset_specific_limit(self):
        monitor = TokenBudgetMonitor(session_limit=1000, daily_limit=5000)
        monitor.record_usage(500)
        monitor.reset("session")
        session = monitor.get_limit("session")
        daily = monitor.get_limit("daily")
        assert session.current_tokens == 0
        assert daily.current_tokens == 500

    def test_reset_all(self):
        monitor = TokenBudgetMonitor(session_limit=1000, daily_limit=5000)
        monitor.record_usage(500)
        monitor.reset()
        for limit in monitor.check_budget().limits:
            assert limit.current_tokens == 0

    def test_zero_max_tokens(self):
        limit = BudgetLimit("test", 0, 0)
        assert limit.usage_pct == 0.0
        assert limit.alert_level == "ok"


class TestCreateBudgetMonitor:
    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("TOKEN_BUDGET_SESSION", "50000")
        monkeypatch.setenv("TOKEN_BUDGET_DAILY", "200000")
        monitor = create_budget_monitor()
        assert "session" in monitor.active_limits
        assert "daily" in monitor.active_limits


# ─── Budget handler tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_check_budget_handler_no_limits():
    from src.handlers.token_optimization_handlers import handle_check_budget

    result = json.loads(await handle_check_budget({}, {"workspace_id": "test_no_limits_unique_1"}))
    assert result["status"] == "success"
    assert result["overall_status"] == "ok"


@pytest.mark.asyncio
async def test_check_budget_handler_with_recording():
    from src.handlers.token_optimization_handlers import handle_check_budget

    result = json.loads(
        await handle_check_budget(
            {},
            {
                "workspace_id": "test_budget_record_unique_1",
                "session_limit": 10000,
                "record_tokens": 5000,
                "tool_name": "ingest_context",
            },
        )
    )
    assert result["status"] == "success"
    assert any(lim["current_tokens"] == 5000 for lim in result["limits"])


# ─── Team Member Stats tests ────────────────────────────────────────────────


class TestTeamMemberStats:
    def test_tokens_saved(self):
        stats = TeamMemberStats("alice", 5, 100000, 15000, 50)
        assert stats.tokens_saved == 85000

    def test_savings_pct(self):
        stats = TeamMemberStats("alice", 5, 100000, 15000, 50)
        assert stats.savings_pct == 85.0

    def test_zero_original(self):
        stats = TeamMemberStats("bob", 1, 0, 0, 0)
        assert stats.savings_pct == 0.0

    def test_to_dict(self):
        stats = TeamMemberStats("alice", 5, 100000, 15000, 50)
        d = stats.to_dict()
        assert d["user_id"] == "alice"
        assert d["tokens_saved"] == 85000
        assert d["savings_pct"] == 85.0


# ─── Team Exporter tests ────────────────────────────────────────────────────


class TestTeamExporter:
    def _sample_exporter(self) -> TeamExporter:
        exporter = TeamExporter()
        exporter.add_member_stats("alice", 10, 500000, 75000, 100)
        exporter.add_member_stats("bob", 5, 200000, 40000, 40)
        return exporter

    def test_build_report(self):
        exporter = self._sample_exporter()
        report = exporter.build_report()
        assert report.total_original_tokens == 700000
        assert report.total_compressed_tokens == 115000
        assert report.total_tokens_saved == 585000
        assert report.total_sessions == 15
        assert len(report.members) == 2

    def test_accumulate_member_stats(self):
        exporter = TeamExporter()
        exporter.add_member_stats("alice", 5, 100000, 15000, 20)
        exporter.add_member_stats("alice", 5, 100000, 15000, 20)
        report = exporter.build_report()
        assert len(report.members) == 1
        assert report.members[0].sessions == 10
        assert report.members[0].total_original_tokens == 200000

    def test_export_json(self):
        exporter = self._sample_exporter()
        data = json.loads(exporter.export_json())
        assert data["total_tokens_saved"] == 585000
        assert len(data["members"]) == 2

    def test_export_csv(self):
        exporter = self._sample_exporter()
        csv_text = exporter.export_csv()
        reader = csv.reader(io.StringIO(csv_text))
        rows = list(reader)
        assert rows[0][0] == "user_id"  # header
        assert rows[-1][0] == "TOTAL"  # summary row
        assert len(rows) == 4  # header + 2 members + total

    def test_export_prometheus(self):
        exporter = self._sample_exporter()
        prom = exporter.export_prometheus()
        assert "gotcontext_tokens_saved_total 585000" in prom
        assert 'gotcontext_member_tokens_saved{user_id="alice"}' in prom
        assert "# TYPE gotcontext_tokens_saved_total counter" in prom

    def test_empty_report(self):
        exporter = TeamExporter()
        report = exporter.build_report()
        assert report.total_tokens_saved == 0
        assert report.overall_savings_pct == 0.0

    def test_members_sorted_by_savings(self):
        exporter = TeamExporter()
        exporter.add_member_stats("low_saver", 1, 1000, 800, 5)
        exporter.add_member_stats("high_saver", 1, 10000, 1000, 50)
        report = exporter.build_report()
        assert report.members[0].user_id == "high_saver"


# ─── Team Export handler tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_export_team_data_json():
    from src.handlers.token_optimization_handlers import handle_export_team_data

    result = json.loads(
        await handle_export_team_data(
            {},
            {
                "members": [
                    {
                        "user_id": "alice",
                        "sessions": 10,
                        "original_tokens": 100000,
                        "compressed_tokens": 15000,
                        "operations": 50,
                    },
                ],
                "format": "json",
            },
        )
    )
    assert result["status"] == "success"
    assert result["format"] == "json"
    assert result["summary"]["total_tokens_saved"] == 85000


@pytest.mark.asyncio
async def test_export_team_data_csv():
    from src.handlers.token_optimization_handlers import handle_export_team_data

    result = json.loads(
        await handle_export_team_data(
            {},
            {
                "members": [
                    {"user_id": "alice", "original_tokens": 50000, "compressed_tokens": 10000},
                ],
                "format": "csv",
            },
        )
    )
    assert "user_id" in result["data"]
    assert "TOTAL" in result["data"]


@pytest.mark.asyncio
async def test_export_team_data_prometheus():
    from src.handlers.token_optimization_handlers import handle_export_team_data

    result = json.loads(
        await handle_export_team_data(
            {},
            {
                "members": [
                    {
                        "user_id": "bob",
                        "original_tokens": 200000,
                        "compressed_tokens": 30000,
                        "operations": 20,
                    },
                ],
                "format": "prometheus",
            },
        )
    )
    assert "gotcontext_tokens_saved_total" in result["data"]
    assert "bob" in result["data"]


@pytest.mark.asyncio
async def test_export_team_data_empty():
    from src.handlers.token_optimization_handlers import handle_export_team_data

    result = json.loads(await handle_export_team_data({}, {"members": [], "format": "json"}))
    assert result["status"] == "success"
    assert result["summary"]["total_tokens_saved"] == 0
