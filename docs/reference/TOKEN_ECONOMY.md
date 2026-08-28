# Token Economy

How Token Saver 5000 tracks compression savings, enforces budgets, and exports team metrics.

| Field | Value |
|-------|-------|
| **Package** | `semantic-modulator` |
| **Version** | 0.11.0 |

## Components

| Module | Role |
|--------|------|
| `src/savings_tracker.py` | Per-session event log and cumulative `SavingsReport` |
| `src/savings_dashboard.py` | Cross-session CLI (`token-saver-stats`) |
| `src/budget_monitor.py` | Per-session / daily / monthly budget caps |
| `src/team_export.py` | JSON / CSV / Prometheus export with escaped `user_id` labels |
| `src/metrics.py` | `compute_cost_savings()` used by compression handlers |

## SavingsTracker

Each compression operation can emit a `SavingsEvent`:

- `original_tokens`, `compressed_tokens`, `tokens_saved`
- `model` — pricing key (default `claude-sonnet-4-6`)
- `cost_without_compression`, `cost_with_compression`, `dollars_saved`
- `compression_ratio`

Events persist via `SessionJournal` when available; the tracker degrades gracefully if journal I/O fails.

`SavingsReport` aggregates lifetime totals, averages, per-tool breakdown (`by_tool`), and ROI helpers:

- `monthly_projected_savings` — extrapolated from session usage rate
- `roi_vs_pro_plan` — `dollars_saved / $29`
- `breakeven_operations` — ops needed to offset Pro plan price

### MCP tools

| Tool | Handler |
|------|---------|
| `get_savings_report` | `token_optimization_handlers` |
| `get_savings_dashboard` | `token_optimization_handlers` |
| `export_team_savings` | `token_optimization_handlers` |

## TokenBudgetMonitor

Configurable limits (env-driven via `src/constants.py`):

- Per-session token cap
- Daily and monthly aggregates

When a limit is exceeded, handlers return a structured refusal before expensive work runs.

| Tool | Purpose |
|------|---------|
| `check_budget` | Query current budget headroom |
| `set_budget_limit` | Set session/daily/monthly caps |

## Pricing

Rates come from `src/cli_benchmark/pricing.py` (`get_model_rates`). Unknown models fall back to conservative defaults and log a warning.

## CLI

```bash
token-saver-stats          # aggregate dashboard across sessions
token-saver-stats --json   # machine-readable output
```

## Related

- [MCP Tool Counts](./MCP_TOOL_COUNTS.md)
- [MCP Tools Guide](../guides/MCP_TOOLS_GUIDE.md) — token-optimization tool category
