"""
Token Optimization Handlers (v0.11.0)

MCP tool handlers for token estimation, client configuration,
and compression profile management. These tools help MCP clients
optimize token usage by configuring compression parameters for
their specific model and context window.

All handlers are model-agnostic -- they benefit any MCP client,
not just Claude Code.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ..token_estimation import TokenEstimator
from ..client_config import ClientConfig, SessionConfigStore
from ..compression_profiles import (
    ProfileManager,
    auto_select_profile,
    get_profile,
    list_profiles,
)
from ..meta_tokens import MetaTokenCompressor
from ..quality_predictor import QualityPredictor
from ..savings_tracker import SavingsTracker

# Module-level singletons for session state
_session_config_store = SessionConfigStore()
_profile_manager = ProfileManager()
_token_estimator = TokenEstimator()

# Module-level journal cache (keyed by session_id)
_session_journals: dict = {}

# Module-level savings tracker cache (keyed by session_id)
_savings_trackers: dict[str, SavingsTracker] = {}


def _get_journal(session_id: str) -> Any:
    """Return the SessionJournal for *session_id*, creating it on first access."""
    from ..session_journal import SessionJournal

    if session_id not in _session_journals:
        _session_journals[session_id] = SessionJournal(session_id)
    return _session_journals[session_id]


def get_session_config_store() -> SessionConfigStore:
    """Get the module-level session config store."""
    return _session_config_store


def get_profile_manager() -> ProfileManager:
    """Get the module-level profile manager."""
    return _profile_manager


async def handle_estimate_tokens(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle estimate_tokens tool call.

    Returns multi-method token estimates for the given text.
    """
    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    estimates = _token_estimator.estimate_all(text)
    return json.dumps(
        {
            "status": "success",
            "text_length": len(text),
            "estimates": estimates,
            "methods": {
                "accurate": "tiktoken cl100k_base (or fallback to bytes/4)",
                "estimated": "len(text) // 4 (matches Claude Code fast estimation)",
                "json_estimated": "len(text) // 2 (for JSON-heavy content)",
                "bytes": "exact UTF-8 byte count",
            },
        }
    )


async def handle_configure_for_client(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle configure_for_client tool call.

    Configures compression parameters for a specific LLM model.
    """
    model_id = args.get("model_id", "default")
    context_window_tokens = args.get("context_window_tokens")
    session_id = args.get("session_id", "default")

    try:
        config = ClientConfig.from_model(
            model_id=model_id,
            context_window_tokens=context_window_tokens,
        )
    except ValueError as e:
        return json.dumps({"error": str(e)})

    _session_config_store.set_config(session_id, config)

    # Journal the configuration for session recovery
    _get_journal(session_id).write_event(
        "configure",
        {
            "model_id": config.model_id,
            "provider": config.provider,
            "context_window_tokens": config.context_window_tokens,
        },
    )

    return json.dumps(
        {
            "status": "success",
            "configuration": {
                "model_id": config.model_id,
                "context_window_tokens": config.context_window_tokens,
                "recommended_skeleton_ratio": round(config.recommended_skeleton_ratio, 4),
                "provider": config.provider,
                "preferred_output_format": config.preferred_output_format,
                "truncation_strategy": config.truncation_strategy,
                "cache_stable_ordering": config.cache_stable_ordering,
            },
            "message": (
                f"Configured for {config.model_id} "
                f"({config.context_window_tokens:,} token context window). "
                f"Recommended skeleton ratio: {config.recommended_skeleton_ratio:.2%}."
            ),
        }
    )


async def handle_set_compression_profile(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle set_compression_profile tool call.

    Sets a named compression profile for the session.
    """
    profile_name = args.get("profile_name")
    session_id = args.get("session_id", "default")

    if not profile_name:
        return json.dumps(
            {
                "error": "profile_name is required",
                "available_profiles": list_profiles(),
            }
        )

    try:
        _profile_manager.set_profile(session_id, profile_name)
        profile = get_profile(profile_name)
    except ValueError as e:
        return json.dumps({"error": str(e), "available_profiles": list_profiles()})

    # Journal the profile change for session recovery
    _get_journal(session_id).write_event("profile", {"profile_name": profile_name})

    return json.dumps(
        {
            "status": "success",
            "active_profile": {
                "name": profile.name,
                "skeleton_ratio": profile.skeleton_ratio,
                "fidelity": profile.fidelity,
                "chunk_size": profile.chunk_size,
                "description": profile.description,
            },
            "message": (
                f"Compression profile set to '{profile.name}': {profile.description}. "
                f"Explicit parameters in subsequent tool calls will override these defaults."
            ),
        }
    )


async def handle_get_compression_profile(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle get_compression_profile tool call.

    Returns the active compression profile for the session.
    """
    session_id = args.get("session_id", "default")
    profile = _profile_manager.get_profile(session_id)

    return json.dumps(
        {
            "status": "success",
            "active_profile": {
                "name": profile.name,
                "skeleton_ratio": profile.skeleton_ratio,
                "fidelity": profile.fidelity,
                "chunk_size": profile.chunk_size,
                "description": profile.description,
            },
            "available_profiles": list_profiles(),
        }
    )


async def handle_compress_meta_tokens(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle compress_meta_tokens tool call.

    Applies lossless meta-token compression to the given text, replacing
    repeated token subsequences with compact dictionary symbols (§1, §2, …).
    """
    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    min_length = int(args.get("min_length", 2))
    min_frequency = int(args.get("min_frequency", 2))
    max_entries = int(args.get("max_entries", 50))

    compressor = MetaTokenCompressor(
        min_length=min_length,
        min_frequency=min_frequency,
        max_entries=max_entries,
    )
    result = compressor.compress(text)

    return json.dumps(
        {
            "status": "success",
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "savings_tokens": result.savings_tokens,
            "compression_ratio": result.compression_ratio,
            "dictionary_entries": len(result.dictionary),
            "dictionary": [
                {
                    "symbol": e.symbol,
                    "expansion": " ".join(e.expansion),
                    "frequency": e.frequency,
                    "savings": e.savings,
                }
                for e in result.dictionary
            ],
            "compressed_text": result.compressed_text,
        }
    )


async def handle_recommend_compression(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle recommend_compression tool call.

    Predicts the quality of each compression profile for the given text and
    recommends the most compressed profile that meets the quality floor.
    """
    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    quality_floor = float(args.get("quality_floor", 0.7))
    query = args.get("query") or None

    predictor = QualityPredictor()
    profile_scores = predictor.predict_all_profiles(text, query=query)
    recommended = auto_select_profile(text, quality_floor=quality_floor, query=query)

    return json.dumps(
        {
            "status": "success",
            "recommended_profile": recommended,
            "quality_floor": quality_floor,
            "profile_scores": {
                name: round(score, 4) for name, score in sorted(profile_scores.items())
            },
            "message": (
                f"Profile '{recommended}' is the most compressed option "
                f"meeting quality floor {quality_floor:.0%}."
            ),
        }
    )


async def handle_recover_session(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle recover_session tool call.

    Returns a compact summary of all prior ingestions, configurations, and tool
    calls for the given session, enabling a new conversation to resume where a
    compacted one left off.
    """
    session_id = args.get("session_id", "default")
    journal = _get_journal(session_id)
    summary = journal.recover()
    return json.dumps(
        {
            "status": "success",
            "session_id": summary.session_id,
            "event_count": summary.event_count,
            "ingested_files": summary.ingested_files,
            "client_config": summary.client_config,
            "active_profile": summary.active_profile,
            "tool_call_stats": summary.tool_call_stats,
            "total_tokens_saved": summary.total_tokens_saved,
        }
    )


async def handle_filter_cli_output(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle filter_cli_output tool call.

    Auto-detects CLI command type and applies an appropriate filtering strategy
    to reduce token usage. Strips ANSI codes, summarizes git diffs, focuses on
    test failures, groups lint errors, removes progress bars, etc.
    """
    from ..cli_output_optimizer import CLIOutputOptimizer

    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    hint = args.get("command_hint") or None
    optimizer = CLIOutputOptimizer()
    result = optimizer.filter(text, command_hint=hint)
    return json.dumps(
        {
            "status": "success",
            "filtered_text": result.filtered_text,
            "command_detected": result.command_detected,
            "strategy_applied": result.strategy_applied,
            "original_lines": result.original_lines,
            "filtered_lines": result.filtered_lines,
            "compression_pct": round(result.compression_pct, 1),
        }
    )


# ---------------------------------------------------------------------------
# Savings Tracker helpers and handlers (v0.14.0)
# ---------------------------------------------------------------------------


def _get_tracker(session_id: str, model: str = "claude-sonnet-4-6") -> SavingsTracker:
    """Return (or create) the SavingsTracker for *session_id*."""
    if session_id not in _savings_trackers:
        _savings_trackers[session_id] = SavingsTracker(session_id, model)
    tracker = _savings_trackers[session_id]
    if model and model != tracker.model:
        tracker.model = model
    return tracker


async def handle_get_savings_report(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle get_savings_report tool call.

    Returns a detailed report of token savings and ROI for the session.
    """
    session_id = args.get("session_id", "default")
    model = args.get("model")
    config = _session_config_store.get_config(session_id)
    tracker = _get_tracker(session_id, model or config.model_id)
    report = tracker.get_report()

    return json.dumps(
        {
            "status": "success",
            "session_id": report.session_id,
            "model": report.model,
            "total_operations": report.total_operations,
            "total_tokens_saved": report.total_tokens_saved,
            "total_dollars_saved": report.total_dollars_saved,
            "avg_compression_ratio": report.avg_compression_ratio,
            "avg_savings_pct": report.avg_savings_pct,
            "monthly_projected_savings": report.monthly_projected_savings,
            "roi_vs_pro_plan": report.roi_vs_pro_plan,
            "breakeven_operations": report.breakeven_operations,
            "by_tool": report.by_tool,
            "message": (
                f"Session savings: {report.total_tokens_saved:,} tokens "
                f"(${report.total_dollars_saved:.2f}). "
                f"Projected monthly: ${report.monthly_projected_savings:.2f} "
                f"({report.roi_vs_pro_plan}x ROI vs $29/mo Pro plan)."
            ),
        }
    )


async def handle_advise_cache_strategy(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle advise_cache_strategy tool call.

    Returns provider-specific caching recommendations for the given model.
    """
    from ..cache_strategy_advisor import advise_cache_strategy

    model_id = args.get("model_id", "default")
    strategy = advise_cache_strategy(model_id)
    return json.dumps(
        {
            "status": "success",
            "provider": strategy.provider,
            "model": strategy.model,
            "cache_type": strategy.cache_type,
            "min_prefix_tokens": strategy.min_prefix_tokens,
            "cache_discount_pct": strategy.cache_discount_pct,
            "ttl": strategy.ttl_description,
            "client_action": strategy.client_action,
            "tips": strategy.tips,
            "supports_cache": strategy.supports_cache,
        }
    )


async def handle_generate_structural_summary(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle generate_structural_summary tool call.

    Returns a compact structural outline of the provided source code, preserving
    imports, class definitions, and function signatures while omitting bodies.
    """
    from ..structural_summary import generate_structural_summary

    text = args.get("text", "")
    if not isinstance(text, str):
        return json.dumps({"error": "text must be a string"})

    file_path = args.get("file_path", "")
    result = generate_structural_summary(text, file_path)
    return json.dumps(
        {
            "status": "success",
            "summary": result.summary_text,
            "original_tokens": result.original_tokens,
            "summary_tokens": result.summary_tokens,
            "savings_pct": result.savings_pct,
            "symbol_count": result.symbol_count,
            "language": result.language,
            "line_count": result.line_count,
        }
    )


async def handle_detect_dead_code(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle detect_dead_code tool call.

    Scans a directory for Python files that are never imported by other files,
    returning a dead file list with estimated token savings.
    """
    from ..dead_code_detector import detect_dead_files

    directory = args.get("directory", ".")
    if not isinstance(directory, str):
        return json.dumps({"error": "directory must be a string"})

    report = detect_dead_files(directory)
    return json.dumps(
        {
            "status": "success",
            "total_files": report.total_files,
            "dead_files": report.dead_files,
            "dead_file_count": report.dead_file_count,
            "live_file_count": report.live_file_count,
            "tokens_saved": report.tokens_saved,
            "entry_points": report.entry_points,
        }
    )


async def handle_get_savings_inline(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle get_savings_inline tool call.

    Returns a compact one-line savings summary for embedding in tool responses.
    """
    session_id = args.get("session_id", "default")
    config = _session_config_store.get_config(session_id)
    tracker = _get_tracker(session_id, config.model_id)
    report = tracker.get_report()

    if report.total_operations == 0:
        return json.dumps(
            {"status": "success", "summary": "No compression operations recorded yet."}
        )

    summary = (
        f"Saved {report.total_tokens_saved:,} tokens "
        f"(${report.total_dollars_saved:.2f}) across {report.total_operations} operations"
    )
    if report.roi_vs_pro_plan > 0:
        summary += f" | {report.roi_vs_pro_plan}x ROI vs Pro plan"

    return json.dumps({"status": "success", "summary": summary})


# Module-level tee store cache (keyed by session_id)
_tee_stores: dict[str, Any] = {}


def _get_tee_store(session_id: str = "default") -> Any:
    """Return the TeeStore for *session_id*, creating it on first access."""
    from ..tee_recovery import create_tee_store

    if session_id not in _tee_stores:
        _tee_stores[session_id] = create_tee_store(persist=True)
    return _tee_stores[session_id]


async def handle_get_original_output(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle get_original_output tool call.

    Retrieves the original (pre-compression) content for a tee entry ID.
    """
    entry_id = args.get("entry_id", "")
    if not entry_id:
        return json.dumps({"status": "error", "error": "entry_id is required"})

    session_id = args.get("session_id", "default")
    store = _get_tee_store(session_id)
    entry = store.get(entry_id)
    if entry is None:
        return json.dumps(
            {
                "status": "error",
                "error": f"No tee entry found with id '{entry_id}'",
            }
        )

    return json.dumps(
        {
            "status": "success",
            "entry_id": entry.entry_id,
            "original_text": entry.original_text,
            "source": entry.source,
            "command_hint": entry.command_hint,
            "compression_pct": round(entry.compression_pct, 1),
            "timestamp": entry.timestamp,
        }
    )


async def handle_list_tee_entries(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle list_tee_entries tool call.

    Lists recent tee entries with metadata (no full text).
    """
    session_id = args.get("session_id", "default")
    limit = args.get("limit", 20)
    source = args.get("source")

    store = _get_tee_store(session_id)
    entries = store.list_entries(limit=limit, source=source)

    return json.dumps(
        {
            "status": "success",
            "entries": entries,
            "stats": store.stats(),
        }
    )


async def handle_tee_store_stats(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle tee_store_stats tool call.

    Returns tee store statistics.
    """
    session_id = args.get("session_id", "default")
    store = _get_tee_store(session_id)
    return json.dumps({"status": "success", **store.stats()})


async def handle_discover_savings(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle discover_savings tool call.

    Analyzes files or text items to discover compression opportunities.
    Accepts either a directory path or a list of text items.
    """
    from ..savings_discover import ContextAnalyzer, format_report

    analyzer = ContextAnalyzer()

    directory = args.get("directory")
    items = args.get("items")

    if directory:
        # Validate path through PathValidator if available
        path_validator = context.get("path_validator")
        if path_validator:
            try:
                directory = path_validator.validate(directory)
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

        max_files = args.get("max_files", 500)
        report = analyzer.scan_directory(directory, max_files=max_files)
    elif items:
        # BUG-E-01 fix: auto-wrap plain strings as {"text": s} dicts
        wrapped = [i if isinstance(i, dict) else {"text": str(i)} for i in items]
        report = analyzer.analyze_items(wrapped)
    else:
        return json.dumps(
            {
                "status": "error",
                "error": "Provide either 'directory' (path) or 'items' (list of text blobs)",
            }
        )

    return json.dumps(
        {
            "status": "success",
            "report": report.to_dict(),
            "formatted": format_report(report),
        }
    )


# Default compression ratio for ROI calculations (conservative)
_DEFAULT_COMPRESSION_RATIO = 0.85

# Pro plan price per user per month
_PRO_PLAN_PRICE = 29.0


async def handle_calculate_roi(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle calculate_roi tool call.

    Calculates ROI of using gotcontext compression vs raw token usage.
    Shows monthly cost comparison and payback analysis.
    """
    from ..cli_benchmark.pricing import PRICING, get_model_rates

    model = args.get("model", "claude-sonnet-4-6")
    tokens_per_day = args.get("tokens_per_day", 500_000)
    team_size = args.get("team_size", 1)
    compression_ratio = args.get("compression_ratio", _DEFAULT_COMPRESSION_RATIO)

    rates = get_model_rates(model)
    input_rate = rates.get("input", 0)  # per million tokens

    # Monthly calculations (22 working days)
    working_days = 22
    monthly_tokens = tokens_per_day * working_days * team_size
    monthly_compressed = int(monthly_tokens * (1 - compression_ratio))

    cost_without = monthly_tokens * input_rate / 1_000_000
    cost_with = monthly_compressed * input_rate / 1_000_000
    tokens_saved = monthly_tokens - monthly_compressed
    dollars_saved = cost_without - cost_with

    pro_plan_cost = _PRO_PLAN_PRICE * team_size
    net_savings = dollars_saved - pro_plan_cost
    roi_multiplier = dollars_saved / pro_plan_cost if pro_plan_cost > 0 else 0

    available_models = sorted(PRICING.keys() - {"default"})

    comparison = (
        f"Without gotcontext: ${cost_without:,.2f}/mo\n"
        f"With gotcontext:    ${cost_with:,.2f}/mo "
        f"({compression_ratio * 100:.0f}% savings)\n"
        f"Pro plan cost:      ${pro_plan_cost:,.2f}/mo "
        f"(${_PRO_PLAN_PRICE:.0f}/user × {team_size} users)\n"
        f"Net savings:        ${net_savings:,.2f}/mo "
        f"({roi_multiplier:.1f}x ROI)\n"
        f"Payback period:     Day 1"
    )

    return json.dumps(
        {
            "status": "success",
            "model": model,
            "tokens_per_day": tokens_per_day,
            "team_size": team_size,
            "compression_ratio": compression_ratio,
            "monthly_tokens": monthly_tokens,
            "monthly_tokens_saved": tokens_saved,
            "cost_without_monthly": round(cost_without, 2),
            "cost_with_monthly": round(cost_with, 2),
            "dollars_saved_monthly": round(dollars_saved, 2),
            "pro_plan_cost_monthly": round(pro_plan_cost, 2),
            "net_savings_monthly": round(net_savings, 2),
            "roi_multiplier": round(roi_multiplier, 1),
            "comparison": comparison,
            "available_models": available_models,
        }
    )


# Module-level budget monitors keyed by scope
_budget_monitors: Dict[str, Any] = {}


async def handle_check_budget(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle check_budget tool call.

    Returns token budget status across all configured limits
    (session, daily, monthly) with projections and alert levels.
    """
    from ..budget_monitor import TokenBudgetMonitor

    scope = args.get("workspace_id", "default")
    if scope not in _budget_monitors:
        _budget_monitors[scope] = TokenBudgetMonitor(
            session_limit=args.get("session_limit", 0),
            daily_limit=args.get("daily_limit", 0),
            monthly_limit=args.get("monthly_limit", 0),
        )

    monitor = _budget_monitors[scope]

    # Record new usage if provided
    new_tokens = args.get("record_tokens", 0)
    if new_tokens > 0:
        monitor.record_usage(new_tokens, args.get("tool_name", ""))

    result = monitor.check_budget()
    return json.dumps({"status": "success", **result.to_dict()})


async def handle_export_team_data(context: Dict[str, Any], args: Dict[str, Any]) -> str:
    """Handle export_team_data tool call.

    Exports aggregated team savings data in JSON, CSV, or Prometheus format.
    """
    from ..team_export import TeamExporter

    exporter = TeamExporter()

    # Accept member stats from args
    members = args.get("members", [])
    for m in members:
        if not isinstance(m, dict):
            continue
        exporter.add_member_stats(
            user_id=m.get("user_id", "unknown"),
            sessions=m.get("sessions", 0),
            original_tokens=m.get("original_tokens", 0),
            compressed_tokens=m.get("compressed_tokens", 0),
            operations=m.get("operations", 0),
        )

    report = exporter.build_report()
    fmt = args.get("format", "json")

    if fmt == "csv":
        exported = exporter.export_csv(report)
    elif fmt == "prometheus":
        exported = exporter.export_prometheus(report)
    else:
        exported = exporter.export_json(report)

    return json.dumps(
        {
            "status": "success",
            "format": fmt,
            "data": exported,
            "summary": report.to_dict(),
        }
    )
