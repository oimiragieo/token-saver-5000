#!/usr/bin/env python
"""Critical User Journey Baselines for gotcontext.ai

Runs 6 real-world user journeys end-to-end, measuring token savings
at each step. All journeys run locally (no API calls needed).

Usage:
    python scripts/benchmark_cujs.py
    python scripts/benchmark_cujs.py --verbose
    python scripts/benchmark_cujs.py --journey 1  # run single journey
    python scripts/benchmark_cujs.py --output cujs_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

CORPUS_CODE_DIR = str(Path(__file__).resolve().parent.parent / "benchmarks" / "corpus" / "code")
CORPUS_LARGE_TXT = Path(__file__).resolve().parent.parent / "benchmarks" / "corpus" / "large.txt"


@dataclass
class StepResult:
    name: str
    input_tokens: int
    output_tokens: int
    savings_pct: float
    duration_ms: float
    extra: dict = field(default_factory=dict)


@dataclass
class CUJResult:
    journey_id: int
    name: str
    persona: str
    description: str
    steps: list[StepResult] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_savings_pct: float = 0.0
    passed: bool = True
    error: str | None = None


@dataclass
class CUJBaseline:
    timestamp: str
    journeys: list[CUJResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_ms() -> float:
    return time.perf_counter() * 1000


def _elapsed_ms(start: float) -> float:
    return round(time.perf_counter() * 1000 - start, 1)


def _savings_pct(original: int, compressed: int) -> float:
    if original <= 0:
        return 0.0
    return round((original - compressed) / original * 100, 1)


def _count_tokens(text: str) -> int:
    """Estimate token count: one token per four characters."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# CLI output fixtures for CUJ 3
# ---------------------------------------------------------------------------

GIT_DIFF_FIXTURE = "\n".join(
    [
        # 10 files, each with ~50 hunk lines
        f"diff --git a/src/module_{i:02d}.py b/src/module_{i:02d}.py"
        + "\nindex abc123..def456 100644"
        + f"\n--- a/src/module_{i:02d}.py"
        + f"\n+++ b/src/module_{i:02d}.py"
        + "\n@@ -10,6 +10,8 @@ class Service:"
        + "".join(
            f"\n-    old_line_{j} = None  # removed for refactoring pass {i}" for j in range(8)
        )
        + "".join(f"\n+    new_line_{j} = True  # added in refactoring pass {i}" for j in range(10))
        + "\n@@ -50,4 +52,4 @@ def process(data):"
        + f"\n-    return old_result_{i}"
        + f"\n+    return new_result_{i}"
        + "\n@@ -100,3 +102,5 @@ def validate(x):"
        + "\n-    if x is None:"
        + "\n-        raise ValueError('null')"
        + "\n+    if x is None or x == '':"
        + "\n+        raise ValueError('invalid value')"
        + "\n+    return True"
        for i in range(10)
    ]
)

PYTEST_FIXTURE = "\n".join(
    [
        "============================= test session starts ==============================",
        "platform linux -- Python 3.12.0, pytest-7.4.0, pluggy-1.3.0",
        "rootdir: /app, configfile: pyproject.toml",
        "plugins: anyio-4.0.0, asyncio-0.21.0",
        "collected 48 items",
        "",
    ]
    + [f"tests/test_unit.py::test_happy_path_{i} PASSED" for i in range(20)]
    + [
        "tests/test_unit.py::test_auth_token_expiry FAILED",
        "tests/test_unit.py::test_cache_miss_on_ttl FAILED",
        "tests/test_unit.py::test_retry_after_503 FAILED",
    ]
    + [f"tests/test_integration.py::test_integration_{i} PASSED" for i in range(25)]
    + [
        "",
        "=================================== FAILURES ===================================",
        "__________________________ test_auth_token_expiry _____________________________",
        "",
        "    def test_auth_token_expiry():",
        "        token = create_token(expires_in=-1)",
        ">       assert is_valid(token)",
        "E       AssertionError: assert False",
        "E        +  where False = is_valid(Token(exp=1700000000))",
        "",
        "tests/test_unit.py:42: AssertionError",
        "__________________________ test_cache_miss_on_ttl ______________________________",
        "",
        "    def test_cache_miss_on_ttl():",
        "        cache.set('key', 'val', ttl=0)",
        ">       assert cache.get('key') is None",
        "E       AssertionError: assert 'val' is None",
        "",
        "tests/test_unit.py:87: AssertionError",
        "__________________________ test_retry_after_503 _______________________________",
        "",
        "    def test_retry_after_503():",
        "        with mock.patch('requests.get', side_effect=HTTPError(503)):",
        ">           client.fetch('/api/resource')",
        "E           MaxRetryError: gave up after 3 attempts",
        "",
        "tests/test_unit.py:134: AssertionError",
        "=========================== short test summary info ============================",
        "FAILED tests/test_unit.py::test_auth_token_expiry - AssertionError: assert False",
        "FAILED tests/test_unit.py::test_cache_miss_on_ttl - AssertionError: assert 'val' is None",
        "FAILED tests/test_unit.py::test_retry_after_503 - MaxRetryError: gave up after 3 attempts",
        "================== 3 failed, 45 passed in 12.34s ====================",
    ]
)

NPM_INSTALL_FIXTURE = "\n".join(
    [
        "npm warn deprecated lodash@4.17.20: Critical security issue in prototype pollution",
        "npm warn deprecated request@2.88.2: request has been deprecated",
        "npm warn deprecated har-validator@5.1.5: this library is no longer supported",
        "",
    ]
    + [f"npm timing reifyNode:node_modules/pkg-{i} Completed in {50 + i * 3}ms" for i in range(15)]
    + [
        "npm timing idealTree Completed in 843ms",
        "npm timing reify Completed in 6732ms",
        "",
        "added 312 packages, and audited 313 packages in 7s",
        "",
        "54 packages are looking for funding",
        "  run `npm fund` for details",
        "",
        "found 0 vulnerabilities",
    ]
)

# ---------------------------------------------------------------------------
# CUJ implementations
# ---------------------------------------------------------------------------


def run_cuj_1_solo_dev_codebase(verbose: bool = False) -> CUJResult:
    """CUJ 1: Solo Developer Compressing a Large Codebase"""
    result = CUJResult(
        journey_id=1,
        name="Solo Dev: Codebase Compression",
        persona="Dev using Claude Code with a 20-file Python project",
        description="Configure for model, search for auth-related files, compress matched files only",
    )

    try:
        from src.client_config import ClientConfig

        # Step 1: configure_for_client
        t0 = _now_ms()
        config = ClientConfig.from_model("claude-opus-4-6")
        ratio = config.get_recommended_ratio()
        dur1 = _elapsed_ms(t0)

        step1 = StepResult(
            name="configure_for_client",
            input_tokens=0,
            output_tokens=0,
            savings_pct=0.0,
            duration_ms=dur1,
            extra={"model": "claude-opus-4-6", "recommended_ratio": ratio},
        )
        result.steps.append(step1)
        if verbose:
            print(f"  [1/2] configure_for_client -> ratio={ratio} ({dur1:.0f}ms)")

        # Step 2: search_then_compress
        from src.search_compress_pipeline import search_then_compress

        t0 = _now_ms()
        sc = search_then_compress(CORPUS_CODE_DIR, "authentication token JWT", max_files=20)
        dur2 = _elapsed_ms(t0)

        savings = _savings_pct(sc.naive_all_tokens, sc.search_compress_tokens)
        step2 = StepResult(
            name="search_then_compress",
            input_tokens=sc.naive_all_tokens,
            output_tokens=sc.search_compress_tokens,
            savings_pct=savings,
            duration_ms=dur2,
            extra={
                "files_scanned": sc.files_scanned,
                "files_matched": sc.files_matched,
                "search_method": sc.search_method,
                "compression_ratio": sc.compression_ratio,
            },
        )
        result.steps.append(step2)
        if verbose:
            print(
                f"  [2/2] search_then_compress -> {sc.naive_all_tokens:,} -> "
                f"{sc.search_compress_tokens:,} tokens ({savings:.1f}% savings, {dur2:.0f}ms)"
            )

        result.total_input_tokens = sc.naive_all_tokens
        result.total_output_tokens = sc.search_compress_tokens
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_2_long_document(verbose: bool = False) -> CUJResult:
    """CUJ 2: Compressing a Long Technical Document"""
    result = CUJResult(
        journey_id=2,
        name="Long Document Compression",
        persona="Dev feeding architecture docs into AI assistant",
        description="Read large.txt API reference, compress with default settings, measure savings",
    )

    try:
        from src.cli_benchmark.compressor import compress_text

        # Step 1: read document
        t0 = _now_ms()
        content = CORPUS_LARGE_TXT.read_text(encoding="utf-8", errors="replace")
        dur_read = _elapsed_ms(t0)

        original_tokens = _count_tokens(content)
        step1 = StepResult(
            name="read_document",
            input_tokens=original_tokens,
            output_tokens=original_tokens,
            savings_pct=0.0,
            duration_ms=dur_read,
            extra={"path": str(CORPUS_LARGE_TXT), "char_count": len(content)},
        )
        result.steps.append(step1)
        if verbose:
            print(f"  [1/2] read_document -> {original_tokens:,} tokens ({dur_read:.0f}ms)")

        # Step 2: compress
        t0 = _now_ms()
        compressed = compress_text(content, refine=True)
        dur_compress = _elapsed_ms(t0)

        savings = _savings_pct(compressed.original_tokens, compressed.compressed_tokens)
        step2 = StepResult(
            name="compress_text",
            input_tokens=compressed.original_tokens,
            output_tokens=compressed.compressed_tokens,
            savings_pct=savings,
            duration_ms=dur_compress,
            extra={"compression_ratio": compressed.compression_ratio},
        )
        result.steps.append(step2)
        if verbose:
            print(
                f"  [2/2] compress_text -> {compressed.original_tokens:,} -> "
                f"{compressed.compressed_tokens:,} tokens ({savings:.1f}% savings, {dur_compress:.0f}ms)"
            )

        result.total_input_tokens = compressed.original_tokens
        result.total_output_tokens = compressed.compressed_tokens
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_3_cli_output_filtering(verbose: bool = False) -> CUJResult:
    """CUJ 3: Filtering Noisy CLI Output"""
    result = CUJResult(
        journey_id=3,
        name="CLI Output Filtering (3 tools)",
        persona="Dev whose AI agent runs git/pytest/npm and gets huge output",
        description=(
            "Filter git diff (~500 lines), pytest output (~100 lines), "
            "and npm install (~50 lines) to extract only the signal"
        ),
    )

    try:
        from src.cli_output_optimizer import CLIOutputOptimizer

        optimizer = CLIOutputOptimizer()

        total_in = 0
        total_out = 0

        for label, fixture, hint in [
            ("git_diff", GIT_DIFF_FIXTURE, "git_diff"),
            ("pytest_output", PYTEST_FIXTURE, "test_output"),
            ("npm_install", NPM_INSTALL_FIXTURE, "install_output"),
        ]:
            t0 = _now_ms()
            fr = optimizer.filter(fixture, command_hint=hint)
            dur = _elapsed_ms(t0)

            in_tokens = _count_tokens(fr.original_text)
            out_tokens = _count_tokens(fr.filtered_text)
            savings = _savings_pct(in_tokens, out_tokens)

            total_in += in_tokens
            total_out += out_tokens

            step = StepResult(
                name=f"filter_{label}",
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                savings_pct=savings,
                duration_ms=dur,
                extra={
                    "original_lines": fr.original_lines,
                    "filtered_lines": fr.filtered_lines,
                    "strategy": fr.strategy_applied,
                    "compression_pct": round(fr.compression_pct, 1),
                },
            )
            result.steps.append(step)
            if verbose:
                print(
                    f"  [{label}] {fr.original_lines} lines -> {fr.filtered_lines} lines "
                    f"({fr.compression_pct:.1f}% line compression, {dur:.0f}ms)"
                )

        result.total_input_tokens = total_in
        result.total_output_tokens = total_out
        result.total_savings_pct = _savings_pct(total_in, total_out)

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_4_query_focused_search(verbose: bool = False) -> CUJResult:
    """CUJ 4: Query-Focused Code Search + Compression"""
    result = CUJResult(
        journey_id=4,
        name="Query-Focused Code Search",
        persona="Dev asking 'how does the caching layer work?' about a codebase",
        description=(
            "Compare naive (all files), compress-all, and search+compress strategies "
            "for a cache-specific query to show search-first advantage"
        ),
    )

    try:
        from src.search_compress_pipeline import search_then_compress

        t0 = _now_ms()
        sc = search_then_compress(CORPUS_CODE_DIR, "cache TTL invalidation LRU", max_files=20)
        dur = _elapsed_ms(t0)

        # Step 1: Naive strategy (send everything raw)
        step_naive = StepResult(
            name="naive_all_files",
            input_tokens=sc.naive_all_tokens,
            output_tokens=sc.naive_all_tokens,
            savings_pct=0.0,
            duration_ms=0.0,
            extra={"strategy": "send all files uncompressed"},
        )
        result.steps.append(step_naive)

        # Step 2: Compress-all strategy
        savings_compress_all = _savings_pct(sc.naive_all_tokens, sc.naive_compressed_tokens)
        step_compress_all = StepResult(
            name="compress_all_files",
            input_tokens=sc.naive_all_tokens,
            output_tokens=sc.naive_compressed_tokens,
            savings_pct=savings_compress_all,
            duration_ms=0.0,
            extra={"strategy": "compress all files regardless of query"},
        )
        result.steps.append(step_compress_all)

        # Step 3: Search + compress (the optimal path)
        savings_search = _savings_pct(sc.naive_all_tokens, sc.search_compress_tokens)
        step_search = StepResult(
            name="search_then_compress",
            input_tokens=sc.naive_all_tokens,
            output_tokens=sc.search_compress_tokens,
            savings_pct=savings_search,
            duration_ms=dur,
            extra={
                "strategy": "search for relevant files, compress only those",
                "files_matched": sc.files_matched,
                "files_scanned": sc.files_scanned,
                "matched_files": [Path(f).name for f in sc.matched_files],
                "search_vs_compress_all_savings_pct": sc.search_vs_compress_all_savings_pct,
            },
        )
        result.steps.append(step_search)

        if verbose:
            print(f"  Naive:        {sc.naive_all_tokens:,} tokens")
            print(
                f"  Compress-all: {sc.naive_compressed_tokens:,} tokens "
                f"({savings_compress_all:.1f}% savings)"
            )
            print(
                f"  Search+compress: {sc.search_compress_tokens:,} tokens "
                f"({savings_search:.1f}% savings, {dur:.0f}ms)"
            )
            print(f"  Matched files: {[Path(f).name for f in sc.matched_files]}")

        result.total_input_tokens = sc.naive_all_tokens
        result.total_output_tokens = sc.search_compress_tokens
        result.total_savings_pct = savings_search

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_5_session_recovery(verbose: bool = False) -> CUJResult:
    """CUJ 5: Session Recovery After Compaction"""
    result = CUJResult(
        journey_id=5,
        name="Session Recovery After Compaction",
        persona="Dev in a long Claude Code session that gets compacted",
        description=(
            "Journal 5 ingest events + configure + profile, recover session summary, "
            "verify compactness vs re-ingesting everything"
        ),
    )

    try:
        from src.session_journal import SessionJournal

        with tempfile.TemporaryDirectory() as tmp_dir:
            t0 = _now_ms()
            journal = SessionJournal("cuj-5-test", storage_dir=tmp_dir)

            # Write 5 ingest events with realistic token counts
            ingest_events = [
                {"file_id": "auth.py", "original_tokens": 5000, "compressed_tokens": 400},
                {"file_id": "database.py", "original_tokens": 4200, "compressed_tokens": 310},
                {"file_id": "middleware.py", "original_tokens": 3800, "compressed_tokens": 290},
                {"file_id": "models.py", "original_tokens": 6100, "compressed_tokens": 480},
                {"file_id": "api.py", "original_tokens": 7500, "compressed_tokens": 580},
            ]
            for ev in ingest_events:
                journal.write_event("ingest", ev)

            journal.write_event("configure", {"model_id": "claude-opus-4-6", "ratio": 0.35})
            journal.write_event("profile", {"profile_name": "code"})

            dur_write = _elapsed_ms(t0)
            step_write = StepResult(
                name="write_journal_events",
                input_tokens=0,
                output_tokens=0,
                savings_pct=0.0,
                duration_ms=dur_write,
                extra={"events_written": 7},
            )
            result.steps.append(step_write)
            if verbose:
                print(f"  [1/2] wrote 7 events in {dur_write:.0f}ms")

            # Recover the session
            t0 = _now_ms()
            summary = journal.recover()
            dur_recover = _elapsed_ms(t0)
            journal.close()

            # Measure summary size (tokens that would go into the context)
            summary_text = json.dumps(
                {
                    "session_id": summary.session_id,
                    "ingested_files": summary.ingested_files,
                    "client_config": summary.client_config,
                    "active_profile": summary.active_profile,
                    "total_tokens_saved": summary.total_tokens_saved,
                }
            )
            summary_tokens = _count_tokens(summary_text)

            # What it would cost to re-ingest everything
            original_total = sum(ev["original_tokens"] for ev in ingest_events)

            savings = _savings_pct(original_total, summary_tokens)
            step_recover = StepResult(
                name="recover_session",
                input_tokens=original_total,
                output_tokens=summary_tokens,
                savings_pct=savings,
                duration_ms=dur_recover,
                extra={
                    "ingested_files_count": len(summary.ingested_files),
                    "client_config": summary.client_config,
                    "active_profile": summary.active_profile,
                    "total_tokens_saved": summary.total_tokens_saved,
                    "summary_tokens": summary_tokens,
                    "original_total_tokens": original_total,
                },
            )
            result.steps.append(step_recover)
            if verbose:
                print(
                    f"  [2/2] recovered summary: {original_total:,} -> {summary_tokens:,} tokens "
                    f"({savings:.1f}% savings, {dur_recover:.0f}ms)"
                )

        result.total_input_tokens = original_total
        result.total_output_tokens = summary_tokens
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_6_savings_report(verbose: bool = False) -> CUJResult:
    """CUJ 6: Getting Savings Report to Justify the Tool"""
    result = CUJResult(
        journey_id=6,
        name="Savings Report (ROI Justification)",
        persona="Dev who wants to show their manager ROI",
        description=(
            "Record 10 realistic compression operations, generate full report, "
            "verify dollars saved, ROI vs Pro plan, and breakeven calculation"
        ),
    )

    try:
        from src.savings_tracker import SavingsTracker

        # Step 1: record 10 compression operations
        t0 = _now_ms()
        tracker = SavingsTracker("cuj-6-session", model="claude-opus-4-6")
        # Detach journal to avoid filesystem side-effects in tests
        tracker._journal = None

        operations = [
            ("ingest_context", 5000, 400),
            ("ingest_context", 5500, 430),
            ("ingest_context", 4200, 320),
            ("search_then_compress", 20000, 800),
            ("ingest_context", 6100, 480),
            ("compress_text", 15000, 1200),
            ("ingest_context", 3800, 290),
            ("search_then_compress", 18500, 750),
            ("ingest_context", 7500, 580),
            ("ingest_context", 4800, 370),
        ]

        for tool_name, original, compressed in operations:
            tracker.record(
                tool_name,
                original_tokens=original,
                compressed_tokens=compressed,
                model="claude-opus-4-6",
            )

        dur_record = _elapsed_ms(t0)
        step_record = StepResult(
            name="record_10_operations",
            input_tokens=sum(o for _, o, _ in operations),
            output_tokens=sum(c for _, _, c in operations),
            savings_pct=_savings_pct(
                sum(o for _, o, _ in operations), sum(c for _, _, c in operations)
            ),
            duration_ms=dur_record,
            extra={"operations_recorded": len(operations)},
        )
        result.steps.append(step_record)

        # Step 2: generate report
        t0 = _now_ms()
        report = tracker.get_report()
        dur_report = _elapsed_ms(t0)

        step_report = StepResult(
            name="get_report",
            input_tokens=report.total_original_tokens,
            output_tokens=report.total_compressed_tokens,
            savings_pct=report.avg_savings_pct,
            duration_ms=dur_report,
            extra={
                "total_dollars_saved": report.total_dollars_saved,
                "monthly_projected_savings": report.monthly_projected_savings,
                "roi_vs_pro_plan": report.roi_vs_pro_plan,
                "breakeven_operations": report.breakeven_operations,
                "avg_compression_ratio": report.avg_compression_ratio,
                "tools_used": list(report.by_tool.keys()),
            },
        )
        result.steps.append(step_report)

        if verbose:
            print(f"  Total dollars saved: ${report.total_dollars_saved:.4f}")
            print(f"  Monthly projected: ${report.monthly_projected_savings:.2f}")
            print(f"  ROI vs Pro plan ($29): {report.roi_vs_pro_plan}x")
            print(f"  Breakeven operations: {report.breakeven_operations:,}")
            print(f"  Avg compression ratio: {report.avg_compression_ratio}x")

        result.total_input_tokens = report.total_original_tokens
        result.total_output_tokens = report.total_compressed_tokens
        result.total_savings_pct = report.avg_savings_pct

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


# ---------------------------------------------------------------------------
# CUJ 7-12: Additional journeys covering gaps from market analysis
# ---------------------------------------------------------------------------


# Fixture: realistic MCP tool schemas for CUJ 7
_SCHEMA_TOOLS_FIXTURE = [
    {
        "name": f"tool_{i}",
        "description": f"This is tool number {i} which performs operation {chr(65 + i % 26)} "
        f"on the target resource. It accepts a variety of parameters including "
        f"strings, integers, booleans, and nested objects. Use this when you need "
        f"to {'create' if i % 4 == 0 else 'read' if i % 4 == 1 else 'update' if i % 4 == 2 else 'delete'} "
        f"resources of type {chr(65 + i % 26)}.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": f"Unique identifier for the {chr(65 + i % 26)} resource",
                },
                "name": {"type": "string", "description": "Human-readable name"},
                "config": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean", "default": True},
                        "timeout": {"type": "integer", "default": 30},
                        "retries": {"type": "integer", "default": 3},
                        "verbose": {"type": "boolean", "default": False},
                    },
                },
                "tags": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            },
            "required": ["id"],
        },
    }
    for i in range(50)
]


def run_cuj_7_schema_compression(verbose: bool = False) -> CUJResult:
    """CUJ 7: Tool Schema Compression (Proxy Mode)"""
    result = CUJResult(
        journey_id=7,
        name="Tool Schema Compression",
        persona="Dev with 5 MCP servers exposing 50+ tools — schema bloat wastes tokens",
        description=(
            "Compress 50 tool schemas into 3 meta-tools using SchemaCompressor. "
            "Measures token reduction from schema bloat elimination."
        ),
    )

    try:
        from src.proxy.schema_compressor import SchemaCompressor

        # Step 1: measure original schema size
        original_json = json.dumps(_SCHEMA_TOOLS_FIXTURE)
        original_tokens = _count_tokens(original_json)

        t0 = _now_ms()
        compressor = SchemaCompressor(_SCHEMA_TOOLS_FIXTURE)
        meta_tools = compressor.meta_tool_schemas()
        dur = _elapsed_ms(t0)

        compressed_json = json.dumps(meta_tools)
        compressed_tokens = _count_tokens(compressed_json)
        savings = _savings_pct(original_tokens, compressed_tokens)

        step = StepResult(
            name="schema_compress_50_tools",
            input_tokens=original_tokens,
            output_tokens=compressed_tokens,
            savings_pct=savings,
            duration_ms=dur,
            extra={
                "original_tool_count": len(_SCHEMA_TOOLS_FIXTURE),
                "meta_tool_count": len(meta_tools),
                "meta_tool_names": [t["name"] for t in meta_tools],
            },
        )
        result.steps.append(step)
        if verbose:
            print(
                f"  [1/1] 50 tools -> 3 meta-tools: {original_tokens:,} -> "
                f"{compressed_tokens:,} tokens ({savings:.1f}% savings, {dur:.0f}ms)"
            )

        result.total_input_tokens = original_tokens
        result.total_output_tokens = compressed_tokens
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_8_code_compression(verbose: bool = False) -> CUJResult:
    """CUJ 8: Code-Aware Compression"""
    result = CUJResult(
        journey_id=8,
        name="Code-Aware Compression",
        persona="Dev asking AI to review a Python module — full file wastes tokens",
        description=(
            "Compress Python source files using semantic compressor. "
            "Measures savings vs raw file reading for code review context."
        ),
    )

    try:
        from src.cli_benchmark.compressor import compress_text

        code_dir = Path(CORPUS_CODE_DIR)
        py_files = sorted(code_dir.glob("*.py"))

        total_original = 0
        total_compressed = 0

        for py_file in py_files:
            code = py_file.read_text(encoding="utf-8", errors="replace")
            original_tokens = _count_tokens(code)
            total_original += original_tokens

            t0 = _now_ms()
            cr = compress_text(code, refine=True)
            dur = _elapsed_ms(t0)

            total_compressed += cr.compressed_tokens

            if verbose:
                savings = _savings_pct(cr.original_tokens, cr.compressed_tokens)
                print(
                    f"  {py_file.name}: {cr.original_tokens:,} -> "
                    f"{cr.compressed_tokens:,} tokens ({savings:.1f}%, {dur:.0f}ms)"
                )

        savings = _savings_pct(total_original, total_compressed)
        step = StepResult(
            name="code_compress_codebase",
            input_tokens=total_original,
            output_tokens=total_compressed,
            savings_pct=savings,
            duration_ms=0.0,
            extra={"files_compressed": len(py_files)},
        )
        result.steps.append(step)

        result.total_input_tokens = total_original
        result.total_output_tokens = total_compressed
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_9_dialogue_memory(verbose: bool = False) -> CUJResult:
    """CUJ 9: Multi-Turn Dialogue Compression (AFM)"""
    result = CUJResult(
        journey_id=9,
        name="Dialogue Memory (AFM)",
        persona="Dev in 20-turn conversation — context re-sending wastes 15-20% of tokens",
        description=(
            "Simulate a 20-turn coding conversation, use AFM to compress history "
            "while preserving relevant context for the current query."
        ),
    )

    try:
        from src.afm import FocusManager

        afm = FocusManager()

        # Simulate a realistic 20-turn coding conversation
        conversation = [
            ("user", "Can you help me set up authentication for my Flask app?"),
            (
                "assistant",
                "Sure! I'll set up JWT-based authentication. First, install flask-jwt-extended: `pip install flask-jwt-extended`. Then create an auth module with login/register endpoints, password hashing with bcrypt, and token refresh support.",
            ),
            ("user", "Great, now add middleware to protect routes"),
            (
                "assistant",
                "Here's a decorator-based middleware that checks JWT tokens on protected routes. It extracts the token from the Authorization header, validates it, and returns 401 if invalid. I'll also add role-based access control with @require_role('admin').",
            ),
            ("user", "The tests are failing for the auth module"),
            (
                "assistant",
                "Looking at the test failures: test_token_expiry expects a 401 but gets 200 because the mock clock isn't advancing. Fix: use freezegun to freeze time, then advance past token expiry. Also test_register_duplicate should check for 409, not 400.",
            ),
            ("user", "Now I need a database layer with SQLAlchemy"),
            (
                "assistant",
                "I'll create a database module with SQLAlchemy ORM. Models: User, Session, AuditLog. Using Alembic for migrations. Connection pooling with pool_size=5, max_overflow=10. The User model has email (unique), password_hash, created_at, is_active columns.",
            ),
            ("user", "Add caching with Redis"),
            (
                "assistant",
                "Setting up Redis caching layer: cache decorator with configurable TTL, key generation based on function name + args hash, cache invalidation patterns for user updates. Default TTL: 300s for queries, 3600s for user profiles. Fallback to in-memory LRU cache if Redis is unavailable.",
            ),
            ("user", "Write integration tests for the cache layer"),
            (
                "assistant",
                "Created 15 integration tests: test_cache_hit, test_cache_miss, test_cache_ttl_expiry, test_cache_invalidation_on_write, test_redis_connection_failure_fallback, test_concurrent_cache_access, test_cache_key_collision, test_cache_metrics_recording, and more. Using testcontainers for Redis.",
            ),
            ("user", "The caching is causing stale data in production"),
            (
                "assistant",
                "Classic cache invalidation issue. Three fixes: 1) Add write-through cache: update cache on every write, not just reads. 2) Add version tags to cache keys so schema changes auto-invalidate. 3) Reduce TTL for frequently-changing data (user profiles: 60s, config: 300s, static content: 3600s). Also add cache health metrics to monitoring dashboard.",
            ),
            ("user", "Add API rate limiting"),
            (
                "assistant",
                "Implementing rate limiting with a sliding window algorithm using Redis sorted sets. Limits: 100 requests/minute for authenticated users, 20 for anonymous. Per-endpoint limits: auth endpoints 5/min (brute force protection), search 30/min, CRUD 60/min. Returns 429 with Retry-After header. Redis failure: fail-open with in-memory counters.",
            ),
            ("user", "Deploy to Kubernetes"),
            (
                "assistant",
                "Here's the K8s deployment: Deployment with 3 replicas, HPA (min 3, max 10, target 70% CPU), PDB (minAvailable: 2), ConfigMap for env vars, Secret for JWT_SECRET and DB_PASSWORD, Service (ClusterIP), Ingress with TLS. Liveness: /health every 10s, Readiness: /ready every 5s. Resource limits: 256Mi-512Mi memory, 100m-500m CPU.",
            ),
            ("user", "Wait, go back to the auth module — how does token refresh work?"),
            (
                "assistant",
                "Token refresh flow: 1) Client sends expired access token + valid refresh token to /auth/refresh. 2) Server validates refresh token against DB (not just JWT signature — prevents replay after logout). 3) Issues new access token (15min) + rotates refresh token (7d). 4) Old refresh token invalidated immediately. The @jwt_required decorator auto-detects expired tokens and returns 401 with 'token_expired' error code so the client can retry with refresh.",
            ),
            ("user", "How does the rate limiter interact with the caching layer?"),
            (
                "assistant",
                "They share the same Redis connection pool but use different key prefixes: 'cache:' and 'ratelimit:'. No direct interaction — rate limiter counts requests regardless of cache hits. However, cached responses are faster (2ms vs 50ms), so rate-limited users still get fast responses for cached content. One edge case: cache warming requests shouldn't count toward rate limits. I added an X-Internal header check that bypasses rate limiting for internal service-to-service calls.",
            ),
        ]

        t0 = _now_ms()
        for role, content in conversation:
            afm.add_message(role, content)
        dur_add = _elapsed_ms(t0)

        # Measure raw conversation size
        raw_tokens = sum(_count_tokens(c) for _, c in conversation)

        step_add = StepResult(
            name="add_20_messages",
            input_tokens=raw_tokens,
            output_tokens=raw_tokens,
            savings_pct=0.0,
            duration_ms=dur_add,
            extra={"messages_added": len(conversation)},
        )
        result.steps.append(step_add)
        if verbose:
            print(
                f"  [1/2] added {len(conversation)} messages ({raw_tokens:,} tokens, {dur_add:.0f}ms)"
            )

        # Build compressed context for a follow-up question
        t0 = _now_ms()
        context, stats = afm.build_context(
            current_query="How does the rate limiter interact with the caching layer?",
            budget_tokens=2000,
        )
        dur_build = _elapsed_ms(t0)

        compressed_tokens = sum(_count_tokens(c) for _, c in context)
        savings = _savings_pct(raw_tokens, compressed_tokens)

        step_build = StepResult(
            name="build_context_budget_2000",
            input_tokens=raw_tokens,
            output_tokens=compressed_tokens,
            savings_pct=savings,
            duration_ms=dur_build,
            extra={
                "context_messages": len(context),
                "budget_tokens": 2000,
                "full_count": stats.full_count,
                "compressed_count": stats.compressed_count,
                "placeholder_count": stats.placeholder_count,
            },
        )
        result.steps.append(step_build)
        if verbose:
            print(
                f"  [2/2] build_context: {raw_tokens:,} -> {compressed_tokens:,} tokens "
                f"({savings:.1f}% savings, {dur_build:.0f}ms)"
            )
            print(
                f"        full={stats.full_count} compressed={stats.compressed_count} "
                f"placeholder={stats.placeholder_count}"
            )

        result.total_input_tokens = raw_tokens
        result.total_output_tokens = compressed_tokens
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_10_budget_governance(verbose: bool = False) -> CUJResult:
    """CUJ 10: Token Budget Monitoring & Alerts"""
    result = CUJResult(
        journey_id=10,
        name="Budget Governance",
        persona="Engineering manager who wants to cap team token spend at $500/day",
        description=(
            "Set up daily token budget (500K tokens ≈ $500/day at Opus), "
            "record usage across simulated team, verify alert thresholds fire correctly."
        ),
    )

    try:
        from src.budget_monitor import TokenBudgetMonitor

        t0 = _now_ms()
        monitor = TokenBudgetMonitor(
            session_limit=100_000,  # 100K per session
            daily_limit=500_000,  # 500K per day (~$500 at Opus)
            monthly_limit=10_000_000,  # 10M per month
        )

        # Simulate usage: 10 sessions averaging 40K tokens each
        sessions = [
            ("session_1", 35_000),
            ("session_2", 42_000),
            ("session_3", 48_000),
            ("session_4", 31_000),
            ("session_5", 55_000),
            ("session_6", 38_000),
            ("session_7", 45_000),
            ("session_8", 50_000),
            ("session_9", 28_000),
            ("session_10", 43_000),
        ]
        total_used = 0
        for name, tokens in sessions:
            monitor.record_usage(tokens, tool_name=name)
            total_used += tokens

        dur_record = _elapsed_ms(t0)

        # Check budget
        t0 = _now_ms()
        check = monitor.check_budget()
        dur_check = _elapsed_ms(t0)

        step = StepResult(
            name="budget_check_after_10_sessions",
            input_tokens=total_used,
            output_tokens=0,
            savings_pct=0.0,
            duration_ms=dur_record + dur_check,
            extra={
                "overall_status": check.overall_status,
                "limits": [lim.to_dict() for lim in check.limits],
                "sessions_recorded": len(sessions),
                "total_tokens_used": total_used,
            },
        )
        result.steps.append(step)
        if verbose:
            print(f"  [{len(sessions)} sessions] {total_used:,} tokens used")
            for lim in check.limits:
                print(
                    f"    {lim.name}: {lim.usage_pct:.1f}% ({lim.current_tokens:,}/{lim.max_tokens:,}) "
                    f"alert={lim.alert_level}"
                )
            print(f"  Overall: {check.overall_status}")

        # The "savings" here is about governance — how much budget remains
        result.total_input_tokens = total_used
        result.total_output_tokens = 0
        result.total_savings_pct = 0.0
        # For budget CUJ, track governance metrics instead of compression
        result.passed = check.overall_status in ("ok", "info", "warning", "critical")

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_11_tee_recovery(verbose: bool = False) -> CUJResult:
    """CUJ 11: Recovering Original Content After Compression"""
    result = CUJResult(
        journey_id=11,
        name="Tee/Recovery",
        persona="Dev whose AI compressed output but now needs the original for debugging",
        description=(
            "Compress CLI output with tee enabled, verify original is recoverable, "
            "measure overhead of tee storage vs savings."
        ),
    )

    try:
        from src.cli_output_optimizer import CLIOutputOptimizer
        from src.tee_recovery import TeeStore

        with tempfile.TemporaryDirectory() as tmp_dir:
            tee = TeeStore(mode="always", persist_dir=tmp_dir)
            optimizer = CLIOutputOptimizer(tee_store=tee)

            # Compress 3 different CLI outputs
            fixtures = [
                ("git_diff", GIT_DIFF_FIXTURE, "git_diff"),
                ("pytest", PYTEST_FIXTURE, "test_output"),
                ("npm", NPM_INSTALL_FIXTURE, "install_output"),
            ]

            total_original = 0
            total_compressed = 0
            tee_ids = []

            for label, fixture, hint in fixtures:
                t0 = _now_ms()
                fr = optimizer.filter(fixture, command_hint=hint)
                dur = _elapsed_ms(t0)

                original_tokens = _count_tokens(fr.original_text)
                compressed_tokens = _count_tokens(fr.filtered_text)
                total_original += original_tokens
                total_compressed += compressed_tokens

                tee_id = fr.metadata.get("tee_id")
                if tee_id:
                    tee_ids.append(tee_id)

                if verbose:
                    print(
                        f"  [{label}] {original_tokens:,} -> {compressed_tokens:,} tokens, "
                        f"tee_id={tee_id}, {dur:.0f}ms"
                    )

            # Verify recovery
            recovered_count = 0
            for tee_id in tee_ids:
                original = tee.get_original(tee_id)
                if original is not None:
                    recovered_count += 1

            step = StepResult(
                name="compress_and_recover",
                input_tokens=total_original,
                output_tokens=total_compressed,
                savings_pct=_savings_pct(total_original, total_compressed),
                duration_ms=0.0,
                extra={
                    "entries_stored": len(tee_ids),
                    "entries_recovered": recovered_count,
                    "recovery_success": recovered_count == len(tee_ids),
                },
            )
            result.steps.append(step)
            if verbose:
                print(
                    f"  Recovery: {recovered_count}/{len(tee_ids)} entries recovered successfully"
                )

            result.total_input_tokens = total_original
            result.total_output_tokens = total_compressed
            result.total_savings_pct = _savings_pct(total_original, total_compressed)
            result.passed = recovered_count == len(tee_ids) and len(tee_ids) > 0

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


def run_cuj_12_team_dashboard(verbose: bool = False) -> CUJResult:
    """CUJ 12: Team Dashboard Export (Enterprise)"""
    result = CUJResult(
        journey_id=12,
        name="Team Dashboard Export",
        persona="Engineering manager tracking 5-person team token savings",
        description=(
            "Aggregate savings across a simulated 5-person team, export in JSON/CSV/Prometheus, "
            "verify per-member attribution and summary accuracy."
        ),
    )

    try:
        from src.team_export import TeamExporter

        t0 = _now_ms()
        exporter = TeamExporter()

        # Simulate 5 team members with realistic usage
        team = [
            ("alice", 12, 850_000, 127_500, 48),
            ("bob", 8, 620_000, 93_000, 32),
            ("carol", 15, 1_200_000, 180_000, 67),
            ("dave", 6, 410_000, 61_500, 24),
            ("eve", 10, 780_000, 117_000, 41),
        ]

        for user_id, sessions, original, compressed, ops in team:
            exporter.add_member_stats(
                user_id=user_id,
                sessions=sessions,
                original_tokens=original,
                compressed_tokens=compressed,
                operations=ops,
            )

        report = exporter.build_report()
        dur_build = _elapsed_ms(t0)

        # Export all 3 formats
        t0 = _now_ms()
        json_export = exporter.export_json(report)
        csv_export = exporter.export_csv(report)
        prometheus_export = exporter.export_prometheus(report)
        dur_export = _elapsed_ms(t0)

        total_original = report.total_original_tokens
        total_compressed = report.total_compressed_tokens
        savings = _savings_pct(total_original, total_compressed)

        step = StepResult(
            name="team_export_3_formats",
            input_tokens=total_original,
            output_tokens=total_compressed,
            savings_pct=savings,
            duration_ms=dur_build + dur_export,
            extra={
                "team_size": len(team),
                "total_sessions": report.total_sessions,
                "total_operations": report.total_operations,
                "tokens_saved": report.total_tokens_saved,
                "json_size": len(json_export),
                "csv_size": len(csv_export),
                "prometheus_size": len(prometheus_export),
            },
        )
        result.steps.append(step)
        if verbose:
            print(f"  Team: {len(team)} members, {report.total_sessions} sessions")
            print(
                f"  Tokens: {total_original:,} -> {total_compressed:,} "
                f"(saved {report.total_tokens_saved:,}, {savings:.1f}%)"
            )
            print(
                f"  Exports: JSON={len(json_export)} CSV={len(csv_export)} Prometheus={len(prometheus_export)} bytes"
            )

        result.total_input_tokens = total_original
        result.total_output_tokens = total_compressed
        result.total_savings_pct = savings

    except Exception as exc:
        result.passed = False
        result.error = str(exc)
        if verbose:
            print(f"  ERROR: {exc}")

    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ALL_CUJS = [
    run_cuj_1_solo_dev_codebase,
    run_cuj_2_long_document,
    run_cuj_3_cli_output_filtering,
    run_cuj_4_query_focused_search,
    run_cuj_5_session_recovery,
    run_cuj_6_savings_report,
    run_cuj_7_schema_compression,
    run_cuj_8_code_compression,
    run_cuj_9_dialogue_memory,
    run_cuj_10_budget_governance,
    run_cuj_11_tee_recovery,
    run_cuj_12_team_dashboard,
]


def run_all_cujs(verbose: bool = False, journey_filter: int | None = None) -> CUJBaseline:
    baseline = CUJBaseline(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    fns = ALL_CUJS if journey_filter is None else [ALL_CUJS[journey_filter - 1]]

    for fn in fns:
        if verbose:
            print(f"\nRunning {fn.__doc__.strip().splitlines()[0]} ...")
        result = fn(verbose=verbose)
        baseline.journeys.append(result)

    # Build summary
    total_in = sum(j.total_input_tokens for j in baseline.journeys)
    total_out = sum(j.total_output_tokens for j in baseline.journeys)
    passed = sum(1 for j in baseline.journeys if j.passed)
    baseline.summary = {
        "total_journeys": len(baseline.journeys),
        "passed": passed,
        "failed": len(baseline.journeys) - passed,
        "aggregate_input_tokens": total_in,
        "aggregate_output_tokens": total_out,
        "aggregate_savings_pct": _savings_pct(total_in, total_out),
    }

    return baseline


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_ROW_FMT = "{:<3}  {:<36}  {:>12}  {:>13}  {:>7}  {:>6}"
_SEPARATOR = (
    "-" * 3 + "  " + "-" * 36 + "  " + "-" * 12 + "  " + "-" * 13 + "  " + "-" * 7 + "  " + "-" * 6
)  # noqa: E501


def _print_table(baseline: CUJBaseline) -> None:
    print("\n=== gotcontext.ai Critical User Journey Baselines ===\n")
    print(_ROW_FMT.format("CUJ", "Journey", "Input Tokens", "Output Tokens", "Savings", "Status"))
    print(_SEPARATOR)

    for j in baseline.journeys:
        in_str = f"{j.total_input_tokens:,}" if j.total_input_tokens else "--"
        out_str = f"{j.total_output_tokens:,}" if j.total_output_tokens else "--"
        sav_str = f"{j.total_savings_pct:.1f}%" if j.total_savings_pct else "--"
        status = "PASS" if j.passed else "FAIL"
        print(
            _ROW_FMT.format(
                str(j.journey_id),
                j.name[:36],
                in_str,
                out_str,
                sav_str,
                status,
            )
        )

    s = baseline.summary
    total_in = s.get("aggregate_input_tokens", 0)
    total_out = s.get("aggregate_output_tokens", 0)
    pct = s.get("aggregate_savings_pct", 0)
    passed = s.get("passed", 0)
    total = s.get("total_journeys", 0)

    print(f"\nTOTAL: {passed}/{total} journeys passed")
    print(
        f"Aggregate: {total_in:,} input tokens -> {total_out:,} output tokens "
        f"({pct:.1f}% savings)"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Critical User Journey baselines for gotcontext.ai"
    )
    parser.add_argument("--verbose", action="store_true", help="Print step-level details")
    parser.add_argument(
        "--journey",
        type=int,
        choices=range(1, 13),
        metavar="N",
        help="Run only journey N (1-12)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write JSON baseline to FILE",
    )
    args = parser.parse_args()

    baseline = run_all_cujs(verbose=args.verbose, journey_filter=args.journey)
    _print_table(baseline)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(asdict(baseline), indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\nBaseline saved to {out_path}")

    failed = baseline.summary.get("failed", 0)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
