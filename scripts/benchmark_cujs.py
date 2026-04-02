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
# Orchestration
# ---------------------------------------------------------------------------

ALL_CUJS = [
    run_cuj_1_solo_dev_codebase,
    run_cuj_2_long_document,
    run_cuj_3_cli_output_filtering,
    run_cuj_4_query_focused_search,
    run_cuj_5_session_recovery,
    run_cuj_6_savings_report,
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
        choices=range(1, 7),
        metavar="N",
        help="Run only journey N (1-6)",
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
