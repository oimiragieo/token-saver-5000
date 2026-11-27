"""Differential and regression tests for Round 1 runtime optimizations."""

from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal, ROUND_HALF_EVEN
import decimal
import json
import math
from pathlib import Path
import concurrent.futures
import random
import re
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli_output_optimizer as cli_opt_mod
from src.cli_output_optimizer import CLIOutputOptimizer
import src.code_budget_knapsack as knapsack_mod
from src.code_budget_knapsack import (
    CodeBlockItem,
    KnapsackResult,
    compress_code_with_knapsack,
    estimate_code_tokens,
    partition_ast_blocks,
    solve_knapsack_exact,
)
from src.proxy.cache_router import CacheRouter, _tokenize
from src.proxy.schema_levels import RouterPolicy
from src.toon_serializer import dequantize_numeric_columns, quantize_numeric_columns

# ---------------------------------------------------------------------------
# Reference implementations (exact pre-optimization semantics)
# ---------------------------------------------------------------------------


def _scalar_knapsack_reference(items: list[CodeBlockItem], budget_tokens: int) -> KnapsackResult:
    if budget_tokens < 0 or not items:
        return KnapsackResult(
            selected_blocks=[],
            total_tokens=0,
            total_value=0.0,
            budget_tokens=budget_tokens,
            reconstructed_code="",
        )
    for item in items:
        if item.token_cost < 0:
            raise ValueError(f"token_cost must be non-negative, got {item.token_cost}")

    free_items: list[CodeBlockItem] = []
    pos_items: list[CodeBlockItem] = []
    for item in items:
        if item.importance_score <= 0:
            continue
        if item.token_cost == 0:
            free_items.append(item)
        elif item.token_cost <= budget_tokens:
            pos_items.append(item)

    chosen_pos: list[CodeBlockItem] = []
    if budget_tokens > 0 and pos_items:
        dp = [0.0] * (budget_tokens + 1)
        chosen_mask = [0] * (budget_tokens + 1)
        for idx, item in enumerate(pos_items):
            cost = item.token_cost
            val = float(item.importance_score)
            bit = 1 << idx
            for w in range(budget_tokens, cost - 1, -1):
                cand = dp[w - cost] + val
                if cand > dp[w]:
                    dp[w] = cand
                    chosen_mask[w] = chosen_mask[w - cost] | bit
        mask = chosen_mask[budget_tokens]
        for idx, item in enumerate(pos_items):
            if mask & (1 << idx):
                chosen_pos.append(item)

    selected = sorted(
        free_items + chosen_pos,
        key=lambda b: (b.start_line, b.end_line, b.block_id),
    )
    reconstructed_code = "\n".join(b.code for b in selected)
    return KnapsackResult(
        selected_blocks=selected,
        total_tokens=sum(b.token_cost for b in selected),
        total_value=sum(float(b.importance_score) for b in selected),
        budget_tokens=budget_tokens,
        reconstructed_code=reconstructed_code,
    )


def _naive_filter_tools_reference(
    router: CacheRouter,
    query: str = "",
    top_k: int = 5,
    yellowlist: list[str] | set[str] | frozenset[str] | None = None,
    respect_preconditions: bool = True,
) -> list[dict[str, Any]]:
    stage_b = [t for t in router._tools if not router._policy.is_blacklisted(t["name"])]
    if router._policy.whitelist:
        stage_w = [t for t in stage_b if router._policy.is_whitelisted(t["name"])]
    else:
        stage_w = stage_b
    if respect_preconditions:
        eligible = [t for t in stage_w if router.is_precondition_satisfied(t["name"])]
    else:
        eligible = stage_w
    if not eligible:
        return []
    q_terms = _tokenize(query)
    scored = [(router._bm25_score(t["name"], q_terms), idx, t) for idx, t in enumerate(eligible)]
    if q_terms:
        scored.sort(key=lambda item: (-item[0], item[1]))
    stage_k = [item[2] for item in scored[: max(1, top_k)]]
    yl = router._default_yellowlist if yellowlist is None else frozenset(yellowlist)
    if not yl:
        return stage_k
    stage_y = [t for t in stage_k if t["name"] not in yl]
    if not stage_y and stage_k:
        return [stage_k[0]]
    return stage_y


@contextmanager
def _ref_decimal_ctx():
    ctx = decimal.getcontext().copy()
    ctx.prec = max(ctx.prec, 64)
    ctx.traps.update(decimal.DefaultContext.traps)
    with decimal.localcontext(ctx) as local_ctx:
        yield local_ctx


def _ref_is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)


def _ref_to_decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Non-finite Decimal: {value!r}")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float: {value!r}")
        return Decimal(str(value))
    return Decimal(value)


def _ref_export_bound(dval: Decimal, prefer_int: bool) -> Any:
    if prefer_int and dval == dval.to_integral_value():
        return int(dval)
    if abs(dval) > Decimal(1 << 53):
        return int(dval) if dval == dval.to_integral_value() else dval
    return float(dval)


def _ref_quantize(rows: list[dict[str, Any]], bits: int = 8) -> dict[str, Any]:
    if not isinstance(bits, int) or isinstance(bits, bool) or bits < 2 or bits > 16:
        raise ValueError(f"bits must be an integer in [2, 16], got {bits!r}")
    if not rows:
        return {"bits": bits, "columns": {}, "rows": []}
    col_names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                col_names.append(key)
    q_max = (1 << bits) - 1
    columns_meta: dict[str, dict[str, Any]] = {}
    col_bounds: dict[str, tuple[Decimal, Decimal, bool]] = {}
    with _ref_decimal_ctx():
        q_max_dec = Decimal(q_max)
        for col in col_names:
            numeric_vals: list[Decimal] = []
            all_ints = True
            for row in rows:
                if col not in row:
                    continue
                val = row[col]
                if _ref_is_numeric(val):
                    dval = _ref_to_decimal(val)
                    if not isinstance(val, int):
                        all_ints = False
                    numeric_vals.append(dval)
            if not numeric_vals:
                continue
            col_min = min(numeric_vals)
            col_max = max(numeric_vals)
            span = col_max - col_min
            eps_max = Decimal(0) if span == 0 else span / q_max_dec
            col_bounds[col] = (col_min, col_max, all_ints)
            columns_meta[col] = {
                "min": _ref_export_bound(col_min, all_ints),
                "max": _ref_export_bound(col_max, all_ints),
                "bits": bits,
                "eps_max": float(eps_max),
            }
        quantized_rows: list[dict[str, Any]] = []
        for row in rows:
            new_row: dict[str, Any] = {}
            for key, val in row.items():
                bounds = col_bounds.get(key)
                if bounds is not None and _ref_is_numeric(val):
                    col_min, col_max, _ = bounds
                    span = col_max - col_min
                    if span == 0:
                        new_row[key] = 0
                    else:
                        dval = _ref_to_decimal(val)
                        scaled = ((dval - col_min) / span) * q_max_dec
                        q_int = int(scaled.to_integral_value(rounding=ROUND_HALF_EVEN))
                        new_row[key] = max(0, min(q_max, q_int))
                else:
                    new_row[key] = val
            quantized_rows.append(new_row)
    return {"bits": bits, "columns": columns_meta, "rows": quantized_rows}


def _ref_dequantize(quantized_payload: dict[str, Any]) -> list[dict[str, Any]]:
    columns_meta = quantized_payload.get("columns", {})
    rows = quantized_payload.get("rows", [])
    restored: list[dict[str, Any]] = []
    with _ref_decimal_ctx():
        for row in rows:
            new_row: dict[str, Any] = {}
            for key, val in row.items():
                meta = columns_meta.get(key)
                if meta is not None and isinstance(val, int) and not isinstance(val, bool):
                    bits = int(meta["bits"])
                    q_max = Decimal((1 << bits) - 1)
                    raw_min = meta["min"]
                    raw_max = meta["max"]
                    prefer_int = isinstance(raw_min, int) and isinstance(raw_max, int)
                    col_min = _ref_to_decimal(raw_min)
                    col_max = _ref_to_decimal(raw_max)
                    span = col_max - col_min
                    if span == 0:
                        new_row[key] = _ref_export_bound(col_min, prefer_int)
                    else:
                        d_val = col_min + (Decimal(val) / q_max) * span
                        new_row[key] = _ref_export_bound(d_val, prefer_int)
                else:
                    new_row[key] = val
            restored.append(new_row)
    return restored


def _assert_exact_value_and_type(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    assert actual == expected


# ---------------------------------------------------------------------------
# 1. CLIOutputOptimizer: RTK caching + precompiled regex parity
# ---------------------------------------------------------------------------


class TestCLIOutputOptimizerPerfAndParity:
    def test_rtk_which_cache_hits_and_invalidates_on_monkeypatch_and_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_opt_mod._RTK_WHICH_CACHE.clear()
        opt = CLIOutputOptimizer()
        long_text = "alpha line\n" * 80  # > 500 chars

        calls = 0

        def fake_which_none(cmd: str) -> str | None:
            nonlocal calls
            calls += 1
            return None

        monkeypatch.setattr(shutil, "which", fake_which_none)
        assert opt._try_rtk(long_text) is None
        assert opt._try_rtk(long_text) is None
        assert opt._try_rtk(long_text) is None
        assert calls == 1

        # Changing PATH invalidates cache even with same shutil.which function
        monkeypatch.setenv("PATH", "/custom/bin")
        assert opt._try_rtk(long_text) is None
        assert calls == 2

        # Changing PATHEXT invalidates cache
        monkeypatch.setenv("PATHEXT", ".COM;.EXE")
        assert opt._try_rtk(long_text) is None
        assert calls == 3

        # Monkeypatching shutil.which to a new function invalidates cache
        calls_v2 = 0

        def fake_which_hit(cmd: str) -> str | None:
            nonlocal calls_v2
            calls_v2 += 1
            return "/custom/bin/rtk"

        monkeypatch.setattr(shutil, "which", fake_which_hit)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(
                returncode=0, stdout=json.dumps({"filtered": "rtk-ok"})
            ),
        )
        assert opt._try_rtk(long_text) == "rtk-ok"
        assert opt._try_rtk(long_text) == "rtk-ok"
        assert calls_v2 == 1

        # Cache stays bounded to _RTK_WHICH_CACHE_MAX entries
        for i in range(15):
            monkeypatch.setenv("PATH", f"/custom/bin/{i}")
            opt._try_rtk(long_text)
        assert len(cli_opt_mod._RTK_WHICH_CACHE) <= cli_opt_mod._RTK_WHICH_CACHE_MAX

    def test_rtk_which_cache_thread_safe_ttl_and_pinned_binary_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_opt_mod._RTK_WHICH_CACHE.clear()
        opt = CLIOutputOptimizer()
        long_text = "alpha line\n" * 80

        now_val = [1000.0]
        monkeypatch.setattr(time, "monotonic", lambda: now_val[0])
        which_calls = 0

        def fake_which(cmd: str) -> str | None:
            nonlocal which_calls
            which_calls += 1
            return None if which_calls == 1 else "/opt/bin/rtk"

        captured_cmds: list[list[str]] = []

        def fake_run(cmd: list[str], **_kw: Any) -> SimpleNamespace:
            captured_cmds.append(cmd)
            return SimpleNamespace(returncode=0, stdout=json.dumps({"filtered": "pinned-ok"}))

        monkeypatch.setattr(shutil, "which", fake_which)
        monkeypatch.setattr(subprocess, "run", fake_run)

        # Negative lookup is cached until negative TTL expires
        assert opt._try_rtk(long_text) is None
        assert opt._try_rtk(long_text) is None
        assert which_calls == 1

        # Advance past 60s negative TTL -> re-probes and picks up newly installed binary
        now_val[0] += cli_opt_mod._RTK_WHICH_NEGATIVE_TTL_S + 1.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(opt._try_rtk, [long_text] * 16))
        assert all(r == "pinned-ok" for r in results)
        assert all(cmd == ["/opt/bin/rtk", "--json"] for cmd in captured_cmds)

    def test_detect_command_unicode_casefold_test_summary_parity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)
        opt = CLIOutputOptimizer()
        assert opt.detect_command("1 pa\u017f\u017fed, 2 pa\u017f\u017fed") == "test_output"
        assert opt.detect_command("1 fa\u0130led, 2 passed") == "test_output"

    def test_differential_all_strategies_byte_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _cmd: None)
        opt = CLIOutputOptimizer()

        pytest_failures = (
            "tests/test_a.py::test_one FAILED\n"
            "tests/test_a.py::test_two PASSED\n"
            "=== FAILURES ===\n"
            "___ test_one ___\n"
            ">   assert x == 1\n"
            "E   assert 2 == 1\n"
            "___ test_two ___\n"
            ">   assert y == 2\n"
            "E   AssertionError: boom\n"
            "=== short test summary info ===\n"
            "========================= 1 failed, 1 passed in 0.23s =========================\n"
        )
        git_diff_text = (
            "diff --git a/foo.py b/foo.py\n"
            "rename from old_foo.py\n"
            "--- a/old_foo.py\n"
            "+++ b/foo.py\n"
            "+line1\n+line2\n+line3\n+line4\n-old1\n"
        )
        git_status_text = (
            "On branch main\n"
            "\tmodified:   src/a.py\n"
            "\tnew file:   src/b.py\n"
            "\tdeleted:    src/c.py\n"
            "\trenamed:    src/d.py -> src/e.py\n"
        )
        docker_ps_text = (
            "CONTAINER ID   IMAGE          COMMAND       CREATED        STATUS          PORTS     NAMES\n"
            '1234567890ab   nginx:latest   "nginx -g"    2 hours ago    Up 2 hours      80/tcp    web\n'
        )
        docker_images_text = (
            "REPOSITORY   TAG       IMAGE ID       CREATED       SIZE\n"
            "python       3.14      abcdef123456   2 days ago    120MB\n"
        )
        lint_text = (
            "src/a.py:1:1: E501 line too long\n"
            "src/a.py:2:1: E501 line too long\n"
            "src/b.py:3:4: F401 unused import\n"
            "Found 3 errors.\n"
        )
        lint_json = json.dumps(
            [
                {"code": "F401", "message": "unused", "filename": "a.py", "fix": {"edit": []}},
                {"code": "E501", "message": "too long", "filename": "b.py"},
            ]
        )
        install_text = (
            "npm warn deprecated pkg@1.0.0\n"
            "\u280b installing\n"
            "npm timing idealTree completed in 12ms\n"
            "added 42 packages, and audited 43 packages in 2s\n"
            "found 0 vulnerabilities\n"
        )
        log_text = "\n".join(["[INFO] tick"] * 12 + ["[WARN] spike"])
        json_text = json.dumps({"ok": True, "items": [{"a": 1}, {"a": 2}], "meta": {"v": 2}})
        progress_text = "Start step\n  25%\n  \u280b  \n  ...  \nEnd step\n"
        tree_text = "├── src/alpha.py\n" "├── src/beta.py\n" "└── tests/test_alpha.py\n"

        cases: list[tuple[str, str | None, str, str]] = [
            (
                pytest_failures,
                None,
                "test_output",
                "FAILED test_one\n  >   assert x == 1\n  E   assert 2 == 1\n"
                "FAILED test_two\n  >   assert y == 2\n  E   AssertionError: boom\n"
                "========================= 1 failed, 1 passed in 0.23s =========================",
            ),
            (
                git_diff_text,
                None,
                "git_diff",
                "[foo.py] (+4 -1) (renamed from old_foo.py)\n"
                "  +line1\n  +line2\n  +line3\n  ... +2 more changes\n"
                "1 file changed, +4 insertions, -1 deletions",
            ),
            (
                git_status_text,
                None,
                "git_status",
                "Modified (1): src/a.py\nNew (1): src/b.py\n"
                "Deleted (1): src/c.py\nRenamed (1): src/d.py -> src/e.py",
            ),
            (
                docker_ps_text,
                None,
                "docker_output",
                "Docker containers (1):\n  web: nginx:latest [Up 2 hours]",
            ),
            (
                docker_images_text,
                None,
                "docker_output",
                "Docker images (1):\n  python:3.14 (120MB)",
            ),
            (
                lint_text,
                None,
                "lint_output",
                "Lint: 3 issues in 2 files\n\nBy rule:\n  E501 (2x): line too long\n"
                "  F401 (1x): unused import\n\nBy file:\n  src/a.py: 2 issues\n"
                "  src/b.py: 1 issues\nFound 3 errors.",
            ),
            (
                lint_json,
                "lint_output",
                "lint_output",
                "Lint: 2 issues in 2 files\n\nBy rule:\n  F401 (1x): unused\n"
                "  E501 (1x): too long\n\nBy file:\n  a.py: 1 issues\n  b.py: 1 issues\n\n"
                "1 auto-fixable (use --fix)",
            ),
            (
                install_text,
                None,
                "install_output",
                "npm warn deprecated pkg@1.0.0\n"
                "added 42 packages, and audited 43 packages in 2s\n"
                "found 0 vulnerabilities",
            ),
            (log_text, None, "log_output", "[INFO] tick (repeated 12 times)\n[WARN] spike"),
            (
                json_text,
                None,
                "json_output",
                "ok: True\nitems: [2 items, +1 more]\n    a: 1\nmeta: {1 keys}",
            ),
            (progress_text, "progress_output", "progress_output", "Start step\nEnd step"),
            (
                tree_text,
                None,
                "tree_output",
                "src/ (2 files): alpha.py, beta.py\ntests/test_alpha.py",
            ),
        ]

        for text, hint, expected_cmd, expected_filtered in cases:
            res = opt.filter(text, command_hint=hint)
            assert res.command_detected == expected_cmd
            assert res.filtered_text == expected_filtered

        # Verify no inline re.search/re.match/re.compile remains inside hot methods
        src_text = Path(cli_opt_mod.__file__).read_text(encoding="utf-8")
        detect_body = src_text.split("def detect_command(", 1)[1].split("def _strip_ansi(", 1)[0]
        assert not re.search(r"\bre\.(search|match|compile|findall)\b", detect_body)


# ---------------------------------------------------------------------------
# 2. code_budget_knapsack: vectorized DP + batch tiktoken parity
# ---------------------------------------------------------------------------


class TestCodeBudgetKnapsackPerfAndParity:
    def test_differential_fuzz_50_cases_against_scalar_dp(self) -> None:
        rng = random.Random(20261127)
        for case_idx in range(55):
            n_items = rng.randint(0, 16)
            items: list[CodeBlockItem] = []
            for i in range(n_items):
                # Include ties, zero-cost items, zero/negative importance, and varying costs
                cost = 0 if rng.random() < 0.15 else rng.randint(1, 25)
                score = rng.choice([0.0, -0.5, 0.25, 0.5, 0.5, 0.75, 1.0, 1.25])
                if rng.random() < 0.5:
                    score = round(rng.uniform(0.05, 2.0), 3)
                items.append(
                    CodeBlockItem(
                        block_id=f"b_{case_idx}_{i}",
                        kind="statement",
                        name=f"fn_{i}",
                        start_line=i + 1,
                        end_line=i + 1,
                        code=f"v_{i} = {i}",
                        token_cost=cost,
                        importance_score=score,
                    )
                )
            budget = rng.choice([-3, 0, 1, 5, 15, 40, 100])
            actual = solve_knapsack_exact(items, budget)
            expected = _scalar_knapsack_reference(items, budget)
            assert actual.selected_blocks == expected.selected_blocks
            assert actual.total_tokens == expected.total_tokens
            assert actual.total_value == expected.total_value
            assert actual.budget_tokens == expected.budget_tokens
            assert actual.reconstructed_code == expected.reconstructed_code

    def test_knapsack_dominance_fast_path_and_dp_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        items = [
            CodeBlockItem(
                f"b{i}", "statement", "f", i + 1, i + 1, f"x{i} = {i}", i + 2, (i % 4) + 0.5
            )
            for i in range(8)
        ]
        # Dominance fast-path: huge budget >= sum(pos_costs) finishes in O(N) without DP allocation
        huge_res = solve_knapsack_exact(items, 10_000_000)
        assert len(huge_res.selected_blocks) == len(items)
        assert huge_res.total_tokens == sum(b.token_cost for b in items)

        # Pathological DP grid exceeding _MAX_DP_CELLS raises ValueError cleanly
        monkeypatch.setattr(knapsack_mod, "_MAX_DP_CELLS", 10)
        with pytest.raises(ValueError, match="exceeds safety cap"):
            solve_knapsack_exact(items, 14)

    def test_partition_ast_blocks_honors_estimate_code_tokens_monkeypatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = "def f(a):\n    return a + 1\n"
        monkeypatch.setattr(knapsack_mod, "estimate_code_tokens", lambda code: 999)
        blocks = partition_ast_blocks(src, query="a")
        assert blocks
        assert all(b.token_cost == 999 for b in blocks)

        monkeypatch.undo()
        est = knapsack_mod._get_estimator()
        monkeypatch.setattr(est, "count_tokens", lambda code: 777)
        blocks2 = partition_ast_blocks(src, query="a")
        assert all(b.token_cost == 777 for b in blocks2)

    @pytest.mark.asyncio
    async def test_compress_codebase_rejects_excessive_budget_tokens(self, tmp_path: Path) -> None:
        from src.handlers.compression_handlers_code_flow import handle_compress_codebase

        ctx = {
            "path_validator": SimpleNamespace(validate=lambda p: str(tmp_path)),
        }
        raw = await handle_compress_codebase(
            ctx, {"directory": str(tmp_path), "budget_tokens": 200_001}
        )
        payload = json.loads(raw)
        assert payload["status"] == "error"
        assert "200000" in payload["error"]

    def test_partition_ast_blocks_batch_tokens_matches_per_block(self) -> None:
        sample = Path(knapsack_mod.__file__).read_text(encoding="utf-8")
        blocks = partition_ast_blocks(sample, query="knapsack budget tokens")
        assert len(blocks) > 10
        for b in blocks:
            assert b.token_cost == estimate_code_tokens(b.code)

        # SyntaxError fallback path also preserves identical token costs
        bad_syntax = "\n".join(f"x_{i} = {i} +" for i in range(25))
        syn_blocks = partition_ast_blocks(bad_syntax, query="x_1")
        assert len(syn_blocks) == 3
        for b in syn_blocks:
            assert b.token_cost == estimate_code_tokens(b.code)

        # compress_code_with_knapsack respects real file budget
        comp = compress_code_with_knapsack(sample, budget_tokens=150, query="solve_knapsack_exact")
        assert comp.total_tokens <= 150
        assert comp.selected_blocks

    def test_partition_ast_blocks_sequential_encode_fallback_and_none_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = "def f(a):\n    return a + 1\n"
        expected = partition_ast_blocks(src, query="a")

        est = knapsack_mod._get_estimator()

        # Verify tight sequential encoder.encode path is used (never encode_batch)
        encode_calls: list[str] = []

        class _SequentialEncoder:
            def encode(self, text: str) -> list[int]:
                encode_calls.append(text)
                return est._encoder.encode(text) if est._encoder else [1] * max(1, len(text) // 4)

            def encode_batch(self, texts: list[str]) -> list[list[int]]:
                raise AssertionError("encode_batch must not be called")

        monkeypatch.setattr(
            knapsack_mod,
            "_get_estimator",
            lambda: SimpleNamespace(
                _encoder=_SequentialEncoder(),
                count_tokens=est.count_tokens,
            ),
        )
        assert partition_ast_blocks(src, query="a") == expected
        assert len(encode_calls) == len(expected)

        # Verify encode() exception falls back cleanly to estimate_code_tokens
        class _BrokenEncoder:
            def encode(self, text: str) -> list[int]:
                raise RuntimeError("boom")

        monkeypatch.setattr(
            knapsack_mod,
            "_get_estimator",
            lambda: SimpleNamespace(
                _encoder=_BrokenEncoder(),
                count_tokens=est.count_tokens,
            ),
        )
        assert partition_ast_blocks(src, query="a") == expected

        # Verify encoder=None falls back cleanly to estimate_code_tokens
        monkeypatch.setattr(
            knapsack_mod,
            "_get_estimator",
            lambda: SimpleNamespace(_encoder=None, count_tokens=lambda t: len(t) // 4),
        )
        none_blocks = partition_ast_blocks(src, query="a")
        assert [b.token_cost for b in none_blocks] == [
            max(1, len(b.code) // 4) for b in none_blocks
        ]


# ---------------------------------------------------------------------------
# 3. CacheRouter: cached policy + inverted BM25 postings parity
# ---------------------------------------------------------------------------


class TestCacheRouterPerfAndParity:
    def _build_catalog(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "name": "read_file",
                "description": "Read a file from local workspace.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
            {
                "name": "bash",
                "description": "Execute shell command.",
                "inputSchema": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
            {
                "name": "auth_login",
                "description": "Authenticate user session token.",
                "inputSchema": {"type": "object", "properties": {"token": {"type": "string"}}},
                "x-effects": ["authenticated"],
            },
            {
                "name": "deploy_prod",
                "description": "Deploy invoice service to production.",
                "inputSchema": {"type": "object", "properties": {"env": {"type": "string"}}},
                "x-preconditions": ["authenticated"],
            },
            {
                "name": "danger_nuke",
                "description": "Delete invoice database.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        for i in range(60):
            tools.append(
                {
                    "name": f"billing_tool_{i}" if i % 7 == 0 else f"svc_op_{i}",
                    "description": (
                        f"Handle billing invoice ledger batch {i}."
                        if i % 5 == 0
                        else f"Execute subsystem task {i} for queue {i % 4}."
                    ),
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "target_id": {"type": "string"},
                            "invoice_ref": {"type": "string"},
                        },
                    },
                }
            )
        return tools

    def test_differential_35_queries_against_naive_full_scan(self) -> None:
        tools = self._build_catalog()
        policy = RouterPolicy(blacklist={"danger_nuke"})
        router = CacheRouter(tools, policy=policy)

        queries = [
            "",
            "   ",
            "invoice",
            "invoice invoice billing",
            "billing ledger batch",
            "subsystem queue 2",
            "nonexistent_xyz_term",
            "deploy production invoice",
            "read file path",
            "shell bash command",
            "authenticate session token",
            "target_id invoice_ref",
            "svc op 13",
            "billing_tool_14",
            "queue 0 queue 0",
        ]

        # Run across pre-auth and post-auth, with/without preconditions, various top_k & yellowlists
        for step in range(2):
            if step == 1:
                router.commit_tool_effects("auth_login", succeeded=True)
            for idx, q in enumerate(queries):
                top_k = (idx % 7) + 1
                respect_pre = (idx % 2) == 0
                yl = (
                    None
                    if idx % 3 == 0
                    else ({"read_file", "bash", "billing_tool_0"} if idx % 3 == 1 else set())
                )
                actual = router.filter_tools(
                    query=q,
                    top_k=top_k,
                    yellowlist=yl,
                    respect_preconditions=respect_pre,
                )
                expected = _naive_filter_tools_reference(
                    router,
                    query=q,
                    top_k=top_k,
                    yellowlist=yl,
                    respect_preconditions=respect_pre,
                )
                assert [t["name"] for t in actual] == [t["name"] for t in expected]
                assert actual == expected

        # Mutate policy (whitelist + expanded blacklist) and verify automatic cache invalidation
        router._policy = RouterPolicy(
            blacklist={"danger_nuke", "svc_op_1"},
            whitelist={"billing_tool_0", "billing_tool_7", "deploy_prod"},
        )
        for q in ("", "billing invoice", "deploy production", "svc op 1"):
            actual = router.filter_tools(query=q, top_k=5)
            expected = _naive_filter_tools_reference(router, query=q, top_k=5)
            assert [t["name"] for t in actual] == [t["name"] for t in expected]
            assert actual == expected

    def test_atomic_policy_cache_tuple_publication_under_concurrent_mutation(self) -> None:
        tools = self._build_catalog()
        router = CacheRouter(tools, policy=RouterPolicy(blacklist={"danger_nuke"}))
        assert isinstance(router._policy_cache, tuple) and len(router._policy_cache) == 2

        policies = [
            RouterPolicy(blacklist={"danger_nuke"}),
            RouterPolicy(whitelist={"billing_tool_0", "billing_tool_7"}),
            RouterPolicy(
                blacklist={"danger_nuke", "bash"}, whitelist={"deploy_prod", "auth_login"}
            ),
        ]
        valid_sets_by_key: dict[tuple[Any, Any, Any], set[str]] = {}
        expected_ordered_results: list[list[str]] = []
        for pol in policies:
            key = router._policy_cache_key(pol)
            allowed = {
                t["name"]
                for t in router._tools
                if not pol.is_blacklisted(t["name"])
                and (not pol.whitelist or pol.is_whitelisted(t["name"]))
            }
            valid_sets_by_key[key] = allowed
            expected_ordered_results.append(
                [
                    t["name"]
                    for t in CacheRouter(tools, policy=pol).filter_tools(
                        query="billing", top_k=3, yellowlist=router._default_yellowlist
                    )
                ]
            )
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer() -> None:
            try:
                for i in range(150):
                    router._policy = policies[i % len(policies)]
                    time.sleep(0)
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)
            finally:
                stop.set()

        def reader() -> None:
            try:
                while not stop.is_set():
                    res = router.filter_tools(query="billing", top_k=3)
                    assert isinstance(res, list)
                    ordered_names = [t["name"] for t in res]
                    assert ordered_names in expected_ordered_results
                    ckey, celigible = router._policy_cache
                    assert ckey in valid_sets_by_key
                    assert {t["name"] for _, t in celigible} == valid_sets_by_key[ckey]
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        wt = threading.Thread(target=writer)
        rts = [threading.Thread(target=reader) for _ in range(4)]
        for t in rts:
            t.start()
        wt.start()
        wt.join()
        for t in rts:
            t.join()
        assert not errors


# ---------------------------------------------------------------------------
# 4. toon_serializer: quantization / dequantization exact bit parity
# ---------------------------------------------------------------------------


class TestToonQuantizationPerfAndParity:
    def test_quantize_dequantize_bit_identical_parity(self) -> None:
        datasets: list[list[dict[str, Any]]] = [
            [],
            [{"a": 1, "b": 1, "flag": True, "label": "only"}],
            [
                {
                    "id": 9007199254740992,
                    "f": -1e308,
                    "d": Decimal("1.234567890123456789"),
                    "b": False,
                    "s": "x",
                },
                {
                    "id": 9007199254740993,
                    "f": 0.0,
                    "d": Decimal("2.345678901234567890"),
                    "b": True,
                    "s": "y",
                },
                {
                    "id": 9007199254740995,
                    "f": 1e308,
                    "d": Decimal("1.234567890123456789"),
                    "b": None,
                    "s": "z",
                },
            ],
            [
                {"mixed": 9007199254740993, "const": 42, "opt": None},
                {"mixed": 9007199254740994.0, "const": 42, "opt": 12.5},
                {"mixed": 9007199254740996, "const": 42, "opt": 12.5},
            ],
        ]
        # Add deterministic random tabular batches with repeated values (LUT hits)
        rng = random.Random(20261127)
        big_rows = []
        for i in range(120):
            big_rows.append(
                {
                    "row": f"r{i}",
                    "score": round(rng.choice([0.1, 0.25, 0.5, 0.75, 0.9999]), 4),
                    "count": rng.choice([0, 10, 20, 30, 100]),
                    "const_f": 3.14,
                    "flag": bool(i % 2),
                    "maybe": None if i % 4 == 0 else rng.uniform(-50.0, 50.0),
                }
            )
        datasets.append(big_rows)

        for rows in datasets:
            for bits in (2, 4, 8, 12, 16):
                actual_q = quantize_numeric_columns(rows, bits=bits)
                expected_q = _ref_quantize(rows, bits=bits)
                assert actual_q["bits"] == expected_q["bits"]
                assert list(actual_q["columns"].keys()) == list(expected_q["columns"].keys())
                for col in actual_q["columns"]:
                    for k in ("min", "max", "bits", "eps_max"):
                        _assert_exact_value_and_type(
                            actual_q["columns"][col][k], expected_q["columns"][col][k]
                        )
                assert actual_q["rows"] == expected_q["rows"]

                actual_d = dequantize_numeric_columns(actual_q)
                expected_d = _ref_dequantize(expected_q)
                assert len(actual_d) == len(expected_d)
                for a_row, e_row in zip(actual_d, expected_d):
                    assert list(a_row.keys()) == list(e_row.keys())
                    for k in a_row:
                        _assert_exact_value_and_type(a_row[k], e_row[k])

    def test_quantize_preserves_first_seen_column_order_with_leading_none(self) -> None:
        rows = [{"a": None, "b": 1}, {"a": 2}]
        actual = quantize_numeric_columns(rows)
        expected = _ref_quantize(rows)
        assert list(actual["columns"].keys()) == ["a", "b"]
        assert list(actual["columns"].keys()) == list(expected["columns"].keys())
        assert actual["rows"] == expected["rows"]

    @pytest.mark.parametrize(
        "bad_row",
        [
            {"x": math.nan},
            {"x": math.inf},
            {"x": -math.inf},
            {"x": Decimal("NaN")},
            {"x": Decimal("Infinity")},
        ],
    )
    def test_quantize_rejects_non_finite_values(self, bad_row: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="Non-finite"):
            quantize_numeric_columns([{"x": 1.0}, bad_row])
