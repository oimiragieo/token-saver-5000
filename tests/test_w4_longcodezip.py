"""CL 1 & CL 3a tests for LongCodeZip exact 0/1 knapsack code compression."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.code_budget_knapsack import (
    CodeBlockItem,
    compress_code_with_knapsack,
    partition_ast_blocks,
    solve_knapsack_exact,
)
from src.code_compressor import CodeSemanticCompressor
from src.handlers.mcp_core import route_tool_call
from src.path_validator import PathValidator
from src.token_estimation import TokenEstimator

SAMPLE_CODE = '''
import os
import math

def compute_discount(price: float, vip: bool = False) -> float:
    """Compute discounted price."""
    if price < 0:
        raise ValueError("negative price")
    discount = 0.0
    for step in range(3):
        discount += 0.05 * step
    try:
        factor = 1.0 - discount
    except ZeroDivisionError:
        factor = 1.0
    if vip:
        return price * factor * 0.8
    return price * factor

class InvoicePrinter:
    def render(self, total: float) -> str:
        return f"Total: {total:.2f}"
'''


def _brute_force_best_value(items: list[CodeBlockItem], budget: int) -> float:
    best = 0.0
    n = len(items)
    for mask in range(1 << n):
        cost = 0
        val = 0.0
        for i in range(n):
            if mask & (1 << i):
                cost += items[i].token_cost
                val += items[i].importance_score
        if cost <= budget and val > best:
            best = val
    return best


def test_knapsack_exact_matches_brute_force():
    items = [
        CodeBlockItem("b0", "import", "os", 1, 1, "import os", 3, 0.35),
        CodeBlockItem("b1", "signature", "f", 3, 3, "def f():", 4, 0.55),
        CodeBlockItem("b2", "branch", "f", 4, 5, "if x:\n return 1", 7, 0.92),
        CodeBlockItem("b3", "loop", "f", 6, 7, "for i in x: pass", 6, 0.61),
        CodeBlockItem("b4", "return", "f", 8, 8, "return 0", 3, 0.74),
        CodeBlockItem("b5", "statement", "f", 9, 9, "x", 2, 0.25),
    ]
    for budget in (0, 3, 5, 8, 10, 15, 25):
        res = solve_knapsack_exact(items, budget)
        expected = _brute_force_best_value(items, budget)
        assert res.total_tokens <= budget
        assert abs(res.total_value - expected) < 1e-12


def test_knapsack_handles_zero_budget_negative_budget_and_zero_cost_items():
    items = [
        CodeBlockItem("z1", "import", "z1", 1, 1, "", 0, 0.4),
        CodeBlockItem("p1", "return", "p1", 2, 2, "return 1", 4, 0.9),
    ]
    neg = solve_knapsack_exact(items, -5)
    assert neg.selected_blocks == []
    assert neg.total_tokens == 0
    assert neg.total_value == 0.0

    zero = solve_knapsack_exact(items, 0)
    assert zero.total_tokens == 0
    assert [b.block_id for b in zero.selected_blocks] == ["z1"]
    assert abs(zero.total_value - 0.4) < 1e-12

    with pytest.raises(ValueError, match="non-negative"):
        solve_knapsack_exact([CodeBlockItem("bad", "block", "bad", 1, 1, "x", -1, 0.5)], 10)


def test_partition_ast_blocks_covers_kinds_and_query_boost():
    blocks = partition_ast_blocks(SAMPLE_CODE, query="vip discount")
    kinds = {b.kind for b in blocks}
    assert {"import", "signature", "branch", "loop", "try", "return"}.issubset(kinds)

    vip_blocks = [b for b in blocks if "vip" in b.code.lower()]
    unrelated_blocks = [b for b in blocks if "invoiceprinter" in b.name.lower()]
    assert vip_blocks and unrelated_blocks
    assert max(b.importance_score for b in vip_blocks) > min(
        b.importance_score for b in unrelated_blocks
    )


def test_compress_code_with_knapsack_respects_budget():
    estimator = TokenEstimator()
    for budget in (10, 25, 60, 120):
        res = compress_code_with_knapsack(SAMPLE_CODE, budget_tokens=budget, query="discount")
        actual_tokens = estimator.count_tokens(res.reconstructed_code)
        assert actual_tokens <= budget
        assert res.total_tokens == actual_tokens
        assert res.total_tokens <= budget
        assert res.budget_tokens == budget
        assert res.total_value >= 0.0
        # Selected blocks stay in source order
        lines = [b.start_line for b in res.selected_blocks]
        assert lines == sorted(lines)


def test_knapsack_trims_when_joined_tokens_exceed_budget():
    estimator = TokenEstimator()
    src = "alpha_one = 1\nbeta_two = 2\ngamma_three = 3"
    res = compress_code_with_knapsack(src, budget_tokens=6)
    actual_tokens = estimator.count_tokens(res.reconstructed_code) if res.reconstructed_code else 0
    assert actual_tokens <= 6
    assert res.total_tokens == actual_tokens


@pytest.mark.asyncio
async def test_mcp_compress_codebase_respects_budget_tokens(tmp_path: Path):
    f1 = tmp_path / "alpha.py"
    f2 = tmp_path / "beta.py"
    f1.write_text(SAMPLE_CODE, encoding="utf-8")
    f2.write_text(SAMPLE_CODE.replace("compute_discount", "compute_tax"), encoding="utf-8")

    estimator = TokenEstimator()
    validator = PathValidator(allowed_base_dirs=[str(tmp_path)])
    raw = await route_tool_call(
        "compress_codebase",
        {"directory": str(tmp_path), "budget_tokens": 40, "query": "discount"},
        context={"path_validator": validator},
    )
    payload = json.loads(raw)
    assert payload["status"] == "success"
    assert payload["budget_tokens"] == 40
    assert payload["compressed_tokens"] <= 40
    assert estimator.count_tokens(payload["compressed_code"]) <= 40
    assert len(payload["compressed_code"]) > 0


def test_code_compressor_wires_optional_budget_tokens():
    with patch("src.code_compressor.EmbeddingManager") as mock_mgr_cls:
        mock_mgr = Mock()
        mock_embedder = Mock()
        mock_embedder.encode.side_effect = lambda texts, **_kw: np.ones(
            (len(texts), 8), dtype=np.float32
        )
        mock_mgr.get_code_embedder.return_value = mock_embedder
        mock_mgr_cls.return_value = mock_mgr
        comp = CodeSemanticCompressor()

    comp.ingest_code_file(SAMPLE_CODE, file_id="inv", filepath="inv.py")
    default_skel = comp.generate_code_skeleton("inv")
    assert "=== CODE SKELETON: inv ===" in default_skel

    budgeted = comp.generate_code_skeleton("inv", budget_tokens=30, query="discount")
    assert "=== CODE SKELETON: inv ===" not in budgeted
    assert len(budgeted) > 0


def test_solve_knapsack_exact_unit_blocks_budget_5():
    items = [
        CodeBlockItem(
            block_id=f"b{i}",
            kind="statement",
            name="f",
            start_line=i + 1,
            end_line=i + 1,
            code=f"x_{i} = {i}",
            token_cost=1,
            importance_score=1.0,
        )
        for i in range(5)
    ]
    res = solve_knapsack_exact(items, budget_tokens=5)
    assert len(res.selected_blocks) == 5
    assert res.total_value == pytest.approx(5.0, abs=1e-12)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_max_files", [True, False, 0, -3, 101, "10", 5.5])
async def test_mcp_compress_codebase_rejects_invalid_max_files(tmp_path: Path, bad_max_files):
    validator = PathValidator(allowed_base_dirs=[str(tmp_path)])
    raw = await route_tool_call(
        "compress_codebase",
        {"directory": str(tmp_path), "max_files": bad_max_files},
        context={"path_validator": validator},
    )
    payload = json.loads(raw)
    assert payload["status"] == "error"
    assert "max_files" in payload["error"]
