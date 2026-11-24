"""CL 3b tests for FlowPlan CFG/DFG summary and delta verifier."""

from __future__ import annotations

import json
import time

import pytest

from src.code_flow_graph import extract_code_flow, verify_code_flow_delta
from src.handlers.mcp_core import route_tool_call, setup_mcp_tools

BASE_CODE = """
def validate_and_score(items, threshold=0.5):
    total = 0.0
    for item in items:
        if item > threshold:
            total += normalize(item)
    try:
        avg = total / len(items)
    except ZeroDivisionError:
        avg = 0.0
    if avg > 0:
        return avg
    return 0.0
"""


def test_extract_code_flow_captures_cfg_dfg_and_latency():
    start = time.perf_counter()
    flow = extract_code_flow(BASE_CODE)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 5.0
    assert "validate_and_score" in flow
    fn = flow["validate_and_score"]
    assert fn.branches == 2
    assert fn.loops == 1
    assert fn.try_blocks == 1
    assert fn.returns == 2
    assert "normalize" in fn.calls
    assert "len" in fn.calls
    assert "items" in fn.defs
    assert "total" in fn.defs
    assert "threshold" in fn.uses
    assert fn.unreachable_lines == []


def test_verify_code_flow_delta_detects_missing_branch_removed_return_dropped_call():
    mutated = """
def validate_and_score(items, threshold=0.5):
    total = 0.0
    for item in items:
        total += item
    try:
        avg = total / len(items)
    except ZeroDivisionError:
        avg = 0.0
    if avg > 0:
        return avg
"""
    report = verify_code_flow_delta(BASE_CODE, mutated)
    assert "validate_and_score" in report.changed_functions
    assert "missing_branch:validate_and_score" in report.missing_expected_changes
    assert "removed_return:validate_and_score" in report.missing_expected_changes
    assert "dropped_call:validate_and_score:normalize" in report.missing_expected_changes
    assert report.summary_tokens > 0


def test_verify_code_flow_delta_detects_unreached_edits():
    mutated = """
def validate_and_score(items, threshold=0.5):
    total = 0.0
    for item in items:
        if item > threshold:
            total += normalize(item)
    try:
        avg = total / len(items)
    except ZeroDivisionError:
        avg = 0.0
    if avg > 0:
        return avg
        dead_var = avg * 2
    return 0.0
    after_final_return = 1
"""
    report = verify_code_flow_delta(BASE_CODE, mutated)
    assert len(report.unreached_edits) == 2
    assert any(entry.startswith("validate_and_score:L") for entry in report.unreached_edits)


def test_verify_code_flow_delta_with_expected_changes_and_syntax_error():
    report = verify_code_flow_delta(
        BASE_CODE,
        BASE_CODE,
        expected_changes={"validate_and_score": {"min_branches": 3, "required_calls": ["audit"]}},
    )
    assert "expected_branch:validate_and_score" in report.missing_expected_changes
    assert "expected_call:validate_and_score:audit" in report.missing_expected_changes

    with pytest.raises(ValueError, match="Invalid Python syntax"):
        extract_code_flow("def broken(:")


@pytest.mark.asyncio
async def test_mcp_verify_code_flow_delta_and_budget_tokens_wiring():
    tools = {t.name: t for t in setup_mcp_tools()}
    assert "verify_code_flow_delta" in tools
    assert tools["verify_code_flow_delta"].inputSchema.get("additionalProperties") is False
    assert "budget_tokens" in tools["compress_codebase"].inputSchema["properties"]

    raw_ok = await route_tool_call(
        "verify_code_flow_delta",
        {"before_code": BASE_CODE, "after_code": BASE_CODE},
        context={},
    )
    payload_ok = json.loads(raw_ok)
    assert payload_ok["status"] == "success"
    assert payload_ok["missing_expected_changes"] == []
    assert payload_ok["unreached_edits"] == []

    raw_bad = await route_tool_call(
        "verify_code_flow_delta",
        {"before_code": BASE_CODE, "after_code": BASE_CODE, "extra_key": 1},
        context={},
    )
    payload_bad = json.loads(raw_bad)
    assert payload_bad["status"] == "error"
    assert "Unknown argument" in payload_bad["error"]

    raw_bad_budget = await route_tool_call(
        "compress_codebase",
        {"directory": ".", "budget_tokens": 0},
        context={},
    )
    payload_bad_budget = json.loads(raw_bad_budget)
    assert payload_bad_budget["status"] == "error"
    assert "budget_tokens" in payload_bad_budget["error"]


def test_extract_code_flow_does_not_recurse_into_nested_scopes():
    code = """
def outer(x):
    def inner(y):
        if y > 0:
            return y + 1
        return y - 1
    class Helper:
        def method(self, z):
            if z:
                return z
            return 0
    return inner(x)
"""
    flow = extract_code_flow(code)
    assert "outer" in flow
    assert "outer.inner" in flow
    assert "outer.Helper.method" in flow
    assert flow["outer"].branches == 0
    assert flow["outer"].returns == 1
    assert flow["outer"].return_expressions == ["inner(x)"]
    assert flow["outer.inner"].branches == 1
    assert flow["outer.Helper.method"].branches == 1


def test_exhaustive_branch_termination_and_preexisting_unreachable_lines():
    code_with_exhaustive_dead = """
def check(x):
    if x > 0:
        return 1
    else:
        raise ValueError("bad")
    dead_after_if_else = 42
"""
    flow = extract_code_flow(code_with_exhaustive_dead)
    assert len(flow["check"].unreachable_lines) == 1

    # Identical before/after must produce zero unreached_edits even with pre-existing dead code.
    same_report = verify_code_flow_delta(code_with_exhaustive_dead, code_with_exhaustive_dead)
    assert same_report.unreached_edits == []

    # Newly introduced unreachable line IS reported.
    with_new_dead = code_with_exhaustive_dead + "\n    second_dead = 99\n"
    new_report = verify_code_flow_delta(code_with_exhaustive_dead, with_new_dead)
    assert len(new_report.unreached_edits) == 1
    assert new_report.unreached_edits[0].startswith("check:L")


def test_branch_swap_and_dfg_expression_swap_detected_within_latency_budget():
    before = """
def calc(a, b):
    if a > b:
        return a - b
    return b - a
"""
    swapped_cond = """
def calc(a, b):
    if a < b:
        return a - b
    return b - a
"""
    swapped_expr = """
def calc(a, b):
    if a > b:
        return b - a
    return a - b
"""
    start = time.perf_counter()
    rep_cond = verify_code_flow_delta(before, swapped_cond)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 5.0
    assert "calc" in rep_cond.changed_functions

    rep_expr = verify_code_flow_delta(before, swapped_expr)
    assert "calc" in rep_expr.changed_functions


def test_verify_code_flow_delta_strict_expected_changes_validation():
    bad_cases = [
        {"validate_and_score": {"min_branches": True}},
        {"validate_and_score": {"min_branches": -1}},
        {"validate_and_score": {"min_branches": "2"}},
        {"validate_and_score": {"required_calls": "normalize"}},
        {"validate_and_score": {"required_calls": [{}]}},
        {"validate_and_score": {"required_returns": [""]}},
        {"validate_and_score": {"required_defs": None}},
        "validate_and_score",
    ]
    for bad in bad_cases:
        with pytest.raises(ValueError):
            verify_code_flow_delta(BASE_CODE, BASE_CODE, expected_changes=bad)


def test_multiline_unreachable_edit_and_unknown_constraint_key():
    before = "def f(x):\n" "    return 0\n" "    dead = (\n" "        1\n" "    )\n"
    after = "def f(x):\n" "    return 0\n" "    dead = (\n" "        2\n" "    )\n"
    report = verify_code_flow_delta(before, after)
    assert "f" in report.changed_functions
    assert any(item.startswith("f:L") for item in report.unreached_edits)

    with pytest.raises(ValueError, match="Unknown constraint keys"):
        verify_code_flow_delta(before, before, expected_changes={"f": {"min_branch": 99}})
