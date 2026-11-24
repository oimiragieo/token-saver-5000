"""FlowPlan per-function CFG/DFG summary and delta verifier."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from .code_budget_knapsack import estimate_code_tokens

_TRY_TYPES: Tuple[type, ...] = tuple(
    t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None
)
_TERMINAL_TYPES: Tuple[type, ...] = (ast.Return, ast.Raise, ast.Break, ast.Continue)


@dataclass(frozen=True)
class FunctionFlowSummary:
    """Compact CFG/DFG summary for a single Python function or method."""

    name: str
    branches: int
    loops: int
    try_blocks: int
    calls: List[str]
    returns: int
    defs: List[str]
    uses: List[str]
    unreachable_lines: List[int]
    branch_conditions: List[str] = field(default_factory=list)
    call_expressions: List[str] = field(default_factory=list)
    return_expressions: List[str] = field(default_factory=list)
    def_use_edges: List[str] = field(default_factory=list)
    unreachable_stmts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlowDeltaReport:
    """Delta verification report comparing before/after code flow summaries."""

    added_functions: List[str]
    removed_functions: List[str]
    changed_functions: List[str]
    missing_expected_changes: List[str]
    unreached_edits: List[str]
    summary_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts = [func.attr]
        cur = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return "<call>"


_NESTED_SCOPE_TYPES: Tuple[type, ...] = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _iter_enclosing_nodes(root: ast.AST) -> Iterable[ast.AST]:
    """Iterate AST nodes within ``root`` without descending into nested scopes."""
    stack: List[ast.AST] = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(list(ast.iter_child_nodes(node))):
            if isinstance(child, _NESTED_SCOPE_TYPES):
                continue
            stack.append(child)


def _collect_expr_names(expr: Optional[ast.AST], ctx_type: type) -> List[str]:
    if expr is None:
        return []
    return [
        sub.id
        for sub in _iter_enclosing_nodes(expr)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ctx_type)
    ]


def _find_unreachable_in_stmts(stmts: Sequence[ast.stmt], out: Dict[int, str]) -> bool:
    """Record unreachable statement ASTs by line and return True if ``stmts`` always terminates."""
    terminated = False
    for stmt in stmts:
        if terminated:
            out[stmt.lineno] = ast.unparse(stmt)
            continue
        if isinstance(stmt, _TERMINAL_TYPES):
            terminated = True
            continue
        if isinstance(stmt, ast.If):
            body_term = _find_unreachable_in_stmts(stmt.body, out)
            else_term = bool(stmt.orelse) and _find_unreachable_in_stmts(stmt.orelse, out)
            if body_term and else_term:
                terminated = True
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            _find_unreachable_in_stmts(stmt.body, out)
            _find_unreachable_in_stmts(stmt.orelse, out)
        elif isinstance(stmt, _TRY_TYPES):
            body_term = _find_unreachable_in_stmts(stmt.body, out)
            handlers_term = all(_find_unreachable_in_stmts(h.body, out) for h in stmt.handlers)
            else_term = (not stmt.orelse) or _find_unreachable_in_stmts(stmt.orelse, out)
            final_term = bool(stmt.finalbody) and _find_unreachable_in_stmts(stmt.finalbody, out)
            if final_term or (body_term and handlers_term and else_term):
                terminated = True
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            if _find_unreachable_in_stmts(stmt.body, out):
                terminated = True
    return terminated


def _summarize_function(
    fn_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    qualname: str,
) -> FunctionFlowSummary:
    branches = 0
    loops = 0
    try_blocks = 0
    returns = 0
    calls: Set[str] = set()
    defs: Set[str] = set()
    uses: Set[str] = set()
    branch_conditions: List[str] = []
    call_expressions: List[str] = []
    return_expressions: List[str] = []
    def_use_edges: List[str] = []

    for arg in (
        list(fn_node.args.posonlyargs) + list(fn_node.args.args) + list(fn_node.args.kwonlyargs)
    ):
        defs.add(arg.arg)
    if fn_node.args.vararg:
        defs.add(fn_node.args.vararg.arg)
    if fn_node.args.kwarg:
        defs.add(fn_node.args.kwarg.arg)

    for node in _iter_enclosing_nodes(fn_node):
        if isinstance(node, (ast.If, ast.IfExp)):
            branches += 1
            branch_conditions.append(ast.unparse(node.test))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            loops += 1
        elif isinstance(node, _TRY_TYPES):
            try_blocks += 1
        elif isinstance(node, ast.Return):
            returns += 1
            return_expressions.append(ast.unparse(node.value) if node.value is not None else "None")
            for u in _collect_expr_names(node.value, ast.Load):
                def_use_edges.append(f"{u}-><return>")
        elif isinstance(node, ast.Call):
            calls.add(_call_name(node))
            call_expressions.append(ast.unparse(node))
        elif isinstance(node, ast.Assign):
            targets = [t for tgt in node.targets for t in _collect_expr_names(tgt, ast.Store)]
            loaded = _collect_expr_names(node.value, ast.Load)
            for tgt_name in targets:
                for src_name in loaded:
                    def_use_edges.append(f"{src_name}->{tgt_name}")
        elif isinstance(node, ast.AugAssign):
            targets = _collect_expr_names(node.target, ast.Store)
            loaded = _collect_expr_names(node.value, ast.Load)
            for tgt_name in targets:
                def_use_edges.append(f"{tgt_name}->{tgt_name}")
                for src_name in loaded:
                    def_use_edges.append(f"{src_name}->{tgt_name}")
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = _collect_expr_names(node.target, ast.Store)
            loaded = _collect_expr_names(node.value, ast.Load)
            for tgt_name in targets:
                for src_name in loaded:
                    def_use_edges.append(f"{src_name}->{tgt_name}")
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                defs.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                uses.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defs.add(node.name)

    unreachable: Dict[int, str] = {}
    _find_unreachable_in_stmts(fn_node.body, unreachable)
    sorted_unreach = sorted(unreachable)

    return FunctionFlowSummary(
        name=qualname,
        branches=branches,
        loops=loops,
        try_blocks=try_blocks,
        calls=sorted(calls),
        returns=returns,
        defs=sorted(defs),
        uses=sorted(uses),
        unreachable_lines=sorted_unreach,
        branch_conditions=branch_conditions,
        call_expressions=call_expressions,
        return_expressions=return_expressions,
        def_use_edges=def_use_edges,
        unreachable_stmts=[unreachable[ln] for ln in sorted_unreach],
    )


def _walk_functions(
    stmts: Sequence[ast.stmt],
    prefix: str,
    out: Dict[str, FunctionFlowSummary],
) -> None:
    for stmt in stmts:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual = f"{prefix}.{stmt.name}" if prefix else stmt.name
            out[qual] = _summarize_function(stmt, qual)
            _walk_functions(stmt.body, qual, out)
        elif isinstance(stmt, ast.ClassDef):
            qual = f"{prefix}.{stmt.name}" if prefix else stmt.name
            _walk_functions(stmt.body, qual, out)


def extract_code_flow(source_code: str) -> Dict[str, FunctionFlowSummary]:
    """Extract per-function CFG/DFG summaries from Python source code."""
    if not source_code or not source_code.strip():
        return {}
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        raise ValueError(f"Invalid Python syntax: {exc}") from exc

    summaries: Dict[str, FunctionFlowSummary] = {}
    _walk_functions(tree.body, "", summaries)
    return summaries


def _validate_non_negative_int(value: Any, field_name: str, fn_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{field_name} for '{fn_name}' must be a non-negative integer, got {value!r}"
        )
    return value


def _validate_str_collection(value: Any, field_name: str, fn_name: str) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            f"{field_name} for '{fn_name}' must be a list, tuple, or set of non-empty strings"
        )
    validated: List[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{field_name} entries for '{fn_name}' must be non-empty strings, got {item!r}"
            )
        validated.append(item)
    return validated


_ALLOWED_CONSTRAINT_KEYS = frozenset(
    {"min_branches", "min_returns", "required_calls", "required_returns", "required_defs"}
)


def _validate_expected_changes(
    expected_changes: Optional[Union[Sequence[str], Dict[str, Any]]],
) -> None:
    if expected_changes is None:
        return
    if isinstance(expected_changes, dict):
        for fn_name, spec in expected_changes.items():
            if not isinstance(fn_name, str) or not fn_name.strip():
                raise ValueError(
                    f"expected_changes keys must be non-empty strings, got {fn_name!r}"
                )
            if isinstance(spec, dict):
                unknown = sorted(set(spec.keys()) - _ALLOWED_CONSTRAINT_KEYS)
                if unknown:
                    raise ValueError(f"Unknown constraint keys for '{fn_name}': {unknown}")
                if "min_branches" in spec:
                    _validate_non_negative_int(spec["min_branches"], "min_branches", fn_name)
                if "min_returns" in spec:
                    _validate_non_negative_int(spec["min_returns"], "min_returns", fn_name)
                for key in ("required_calls", "required_returns", "required_defs"):
                    if key in spec:
                        _validate_str_collection(spec[key], key, fn_name)
            elif not isinstance(spec, bool):
                raise ValueError(
                    f"expected_changes['{fn_name}'] must be a constraint dict or bool, got {type(spec).__name__}"
                )
        return
    if isinstance(expected_changes, (str, bytes)) or not isinstance(
        expected_changes, (list, tuple, set)
    ):
        raise ValueError("expected_changes must be a dict or sequence of non-empty function names")
    for item in expected_changes:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"expected_changes entries must be non-empty strings, got {item!r}")


def _line_text_at(lines: List[str], line_no: int) -> str:
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def verify_code_flow_delta(
    before_code: str,
    after_code: str,
    expected_changes: Optional[Union[Sequence[str], Dict[str, Any]]] = None,
) -> FlowDeltaReport:
    """Verify CFG/DFG delta between before_code and after_code."""
    _validate_expected_changes(expected_changes)
    before_flow = extract_code_flow(before_code)
    after_flow = extract_code_flow(after_code)

    before_names = set(before_flow)
    after_names = set(after_flow)

    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    common = sorted(before_names & after_names)
    changed = [name for name in common if before_flow[name] != after_flow[name]]

    missing_expected: List[str] = []
    for name in common:
        b = before_flow[name]
        a = after_flow[name]
        if a.branches < b.branches:
            missing_expected.append(f"missing_branch:{name}")
        if a.returns < b.returns:
            missing_expected.append(f"removed_return:{name}")
        dropped_calls = sorted(set(b.calls) - set(a.calls))
        for call in dropped_calls:
            missing_expected.append(f"dropped_call:{name}:{call}")

    if isinstance(expected_changes, dict):
        touched = set(added) | set(changed)
        for fn_name, spec in expected_changes.items():
            if fn_name not in after_flow:
                missing_expected.append(f"expected_function:{fn_name}")
                continue
            summary = after_flow[fn_name]
            if isinstance(spec, dict):
                if "min_branches" in spec and summary.branches < spec["min_branches"]:
                    missing_expected.append(f"expected_branch:{fn_name}")
                if "min_returns" in spec and summary.returns < spec["min_returns"]:
                    missing_expected.append(f"expected_return:{fn_name}")
                for req_call in spec.get("required_calls", ()):
                    if req_call not in summary.calls:
                        missing_expected.append(f"expected_call:{fn_name}:{req_call}")
                for req_ret in spec.get("required_returns", ()):
                    if req_ret not in summary.return_expressions:
                        missing_expected.append(f"expected_return_expr:{fn_name}:{req_ret}")
                for req_def in spec.get("required_defs", ()):
                    if req_def not in summary.defs:
                        missing_expected.append(f"expected_def:{fn_name}:{req_def}")
            elif fn_name not in touched:
                missing_expected.append(f"expected_change:{fn_name}")
    elif expected_changes is not None:
        touched = set(added) | set(changed)
        for item in expected_changes:
            if item not in touched:
                missing_expected.append(f"expected_change:{item}")

    unreached_edits: List[str] = []
    for name in sorted(after_flow):
        after_unreach = after_flow[name].unreachable_lines
        if not after_unreach:
            continue
        if name not in before_flow:
            for line in after_unreach:
                unreached_edits.append(f"{name}:L{line}")
            continue
        before_counts = Counter(before_flow[name].unreachable_stmts)
        for line, stmt_text in zip(after_unreach, after_flow[name].unreachable_stmts):
            if before_counts[stmt_text] > 0:
                before_counts[stmt_text] -= 1
            else:
                unreached_edits.append(f"{name}:L{line}")

    compact_payload = json.dumps(
        {
            "added": added,
            "removed": removed,
            "changed": changed,
            "missing": missing_expected,
            "unreached": unreached_edits,
        },
        separators=(",", ":"),
    )
    summary_tokens = estimate_code_tokens(compact_payload)

    return FlowDeltaReport(
        added_functions=added,
        removed_functions=removed,
        changed_functions=changed,
        missing_expected_changes=missing_expected,
        unreached_edits=unreached_edits,
        summary_tokens=summary_tokens,
    )
