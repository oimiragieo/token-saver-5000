"""LongCodeZip two-stage adaptive code compression via exact 0/1 knapsack."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import List, Sequence, Set

from .token_estimation import TokenEstimator

_TOKEN_ESTIMATOR: TokenEstimator | None = None

_ROLE_WEIGHTS: dict[str, float] = {
    "import": 0.35,
    "signature": 0.55,
    "return": 0.50,
    "branch": 0.45,
    "try": 0.45,
    "loop": 0.40,
    "statement": 0.25,
    "block": 0.30,
}

_TRY_TYPES: tuple[type, ...] = tuple(
    t for t in (ast.Try, getattr(ast, "TryStar", None)) if t is not None
)


def _get_estimator() -> TokenEstimator:
    global _TOKEN_ESTIMATOR
    if _TOKEN_ESTIMATOR is None:
        _TOKEN_ESTIMATOR = TokenEstimator()
    return _TOKEN_ESTIMATOR


def estimate_code_tokens(code: str) -> int:
    """Estimate token cost for a code block (at least 1 token for non-empty code)."""
    if not code or not code.strip():
        return 0
    return max(1, _get_estimator().count_tokens(code))


@dataclass(frozen=True)
class CodeBlockItem:
    """A single AST-partitioned code block candidate for knapsack selection."""

    block_id: str
    kind: str
    name: str
    start_line: int
    end_line: int
    code: str
    token_cost: int
    importance_score: float


@dataclass(frozen=True)
class KnapsackResult:
    """Result of exact 0/1 knapsack block selection."""

    selected_blocks: List[CodeBlockItem]
    total_tokens: int
    total_value: float
    budget_tokens: int
    reconstructed_code: str


def solve_knapsack_exact(items: Sequence[CodeBlockItem], budget_tokens: int) -> KnapsackResult:
    """Solve exact 0/1 knapsack over code blocks for a given token budget.

    Guarantees ``total_tokens <= max(0, budget_tokens)`` and exact maximum
    ``total_value`` among all feasible subsets. Zero-cost items with positive
    ``importance_score`` are included whenever ``budget_tokens >= 0``.
    """
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

    free_items: List[CodeBlockItem] = []
    pos_items: List[CodeBlockItem] = []
    for item in items:
        if item.importance_score <= 0:
            continue
        if item.token_cost == 0:
            free_items.append(item)
        elif item.token_cost <= budget_tokens:
            pos_items.append(item)

    chosen_pos: List[CodeBlockItem] = []
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
    total_tokens = sum(b.token_cost for b in selected)
    total_value = sum(float(b.importance_score) for b in selected)
    return KnapsackResult(
        selected_blocks=selected,
        total_tokens=total_tokens,
        total_value=total_value,
        budget_tokens=budget_tokens,
        reconstructed_code=reconstructed_code,
    )


def _classify_stmt(node: ast.AST) -> str:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "import"
    if isinstance(node, ast.If):
        return "branch"
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return "loop"
    if isinstance(node, _TRY_TYPES):
        return "try"
    if isinstance(node, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom)):
        return "return"
    if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Yield, ast.YieldFrom)):
        return "return"
    return "statement"


def _tokenize_terms(text: str) -> Set[str]:
    raw = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text)
    terms: Set[str] = set()
    for tok in raw:
        lower = tok.lower()
        terms.add(lower)
        for part in lower.split("_"):
            if part:
                terms.add(part)
    return terms


def _score_block(
    code: str,
    name: str,
    kind: str,
    node: ast.AST | None,
    query_terms: Set[str],
) -> float:
    role_score = _ROLE_WEIGHTS.get(kind, 0.30)

    if node is not None:
        id_count = sum(
            1 for sub in ast.walk(node) if isinstance(sub, (ast.Name, ast.Attribute, ast.arg))
        )
    else:
        id_count = len(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code))

    line_count = max(1, len(code.splitlines()))
    id_density = min(1.0, id_count / (line_count * 4.0)) * 0.25

    query_overlap = 0.0
    if query_terms:
        block_terms = _tokenize_terms(f"{name} {code}")
        overlap = len(query_terms & block_terms)
        query_overlap = (overlap / len(query_terms)) * 0.80

    return round(role_score + id_density + query_overlap, 6)


def _slice_lines(lines: List[str], start_line: int, end_line: int) -> str:
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def _emit_block(
    blocks: List[CodeBlockItem],
    lines: List[str],
    kind: str,
    name: str,
    start_line: int,
    end_line: int,
    node: ast.AST | None,
    query_terms: Set[str],
) -> None:
    code = _slice_lines(lines, start_line, end_line)
    if not code.strip():
        return
    idx = len(blocks)
    blocks.append(
        CodeBlockItem(
            block_id=f"{kind}:{name}:{start_line}-{end_line}:{idx}",
            kind=kind,
            name=name,
            start_line=start_line,
            end_line=end_line,
            code=code,
            token_cost=estimate_code_tokens(code),
            importance_score=_score_block(code, name, kind, node, query_terms),
        )
    )


def _partition_function(
    blocks: List[CodeBlockItem],
    lines: List[str],
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualname: str,
    query_terms: Set[str],
) -> None:
    fn_start = fn_node.lineno
    if fn_node.body:
        first_body_line = fn_node.body[0].lineno
        sig_end = max(fn_start, first_body_line - 1)
    else:
        sig_end = fn_node.end_lineno or fn_start

    _emit_block(blocks, lines, "signature", qualname, fn_start, sig_end, None, query_terms)

    for stmt in fn_node.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _partition_function(blocks, lines, stmt, f"{qualname}.{stmt.name}", query_terms)
            continue
        kind = _classify_stmt(stmt)
        s_line = stmt.lineno
        e_line = stmt.end_lineno or s_line
        _emit_block(blocks, lines, kind, qualname, s_line, e_line, stmt, query_terms)


def partition_ast_blocks(source_code: str, query: str = "") -> List[CodeBlockItem]:
    """Partition Python source code into scored AST blocks for knapsack selection."""
    if not source_code or not source_code.strip():
        return []

    lines = source_code.splitlines()
    query_terms = _tokenize_terms(query) if query else set()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        blocks: List[CodeBlockItem] = []
        chunk_size = 10
        for start in range(1, len(lines) + 1, chunk_size):
            end = min(len(lines), start + chunk_size - 1)
            _emit_block(
                blocks, lines, "block", f"lines_{start}_{end}", start, end, None, query_terms
            )
        return blocks

    blocks = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _partition_function(blocks, lines, stmt, stmt.name, query_terms)
        elif isinstance(stmt, ast.ClassDef):
            cls_start = stmt.lineno
            if stmt.body:
                cls_sig_end = max(cls_start, stmt.body[0].lineno - 1)
            else:
                cls_sig_end = stmt.end_lineno or cls_start
            _emit_block(
                blocks, lines, "signature", stmt.name, cls_start, cls_sig_end, None, query_terms
            )
            for member in stmt.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _partition_function(
                        blocks, lines, member, f"{stmt.name}.{member.name}", query_terms
                    )
                else:
                    kind = _classify_stmt(member)
                    s_line = member.lineno
                    e_line = member.end_lineno or s_line
                    _emit_block(blocks, lines, kind, stmt.name, s_line, e_line, member, query_terms)
        else:
            kind = _classify_stmt(stmt)
            s_line = stmt.lineno
            e_line = stmt.end_lineno or s_line
            _emit_block(blocks, lines, kind, kind, s_line, e_line, stmt, query_terms)

    return blocks


def compress_code_with_knapsack(
    source_code: str,
    budget_tokens: int,
    query: str = "",
) -> KnapsackResult:
    """Two-stage LongCodeZip compression: AST block partitioning + exact 0/1 knapsack."""
    blocks = partition_ast_blocks(source_code, query=query)
    result = solve_knapsack_exact(blocks, budget_tokens)
    selected = list(result.selected_blocks)
    estimator = _get_estimator()
    reconstructed_code = result.reconstructed_code
    actual_tokens = estimator.count_tokens(reconstructed_code) if reconstructed_code else 0
    while selected and actual_tokens > budget_tokens:
        drop_idx = min(
            range(len(selected)),
            key=lambda i: (
                float(selected[i].importance_score),
                -selected[i].token_cost,
                -i,
            ),
        )
        selected.pop(drop_idx)
        reconstructed_code = "\n".join(b.code for b in selected)
        actual_tokens = estimator.count_tokens(reconstructed_code) if reconstructed_code else 0
    return KnapsackResult(
        selected_blocks=selected,
        total_tokens=actual_tokens,
        total_value=sum(float(b.importance_score) for b in selected),
        budget_tokens=budget_tokens,
        reconstructed_code=reconstructed_code,
    )
