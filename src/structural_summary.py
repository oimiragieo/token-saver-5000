"""Structural summary generator inspired by repowise.

Generates compact code outlines: imports + class hierarchy + function
signatures. Preserves API surface in ~10% of original tokens.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StructuralSummary:
    file_path: str
    language: str
    line_count: int
    symbol_count: int
    summary_text: str
    summary_tokens: int
    original_tokens: int
    savings_pct: float


def generate_structural_summary(code: str, file_path: str = "") -> StructuralSummary:
    """Generate a compact structural outline of a code file.

    For Python: uses AST to extract imports, class definitions (with
    field annotations), and function/method signatures with type hints.
    Bodies replaced with `...`.

    For other languages: regex-based extraction of function/class patterns.
    """
    language = _detect_language(file_path)
    line_count = len(code.splitlines())
    original_tokens = len(code) // 4

    if language == "python":
        summary_text = _summarize_python(code, file_path)
    else:
        summary_text = _summarize_generic(code, file_path)

    summary_tokens = len(summary_text) // 4
    symbol_count = summary_text.count("def ") + summary_text.count("class ")
    savings_pct = round((original_tokens - summary_tokens) / max(1, original_tokens) * 100, 1)

    return StructuralSummary(
        file_path=file_path,
        language=language,
        line_count=line_count,
        symbol_count=symbol_count,
        summary_text=summary_text,
        summary_tokens=summary_tokens,
        original_tokens=original_tokens,
        savings_pct=savings_pct,
    )


def _detect_language(file_path: str) -> str:
    ext = Path(file_path).suffix.lower() if file_path else ""
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
    }.get(ext, "unknown")


def _summarize_python(code: str, file_path: str) -> str:
    """AST-based Python structural summary."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _summarize_generic(code, file_path)

    lines = []
    name = Path(file_path).name if file_path else "module"
    line_count = len(code.splitlines())

    # Header
    lines.append(f"# {name} ({line_count} lines)")
    lines.append("")

    # Imports (deduplicated, compact)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [a.name for a in node.names]
            if len(names) <= 3:
                imports.append(f"from {module} import {', '.join(names)}")
            else:
                imports.append(f"from {module} import ({len(names)} names)")

    if imports:
        # Group simple imports together
        simple = [i for i in imports if not i.startswith("from ")]
        from_imports = [i for i in imports if i.startswith("from ")]
        if simple:
            lines.append(f"import {', '.join(simple)}")
        lines.extend(from_imports)
        lines.append("")

    # Top-level definitions
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            lines.extend(_summarize_class(node))
            lines.append("")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _function_signature(node)
            lines.append(f"{sig}: ...")
        elif isinstance(node, ast.Assign):
            # Module-level constants
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    lines.append(f"{target.id} = ...")

    return "\n".join(lines).strip()


def _summarize_class(node: ast.ClassDef) -> list[str]:
    """Summarize a class: name, bases, annotations, method signatures."""
    lines = []

    # Class header with bases
    bases = [_name_str(b) for b in node.bases]
    base_str = f"({', '.join(bases)})" if bases else ""
    lines.append(f"class {node.name}{base_str}:")

    # Field annotations (dataclass-style)
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            ann = _annotation_str(item.annotation)
            if item.value is not None:
                lines.append(f"    {item.target.id}: {ann} = ...")
            else:
                lines.append(f"    {item.target.id}: {ann}")

    # Method signatures
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _function_signature(item, indent=4)
            lines.append(f"{sig}: ...")

    # If empty class
    if len(lines) == 1:
        lines.append("    ...")

    return lines


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef, indent: int = 0) -> str:
    """Extract function signature with type hints."""
    prefix = " " * indent
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""

    # Parameters
    params = []
    args = node.args

    # Regular args
    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        p = arg.arg
        if arg.annotation:
            p += f": {_annotation_str(arg.annotation)}"
        if i >= defaults_offset:
            p += " = ..."
        params.append(p)

    # *args
    if args.vararg:
        p = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            p += f": {_annotation_str(args.vararg.annotation)}"
        params.append(p)

    # **kwargs
    if args.kwarg:
        p = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            p += f": {_annotation_str(args.kwarg.annotation)}"
        params.append(p)

    # Return type
    ret = ""
    if node.returns:
        ret = f" -> {_annotation_str(node.returns)}"

    return f"{prefix}{async_prefix}def {node.name}({', '.join(params)}){ret}"


def _annotation_str(node: ast.expr) -> str:
    """Convert AST annotation node to string."""
    try:
        return ast.unparse(node)
    except Exception:
        return "Any"


def _name_str(node: ast.expr) -> str:
    """Convert AST name node to string."""
    try:
        return ast.unparse(node)
    except Exception:
        if isinstance(node, ast.Name):
            return node.id
        return "?"


def _summarize_generic(code: str, file_path: str) -> str:
    """Regex-based summary for non-Python files."""
    lines = []
    name = Path(file_path).name if file_path else "file"
    line_count = len(code.splitlines())
    lines.append(f"# {name} ({line_count} lines)")
    lines.append("")

    # Find function/method definitions
    func_pattern = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:def|function|func|fn|pub fn|pub async fn)\s+(\w+)",
        re.MULTILINE,
    )
    # Find class/struct/interface definitions
    class_pattern = re.compile(
        r"^\s*(?:export\s+)?(?:class|struct|interface|type|enum)\s+(\w+)",
        re.MULTILINE,
    )
    # Find import/use statements
    import_pattern = re.compile(
        r"^(?:import|from|use|require|include)\s+.+$",
        re.MULTILINE,
    )

    imports = import_pattern.findall(code)
    if imports:
        for imp in imports[:10]:  # cap at 10
            lines.append(imp.strip())
        if len(imports) > 10:
            lines.append(f"# ... and {len(imports) - 10} more imports")
        lines.append("")

    classes = class_pattern.findall(code)
    for cls in classes:
        lines.append(f"class {cls}: ...")

    funcs = func_pattern.findall(code)
    for func in funcs:
        lines.append(f"def {func}(...): ...")

    return "\n".join(lines).strip()
