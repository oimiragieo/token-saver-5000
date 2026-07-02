"""Audit #194 regression lock: async def functions/methods must be chunked.

`chunk_python_code` used `isinstance(node, ast.FunctionDef)`, which EXCLUDES
`ast.AsyncFunctionDef` (a sibling type, not a subclass). Every top-level
`async def` therefore produced NO chunk — silently dropping most functions in an
async-heavy file (FastAPI routers = module-level `async def`).

Pure-AST path → model-free via object.__new__ (no embedding model needed).
"""

from __future__ import annotations

from src.code_compressor import CodeSemanticCompressor


def _bare() -> CodeSemanticCompressor:
    # chunk_python_code uses only its (code, file_id) args — no model/instance state.
    return object.__new__(CodeSemanticCompressor)


def test_top_level_async_def_is_chunked():
    c = _bare()
    code = "async def handler(x):\n    return x + 1\n\n\ndef sync_fn(y):\n    return y\n"
    chunks = c.chunk_python_code(code, "mod")
    names = {ch.name for ch in chunks}
    assert "handler" in names, f"async def was dropped: {names}"
    assert "sync_fn" in names, f"sync def missing: {names}"


def test_async_method_counted_in_class_methods():
    c = _bare()
    code = (
        "class A:\n"
        "    async def run(self):\n"
        "        return 1\n\n"
        "    def stop(self):\n"
        "        return 2\n"
    )
    chunks = c.chunk_python_code(code, "mod")
    class_chunks = [ch for ch in chunks if ch.chunk_type == "class"]
    assert class_chunks, "no class chunk produced"
    methods = class_chunks[0].dependencies
    assert "run" in methods and "stop" in methods, f"async method dropped: {methods}"
