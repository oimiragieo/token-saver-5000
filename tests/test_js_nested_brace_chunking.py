"""#212(b) (backlog master-plan Wave 1, 2026-07-12): the JS/TS chunker captured a
function body with the regex `\\{([^}]*)\\}`, whose `[^}]*` STOPS at the first inner
`}`. So any JS/TS function containing a nested block (if/for/object-literal/nested-fn)
had its body TRUNCATED and the tail SILENTLY LOST from the chunk. This locks a
brace-depth-balanced, string/comment-aware scanner that keeps the full function body.

Model-free via object.__new__(CodeSemanticCompressor) -- chunk_javascript_code uses only
`re` + the static brace matcher, no model/instance state.
"""

from src.code_compressor import CodeSemanticCompressor


def _cc():
    return object.__new__(CodeSemanticCompressor)


def _body(code: str, name: str) -> str:
    chunks = _cc().chunk_javascript_code(code, "f.js")
    hit = [ch for ch in chunks if ch.name == name]
    assert hit, f"function {name!r} not chunked; got {[c.name for c in chunks]}"
    return hit[0].code


def test_nested_if_block_keeps_tail():
    code = (
        "function process(data) {\n"
        "  if (data.valid) {\n"
        "    return transform(data);\n"
        "  }\n"
        "  return null;\n"
        "}\n"
    )
    body = _body(code, "process")
    assert "return null;" in body, f"tail lost: {body!r}"
    assert body.count("{") == body.count("}"), f"unbalanced (truncated): {body!r}"


def test_object_literal_braces_preserved():
    code = "const build = () => {\n  return { a: 1, nested: { b: 2 } };\n};\n"
    body = _body(code, "build")
    assert "nested: { b: 2 }" in body, body
    assert body.count("{") == body.count("}"), body


def test_brace_inside_string_does_not_close_early():
    code = 'function f() {\n  const s = "}";\n  return s + `${x}`;\n}\n'
    body = _body(code, "f")
    assert "return s" in body, f"string-brace closed the body early: {body!r}"


def test_brace_inside_comment_does_not_close_early():
    code = "function g() {\n  // closing } in a comment\n  return 1;\n}\n"
    body = _body(code, "g")
    assert "return 1;" in body, body


def test_unbalanced_function_does_not_crash():
    # A truncated / malformed source must not raise -- just skip the unbalanced fn.
    code = "function broken(a) {\n  if (a) {\n    return a;\n"  # never closes
    chunks = _cc().chunk_javascript_code(code, "f.js")
    assert isinstance(chunks, list)  # no exception, graceful skip


def test_two_top_level_functions_both_captured_full():
    code = (
        "function one() {\n  if (a) { return 1; }\n  return 2;\n}\n\n"
        "function two() {\n  for (;;) { break; }\n  return 3;\n}\n"
    )
    assert "return 2;" in _body(code, "one")
    assert "return 3;" in _body(code, "two")


def test_simple_function_unchanged_regression():
    code = "function add(a, b) {\n  return a + b;\n}\n"
    body = _body(code, "add")
    assert "return a + b;" in body


# --- codex round-2 findings: regex literals must not truncate/drop the body ---


def test_regex_char_class_brace_not_structural():
    # `/[}]/` -- the `}` lives in a regex char class and must NOT close the function.
    code = "function f() {\n  const r = /[}]/;\n  return 1;\n}\n"
    body = _body(code, "f")
    assert "return 1;" in body, f"regex-brace truncated body: {body!r}"


def test_regex_looking_like_block_comment_not_dropped():
    # `/\/*/` -- the `/*` is inside a regex, not a block comment; the old scanner jumped
    # to EOF and DROPPED the whole function.
    code = "function f() {\n  const r = /\\/*/;\n  return 1;\n}\n"
    body = _body(code, "f")
    assert "return 1;" in body, f"function dropped via regex-as-comment: {body!r}"


def test_return_regex_with_brace_not_truncated():
    # `return /[}]/` -- `return` is a regex-preceding keyword, so the `/` is a regex.
    code = "function f() {\n  return /[}]/.test(x);\n}\n"
    body = _body(code, "f")
    assert ".test(x)" in body, f"return-regex truncated body: {body!r}"


def test_division_not_misread_as_regex():
    # `a / b` is division, not a regex -- must not swallow following braces.
    code = "function calc() {\n  const x = a / b;\n  if (x) { return x; }\n  return 0;\n}\n"
    body = _body(code, "calc")
    assert "return 0;" in body, body
    assert body.count("{") == body.count("}"), body
