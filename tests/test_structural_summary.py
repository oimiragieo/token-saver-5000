"""Tests for structural_summary.py -- AST-based code outline generator.

Uses the real benchmark corpus at benchmarks/corpus/code/auth.py
as the primary test fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Path to the benchmark corpus
CORPUS_DIR = Path(__file__).parent.parent / "benchmarks" / "corpus" / "code"
AUTH_PY = CORPUS_DIR / "auth.py"


def _auth_code() -> str:
    return AUTH_PY.read_text(encoding="utf-8")


def _auth_path() -> str:
    return str(AUTH_PY)


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------
from src.structural_summary import (
    StructuralSummary,
    generate_structural_summary,
    _detect_language,
    _summarize_python,
    _summarize_generic,
)

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


def test_detect_python():
    assert _detect_language("foo.py") == "python"


def test_detect_javascript():
    assert _detect_language("foo.js") == "javascript"


def test_detect_typescript():
    assert _detect_language("bar.ts") == "typescript"


def test_detect_unknown():
    assert _detect_language("README.md") == "unknown"


def test_detect_no_extension():
    assert _detect_language("") == "unknown"


# ---------------------------------------------------------------------------
# Python corpus: auth.py
# ---------------------------------------------------------------------------


def test_summary_contains_class_name():
    """AuthConfig dataclass should appear in the summary."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert "AuthConfig" in result.summary_text


def test_summary_contains_function_names():
    """generate_token and verify_token must be in the outline."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert "generate_token" in result.summary_text
    assert "verify_token" in result.summary_text


def test_summary_no_function_bodies():
    """Function bodies should not leak into the summary."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    # These are implementation details in auth.py bodies
    assert "pbkdf2_hmac" not in result.summary_text
    assert "hmac.new" not in result.summary_text


def test_summary_has_imports():
    """Import lines from auth.py should appear (hashlib, hmac, json, time, etc.)."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    # At least one import should be present
    assert "import" in result.summary_text.lower()


def test_summary_preserves_type_hints():
    """Type annotations in function signatures should be retained."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    # auth.py uses str, list[str], dict, Any etc.
    assert "str" in result.summary_text


def test_summary_savings_above_80pct():
    """Structural summary should achieve at least 80% token savings on auth.py."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert result.savings_pct >= 80.0, f"savings_pct={result.savings_pct} < 80%"


def test_summary_token_count_smaller():
    """summary_tokens must be strictly less than original_tokens."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert result.summary_tokens < result.original_tokens


def test_summary_result_fields():
    """All StructuralSummary fields must be populated."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert isinstance(result, StructuralSummary)
    assert result.file_path == _auth_path()
    assert result.language == "python"
    assert result.line_count > 0
    assert result.symbol_count > 0
    assert result.summary_text != ""
    assert result.summary_tokens >= 0
    assert result.original_tokens > 0


def test_summary_line_count_correct():
    """line_count should match the actual number of lines in the source."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    assert result.line_count == len(code.splitlines())


def test_summary_dataclass_fields_shown():
    """Dataclass field annotations (e.g. AuthConfig) should appear in summary."""
    code = _auth_code()
    result = generate_structural_summary(code, _auth_path())
    # AuthConfig has 'secret_key', 'algorithm', etc.
    assert "secret_key" in result.summary_text or "algorithm" in result.summary_text


def test_summary_empty_file():
    """Empty string should not raise and should return a valid (minimal) result."""
    result = generate_structural_summary("", "empty.py")
    assert isinstance(result, StructuralSummary)
    assert result.line_count == 0
    assert result.original_tokens == 0


def test_summary_syntax_error_fallback():
    """Syntax errors should fall back to regex-based generic summarizer."""
    bad_code = "def broken(\n    pass\n"  # invalid Python
    result = generate_structural_summary(bad_code, "broken.py")
    assert isinstance(result, StructuralSummary)
    # Falls back to generic -- should not raise
    assert result.summary_text is not None


def test_summary_generic_fallback_non_python():
    """Non-Python files use the regex fallback."""
    js_code = """
import { foo } from 'bar';
class MyClass {
  constructor() {}
  myMethod() { return 42; }
}
function helper(x) { return x + 1; }
"""
    result = generate_structural_summary(js_code, "app.js")
    assert result.language == "javascript"
    assert "MyClass" in result.summary_text or "helper" in result.summary_text
