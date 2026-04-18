"""Tests for ``_validate_ingest`` — file_id character class.

Callers routinely pass realistic identifiers like ``src/auth.py`` or
``doc-auth-layer``. The historical regex ``^[a-zA-Z0-9_]+$`` was too
strict and rejected those, bouncing every MCP ``ingest_context`` call
with a cryptic "must contain only alphanumeric characters and
underscores" message. These tests pin the relaxed character class so
it doesn't drift back.
"""

from __future__ import annotations

from src.validation_hooks import validate_tool_input


class TestIngestFileIdCharacterClass:
    """The relaxed character class for ``file_id``: alphanumeric, ``_-./``.

    No whitespace, no shell metacharacters, no ``..`` parent references.
    """

    def test_simple_alphanumeric_accepted(self):
        assert validate_tool_input("ingest_context", {"text": "hi", "file_id": "abc123"}) == []

    def test_underscore_accepted(self):
        assert validate_tool_input("ingest_context", {"text": "hi", "file_id": "my_doc_1"}) == []

    def test_hyphen_accepted(self):
        """Hyphen is the most common missing character — doc-123 is idiomatic."""
        assert (
            validate_tool_input("ingest_context", {"text": "hi", "file_id": "doc-auth-layer"}) == []
        )

    def test_dot_accepted(self):
        """File extensions are common in file_ids — ``report.md``, ``auth.py``."""
        assert validate_tool_input("ingest_context", {"text": "hi", "file_id": "auth.py"}) == []

    def test_forward_slash_accepted(self):
        """Repo-relative paths are the most natural file_id source."""
        assert validate_tool_input("ingest_context", {"text": "hi", "file_id": "src/auth.py"}) == []

    def test_combined_punctuation_accepted(self):
        assert (
            validate_tool_input(
                "ingest_context", {"text": "hi", "file_id": "apps/web/src/Home-v2.tsx"}
            )
            == []
        )

    def test_space_rejected(self):
        """Spaces break shell tooling; still rejected."""
        errors = validate_tool_input("ingest_context", {"text": "hi", "file_id": "my file"})
        assert errors, f"expected rejection, got: {errors}"

    def test_backslash_rejected(self):
        """Windows-style separators could confuse PathValidator; rejected."""
        errors = validate_tool_input("ingest_context", {"text": "hi", "file_id": "a\\b"})
        assert errors, f"expected rejection, got: {errors}"

    def test_dotdot_parent_rejected(self):
        """Parent-directory tokens are a path-traversal footgun; rejected."""
        errors = validate_tool_input("ingest_context", {"text": "hi", "file_id": "../secret"})
        assert errors, f"expected rejection, got: {errors}"

    def test_shell_metacharacters_rejected(self):
        """Shell metacharacters have no business in an identifier."""
        for bad in ("$abc", "a;b", "a|b", "a&b", "a`b`", "a$(b)"):
            assert validate_tool_input("ingest_context", {"text": "hi", "file_id": bad}), bad

    def test_empty_file_id_accepted(self):
        """file_id is optional — empty string is treated as "not provided"."""
        assert validate_tool_input("ingest_context", {"text": "hi", "file_id": ""}) == []
