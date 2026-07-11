"""Dogfood 2026-07-11: a live compress of httpx's mkdocs docs surfaced
`Summary: "!!!"`. A python-markdown/mkdocs admonition marker (`!!! note`)
fragments on the sentence split into a bare "!!!" candidate, which
``_generate_summary`` accepted as the summary (non-empty check only).

Fix: strip admonition markers before splitting + require the chosen summary
sentence to contain a letter/digit (skip punctuation-only fragments).

Model-free: ``_generate_summary`` uses only its argument + the ``re`` module,
so ``object.__new__`` skips the model load (per the compression-engine-sota
skill's model-free testing pattern).
"""

from __future__ import annotations

import re

from src.semantic_compressor import SemanticCompressor


def _bare() -> SemanticCompressor:
    return object.__new__(SemanticCompressor)


class TestSummaryAdmonitionStrip:
    def test_admonition_marker_does_not_become_summary(self):
        c = _bare()
        s = c._generate_summary("!!! note\n    You can set a custom timeout with httpx.Timeout.")
        assert s.strip() != "!!!"
        assert re.search(r"[A-Za-z0-9]", s)  # substantive, not punctuation-only
        assert "timeout" in s.lower()

    def test_admonition_with_type_and_title_stripped(self):
        c = _bare()
        s = c._generate_summary('!!! warning "Timeout configuration"\n    Use httpx.Timeout(10.0).')
        assert s.strip() not in ("!!!", "!!! warning", "warning")
        assert re.search(r"[A-Za-z0-9]", s)
        assert "httpx" in s.lower()

    def test_collapsible_admonition_marker_stripped(self):
        c = _bare()
        s = c._generate_summary("??? tip\n    Reuse a Client for connection pooling.")
        assert s.strip() != "???"
        assert "client" in s.lower()

    def test_bare_punctuation_fragment_skipped(self):
        c = _bare()
        s = c._generate_summary("!!!\n\nReal content sentence follows.")
        assert re.search(r"[A-Za-z0-9]", s)
        assert "Real content" in s

    def test_normal_prose_unchanged(self):
        c = _bare()
        s = c._generate_summary("The client sends requests using .get() and .post().")
        assert s.startswith("The client sends requests")
