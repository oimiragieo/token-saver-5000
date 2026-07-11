"""World-Class Compression Sprint — Task 4: summary + entity quality (2026-07-02).

Pure-function regression locks for the dogfood-visible perceived-quality bugs:
- ``_extract_key_entities`` surfaced "Key entities: This" garbage (a capitalized word
  after a colon — the colon isn't a sentence end, so the sentence-start guard missed it).
- ``_generate_summary`` leaked markdown noise (headings / inline links / backticks) into
  the "Summary:" line.

Both are pure text functions, so these run MODEL-FREE via ``object.__new__`` — no
embedding model needed (avoids the HF-cache dependency entirely).
"""

from __future__ import annotations

from src.semantic_compressor import SemanticCompressor


def _bare_compressor() -> SemanticCompressor:
    # Bypass __init__ (no model load): _extract_key_entities + _generate_summary use
    # only their arguments, no instance/model state.
    return object.__new__(SemanticCompressor)


class TestEntityStopwordFilter:
    def test_colon_prefixed_common_word_not_an_entity(self):
        c = _bare_compressor()
        ents = c._extract_key_entities("Section 2: This paragraph covers Quantum Computing.")
        assert "This" not in ents, ents  # the dogfood garbage
        assert "Section" not in ents, ents
        assert "Quantum" in ents and "Computing" in ents, ents

    def test_real_entities_still_extracted(self):
        c = _bare_compressor()
        ents = c._extract_key_entities("The United States and European Union discussed NATO.")
        assert "United" in ents and "NATO" in ents, ents
        assert "The" not in ents, ents

    def test_trailing_punctuation_stripped(self):
        c = _bare_compressor()
        ents = c._extract_key_entities("We deploy on Redis, Postgres; plus Fly infra.")
        assert "Redis" in ents and "Postgres" in ents, ents  # not "Redis," / "Postgres;"


class TestEntityMarkdownLinkStrip:
    """#286 (dogfood 2026-07-11): a `[text](url)` markdown link leaked its URL into
    the Key entities line ("Apache-2.0)](https://github.com/oimiragieo/tensor-grep").
    Entities must be words, not URLs / paths / markdown-link fragments."""

    def _assert_no_url_fragments(self, ents):
        for e in ents:
            assert "/" not in e, f"URL/path fragment leaked as entity: {e!r} in {ents}"
            assert "](" not in e, f"markdown-link fragment leaked as entity: {e!r} in {ents}"
            assert not e.lower().startswith(("http:", "https:")), f"bare URL as entity: {e!r}"

    def test_markdown_link_url_not_an_entity(self):
        c = _bare_compressor()
        text = (
            "See our OSS project "
            "[tensor-grep (open source, Apache-2.0)](https://github.com/oimiragieo/tensor-grep) "
            "for Fast code search."
        )
        ents = c._extract_key_entities(text)
        self._assert_no_url_fragments(ents)
        # The link's visible text is still mined for real entities.
        assert "Apache-2.0" in ents, ents

    def test_bare_url_not_an_entity(self):
        c = _bare_compressor()
        # A capitalized-scheme URL must not survive as an entity.
        ents = c._extract_key_entities("Docs live at Https://Example.com/Guide for reference.")
        self._assert_no_url_fragments(ents)

    def test_regression_exact_dogfood_fragment(self):
        c = _bare_compressor()
        # The exact llms.txt shape that produced the leak.
        text = (
            "GitHub (SDKs + plugin + benchmarks) - "
            "[tensor-grep (open source, Apache-2.0)](https://github.com/oimiragieo/tensor-grep), Fast"
        )
        ents = c._extract_key_entities(text)
        self._assert_no_url_fragments(ents)

    def test_slash_entities_not_dropped(self):
        """codex 2026-07-11: the URL guard must key on '://', not a bare '/', so
        legitimate slash-bearing tech entities survive."""
        c = _bare_compressor()
        ents = c._extract_key_entities("Our stack speaks TCP/IP and runs CI/CD pipelines.")
        assert "TCP/IP" in ents, ents
        assert "CI/CD" in ents, ents


class TestSummaryMarkdownStrip:
    def test_heading_marker_stripped(self):
        c = _bare_compressor()
        s = c._generate_summary("## Authentication\n\nThe login flow validates tokens.")
        assert not s.lstrip().startswith("#"), s
        assert "Authentication" in s or "login flow" in s, s

    def test_inline_link_flattened(self):
        c = _bare_compressor()
        s = c._generate_summary("See [the docs](https://x.io/a) for details right now.")
        assert "](https" not in s and "the docs" in s, s

    def test_backticks_removed(self):
        c = _bare_compressor()
        s = c._generate_summary("The `compress()` call returns a skeleton object here.")
        assert "`" not in s, s

    def test_plain_text_first_sentence_unchanged(self):
        c = _bare_compressor()
        # Regression: plain prose must still yield the first sentence verbatim.
        s = c._generate_summary("First sentence is here. Second follows.")
        assert s == "First sentence is here.", s
