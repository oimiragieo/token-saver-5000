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

    def test_python_literals_not_entities(self):
        """#306 (dogfood 2026-07-13): code ingested as text surfaced `None` as a
        Key entity (`last = None`). Python literals None/True/False are never
        domain entities; a real class name (TokenBucket) is still kept."""
        c = _bare_compressor()
        ents = c._extract_key_entities("The bucket returns None when True or False here.")
        assert "None" not in ents, ents
        assert "True" not in ents, ents
        assert "False" not in ents, ents
        ents2 = c._extract_key_entities("We use the TokenBucket limiter and it returns None.")
        assert "TokenBucket" in ents2, ents2  # real class name survives
        assert "None" not in ents2, ents2

    def test_section_heading_words_not_entities(self):
        """#360 (dogfood 2026-07-16): generic section-heading words (Overview,
        Design) leaked into Key entities from markdown `## Overview` / `## Design`
        blocks. A bare structural heading is never a useful domain entity; real
        entities (Redis) still survive."""
        c = _bare_compressor()
        ents = c._extract_key_entities("Read the Overview and Design sections for Redis details.")
        assert "Overview" not in ents, ents
        assert "Design" not in ents, ents
        assert "Redis" in ents, ents  # real entity survives

    def test_common_adverbs_and_allcaps_emphasis_not_entities(self):
        """#360 (dogfood 2026-07-16): common adverbs capitalized mid-sentence
        (Only) and all-caps emphasis of function words (NOT — the exact-match
        stopword check missed it because only title-case "Not" was listed) leaked
        as Key entities. A real acronym (IT = Information Technology) must NOT be
        dropped by the all-caps additions."""
        c = _bare_compressor()
        ents = c._extract_key_entities("You must NOT deploy Only to Postgres.")
        assert "NOT" not in ents, ents
        assert "Only" not in ents, ents
        assert "Postgres" in ents, ents  # real entity survives
        # Guard against over-filtering: a genuine all-caps acronym still surfaces.
        ents2 = c._extract_key_entities("The IT department runs Kubernetes.")
        assert "IT" in ents2, ents2
        assert "Kubernetes" in ents2, ents2


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

    def test_bold_emphasis_markers_stripped(self):
        """#287 (dogfood 2026-07-11): markdown emphasis asterisks must not leak into
        entities ('JWT**' surfaced as an entity). Strip leading/trailing '*' only;
        keep mid-word underscores (gc_kb_query), slashes (TCP/IP) AND legit
        trailing-underscore identifiers (codex: don't strip '_') intact."""
        c = _bare_compressor()
        ents = c._extract_key_entities("The token is a JWT** bearer for the Config service.")
        assert "JWT" in ents, ents
        assert "JWT**" not in ents, ents
        # Mid-word underscore/slash identifiers preserved; trailing '_' NOT stripped.
        ents2 = c._extract_key_entities("We deploy Gc_Kb_Query over TCP/IP, keep Type_ intact.")
        assert "Gc_Kb_Query" in ents2, ents2
        assert "TCP/IP" in ents2, ents2
        assert "Type_" in ents2, ents2  # legit trailing-underscore identifier kept


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

    def test_bold_italic_emphasis_stripped(self):
        """Dogfood 2026-07-12: a live read_skeleton summary leaked `**` markers
        (`enforces a **token-bucket** rate limiter`). Strip paired asterisk
        emphasis; keep inner words AND snake_case identifiers intact."""
        c = _bare_compressor()
        s = c._generate_summary("The API enforces a **token-bucket** rate limiter here.")
        assert "**" not in s, s
        assert "*" not in s, s
        assert "token-bucket" in s, s

    def test_italic_emphasis_stripped_but_underscores_kept(self):
        c = _bare_compressor()
        s = c._generate_summary("The *async* gc_kb_query handler runs first here.")
        assert "*" not in s, s
        assert "async" in s, s
        assert "gc_kb_query" in s, s  # snake_case survives (underscores untouched)
