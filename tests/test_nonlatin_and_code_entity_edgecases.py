"""Dogfood 2026-07-12 (#212 non-English + code edge cases): two perceived-quality
leaks in the #286/#287 lineage, verified on the live prod MCP then TDD'd model-free.

BUG 1 (_generate_summary / _extractive_anchor_content): the substantive-content
guard used the ASCII-only `[A-Za-z0-9]`, so a pure non-Latin sentence (CJK, Cyrillic,
Arabic) was treated as non-substantive -> every candidate was skipped -> the summary
fell through to the `cleaned.strip()` fallback that RETAINS the leading markdown
marker. Result: `Summary: "# 部署手册"` (the '#' leaked). English stripped it fine.
Fix: `[^\W_]` (Unicode letter/digit, no '_') so non-Latin text is substantive and
gets the markdown-heading strip -- while '!!!' / '---' / '___' stay skipped.

BUG 2 (_extract_key_entities): `.strip()` only touches word ends, so a code fragment
`SettlementError(f"failed` (embedded '(' + '"') survived every guard and surfaced as a
garbled 'Key entity'. Fix: truncate at the first embedded paren/quote -> recover the
clean identifier `SettlementError`; legit entities (Fly.io, TCP/IP, gc_kb_query) have
no parens/quotes and are unaffected.

All model-free via object.__new__(SemanticCompressor) (no model load / HF cache).
"""

from src.semantic_compressor import SemanticCompressor


def _c() -> SemanticCompressor:
    c = object.__new__(SemanticCompressor)
    # _extractive_anchor_content -> _count_tokens reads these; the word-count
    # fallback keeps the function model-free (no tiktoken / no model load).
    c.use_tiktoken = False
    c.tokenizer = None
    return c


def test_nonlatin_summary_strips_markdown_heading():
    c = _c()
    # CJK + Cyrillic headings must no longer leak the '#' marker.
    assert c._generate_summary("# 部署手册") == "部署手册"
    assert c._generate_summary("## Раздел безопасности") == "Раздел безопасности"


def test_latin_summary_still_strips_heading_regression():
    c = _c()
    assert c._generate_summary("# Deployment Runbook") == "Deployment Runbook"
    assert c._generate_summary("## Section: overview here.") == "Section: overview here."


def test_punctuation_only_fragment_still_skipped():
    c = _c()
    # A pure-punctuation first fragment must be skipped in favour of the real sentence
    # (the '!!!'/'---' guard must survive the [^\W_] change).
    out = c._generate_summary("!!! note\nReal content follows here.")
    assert "Real content" in out
    # '___' (underscore-only) is NOT substantive either.
    out2 = c._generate_summary("___\nActual body text.")
    assert "Actual body text" in out2


def test_extractive_anchor_keeps_nonlatin_sentence():
    c = _c()
    # _extractive_anchor_content must also treat non-Latin text as substantive.
    out = c._extractive_anchor_content("概述：服务运行在 Fly.io 上。", token_budget=50)
    assert "概述" in out and "#" not in out


def test_code_syntax_entity_recovers_clean_identifier():
    c = _c()
    ents = c._extract_key_entities(
        'The processor calls raise SettlementError(f"failed after {n}") when done'
    )
    # No entity may contain raw code syntax; the clean identifier is recovered.
    assert all(ch not in e for e in ents for ch in '()"'), ents
    assert "SettlementError" in ents


def test_legit_entities_with_dots_slashes_underscores_preserved():
    c = _c()
    ents = c._extract_key_entities(
        "We deploy on Fly.io using CI/CD and the gc_kb_query MCP tool via TCP/IP."
    )
    assert "Fly.io" in ents
    assert "CI/CD" in ents
    assert "TCP/IP" in ents


def test_apostrophe_entity_not_truncated():
    # droid gate: the code-syntax truncation must NOT split on "'" or a legit
    # possessive/name entity like O'Reilly would collapse to "O".
    c = _c()
    ents = c._extract_key_entities("The book was published by O'Reilly Media last year.")
    assert "O'Reilly" in ents
