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


def test_summary_strips_blockquote_markers():
    # Dogfood 2026-07-14 (our OWN llms.txt): a multi-line markdown blockquote
    # (`> line one\n> line two`) whitespace-collapsed into one candidate and leaked
    # the '>' markers mid-summary ("... graph-based > PageRank-ranked ...").
    c = _c()
    out = c._generate_summary(
        "# gotcontext.ai\n\n> Semantic compression API cuts token usage\n> via PageRank ranking."
    )
    assert ">" not in out, f"blockquote marker leaked into summary: {out!r}"
    assert "Semantic compression API" in out
    # Nested blockquote markers ('>> ') are stripped too.
    assert ">" not in c._generate_summary(">> nested quoted line of prose here.")


def test_summary_keeps_midline_greater_than_comparison():
    # A mid-line '>' (comparison / HTML) is NOT a line-initial blockquote marker,
    # so it must survive the strip (only line-initial '>' is cut).
    c = _c()
    out = c._generate_summary("peak RSS must stay O(input) where a > b holds for all inputs.")
    assert ">" in out


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


def test_mid_content_heading_marker_stripped_all_langs():
    # dogfood 2026-07-12: a chunk with a MID-content heading (not just leading)
    # must not leak the '##' — the line-initial multiline strip handles it for any
    # language, incl. CJK where the '.!?' splitter can't segment on full-width '。'.
    c = _c()
    cjk = "# 部署手册\n\n## 概述\n服务运行在 Fly.io 上。数据库使用 Supabase。"
    s = c._generate_summary(cjk, max_length=200)
    assert "#" not in s, s
    a = c._extractive_anchor_content(cjk, token_budget=120)
    assert "#" not in a, a
    # English mid-heading too.
    en = c._generate_summary("# Title\n\n## Section\nReal body text here.", max_length=200)
    assert "#" not in en, en


def test_hash_not_at_line_start_preserved():
    # 'C#'/'F#' are NOT line-start headings and must survive the strip.
    c = _c()
    s = c._generate_summary("We use the C# language and F# too.")
    assert "C#" in s and "F#" in s


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


def test_cjk_summary_selects_first_sentence_not_whole_paragraph():
    # dogfood 2026-07-12 (#212 remainder): the '.!?' splitter can't segment on the
    # full-width CJK terminator '。', so a multi-sentence CJK paragraph collapsed into
    # ONE candidate and the summary returned the WHOLE blob. Split on '。！？' too so a
    # CJK summary is the first CJK sentence -- matching the English "first sentence"
    # contract. 'Fly.io' internal '.' has no trailing space so it still never splits.
    c = _c()
    s = c._generate_summary("服务运行在 Fly.io 上。数据库使用 Supabase。数据缓存用 Redis。")
    assert s == "服务运行在 Fly.io 上。", s


def test_japanese_summary_selects_first_sentence():
    c = _c()
    s = c._generate_summary("これはテストです。二番目の文はここにあります。")
    assert s == "これはテストです。", s


def test_fullwidth_question_exclamation_terminators_split():
    # Full-width '！' and '？' terminate a CJK sentence too.
    c = _c()
    assert c._generate_summary("你好吗？我很好。") == "你好吗？"
    assert c._generate_summary("太好了！下一句在这里。") == "太好了！"


def test_cjk_split_does_not_regress_latin_first_sentence():
    # Latin behavior must be byte-identical: '。！？' never appear in Latin text, so
    # the ASCII '(?<=[.!?])\s+' path is unchanged.
    c = _c()
    assert c._generate_summary("First sentence here. Second one follows.") == "First sentence here."
    # 'Fly.io' internal dot (no trailing space) must not split.
    assert c._generate_summary("Deploy on Fly.io today. More text.") == "Deploy on Fly.io today."


def test_extractive_anchor_respects_cjk_sentence_boundary():
    # The anchor render keeps whole sentences up to budget; with CJK sentences now
    # individually segmented, a budget-1 render lands the first sentence cleanly on
    # '。' instead of spilling the whole blob.
    c = _c()
    out = c._extractive_anchor_content("第一句在这里。第二句在这里。第三句在这里。", token_budget=1)
    assert out == "第一句在这里。", out
