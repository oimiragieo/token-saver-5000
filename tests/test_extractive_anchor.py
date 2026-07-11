"""Model-free unit tests for `_extractive_anchor_content` (world-class audit #1).

A kept anchor should render budgeted extractive content (leading sentences up to a
token budget), not a 1-sentence `_generate_summary`. Tested via `object.__new__`
with `use_tiktoken=False` so `_count_tokens` uses its deterministic word-count
fallback — no model load, no HF cache (see the `compression-engine-sota` skill).
"""

from __future__ import annotations

from src.semantic_compressor import SemanticCompressor


def _bare() -> SemanticCompressor:
    svc = object.__new__(SemanticCompressor)
    svc.use_tiktoken = False  # force _count_tokens' word-count fallback (model-free)
    svc.tokenizer = None
    return svc


def test_empty_text_or_zero_budget_returns_empty():
    svc = _bare()
    assert svc._extractive_anchor_content("", 100) == ""
    assert svc._extractive_anchor_content("Some text.", 0) == ""
    assert svc._extractive_anchor_content("Some text.", -5) == ""


def test_keeps_multiple_sentences_up_to_budget():
    svc = _bare()
    text = "First sentence here. Second sentence follows. Third one too. Fourth extra."
    out = svc._extractive_anchor_content(text, 100)
    # The audit-#1 improvement: MORE than one sentence when the budget allows.
    assert "First sentence here." in out
    assert "Second sentence follows." in out
    assert out.count(".") >= 2


def test_at_least_one_sentence_even_if_over_budget():
    svc = _bare()
    text = "This single sentence has many words that clearly exceed a tiny budget."
    out = svc._extractive_anchor_content(text, 1)  # tiny budget
    assert out  # non-empty: at least one sentence is always kept
    assert out.startswith("This single sentence")


def test_strips_markdown_noise():
    svc = _bare()
    text = "## Heading here. See [the docs](http://x.test) and `code` inline."
    out = svc._extractive_anchor_content(text, 100)
    assert "##" not in out
    assert "http://x.test" not in out
    assert "`" not in out
    assert "the docs" in out  # link text preserved, url dropped


def test_budget_truncates_before_second_sentence():
    svc = _bare()
    # Each sentence = 4 words = int(4 * 1.3) = 5 tokens. Budget 6: first fits (5),
    # second (5 more -> 10 > 6) is dropped.
    text = "Aaa bbb ccc ddd. Eee fff ggg hhh. Iii jjj kkk lll."
    out = svc._extractive_anchor_content(text, 6)
    assert "Aaa bbb ccc ddd." in out
    assert "Eee fff ggg hhh." not in out


def test_deterministic_source_order_preserved():
    svc = _bare()
    text = "One statement. Two statement. Three statement."
    out = svc._extractive_anchor_content(text, 100)
    assert out.index("One") < out.index("Two") < out.index("Three")
