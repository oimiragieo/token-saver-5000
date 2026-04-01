"""Tests for src/token_refiner.py.

Covers:
- Filler word removal (articles, hedge words)
- Preservation of semantic anchors (numbers, identifiers, URLs, paths, acronyms)
- Sentence boundary protection
- Short sentence passthrough
- Edge cases (empty, whitespace-only, target_ratio=1.0)
- RefinedResult dataclass fields
- Compression ratio sanity on realistic tech text
"""

from __future__ import annotations

from src.token_refiner import FILLER_WORDS, PRESERVE_PATTERNS, RefinedResult, TokenRefiner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_refiner() -> TokenRefiner:
    return TokenRefiner()


# ---------------------------------------------------------------------------
# score_token tests
# ---------------------------------------------------------------------------


class TestScoreToken:
    def test_score_token_number(self):
        refiner = make_refiner()
        assert refiner.score_token("42") == 1.0

    def test_score_token_float_number(self):
        refiner = make_refiner()
        assert refiner.score_token("3.14") == 1.0

    def test_score_token_filler(self):
        refiner = make_refiner()
        assert refiner.score_token("basically") == 0.1

    def test_score_token_article(self):
        refiner = make_refiner()
        assert refiner.score_token("the") == 0.1

    def test_score_token_content(self):
        refiner = make_refiner()
        assert refiner.score_token("architecture") == 0.7

    def test_score_token_snake_case_identifier(self):
        refiner = make_refiner()
        assert refiner.score_token("get_user_name") == 1.0

    def test_score_token_camelcase_identifier(self):
        refiner = make_refiner()
        assert refiner.score_token("getUserName") == 1.0

    def test_score_token_url(self):
        refiner = make_refiner()
        assert refiner.score_token("https://example.com") == 1.0

    def test_score_token_file_path(self):
        refiner = make_refiner()
        assert refiner.score_token("/usr/bin/python") == 1.0

    def test_score_token_acronym(self):
        refiner = make_refiner()
        assert refiner.score_token("API") == 1.0
        assert refiner.score_token("HTTP") == 1.0

    def test_score_token_dollar_variable(self):
        refiner = make_refiner()
        assert refiner.score_token("$HOME") == 1.0

    def test_score_token_at_tag(self):
        refiner = make_refiner()
        assert refiner.score_token("@param") == 1.0

    def test_score_token_hash_tag(self):
        refiner = make_refiner()
        assert refiner.score_token("#TODO") == 1.0

    def test_score_token_punctuation_only_returns_half(self):
        refiner = make_refiner()
        assert refiner.score_token(".,;") == 0.5

    def test_score_token_short_word(self):
        refiner = make_refiner()
        # Two-character word that isn't a filler: score == 0.3
        score = refiner.score_token("ox")
        assert score == 0.3

    def test_score_token_filler_with_trailing_punctuation(self):
        # "the," should still be recognised as a filler after stripping punctuation
        refiner = make_refiner()
        assert refiner.score_token("the,") == 0.1


# ---------------------------------------------------------------------------
# score_tokens tests
# ---------------------------------------------------------------------------


class TestScoreTokens:
    def test_score_tokens_returns_pairs(self):
        refiner = make_refiner()
        result = refiner.score_tokens("hello world 42")
        assert isinstance(result, list)
        assert len(result) == 3
        for token, score in result:
            assert isinstance(token, str)
            assert isinstance(score, float)

    def test_score_tokens_empty_string(self):
        refiner = make_refiner()
        assert refiner.score_tokens("") == []

    def test_score_tokens_preserves_order(self):
        refiner = make_refiner()
        tokens_text = "the quick brown fox"
        pairs = refiner.score_tokens(tokens_text)
        reconstructed = " ".join(t for t, _ in pairs)
        assert reconstructed == tokens_text


# ---------------------------------------------------------------------------
# refine — basic removal tests
# ---------------------------------------------------------------------------


class TestRefineRemoval:
    def test_refine_removes_articles(self):
        refiner = make_refiner()
        # Sentence long enough to trigger removal; "the" and "a" are fillers
        text = "the cat sat on a mat here today now"
        result = refiner.refine(text, target_ratio=0.6)
        assert "the" not in result.refined_text.split() or "a" not in result.refined_text.split()
        assert result.removed_count > 0

    def test_refine_removes_fillers(self):
        refiner = make_refiner()
        # Contains multiple filler words that should be candidates
        text = "basically it is very good and essentially works well quite well now"
        result = refiner.refine(text, target_ratio=0.5)
        refined_words = set(result.refined_text.split())
        # At least one filler should be gone
        fillers_in_original = {"basically", "very", "essentially", "quite"}
        remaining_fillers = fillers_in_original & refined_words
        assert len(remaining_fillers) < len(fillers_in_original)

    def test_refine_reduces_token_count(self):
        refiner = make_refiner()
        text = "basically the system is very good and will work well quite smoothly"
        result = refiner.refine(text, target_ratio=0.6)
        assert result.refined_tokens < result.original_tokens


# ---------------------------------------------------------------------------
# refine — preservation tests
# ---------------------------------------------------------------------------


class TestRefinePreservation:
    def test_refine_preserves_numbers(self):
        refiner = make_refiner()
        text = "there are 3 errors in line 42 and the system crashed basically"
        result = refiner.refine(text, target_ratio=0.5)
        refined_words = result.refined_text.split()
        assert "3" in refined_words
        assert "42" in refined_words

    def test_refine_preserves_code_identifiers(self):
        refiner = make_refiner()
        text = "call get_user_name then process_data to handle the request basically"
        result = refiner.refine(text, target_ratio=0.5)
        refined_words = result.refined_text.split()
        assert "get_user_name" in refined_words
        assert "process_data" in refined_words

    def test_refine_preserves_camelcase(self):
        refiner = make_refiner()
        text = "the getUserName function returns basically a value from the database"
        result = refiner.refine(text, target_ratio=0.5)
        assert "getUserName" in result.refined_text.split()

    def test_refine_preserves_urls(self):
        refiner = make_refiner()
        text = "see https://example.com for the documentation of this feature"
        result = refiner.refine(text, target_ratio=0.5)
        assert "https://example.com" in result.refined_text.split()

    def test_refine_preserves_paths(self):
        refiner = make_refiner()
        text = "the binary is at /usr/bin/python which is basically standard"
        result = refiner.refine(text, target_ratio=0.5)
        assert "/usr/bin/python" in result.refined_text.split()

    def test_refine_preserves_acronyms(self):
        refiner = make_refiner()
        text = "the API uses HTTP and the system basically handles it well"
        result = refiner.refine(text, target_ratio=0.5)
        refined_words = result.refined_text.split()
        assert "API" in refined_words
        assert "HTTP" in refined_words

    def test_refine_preserves_sentence_boundaries(self):
        refiner = make_refiner()
        # Each sentence starts with "the" and ends with a content word — boundary
        # protection means the first and last tokens of each sentence are never removed.
        text = "the system works well. the process runs fast."
        result = refiner.refine(text, target_ratio=0.5)
        # Sentences are joined by spaces after split; just verify both final words present
        assert "well." in result.refined_text or "works" in result.refined_text
        assert "fast." in result.refined_text or "runs" in result.refined_text


# ---------------------------------------------------------------------------
# refine — short sentence passthrough
# ---------------------------------------------------------------------------


class TestRefineShortSentences:
    def test_refine_short_sentence_kept_entirely(self):
        refiner = make_refiner()
        # Exactly 3 tokens — must be kept as-is
        text = "the cat sat"
        result = refiner.refine(text, target_ratio=0.5)
        assert result.refined_text == text
        assert result.removed_count == 0

    def test_refine_two_token_sentence_kept(self):
        refiner = make_refiner()
        text = "basically works"
        result = refiner.refine(text, target_ratio=0.5)
        assert result.refined_text == text

    def test_refine_does_not_remove_all_tokens(self):
        refiner = make_refiner()
        # Even with aggressive ratio, at least 2 tokens must remain per sentence
        text = "the the the the the the the the the ten"
        result = refiner.refine(text, target_ratio=0.1)
        # Last token "ten" is protected as boundary; first "the" also protected.
        assert result.refined_tokens >= 2


# ---------------------------------------------------------------------------
# refine — edge cases
# ---------------------------------------------------------------------------


class TestRefineEdgeCases:
    def test_refine_empty_input(self):
        refiner = make_refiner()
        result = refiner.refine("")
        assert result.original_text == ""
        assert result.refined_text == ""
        assert result.original_tokens == 0
        assert result.refined_tokens == 0
        assert result.removed_count == 0
        assert result.compression_ratio == 1.0

    def test_refine_whitespace_only(self):
        refiner = make_refiner()
        result = refiner.refine("   ")
        assert result.refined_text == "   "
        assert result.removed_count == 0

    def test_refine_noop_when_ratio_1(self):
        refiner = make_refiner()
        text = "basically the system is very good"
        result = refiner.refine(text, target_ratio=1.0)
        assert result.refined_text == text
        assert result.removed_count == 0
        assert result.compression_ratio == 1.0

    def test_refine_noop_when_ratio_above_1(self):
        refiner = make_refiner()
        text = "basically the system is very good"
        result = refiner.refine(text, target_ratio=1.5)
        assert result.refined_text == text
        assert result.removed_count == 0


# ---------------------------------------------------------------------------
# refine — compression ratio
# ---------------------------------------------------------------------------


class TestRefineCompressionRatio:
    def test_refine_compression_ratio_typical_tech_text(self):
        refiner = make_refiner()
        # Realistic tech text with a mix of fillers and content words
        text = (
            "the system basically uses an API that is very fast and will handle "
            "essentially all requests. the architecture should be quite modular and "
            "perhaps can be extended with additional components that are really useful."
        )
        result = refiner.refine(text, target_ratio=0.6)
        # Expect at least some compression (ratio > 1.0) on filler-heavy text
        assert result.compression_ratio >= 1.1

    def test_refine_compression_ratio_field_is_float(self):
        refiner = make_refiner()
        text = "the system basically works well and is very reliable"
        result = refiner.refine(text, target_ratio=0.6)
        assert isinstance(result.compression_ratio, float)

    def test_refine_compression_ratio_equals_original_over_refined(self):
        refiner = make_refiner()
        text = "basically the entire architecture is essentially very robust and modular"
        result = refiner.refine(text, target_ratio=0.5)
        expected = round(result.original_tokens / max(1, result.refined_tokens), 2)
        assert result.compression_ratio == expected


# ---------------------------------------------------------------------------
# RefinedResult dataclass
# ---------------------------------------------------------------------------


class TestRefinedResult:
    def test_refined_result_has_stats(self):
        refiner = make_refiner()
        text = "basically the system is very good indeed for everyone"
        result = refiner.refine(text)
        assert isinstance(result, RefinedResult)
        # All fields must be populated
        assert isinstance(result.original_text, str)
        assert isinstance(result.refined_text, str)
        assert isinstance(result.original_tokens, int)
        assert isinstance(result.refined_tokens, int)
        assert isinstance(result.removed_count, int)
        assert isinstance(result.compression_ratio, float)

    def test_refined_result_token_counts_consistent(self):
        refiner = make_refiner()
        text = "basically the system is very good indeed for everyone here"
        result = refiner.refine(text, target_ratio=0.6)
        assert result.original_tokens == result.refined_tokens + result.removed_count

    def test_refined_result_original_text_unchanged(self):
        refiner = make_refiner()
        text = "basically the system is very good"
        result = refiner.refine(text)
        assert result.original_text == text


# ---------------------------------------------------------------------------
# Custom filler / pattern injection
# ---------------------------------------------------------------------------


class TestCustomConfiguration:
    def test_custom_filler_words_applied(self):
        custom_fillers = frozenset({"foo", "bar"})
        refiner = TokenRefiner(filler_words=custom_fillers)
        text = "foo the system bar works architecture pipeline correctly indeed"
        refiner.refine(text, target_ratio=0.6)
        # Default fillers like "the" still score 0.7 (content); custom ones are 0.1
        # "foo" and "bar" should be candidates; "the" should score 0.7 now
        assert refiner.score_token("foo") == 0.1
        assert refiner.score_token("bar") == 0.1
        # "the" is no longer a filler in this custom config — 3 chars, no preserve pattern
        # → falls through to the content branch: score == 0.7
        assert refiner.score_token("the") == 0.7

    def test_default_filler_words_constant_is_frozenset(self):
        assert isinstance(FILLER_WORDS, frozenset)
        assert "the" in FILLER_WORDS
        assert "basically" in FILLER_WORDS

    def test_default_preserve_patterns_is_list(self):
        assert isinstance(PRESERVE_PATTERNS, list)
        assert len(PRESERVE_PATTERNS) > 0
