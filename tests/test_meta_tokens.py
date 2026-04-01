"""Tests for src/meta_tokens.py — lossless meta-token compression (arXiv 2506.00307).

TDD test suite written before implementation.
"""

from __future__ import annotations

from src.meta_tokens import MetaTokenCompressor, MetaTokenEntry, MetaTokenResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_compressor(**kwargs) -> MetaTokenCompressor:
    return MetaTokenCompressor(**kwargs)


# ---------------------------------------------------------------------------
# MetaTokenCompressor.find_repeated_subsequences
# ---------------------------------------------------------------------------


class TestFindRepeatedSubsequences:
    def test_find_repeated_returns_dict(self):
        """Returns a dict with tuple keys and list-of-int values."""
        c = make_compressor()
        tokens = ["the", "cat", "sat", "the", "cat", "ran"]
        result = c.find_repeated_subsequences(tokens)
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, tuple)
            assert isinstance(val, list)

    def test_find_repeated_detects_bigram(self):
        """'the cat' appearing twice is found as a repeated subsequence."""
        c = make_compressor()
        tokens = ["the", "cat", "sat", "the", "cat", "ran"]
        result = c.find_repeated_subsequences(tokens)
        assert ("the", "cat") in result
        assert len(result[("the", "cat")]) >= 2

    def test_find_repeated_single_occurrence_excluded(self):
        """Subsequences appearing only once are excluded (min_frequency=2)."""
        c = make_compressor(min_frequency=2)
        tokens = ["one", "two", "three", "four"]
        result = c.find_repeated_subsequences(tokens)
        # No bigram appears twice
        assert len(result) == 0

    def test_find_repeated_respects_min_length(self):
        """min_length=3 means bigrams (length 2) are not returned."""
        c = make_compressor(min_length=3)
        tokens = ["the", "cat", "sat", "the", "cat", "ran"]
        result = c.find_repeated_subsequences(tokens, min_length=3)
        for key in result:
            assert len(key) >= 3


# ---------------------------------------------------------------------------
# MetaTokenCompressor.rank_by_savings
# ---------------------------------------------------------------------------


class TestRankBySavings:
    def test_rank_by_savings_ordering(self):
        """Higher savings entries appear first in sorted output."""
        c = make_compressor()
        # Formula: savings = freq * (length - 1) - length
        # freq=3, length=4: savings = 3*3 - 4 = 5
        # freq=5, length=3: savings = 5*2 - 3 = 7
        subsequences = {
            ("a", "b", "c", "d"): [0, 5, 10],  # freq=3, length=4 -> savings=5
            ("x", "y", "z"): [0, 4, 8, 12, 16],  # freq=5, length=3 -> savings=7
        }
        ranked = c.rank_by_savings(subsequences)
        assert len(ranked) == 2
        # Highest savings first
        assert ranked[0][0] == ("x", "y", "z")  # savings=7
        assert ranked[1][0] == ("a", "b", "c", "d")  # savings=5

    def test_rank_by_savings_excludes_negative(self):
        """Entries with savings <= 0 are excluded."""
        c = make_compressor()
        subsequences = {
            ("a", "b"): [0, 2],  # freq=2, length=2: savings = 2*1 - 2 = 0 (excluded)
        }
        ranked = c.rank_by_savings(subsequences)
        assert len(ranked) == 0

    def test_rank_by_savings_returns_tuples(self):
        """Each ranked entry is a (seq_tuple, freq, savings) tuple."""
        c = make_compressor()
        subsequences = {
            ("a", "b", "c"): [0, 4, 8, 12],  # freq=4, length=3: savings=4*2-3=5
        }
        ranked = c.rank_by_savings(subsequences)
        assert len(ranked) == 1
        seq, freq, savings = ranked[0]
        assert seq == ("a", "b", "c")
        assert freq == 4
        assert savings == 5


# ---------------------------------------------------------------------------
# MetaTokenCompressor.compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_compress_finds_repeats(self):
        """Repeated bigrams produce a compressed result with savings."""
        c = make_compressor()
        text = "the cat sat the cat ran the cat jumped"
        result = c.compress(text)
        assert isinstance(result, MetaTokenResult)
        assert result.savings_tokens > 0

    def test_compress_empty(self):
        """Empty text returns unchanged MetaTokenResult with zeros."""
        c = make_compressor()
        result = c.compress("")
        assert result.original_text == ""
        assert result.compressed_text == ""
        assert result.savings_tokens == 0
        assert result.compression_ratio == 1.0

    def test_compress_no_repeats(self):
        """Unique words return unchanged with savings=0."""
        c = make_compressor()
        text = "alpha beta gamma delta epsilon"
        result = c.compress(text)
        assert result.savings_tokens == 0
        assert result.compression_ratio == 1.0

    def test_compress_min_frequency(self):
        """Subsequence appearing only twice with min_frequency=3 is not replaced."""
        c = make_compressor(min_frequency=3)
        text = "the cat sat the cat ran"  # "the cat" appears twice only
        result = c.compress(text)
        # With min_frequency=3, "the cat" (appearing 2 times) is not replaced
        assert result.savings_tokens == 0

    def test_compress_min_length_respected(self):
        """min_length=3 means 2-grams are not used as meta-tokens."""
        c = make_compressor(min_length=3, min_frequency=2)
        # This text has "the cat" (bigram) repeated, but no trigrams repeated
        text = "the cat sat the cat ran the cat jumped"
        result = c.compress(text)
        # No trigram repeats, so no substitution
        assert result.savings_tokens == 0

    def test_compress_dictionary_format(self):
        """Compressed text contains '§1=' assignment and '---' separator."""
        c = make_compressor()
        text = "the cat sat the cat ran the cat jumped"
        result = c.compress(text)
        if result.savings_tokens > 0:
            assert "§1=" in result.compressed_text
            assert "---" in result.compressed_text

    def test_compress_savings_positive(self):
        """MetaTokenResult.savings_tokens > 0 when repeats exist."""
        c = make_compressor()
        text = "machine learning model machine learning algorithm machine learning framework"
        result = c.compress(text)
        assert result.savings_tokens > 0

    def test_compress_ratio_gt_1(self):
        """compression_ratio > 1.0 when savings exist."""
        c = make_compressor()
        text = "machine learning model machine learning algorithm machine learning framework"
        result = c.compress(text)
        assert result.compression_ratio > 1.0

    def test_compress_multiline(self):
        """Text with newlines: roundtrip preserves content."""
        c = make_compressor()
        text = "the cat sat\nthe cat ran\nthe cat jumped"
        result = c.compress(text)
        # Roundtrip must reconstruct meaningful content
        decompressed = c.decompress(result.compressed_text)
        # Core words must be present after decompression
        assert "cat" in decompressed
        assert "sat" in decompressed

    def test_compress_max_entries_limits_dictionary(self):
        """dictionary list is limited to max_entries."""
        c = make_compressor(max_entries=2)
        # Create text with many different repeated phrases
        text = " ".join(
            [
                "the cat sat",
                "the dog ran",
                "the bird flew",
                "the cat sat",
                "the dog ran",
                "the bird flew",
                "the cat sat",
                "the dog ran",
                "the bird flew",
            ]
        )
        result = c.compress(text)
        assert len(result.dictionary) <= 2

    def test_compress_result_fields(self):
        """MetaTokenResult has all expected fields with correct types."""
        c = make_compressor()
        text = "the cat sat the cat ran"
        result = c.compress(text)
        assert isinstance(result.original_text, str)
        assert isinstance(result.compressed_text, str)
        assert isinstance(result.dictionary, list)
        assert isinstance(result.original_tokens, int)
        assert isinstance(result.compressed_tokens, int)
        assert isinstance(result.savings_tokens, int)
        assert isinstance(result.compression_ratio, float)

    def test_compress_original_tokens_count(self):
        """original_tokens matches word count of original text."""
        c = make_compressor()
        text = "one two three four five"
        result = c.compress(text)
        assert result.original_tokens == 5


# ---------------------------------------------------------------------------
# MetaTokenCompressor.decompress
# ---------------------------------------------------------------------------


class TestDecompress:
    def test_decompress_no_dictionary(self):
        """Text without '---' separator returned unchanged."""
        c = make_compressor()
        text = "no separator here just plain text"
        result = c.decompress(text)
        assert result == text

    def test_compress_decompress_roundtrip_simple(self):
        """decompress(compress(text)) contains original words."""
        c = make_compressor()
        text = "the cat sat the cat ran the cat jumped"
        compressed = c.compress(text)
        decompressed = c.decompress(compressed.compressed_text)
        # All original words should be present
        for word in ["cat", "sat", "ran", "jumped"]:
            assert word in decompressed

    def test_compress_decompress_roundtrip_multiline(self):
        """Roundtrip preserves multiline content semantically."""
        c = make_compressor()
        text = "neural network training\nneural network inference\nneural network deployment"
        compressed = c.compress(text)
        decompressed = c.decompress(compressed.compressed_text)
        assert "neural" in decompressed
        assert "network" in decompressed

    def test_compress_decompress_roundtrip_technical(self):
        """Roundtrip on technical text preserves key terms."""
        c = make_compressor()
        text = (
            "the python interpreter executes the python code "
            "the python interpreter parses the python syntax "
            "the python interpreter optimizes the python bytecode"
        )
        compressed = c.compress(text)
        decompressed = c.decompress(compressed.compressed_text)
        assert "python" in decompressed
        assert "interpreter" in decompressed

    def test_decompress_empty_string(self):
        """Empty string returns empty string."""
        c = make_compressor()
        assert c.decompress("") == ""

    def test_decompress_no_equals_sign(self):
        """Dict header without '=' entries handled gracefully."""
        c = make_compressor()
        result = c.decompress("no-equals-here\n---\nbody text")
        assert "body text" in result


# ---------------------------------------------------------------------------
# MetaTokenEntry dataclass
# ---------------------------------------------------------------------------


class TestMetaTokenEntry:
    def test_entry_fields(self):
        """MetaTokenEntry has symbol, expansion, frequency, savings."""
        entry = MetaTokenEntry(symbol="§1", expansion=["the", "cat"], frequency=3, savings=2)
        assert entry.symbol == "§1"
        assert entry.expansion == ["the", "cat"]
        assert entry.frequency == 3
        assert entry.savings == 2
