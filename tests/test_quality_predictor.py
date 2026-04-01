"""Tests for src/quality_predictor.py — PoC compression quality prediction.

TDD test suite written before implementation.
"""

from __future__ import annotations

from src.quality_predictor import QualityPredictor
from src.compression_profiles import auto_select_profile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_predictor() -> QualityPredictor:
    return QualityPredictor()


SAMPLE_TEXT = (
    "The Python language is a high-level programming language. "
    "Python supports multiple programming paradigms. "
    "Python was created by Guido van Rossum in 1991. "
    "The language is widely used in data science and machine learning. "
    "Python has a large standard library and active community."
)


# ---------------------------------------------------------------------------
# predict_quality
# ---------------------------------------------------------------------------


class TestPredictQuality:
    def test_full_text_quality_near_1(self):
        """Same text as original predicts quality close to 1.0."""
        p = make_predictor()
        score = p.predict_quality(SAMPLE_TEXT, SAMPLE_TEXT)
        assert score >= 0.85, f"Expected >= 0.85, got {score}"

    def test_empty_compressed_quality_low(self):
        """Empty compressed text predicts low quality."""
        p = make_predictor()
        score = p.predict_quality(SAMPLE_TEXT, "")
        assert score < 0.3, f"Expected < 0.3, got {score}"

    def test_quality_monotonic(self):
        """More text kept = higher or equal quality score."""
        p = make_predictor()
        # More words retained -> higher quality
        full = SAMPLE_TEXT
        half = " ".join(SAMPLE_TEXT.split()[: len(SAMPLE_TEXT.split()) // 2])
        quarter = " ".join(SAMPLE_TEXT.split()[: len(SAMPLE_TEXT.split()) // 4])

        score_full = p.predict_quality(full, full)
        score_half = p.predict_quality(full, half)
        score_quarter = p.predict_quality(full, quarter)

        assert score_full >= score_half, f"full={score_full} should >= half={score_half}"
        assert score_half >= score_quarter, f"half={score_half} should >= quarter={score_quarter}"

    def test_predictor_empty_text(self):
        """Empty original text handled gracefully, returns 1.0 (nothing to lose)."""
        p = make_predictor()
        score = p.predict_quality("", "")
        assert 0.0 <= score <= 1.0

    def test_quality_score_in_range(self):
        """Quality score is always in [0.0, 1.0]."""
        p = make_predictor()
        score = p.predict_quality(SAMPLE_TEXT, "python machine learning")
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    def test_entity_extracts_capitalized(self):
        """Capitalized words not at sentence start are extracted as entities."""
        p = make_predictor()
        text = "I use Python and TensorFlow for machine learning."
        entities = p.extract_entities(text)
        # Python and TensorFlow should be found (capitalized mid-sentence)
        assert "Python" in entities or "TensorFlow" in entities

    def test_entity_empty_text(self):
        """Empty text returns empty list."""
        p = make_predictor()
        assert p.extract_entities("") == []


# ---------------------------------------------------------------------------
# compute_entity_retention
# ---------------------------------------------------------------------------


class TestComputeEntityRetention:
    def test_entity_retention_all_present(self):
        """All entities present in compressed text returns 1.0."""
        p = make_predictor()
        entities = ["Python", "TensorFlow", "Guido"]
        compressed = "Python and TensorFlow created by Guido"
        score = p.compute_entity_retention(entities, compressed)
        assert score == 1.0

    def test_entity_retention_none_present(self):
        """No entities present in compressed text returns 0.0."""
        p = make_predictor()
        entities = ["Python", "TensorFlow", "Guido"]
        compressed = "some unrelated text here"
        score = p.compute_entity_retention(entities, compressed)
        assert score == 0.0

    def test_entity_retention_partial(self):
        """Partial entity presence returns intermediate score."""
        p = make_predictor()
        entities = ["Python", "TensorFlow", "Guido"]
        compressed = "Python is great"
        score = p.compute_entity_retention(entities, compressed)
        assert 0.0 < score < 1.0

    def test_entity_retention_empty_entities(self):
        """Empty entities list returns 1.0 (nothing to retain)."""
        p = make_predictor()
        score = p.compute_entity_retention([], "any text")
        assert score == 1.0


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_coverage_full(self):
        """All sentences covered returns 1.0."""
        p = make_predictor()
        sentences = ["the cat sat", "the dog ran"]
        compressed = "cat dog sat ran"
        score = p.compute_coverage(sentences, compressed)
        assert score == 1.0

    def test_coverage_partial(self):
        """Only some sentences with a word in compressed < 1.0."""
        p = make_predictor()
        sentences = ["python is fast", "weather is cold", "science is hard"]
        # Only 'python' appears in compressed
        compressed = "python algorithms"
        score = p.compute_coverage(sentences, compressed)
        assert 0.0 < score < 1.0

    def test_coverage_none(self):
        """No sentence words in compressed returns 0.0."""
        p = make_predictor()
        sentences = ["one two three", "four five six"]
        compressed = "alpha beta gamma"
        score = p.compute_coverage(sentences, compressed)
        assert score == 0.0

    def test_coverage_empty_sentences(self):
        """Empty sentence list returns 1.0."""
        p = make_predictor()
        score = p.compute_coverage([], "any text")
        assert score == 1.0


# ---------------------------------------------------------------------------
# simulate_compression
# ---------------------------------------------------------------------------


class TestSimulateCompression:
    def test_simulate_returns_string(self):
        """simulate_compression returns a non-empty string for non-empty input."""
        p = make_predictor()
        result = p.simulate_compression(SAMPLE_TEXT, "balanced")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_simulate_full_profile_returns_most(self):
        """'full' profile returns more words than 'minimal' profile."""
        p = make_predictor()
        full_result = p.simulate_compression(SAMPLE_TEXT, "full")
        minimal_result = p.simulate_compression(SAMPLE_TEXT, "minimal")
        assert len(full_result.split()) >= len(minimal_result.split())

    def test_simulate_unknown_profile_defaults_balanced(self):
        """Unknown profile name defaults to balanced behavior gracefully."""
        p = make_predictor()
        # Should not raise
        result = p.simulate_compression(SAMPLE_TEXT, "unknown_profile")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# predict_all_profiles
# ---------------------------------------------------------------------------


class TestPredictAllProfiles:
    def test_predict_all_returns_dict(self):
        """predict_all_profiles returns a dict mapping profile name -> score."""
        p = make_predictor()
        results = p.predict_all_profiles(SAMPLE_TEXT)
        assert isinstance(results, dict)
        assert len(results) > 0
        for key, value in results.items():
            assert isinstance(key, str)
            assert 0.0 <= value <= 1.0

    def test_predict_all_includes_standard_profiles(self):
        """Result includes at least minimal, balanced, full profiles."""
        p = make_predictor()
        results = p.predict_all_profiles(SAMPLE_TEXT)
        assert "minimal" in results
        assert "balanced" in results
        assert "full" in results

    def test_predict_all_full_scores_highest(self):
        """'full' profile typically scores highest."""
        p = make_predictor()
        results = p.predict_all_profiles(SAMPLE_TEXT)
        full_score = results["full"]
        minimal_score = results["minimal"]
        assert full_score >= minimal_score


# ---------------------------------------------------------------------------
# auto_select_profile (from compression_profiles.py)
# ---------------------------------------------------------------------------


class TestAutoSelectProfile:
    def test_auto_select_high_floor_returns_full(self):
        """quality_floor=0.9 returns 'full' or 'detailed' profile."""
        profile = auto_select_profile(SAMPLE_TEXT, quality_floor=0.9)
        assert profile in ("full", "detailed"), f"Got: {profile}"

    def test_auto_select_low_floor_returns_minimal(self):
        """quality_floor=0.1 returns 'minimal' profile."""
        profile = auto_select_profile(SAMPLE_TEXT, quality_floor=0.1)
        assert profile == "minimal"

    def test_auto_select_medium_floor(self):
        """quality_floor=0.5 returns a middle-ish profile."""
        profile = auto_select_profile(SAMPLE_TEXT, quality_floor=0.5)
        assert profile in ("minimal", "summary", "balanced", "detailed", "full")

    def test_auto_select_returns_string(self):
        """auto_select_profile always returns a string profile name."""
        profile = auto_select_profile(SAMPLE_TEXT, quality_floor=0.7)
        assert isinstance(profile, str)

    def test_quality_floor_zero_returns_minimal(self):
        """quality_floor=0.0 returns the most compressed profile."""
        profile = auto_select_profile(SAMPLE_TEXT, quality_floor=0.0)
        assert profile == "minimal"

    def test_auto_select_empty_text(self):
        """Empty text handled gracefully, returns a valid profile."""
        profile = auto_select_profile("", quality_floor=0.5)
        assert profile in ("minimal", "summary", "balanced", "detailed", "full")
