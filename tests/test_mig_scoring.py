"""Tests for MIG (Marginal Information Gain) scoring in src/token_refiner.py.

Based on COMI arXiv 2602.01719 — tests written TDD-first.
"""

from __future__ import annotations

from src.token_refiner import MIGConfig, MIGScorer, TokenRefiner

# ---------------------------------------------------------------------------
# MIGConfig defaults
# ---------------------------------------------------------------------------


class TestMIGConfig:
    def test_mig_config_defaults(self):
        """MIGConfig() has lambda_redundancy=0.5 by default."""
        cfg = MIGConfig()
        assert cfg.lambda_redundancy == 0.5

    def test_mig_config_min_corpus_tokens(self):
        """MIGConfig() has min_corpus_tokens=10 by default."""
        cfg = MIGConfig()
        assert cfg.min_corpus_tokens == 10

    def test_mig_config_custom(self):
        """Custom lambda is respected."""
        cfg = MIGConfig(lambda_redundancy=0.8)
        assert cfg.lambda_redundancy == 0.8


# ---------------------------------------------------------------------------
# MIGScorer.score_tokens_mig
# ---------------------------------------------------------------------------


class TestMIGScorer:
    def test_mig_scorer_returns_pairs(self):
        """Returns a list of (str, float) pairs."""
        scorer = MIGScorer()
        tokens = ["python", "weather", "algorithm"]
        result = scorer.score_tokens_mig(tokens, query="python")
        assert isinstance(result, list)
        for item in result:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_mig_empty_tokens(self):
        """Empty token list returns empty list."""
        scorer = MIGScorer()
        result = scorer.score_tokens_mig([], query="anything")
        assert result == []

    def test_mig_single_token(self):
        """Single token returns one scored pair."""
        scorer = MIGScorer()
        result = scorer.score_tokens_mig(["python"], query="python")
        assert len(result) == 1
        token, score = result[0]
        assert token == "python"
        assert 0.0 <= score <= 1.0

    def test_mig_relevance_scores_query_tokens_higher(self):
        """Token matching query scores higher than unrelated token."""
        scorer = MIGScorer()
        # Use tokens clearly related to vs. unrelated to query
        tokens = ["python", "programming", "sunny", "weather"]
        result = scorer.score_tokens_mig(tokens, query="python programming")
        score_map = dict(result)
        # python/programming should score higher than weather on average
        relevant_avg = (score_map["python"] + score_map["programming"]) / 2
        irrelevant_avg = (score_map["sunny"] + score_map["weather"]) / 2
        assert relevant_avg > irrelevant_avg

    def test_mig_preserves_protected_patterns(self):
        """Numbers and identifiers matching PRESERVE_PATTERNS score 1.0."""
        scorer = MIGScorer()
        tokens = ["42", "my_variable", "normal_word"]
        result = scorer.score_tokens_mig(tokens, query="something")
        score_map = dict(result)
        assert score_map["42"] == 1.0
        assert score_map["my_variable"] == 1.0

    def test_mig_scores_in_range(self):
        """All scores are in [0.0, 1.0]."""
        scorer = MIGScorer()
        tokens = ["the", "cat", "sat", "on", "the", "mat"]
        result = scorer.score_tokens_mig(tokens, query="feline")
        for _, score in result:
            assert 0.0 <= score <= 1.0, f"Score {score} out of range"

    def test_mig_token_count_preserved(self):
        """Output has same number of entries as input tokens."""
        scorer = MIGScorer()
        tokens = ["alpha", "beta", "gamma", "delta", "epsilon"]
        result = scorer.score_tokens_mig(tokens, query="test")
        assert len(result) == len(tokens)

    def test_mig_token_identity_preserved(self):
        """Output tokens match input tokens in the same order."""
        scorer = MIGScorer()
        tokens = ["first", "second", "third"]
        result = scorer.score_tokens_mig(tokens, query="first")
        out_tokens = [t for t, _ in result]
        assert out_tokens == tokens


# ---------------------------------------------------------------------------
# TokenRefiner integration with MIG
# ---------------------------------------------------------------------------


class TestTokenRefinerMIG:
    def test_refiner_scoring_mode_default_is_heuristic(self):
        """Default TokenRefiner uses heuristic scoring (scoring_mode='heuristic')."""
        refiner = TokenRefiner()
        assert refiner.scoring_mode == "heuristic"

    def test_refiner_with_mig_and_query(self):
        """scoring_mode='mig' + query= returns a RefinedResult."""
        refiner = TokenRefiner(scoring_mode="mig")
        text = "the python interpreter is fast the python code is efficient"
        result = refiner.refine(text, target_ratio=0.7, query="python speed")
        assert result.refined_text != ""
        assert result.original_tokens > 0

    def test_refiner_backward_compat(self):
        """Existing refine() calls without query still work (backward compat)."""
        refiner = TokenRefiner()
        text = "the cat sat on the mat the cat ran away"
        result = refiner.refine(text, target_ratio=0.7)
        # Must complete without error and return RefinedResult
        assert result.original_tokens > 0
        assert isinstance(result.refined_text, str)

    def test_refiner_mig_mode_no_query_falls_back(self):
        """MIG mode without query falls back to heuristic scoring gracefully."""
        refiner = TokenRefiner(scoring_mode="mig")
        text = "the python code is fast and efficient"
        # No query provided — should fall back without raising
        result = refiner.refine(text, target_ratio=0.7)
        assert isinstance(result.refined_text, str)

    def test_refiner_mig_config_accepted(self):
        """TokenRefiner accepts mig_config parameter."""
        cfg = MIGConfig(lambda_redundancy=0.8)
        refiner = TokenRefiner(scoring_mode="mig", mig_config=cfg)
        assert refiner.mig_config is not None
        assert refiner.mig_config.lambda_redundancy == 0.8

    def test_mig_falls_back_without_query(self):
        """MIGScorer falls back gracefully when query is empty."""
        scorer = MIGScorer()
        tokens = ["alpha", "beta", "gamma"]
        result = scorer.score_tokens_mig(tokens, query="")
        # Should return scored pairs without error
        assert len(result) == len(tokens)
        for _, score in result:
            assert 0.0 <= score <= 1.0
