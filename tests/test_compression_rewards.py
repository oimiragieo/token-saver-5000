"""
Tests for Compression Rewards System

Tests for:
- Individual reward component calculations
- CompressionReward aggregation
- CompressionRewardCalculator
- Progressive reward shaping
"""

import numpy as np

from src.compression_rewards import (
    SchemaValidationResult,
    SemanticPreservationResult,
    FidelityAdherenceResult,
    CompositionIntegrityResult,
    MemoryDisciplineResult,
    CompressionReward,
    CompressionRewardCalculator,
    ProgressiveRewardShaper,
    RewardComponent,
    FIDELITY_TARGET_RATIOS,
    FIDELITY_TOKEN_BUDGETS,
)


class TestSchemaValidationResult:
    """Tests for schema validation"""

    def test_all_valid(self):
        """Test all valid result"""
        result = SchemaValidationResult()
        assert result.all_valid
        assert result.score == 1.0

    def test_input_invalid(self):
        """Test invalid input"""
        result = SchemaValidationResult(input_valid=False, input_errors=["Empty input"])
        assert not result.all_valid
        assert result.score == 0.5

    def test_output_invalid(self):
        """Test invalid output"""
        result = SchemaValidationResult(output_valid=False, output_errors=["Malformed output"])
        assert not result.all_valid
        assert result.score == 0.5

    def test_both_invalid(self):
        """Test both invalid"""
        result = SchemaValidationResult(input_valid=False, output_valid=False)
        assert not result.all_valid
        assert result.score == 0.0


class TestSemanticPreservationResult:
    """Tests for semantic preservation"""

    def test_perfect_preservation(self):
        """Test perfect semantic preservation"""
        result = SemanticPreservationResult(
            ssim_score=1.0,
            embedding_similarity=1.0,
            structure_preservation=1.0,
            keyword_retention=1.0,
        )
        assert result.score == 1.0

    def test_weighted_score(self):
        """Test weighted score calculation"""
        result = SemanticPreservationResult(
            ssim_score=0.8,  # weight: 0.35
            embedding_similarity=0.9,  # weight: 0.30
            structure_preservation=0.7,  # weight: 0.20
            keyword_retention=0.6,  # weight: 0.15
        )
        expected = 0.35 * 0.8 + 0.30 * 0.9 + 0.20 * 0.7 + 0.15 * 0.6
        assert abs(result.score - expected) < 0.001

    def test_zero_scores(self):
        """Test all zero scores"""
        result = SemanticPreservationResult()
        assert result.score == 0.0


class TestFidelityAdherenceResult:
    """Tests for fidelity adherence"""

    def test_perfect_adherence(self):
        """Test perfect fidelity adherence"""
        result = FidelityAdherenceResult(
            target_fidelity="BALANCED",
            achieved_ratio=5.0,
            target_ratio=5.0,
            within_budget=True,
            budget_utilization=0.8,
        )
        assert result.ratio_score == 1.0
        assert result.score == 1.0

    def test_within_tolerance(self):
        """Test ratio within tolerance"""
        result = FidelityAdherenceResult(
            achieved_ratio=4.5, target_ratio=5.0, within_budget=True  # 10% below target
        )
        # 10% deviation = 0.9 ratio score
        assert result.ratio_score == 0.9
        assert result.score > 0.9

    def test_budget_exceeded(self):
        """Test budget exceeded"""
        result = FidelityAdherenceResult(achieved_ratio=5.0, target_ratio=5.0, within_budget=False)
        # Perfect ratio (1.0) but budget missed (0.5)
        assert result.score == 0.75


class TestCompositionIntegrityResult:
    """Tests for composition integrity"""

    def test_perfect_integrity(self):
        """Test perfect composition integrity"""
        result = CompositionIntegrityResult()
        assert result.score == 1.0

    def test_edge_inconsistency(self):
        """Test edge inconsistency"""
        result = CompositionIntegrityResult(edge_consistency=False)
        assert result.score < 1.0

    def test_orphan_nodes_penalty(self):
        """Test orphan nodes penalty"""
        result = CompositionIntegrityResult(orphan_nodes=3)
        # Penalty is 0.1 per orphan
        assert result.score < 1.0

    def test_disconnected_graph(self):
        """Test disconnected graph (partial credit)"""
        result = CompositionIntegrityResult(graph_connected=False)
        # Should get partial credit (0.5) instead of 0
        assert result.score > 0.5


class TestMemoryDisciplineResult:
    """Tests for memory discipline"""

    def test_no_growth(self):
        """Test no context growth"""
        result = MemoryDisciplineResult(
            context_growth_rate=0.0,
            eviction_efficiency=1.0,
            peak_memory_mb=50,
            memory_budget_mb=100,
        )
        assert result.growth_score == 1.0
        assert result.memory_score == 1.0
        assert result.score == 1.0

    def test_high_growth(self):
        """Test high context growth"""
        result = MemoryDisciplineResult(
            context_growth_rate=0.5,  # 50% growth
            eviction_efficiency=1.0,
            peak_memory_mb=50,
            memory_budget_mb=100,
        )
        assert result.growth_score == 0.0
        assert result.score < 1.0

    def test_memory_over_budget(self):
        """Test memory over budget"""
        result = MemoryDisciplineResult(
            context_growth_rate=0.0,
            eviction_efficiency=1.0,
            peak_memory_mb=120,  # 120% of budget
            memory_budget_mb=100,
        )
        assert result.memory_score < 1.0


class TestCompressionReward:
    """Tests for CompressionReward aggregation"""

    def test_perfect_reward(self):
        """Test perfect reward score"""
        reward = CompressionReward(
            schema=SchemaValidationResult(),
            semantic=SemanticPreservationResult(
                ssim_score=1.0,
                embedding_similarity=1.0,
                structure_preservation=1.0,
                keyword_retention=1.0,
            ),
            fidelity=FidelityAdherenceResult(
                achieved_ratio=5.0, target_ratio=5.0, within_budget=True
            ),
            composition=CompositionIntegrityResult(),
            memory=MemoryDisciplineResult(eviction_efficiency=1.0),
        )

        assert reward.total_reward == 1.0
        assert reward.passes_threshold()

    def test_component_scores(self):
        """Test component scores dict"""
        reward = CompressionReward()
        scores = reward.component_scores

        assert RewardComponent.SCHEMA in scores
        assert RewardComponent.SEMANTIC in scores
        assert RewardComponent.FIDELITY in scores
        assert RewardComponent.COMPOSITION in scores
        assert RewardComponent.MEMORY in scores

    def test_weakest_component(self):
        """Test weakest component identification"""
        reward = CompressionReward(
            schema=SchemaValidationResult(input_valid=False),  # 0.5
            semantic=SemanticPreservationResult(ssim_score=0.9, embedding_similarity=0.9),
            fidelity=FidelityAdherenceResult(
                achieved_ratio=5.0, target_ratio=5.0, within_budget=True
            ),
            composition=CompositionIntegrityResult(),
            memory=MemoryDisciplineResult(eviction_efficiency=1.0),
        )

        weakest, score = reward.weakest_component
        assert weakest == RewardComponent.SCHEMA
        assert score == 0.5

    def test_to_dict(self):
        """Test serialization to dict"""
        reward = CompressionReward()
        d = reward.to_dict()

        assert "total_reward" in d
        assert "component_scores" in d
        assert "weakest_component" in d
        assert "schema" in d
        assert "semantic" in d

    def test_passes_threshold(self):
        """Test threshold checking"""
        low_reward = CompressionReward(
            schema=SchemaValidationResult(input_valid=False, output_valid=False)
        )
        assert not low_reward.passes_threshold(0.7)

        # With defaults (all 1.0 or close)
        high_reward = CompressionReward(
            semantic=SemanticPreservationResult(
                ssim_score=0.9,
                embedding_similarity=0.9,
                structure_preservation=0.9,
                keyword_retention=0.9,
            )
        )
        assert high_reward.passes_threshold(0.5)


class TestCompressionRewardCalculator:
    """Tests for CompressionRewardCalculator"""

    def test_calculate_basic(self):
        """Test basic calculation"""
        calc = CompressionRewardCalculator()

        reward = calc.calculate(
            input_text="Hello world this is a test document",
            output_text="Hello test document",
            input_tokens=8,
            output_tokens=3,
            fidelity_level="BALANCED",
            node_map={"node1": "description"},
        )

        assert isinstance(reward, CompressionReward)
        assert 0.0 <= reward.total_reward <= 1.0

    def test_empty_input_fails(self):
        """Test that empty input fails schema validation"""
        calc = CompressionRewardCalculator()

        reward = calc.calculate(
            input_text="",
            output_text="output",
            input_tokens=0,
            output_tokens=1,
            fidelity_level="BALANCED",
        )

        assert reward.schema.input_valid is False
        assert reward.schema_score < 1.0

    def test_with_embeddings(self):
        """Test with embedding vectors"""
        calc = CompressionRewardCalculator()

        input_emb = np.array([1.0, 0.0, 0.0])
        output_emb = np.array([0.9, 0.1, 0.0])

        reward = calc.calculate(
            input_text="test input",
            output_text="test output",
            input_tokens=10,
            output_tokens=5,
            fidelity_level="BALANCED",
            input_embedding=input_emb,
            output_embedding=output_emb,
        )

        # Should have high embedding similarity
        assert reward.semantic.embedding_similarity > 0.8

    def test_with_ssim(self):
        """Test with pre-calculated SSIM"""
        calc = CompressionRewardCalculator()

        reward = calc.calculate(
            input_text="test",
            output_text="test",
            input_tokens=10,
            output_tokens=5,
            fidelity_level="BALANCED",
            ssim_score=0.95,
        )

        assert reward.semantic.ssim_score == 0.95

    def test_with_graph_edges(self):
        """Test with graph edges"""
        calc = CompressionRewardCalculator()

        node_map = {"n1": "desc1", "n2": "desc2"}
        edges = [("n1", "n2")]

        reward = calc.calculate(
            input_text="test input text",
            output_text="test output",
            input_tokens=10,
            output_tokens=5,
            fidelity_level="BALANCED",
            node_map=node_map,
            graph_edges=edges,
        )

        assert reward.composition.edge_consistency is True

    def test_batch_calculate(self):
        """Test batch calculation"""
        calc = CompressionRewardCalculator()

        operations = [
            {
                "input_text": f"document {i}",
                "output_text": f"doc {i}",
                "input_tokens": 10,
                "output_tokens": 5,
                "fidelity_level": "BALANCED",
            }
            for i in range(5)
        ]

        rewards = calc.calculate_batch(operations)
        assert len(rewards) == 5
        assert all(isinstance(r, CompressionReward) for r in rewards)

    def test_aggregate_rewards(self):
        """Test reward aggregation"""
        calc = CompressionRewardCalculator()

        rewards = [
            calc.calculate(
                input_text=f"doc {i}",
                output_text=f"d{i}",
                input_tokens=10,
                output_tokens=5,
                fidelity_level="BALANCED",
            )
            for i in range(10)
        ]

        stats = calc.aggregate_rewards(rewards)

        assert stats["count"] == 10
        assert "total_reward" in stats
        assert "mean" in stats["total_reward"]
        assert "component_means" in stats
        assert "pass_rate" in stats


class TestProgressiveRewardShaper:
    """Tests for progressive reward shaping"""

    def test_initial_phase(self):
        """Test initial phase weights"""
        shaper = ProgressiveRewardShaper(total_phases=4)
        weights = shaper.get_weights_for_phase(0)

        # Phase 0 should prioritize schema
        assert weights[RewardComponent.SCHEMA] > weights[RewardComponent.SEMANTIC]

    def test_final_phase(self):
        """Test final phase weights"""
        shaper = ProgressiveRewardShaper(total_phases=4)
        weights = shaper.get_weights_for_phase(3)

        # Phase 3 should prioritize efficiency
        assert weights[RewardComponent.MEMORY] >= 0.15

    def test_phase_advancement(self):
        """Test phase advancement"""
        shaper = ProgressiveRewardShaper(total_phases=4)
        assert shaper.current_phase == 0

        shaper.advance_phase()
        assert shaper.current_phase == 1

        shaper.advance_phase()
        assert shaper.current_phase == 2

    def test_phase_cap(self):
        """Test phase doesn't exceed maximum"""
        shaper = ProgressiveRewardShaper(total_phases=2)
        shaper.advance_phase()
        shaper.advance_phase()
        shaper.advance_phase()

        assert shaper.current_phase == 1  # Capped at max - 1

    def test_get_current_calculator(self):
        """Test getting calculator with current phase weights"""
        shaper = ProgressiveRewardShaper()
        calc = shaper.get_current_calculator()

        assert isinstance(calc, CompressionRewardCalculator)
        assert calc.weights == shaper.get_weights_for_phase(0)


class TestFidelityConstants:
    """Tests for fidelity constants"""

    def test_target_ratios(self):
        """Test target ratios are defined"""
        assert "ABSTRACT" in FIDELITY_TARGET_RATIOS
        assert "RAW" in FIDELITY_TARGET_RATIOS
        assert FIDELITY_TARGET_RATIOS["ABSTRACT"] > FIDELITY_TARGET_RATIOS["RAW"]

    def test_token_budgets(self):
        """Test token budgets are defined"""
        assert "ABSTRACT" in FIDELITY_TOKEN_BUDGETS
        assert "RAW" in FIDELITY_TOKEN_BUDGETS
        assert FIDELITY_TOKEN_BUDGETS["ABSTRACT"] < FIDELITY_TOKEN_BUDGETS["DETAILED"]
