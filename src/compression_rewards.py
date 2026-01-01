"""
Decomposed Compression Rewards System

Implements multi-component reward design inspired by ASG-SI paper:
- Tool Validity → Schema correctness
- Outcome Verification → Semantic preservation
- Skill Reuse → Fidelity adherence
- Composition Integrity → Graph consistency
- Memory Discipline → Context growth control

Each component produces a 0.0-1.0 score that can be weighted and combined.
The decomposed structure enables attribution of quality issues to specific
aspects of compression.

Usage:
    from src.compression_rewards import CompressionRewardCalculator

    calculator = CompressionRewardCalculator()
    reward = calculator.calculate(
        input_doc=document,
        output_skeleton=skeleton,
        fidelity_level=FidelityLevel.BALANCED,
        context_growth=0.15
    )
    print(f"Total reward: {reward.total_reward:.3f}")
    print(f"Schema score: {reward.schema_score:.3f}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.semantic_compressor import SkeletonResponse, FidelityLevel

logger = logging.getLogger(__name__)


# Expected compression ratios per fidelity level
FIDELITY_TARGET_RATIOS = {
    "ABSTRACT": 20.0,   # ~95% reduction
    "OUTLINE": 10.0,    # ~90% reduction
    "STRUCTURE": 5.0,   # ~80% reduction
    "DETAILED": 2.5,    # ~60% reduction
    "RAW": 1.0,         # No reduction
}

# Token budgets per fidelity level (approximate)
FIDELITY_TOKEN_BUDGETS = {
    "ABSTRACT": 50,
    "OUTLINE": 100,
    "STRUCTURE": 200,
    "DETAILED": 500,
    "RAW": float("inf"),
}


class RewardComponent(Enum):
    """Individual reward components"""
    SCHEMA = "schema"
    SEMANTIC = "semantic"
    FIDELITY = "fidelity"
    COMPOSITION = "composition"
    MEMORY = "memory"


@dataclass
class SchemaValidationResult:
    """Result of schema validation checks"""
    input_valid: bool = True
    input_errors: List[str] = field(default_factory=list)
    output_valid: bool = True
    output_errors: List[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return self.input_valid and self.output_valid

    @property
    def score(self) -> float:
        """Calculate schema score (0.0-1.0)"""
        input_score = 1.0 if self.input_valid else 0.0
        output_score = 1.0 if self.output_valid else 0.0
        return (input_score + output_score) / 2.0


@dataclass
class SemanticPreservationResult:
    """Result of semantic preservation checks"""
    ssim_score: float = 0.0
    embedding_similarity: float = 0.0
    structure_preservation: float = 0.0
    keyword_retention: float = 0.0

    @property
    def score(self) -> float:
        """Calculate weighted semantic score"""
        weights = [0.35, 0.30, 0.20, 0.15]
        scores = [
            self.ssim_score,
            self.embedding_similarity,
            self.structure_preservation,
            self.keyword_retention
        ]
        return sum(w * s for w, s in zip(weights, scores))


@dataclass
class FidelityAdherenceResult:
    """Result of fidelity adherence checks"""
    target_fidelity: str = "BALANCED"
    achieved_ratio: float = 1.0
    target_ratio: float = 1.0
    within_budget: bool = True
    budget_utilization: float = 1.0

    @property
    def ratio_score(self) -> float:
        """Score based on achieving target compression"""
        if self.target_ratio == 0:
            return 0.0
        # Allow 20% tolerance
        ratio_diff = abs(self.achieved_ratio - self.target_ratio) / self.target_ratio
        return max(0.0, 1.0 - ratio_diff)

    @property
    def score(self) -> float:
        """Calculate fidelity adherence score"""
        budget_score = 1.0 if self.within_budget else 0.5
        return (self.ratio_score + budget_score) / 2.0


@dataclass
class CompositionIntegrityResult:
    """Result of composition integrity checks"""
    node_consistency: bool = True
    edge_consistency: bool = True
    node_id_valid: bool = True
    graph_connected: bool = True
    orphan_nodes: int = 0

    @property
    def score(self) -> float:
        """Calculate composition integrity score"""
        scores = [
            1.0 if self.node_consistency else 0.0,
            1.0 if self.edge_consistency else 0.0,
            1.0 if self.node_id_valid else 0.0,
            1.0 if self.graph_connected else 0.5,  # Partial credit
            max(0.0, 1.0 - self.orphan_nodes * 0.1),  # Penalty for orphans
        ]
        return sum(scores) / len(scores)


@dataclass
class MemoryDisciplineResult:
    """Result of memory discipline checks"""
    context_growth_rate: float = 0.0  # New tokens / total tokens
    eviction_efficiency: float = 1.0  # LRU effectiveness
    peak_memory_mb: float = 0.0
    memory_budget_mb: float = 100.0

    @property
    def growth_score(self) -> float:
        """Score based on context growth (lower is better)"""
        # Ideal: 0% growth, Bad: 50%+ growth
        return max(0.0, 1.0 - self.context_growth_rate * 2)

    @property
    def memory_score(self) -> float:
        """Score based on memory usage"""
        if self.memory_budget_mb == 0:
            return 0.0
        utilization = self.peak_memory_mb / self.memory_budget_mb
        return max(0.0, 1.0 - max(0, utilization - 0.8) * 5)

    @property
    def score(self) -> float:
        """Calculate memory discipline score"""
        return (
            self.growth_score * 0.4 +
            self.eviction_efficiency * 0.3 +
            self.memory_score * 0.3
        )


@dataclass
class CompressionReward:
    """
    Multi-component reward for compression quality.

    Implements ASG-SI's decomposed reward design with five components:
    1. Schema Validity - Input/output schema correctness
    2. Semantic Preservation - Quality of compression
    3. Fidelity Adherence - Meeting compression targets
    4. Composition Integrity - Graph consistency
    5. Memory Discipline - Context growth control
    """
    schema: SchemaValidationResult = field(default_factory=SchemaValidationResult)
    semantic: SemanticPreservationResult = field(default_factory=SemanticPreservationResult)
    fidelity: FidelityAdherenceResult = field(default_factory=FidelityAdherenceResult)
    composition: CompositionIntegrityResult = field(default_factory=CompositionIntegrityResult)
    memory: MemoryDisciplineResult = field(default_factory=MemoryDisciplineResult)

    # Component weights (must sum to 1.0)
    weights: Dict[RewardComponent, float] = field(default_factory=lambda: {
        RewardComponent.SCHEMA: 0.15,
        RewardComponent.SEMANTIC: 0.35,
        RewardComponent.FIDELITY: 0.20,
        RewardComponent.COMPOSITION: 0.15,
        RewardComponent.MEMORY: 0.15,
    })

    @property
    def schema_score(self) -> float:
        return self.schema.score

    @property
    def semantic_score(self) -> float:
        return self.semantic.score

    @property
    def fidelity_score(self) -> float:
        return self.fidelity.score

    @property
    def composition_score(self) -> float:
        return self.composition.score

    @property
    def memory_score(self) -> float:
        return self.memory.score

    @property
    def component_scores(self) -> Dict[RewardComponent, float]:
        """Get all component scores"""
        return {
            RewardComponent.SCHEMA: self.schema_score,
            RewardComponent.SEMANTIC: self.semantic_score,
            RewardComponent.FIDELITY: self.fidelity_score,
            RewardComponent.COMPOSITION: self.composition_score,
            RewardComponent.MEMORY: self.memory_score,
        }

    @property
    def total_reward(self) -> float:
        """Calculate weighted total reward"""
        total = 0.0
        for component, weight in self.weights.items():
            total += weight * self.component_scores[component]
        return total

    @property
    def weakest_component(self) -> Tuple[RewardComponent, float]:
        """Identify the weakest component for improvement focus"""
        scores = self.component_scores
        weakest = min(scores.items(), key=lambda x: x[1])
        return weakest

    def passes_threshold(self, threshold: float = 0.7) -> bool:
        """Check if reward passes minimum threshold"""
        return self.total_reward >= threshold

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "total_reward": self.total_reward,
            "component_scores": {k.value: v for k, v in self.component_scores.items()},
            "passes_threshold": self.passes_threshold(),
            "weakest_component": self.weakest_component[0].value,
            "schema": {
                "input_valid": self.schema.input_valid,
                "output_valid": self.schema.output_valid,
                "score": self.schema_score
            },
            "semantic": {
                "ssim_score": self.semantic.ssim_score,
                "embedding_similarity": self.semantic.embedding_similarity,
                "structure_preservation": self.semantic.structure_preservation,
                "keyword_retention": self.semantic.keyword_retention,
                "score": self.semantic_score
            },
            "fidelity": {
                "target_fidelity": self.fidelity.target_fidelity,
                "achieved_ratio": self.fidelity.achieved_ratio,
                "within_budget": self.fidelity.within_budget,
                "score": self.fidelity_score
            },
            "composition": {
                "node_consistency": self.composition.node_consistency,
                "edge_consistency": self.composition.edge_consistency,
                "orphan_nodes": self.composition.orphan_nodes,
                "score": self.composition_score
            },
            "memory": {
                "context_growth_rate": self.memory.context_growth_rate,
                "eviction_efficiency": self.memory.eviction_efficiency,
                "score": self.memory_score
            }
        }


class CompressionRewardCalculator:
    """
    Calculator for decomposed compression rewards.

    Evaluates compression operations across five dimensions and
    produces a CompressionReward with individual component scores.
    """

    def __init__(
        self,
        weights: Optional[Dict[RewardComponent, float]] = None,
        memory_budget_mb: float = 100.0
    ):
        """
        Initialize reward calculator.

        Args:
            weights: Optional custom weights for components
            memory_budget_mb: Memory budget for memory discipline scoring
        """
        self.weights = weights or {
            RewardComponent.SCHEMA: 0.15,
            RewardComponent.SEMANTIC: 0.35,
            RewardComponent.FIDELITY: 0.20,
            RewardComponent.COMPOSITION: 0.15,
            RewardComponent.MEMORY: 0.15,
        }
        self.memory_budget_mb = memory_budget_mb

    def calculate(
        self,
        input_text: str,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
        fidelity_level: str = "BALANCED",
        node_map: Optional[Dict[str, str]] = None,
        graph_edges: Optional[List[Tuple[str, str]]] = None,
        input_embedding: Optional[np.ndarray] = None,
        output_embedding: Optional[np.ndarray] = None,
        ssim_score: Optional[float] = None,
        context_growth_rate: float = 0.0,
        eviction_efficiency: float = 1.0,
        peak_memory_mb: float = 0.0,
    ) -> CompressionReward:
        """
        Calculate decomposed compression reward.

        Args:
            input_text: Original document text
            output_text: Compressed skeleton text
            input_tokens: Original token count
            output_tokens: Compressed token count
            fidelity_level: Target fidelity level
            node_map: Node ID to description mapping
            graph_edges: List of (source, target) edges
            input_embedding: Embedding of original document
            output_embedding: Embedding of compressed skeleton
            ssim_score: Pre-calculated SSIM score
            context_growth_rate: Rate of context growth
            eviction_efficiency: LRU eviction efficiency
            peak_memory_mb: Peak memory usage

        Returns:
            CompressionReward with all component scores
        """
        # Validate schema
        schema = self._validate_schema(input_text, output_text, node_map)

        # Calculate semantic preservation
        semantic = self._calculate_semantic(
            input_text, output_text,
            input_embedding, output_embedding,
            ssim_score
        )

        # Calculate fidelity adherence
        fidelity = self._calculate_fidelity(
            input_tokens, output_tokens, fidelity_level
        )

        # Calculate composition integrity
        composition = self._calculate_composition(node_map, graph_edges)

        # Calculate memory discipline
        memory = MemoryDisciplineResult(
            context_growth_rate=context_growth_rate,
            eviction_efficiency=eviction_efficiency,
            peak_memory_mb=peak_memory_mb,
            memory_budget_mb=self.memory_budget_mb
        )

        return CompressionReward(
            schema=schema,
            semantic=semantic,
            fidelity=fidelity,
            composition=composition,
            memory=memory,
            weights=self.weights
        )

    def _validate_schema(
        self,
        input_text: str,
        output_text: str,
        node_map: Optional[Dict[str, str]]
    ) -> SchemaValidationResult:
        """Validate input/output schema"""
        result = SchemaValidationResult()

        # Input validation
        if not input_text:
            result.input_valid = False
            result.input_errors.append("Empty input text")
        if not isinstance(input_text, str):
            result.input_valid = False
            result.input_errors.append("Input must be string")

        # Output validation
        if not output_text:
            result.output_valid = False
            result.output_errors.append("Empty output text")
        if not isinstance(output_text, str):
            result.output_valid = False
            result.output_errors.append("Output must be string")
        if node_map is not None and not isinstance(node_map, dict):
            result.output_valid = False
            result.output_errors.append("Node map must be dict")

        return result

    def _calculate_semantic(
        self,
        input_text: str,
        output_text: str,
        input_embedding: Optional[np.ndarray],
        output_embedding: Optional[np.ndarray],
        ssim_score: Optional[float]
    ) -> SemanticPreservationResult:
        """Calculate semantic preservation metrics"""
        result = SemanticPreservationResult()

        # SSIM score (if provided)
        if ssim_score is not None:
            result.ssim_score = max(0.0, min(1.0, ssim_score))
        else:
            result.ssim_score = 0.5  # Default neutral score

        # Embedding similarity
        if input_embedding is not None and output_embedding is not None:
            try:
                # Normalize embeddings
                in_norm = input_embedding / (np.linalg.norm(input_embedding) + 1e-8)
                out_norm = output_embedding / (np.linalg.norm(output_embedding) + 1e-8)
                similarity = float(np.dot(in_norm, out_norm))
                result.embedding_similarity = max(0.0, min(1.0, similarity))
            except Exception:
                result.embedding_similarity = 0.5
        else:
            result.embedding_similarity = 0.5

        # Keyword retention (simple heuristic)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        if input_words:
            retention = len(input_words & output_words) / len(input_words)
            result.keyword_retention = retention
        else:
            result.keyword_retention = 0.0

        # Structure preservation (length ratio as proxy)
        if len(input_text) > 0:
            length_ratio = len(output_text) / len(input_text)
            # Ideal: some compression but not too much
            result.structure_preservation = min(1.0, length_ratio * 2)
        else:
            result.structure_preservation = 0.0

        return result

    def _calculate_fidelity(
        self,
        input_tokens: int,
        output_tokens: int,
        fidelity_level: str
    ) -> FidelityAdherenceResult:
        """Calculate fidelity adherence"""
        result = FidelityAdherenceResult(target_fidelity=fidelity_level)

        # Get target ratio for fidelity level
        target_ratio = FIDELITY_TARGET_RATIOS.get(fidelity_level, 5.0)
        result.target_ratio = target_ratio

        # Calculate achieved ratio
        if output_tokens > 0:
            result.achieved_ratio = input_tokens / output_tokens
        else:
            result.achieved_ratio = float("inf")

        # Check budget
        budget = FIDELITY_TOKEN_BUDGETS.get(fidelity_level, float("inf"))
        result.within_budget = output_tokens <= budget
        if budget > 0 and budget != float("inf"):
            result.budget_utilization = output_tokens / budget
        else:
            result.budget_utilization = 0.0

        return result

    def _calculate_composition(
        self,
        node_map: Optional[Dict[str, str]],
        graph_edges: Optional[List[Tuple[str, str]]]
    ) -> CompositionIntegrityResult:
        """Calculate composition integrity"""
        result = CompositionIntegrityResult()

        if node_map is None:
            # No node map means we can't fully validate
            return result

        node_ids = set(node_map.keys())

        # Check node ID validity
        result.node_id_valid = all(
            isinstance(nid, str) and len(nid) > 0
            for nid in node_ids
        )

        # Check edge consistency
        if graph_edges:
            orphan_nodes = set()
            for src, tgt in graph_edges:
                if src not in node_ids:
                    result.edge_consistency = False
                    orphan_nodes.add(src)
                if tgt not in node_ids:
                    result.edge_consistency = False
                    orphan_nodes.add(tgt)
            result.orphan_nodes = len(orphan_nodes)
        else:
            result.edge_consistency = True

        # Check connectivity (simplified - assume connected if edges exist)
        result.graph_connected = bool(graph_edges) or len(node_map) <= 1

        return result

    def calculate_batch(
        self,
        operations: List[Dict[str, Any]]
    ) -> List[CompressionReward]:
        """Calculate rewards for a batch of operations"""
        return [self.calculate(**op) for op in operations]

    def aggregate_rewards(
        self,
        rewards: List[CompressionReward]
    ) -> Dict[str, Any]:
        """Aggregate statistics from multiple rewards"""
        if not rewards:
            return {"count": 0}

        total_rewards = [r.total_reward for r in rewards]
        component_scores = {
            comp: [r.component_scores[comp] for r in rewards]
            for comp in RewardComponent
        }

        return {
            "count": len(rewards),
            "total_reward": {
                "mean": np.mean(total_rewards),
                "std": np.std(total_rewards),
                "min": min(total_rewards),
                "max": max(total_rewards),
            },
            "component_means": {
                comp.value: np.mean(scores)
                for comp, scores in component_scores.items()
            },
            "pass_rate": sum(1 for r in rewards if r.passes_threshold()) / len(rewards),
            "weakest_components": self._count_weakest(rewards),
        }

    def _count_weakest(
        self,
        rewards: List[CompressionReward]
    ) -> Dict[str, int]:
        """Count how often each component is the weakest"""
        counts = {comp.value: 0 for comp in RewardComponent}
        for reward in rewards:
            weakest, _ = reward.weakest_component
            counts[weakest.value] += 1
        return counts


# Progressive reward shaping (ASG-SI concept)
class ProgressiveRewardShaper:
    """
    Implements progressive reward shaping from ASG-SI.

    Early phases prioritize structural validity,
    later phases emphasize correctness and efficiency.
    """

    def __init__(self, total_phases: int = 4):
        self.total_phases = total_phases
        self.current_phase = 0

    def get_weights_for_phase(self, phase: int) -> Dict[RewardComponent, float]:
        """Get component weights for current training phase"""
        # Phase 0: Prioritize schema validity
        # Phase 1: Balance schema and semantic
        # Phase 2: Full balance
        # Phase 3: Prioritize semantic and efficiency

        phase_weights = [
            # Phase 0: Schema focus
            {
                RewardComponent.SCHEMA: 0.40,
                RewardComponent.SEMANTIC: 0.20,
                RewardComponent.FIDELITY: 0.15,
                RewardComponent.COMPOSITION: 0.15,
                RewardComponent.MEMORY: 0.10,
            },
            # Phase 1: Balanced
            {
                RewardComponent.SCHEMA: 0.25,
                RewardComponent.SEMANTIC: 0.30,
                RewardComponent.FIDELITY: 0.20,
                RewardComponent.COMPOSITION: 0.15,
                RewardComponent.MEMORY: 0.10,
            },
            # Phase 2: Semantic focus
            {
                RewardComponent.SCHEMA: 0.15,
                RewardComponent.SEMANTIC: 0.35,
                RewardComponent.FIDELITY: 0.20,
                RewardComponent.COMPOSITION: 0.15,
                RewardComponent.MEMORY: 0.15,
            },
            # Phase 3: Efficiency focus
            {
                RewardComponent.SCHEMA: 0.10,
                RewardComponent.SEMANTIC: 0.30,
                RewardComponent.FIDELITY: 0.25,
                RewardComponent.COMPOSITION: 0.15,
                RewardComponent.MEMORY: 0.20,
            },
        ]

        phase = min(phase, len(phase_weights) - 1)
        return phase_weights[phase]

    def advance_phase(self) -> None:
        """Advance to next training phase"""
        if self.current_phase < self.total_phases - 1:
            self.current_phase += 1

    def get_current_calculator(self) -> CompressionRewardCalculator:
        """Get calculator with current phase weights"""
        weights = self.get_weights_for_phase(self.current_phase)
        return CompressionRewardCalculator(weights=weights)
