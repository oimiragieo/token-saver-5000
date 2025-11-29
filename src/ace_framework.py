"""
ACE Framework for Token Saver 5000

Implements Agentic Context Engineering (arXiv:2510.04618v1) for evolving
contexts as playbooks through Generator/Reflector/Curator architecture.

Key Features:
- Generator: Produces reasoning trajectories through semantic space
- Reflector: Distills insights from successes/errors with iterative refinement
- Curator: Integrates insights via delta updates with semantic deduplication
- Grow-and-Refine: Balances expansion and deduplication to prevent context collapse
- Delta Updates: Incremental, localized changes instead of monolithic rewrites

Integration with Token Saver 5000:
- ACE bullets provide meta-level guidance for semantic node selection
- Delta updates complement fidelity modulation (both avoid monolithic rewrites)
- Reflector enables self-correction alongside blind spot detection
- Grow-and-refine prevents information loss during compression

References:
- Paper: Agentic Context Engineering (arXiv:2510.04618v1)
- Section 3: Delta Updates and Semantic Deduplication
- Section 4: Empirical Results (32% quality boost with 4x shorter contexts)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Structures
# ============================================================================


class BulletType(Enum):
    """Types of bullets in ACE playbook (Section 2.2 of paper)"""

    PRINCIPLE = "principle"  # High-level guidance (e.g., "Be concise")
    STRATEGY = "strategy"  # Tactical approaches (e.g., "Use examples")
    TACTIC = "tactic"  # Specific techniques (e.g., "List pros/cons")
    CONSTRAINT = "constraint"  # Hard requirements (e.g., "No hallucinations")
    PREFERENCE = "preference"  # Soft preferences (e.g., "Prefer bullet points")
    LEARNED = "learned"  # Insights from reflection (e.g., "User prefers X")


@dataclass
class ACEBullet:
    """
    Individual playbook entry with performance tracking.

    Corresponds to a single bullet in the ACE context. Each bullet has:
    - Text content and embedding for semantic operations
    - Type classification (principle, strategy, etc.)
    - Confidence score updated through successes/failures
    - Performance tracking (usage count, success rate)
    - Metadata for versioning and provenance

    Args:
        text: The bullet content (e.g., "Be concise and direct")
        bullet_type: Classification of this bullet
        embedding: Vector representation for semantic operations
        confidence: Belief in this bullet's utility (0.0 to 1.0)
        success_count: Number of successful applications
        failure_count: Number of failed applications
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
        source: Origin of this bullet (e.g., "reflection", "manual")
        metadata: Additional context (e.g., {"domain": "code_review"})
    """

    text: str
    bullet_type: BulletType
    embedding: np.ndarray
    confidence: float = 0.5  # Default: neutral confidence
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)
    bullet_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def update_performance(self, success: bool, confidence_boost: float = 0.05):
        """
        Update bullet performance based on application result.

        Implements confidence adjustment from Section 3.3 of paper.

        Args:
            success: Whether this bullet led to successful outcome
            confidence_boost: Amount to adjust confidence (default: 0.05)
        """
        if success:
            self.success_count += 1
            self.confidence = min(1.0, self.confidence + confidence_boost)
        else:
            self.failure_count += 1
            self.confidence = max(0.0, self.confidence - confidence_boost)

        self.updated_at = time.time()

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    @property
    def total_usage(self) -> int:
        """Total number of times this bullet has been applied"""
        return self.success_count + self.failure_count

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bullet to dictionary"""
        return {
            "bullet_id": self.bullet_id,
            "text": self.text,
            "bullet_type": self.bullet_type.value,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_usage": self.total_usage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class ACEContext:
    """
    Collection of bullets forming a playbook with versioning.

    This is the core "context" in ACE that evolves over time through
    delta updates. Maintains:
    - Mapping of bullet IDs to ACEBullet objects
    - Version counter for tracking evolution
    - Delta history for auditability
    - Performance statistics

    Args:
        bullets: Dictionary mapping bullet_id -> ACEBullet
        version: Current version number
        delta_history: List of delta update descriptions
        created_at: Timestamp of context creation
        updated_at: Timestamp of last update
        metadata: Additional context-level metadata
    """

    bullets: Dict[str, ACEBullet] = field(default_factory=dict)
    version: int = 1
    delta_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_bullet(self, bullet: ACEBullet, delta_description: str = ""):
        """
        Add a new bullet to the context (grow operation).

        Args:
            bullet: ACEBullet to add
            delta_description: Human-readable description of this change
        """
        self.bullets[bullet.bullet_id] = bullet
        self.version += 1
        self.updated_at = time.time()

        self.delta_history.append(
            {
                "version": self.version,
                "operation": "add",
                "bullet_id": bullet.bullet_id,
                "description": delta_description or f"Added bullet: {bullet.text[:50]}...",
                "timestamp": self.updated_at,
            }
        )

    def update_bullet(self, bullet_id: str, updated_bullet: ACEBullet, delta_description: str = ""):
        """
        Update an existing bullet (refine operation).

        Args:
            bullet_id: ID of bullet to update
            updated_bullet: New version of the bullet
            delta_description: Human-readable description of this change
        """
        if bullet_id not in self.bullets:
            raise KeyError(f"Bullet {bullet_id} not found in context")

        old_text = self.bullets[bullet_id].text
        self.bullets[bullet_id] = updated_bullet
        self.version += 1
        self.updated_at = time.time()

        self.delta_history.append(
            {
                "version": self.version,
                "operation": "update",
                "bullet_id": bullet_id,
                "old_text": old_text,
                "new_text": updated_bullet.text,
                "description": delta_description
                or f"Updated bullet: {old_text[:30]}... -> {updated_bullet.text[:30]}...",
                "timestamp": self.updated_at,
            }
        )

    def remove_bullet(self, bullet_id: str, delta_description: str = ""):
        """
        Remove a bullet from the context (prune operation).

        Args:
            bullet_id: ID of bullet to remove
            delta_description: Human-readable description of this change
        """
        if bullet_id not in self.bullets:
            raise KeyError(f"Bullet {bullet_id} not found in context")

        removed_text = self.bullets[bullet_id].text
        del self.bullets[bullet_id]
        self.version += 1
        self.updated_at = time.time()

        self.delta_history.append(
            {
                "version": self.version,
                "operation": "remove",
                "bullet_id": bullet_id,
                "description": delta_description or f"Removed bullet: {removed_text[:50]}...",
                "timestamp": self.updated_at,
            }
        )

    def get_bullets_by_type(self, bullet_type: BulletType) -> List[ACEBullet]:
        """Get all bullets of a specific type"""
        return [b for b in self.bullets.values() if b.bullet_type == bullet_type]

    def get_top_bullets(self, k: int = 10, min_confidence: float = 0.3) -> List[ACEBullet]:
        """
        Get top-k bullets by confidence score.

        Args:
            k: Number of bullets to return
            min_confidence: Minimum confidence threshold

        Returns:
            List of top bullets sorted by confidence
        """
        filtered = [b for b in self.bullets.values() if b.confidence >= min_confidence]
        return sorted(filtered, key=lambda b: b.confidence, reverse=True)[:k]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get aggregate performance statistics"""
        if not self.bullets:
            return {
                "total_bullets": 0,
                "avg_confidence": 0.0,
                "avg_success_rate": 0.0,
                "total_usage": 0,
            }

        bullets_list = list(self.bullets.values())
        return {
            "total_bullets": len(bullets_list),
            "avg_confidence": np.mean([b.confidence for b in bullets_list]),
            "avg_success_rate": np.mean([b.success_rate for b in bullets_list]),
            "total_usage": sum(b.total_usage for b in bullets_list),
            "by_type": {
                bt.value: len([b for b in bullets_list if b.bullet_type == bt]) for bt in BulletType
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context to dictionary"""
        return {
            "context_id": self.context_id,
            "version": self.version,
            "bullets": {bid: bullet.to_dict() for bid, bullet in self.bullets.items()},
            "delta_history": self.delta_history[-10:],  # Last 10 deltas only
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "stats": self.get_performance_stats(),
        }


# ============================================================================
# ACE Components: Generator, Reflector, Curator
# ============================================================================


class ACEGenerator:
    """
    Generates reasoning trajectories through semantic space.

    Implements the Generator component from Section 2.3 of the ACE paper.
    Given a task and context, produces a step-by-step reasoning trajectory
    that applies relevant bullets from the playbook.

    Args:
        embedding_manager: Shared embedding manager for semantic operations
    """

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.text_model = self.embedding_manager.get_text_embedder()
        # v0.8.0 audit: Log simulation mode notice
        logger.info(
            "ACEGenerator initialized in SIMULATION MODE - "
            "reasoning is templated, not LLM-generated. "
            "See _generate_step_reasoning() for production integration."
        )

    def generate_trajectory(
        self,
        task: str,
        context: ACEContext,
        max_steps: int = 5,
        top_k_bullets: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Generate reasoning trajectory for a task using playbook bullets.

        Args:
            task: The task or query to reason about
            context: Current ACE playbook context
            max_steps: Maximum trajectory steps
            top_k_bullets: Number of relevant bullets to consider per step

        Returns:
            List of trajectory steps, each containing:
            - step_number: Sequential step index
            - relevant_bullets: Bullets applied in this step
            - reasoning: Generated reasoning text
            - confidence: Aggregate confidence for this step
        """
        logger.info(f"Generating trajectory for task: {task[:100]}...")

        # Embed the task
        task_embedding = self.text_model.encode(task, convert_to_numpy=True)

        trajectory = []

        for step in range(max_steps):
            # Find most relevant bullets for current task
            relevant_bullets = self._find_relevant_bullets(
                task_embedding, context, top_k=top_k_bullets
            )

            if not relevant_bullets:
                logger.warning(f"No relevant bullets found at step {step}")
                break

            # Generate reasoning using these bullets
            reasoning = self._generate_step_reasoning(task, relevant_bullets, step)

            # Calculate aggregate confidence
            avg_confidence = np.mean([b.confidence for b in relevant_bullets])

            trajectory.append(
                {
                    "step_number": step + 1,
                    "relevant_bullets": [b.to_dict() for b in relevant_bullets],
                    "reasoning": reasoning,
                    "confidence": float(avg_confidence),
                }
            )

        logger.info(f"Generated trajectory with {len(trajectory)} steps")
        return trajectory

    def _find_relevant_bullets(
        self, task_embedding: np.ndarray, context: ACEContext, top_k: int = 5
    ) -> List[ACEBullet]:
        """Find most semantically relevant bullets for the task"""
        if not context.bullets:
            return []

        bullets_list = list(context.bullets.values())

        # Calculate similarity between task and each bullet
        similarities = []
        for bullet in bullets_list:
            sim = cosine_similarity(task_embedding.reshape(1, -1), bullet.embedding.reshape(1, -1))[
                0, 0
            ]
            # Weight by confidence (high-confidence bullets prioritized)
            weighted_score = sim * (0.5 + 0.5 * bullet.confidence)
            similarities.append((bullet, weighted_score))

        # Sort by weighted score and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [bullet for bullet, _ in similarities[:top_k]]

    def _generate_step_reasoning(self, task: str, bullets: List[ACEBullet], step: int) -> str:
        """
        Generate reasoning text for a trajectory step.

        In a production implementation, this would invoke an LLM with the
        task and bullets as context. For now, we generate a structured
        description of the reasoning process.

        Args:
            task: The task being reasoned about
            bullets: Relevant bullets for this step
            step: Step number

        Returns:
            Reasoning text for this step
        """
        reasoning_parts = [f"Step {step + 1}: Applying {len(bullets)} relevant guidelines"]

        for i, bullet in enumerate(bullets, 1):
            reasoning_parts.append(
                f"  {i}. [{bullet.bullet_type.value}] {bullet.text} "
                f"(confidence: {bullet.confidence:.2f})"
            )

        # SIMULATION MODE: In production, this would be replaced with LLM call
        # Log warning to clarify this is not real LLM reasoning (v0.8.0 audit fix)
        logger.debug(
            "[ACE SIMULATION MODE] Generating templated reasoning - "
            "not invoking LLM. For production use, implement LLM integration."
        )
        reasoning_parts.append(
            f"\nBased on these guidelines, approaching the task: {task[:100]}..."
        )

        return "\n".join(reasoning_parts)


class ACEReflector:
    """
    Distills insights from reasoning trajectories.

    Implements the Reflector component from Section 2.4 of the ACE paper.
    Analyzes trajectories to extract insights about what worked (successes)
    and what didn't (errors), then formulates new bullets or updates.

    Args:
        embedding_manager: Shared embedding manager for semantic operations
    """

    def __init__(self, embedding_manager: Optional[EmbeddingManager] = None):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.text_model = self.embedding_manager.get_text_embedder()

    def reflect_on_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        outcome: str,
        success: bool,
    ) -> List[Dict[str, Any]]:
        """
        Reflect on a trajectory to extract insights.

        Args:
            trajectory: Generated reasoning trajectory
            outcome: Actual outcome (text describing what happened)
            success: Whether the trajectory led to success

        Returns:
            List of insights, each containing:
            - text: The insight content
            - bullet_type: Suggested type for this insight
            - confidence: Initial confidence (0.0 to 1.0)
            - source: "reflection_success" or "reflection_failure"
            - reasoning: Why this insight was extracted
        """
        logger.info(f"Reflecting on trajectory (success={success})")

        insights = []

        if success:
            insights.extend(self._extract_success_insights(trajectory, outcome))
        else:
            insights.extend(self._extract_failure_insights(trajectory, outcome))

        logger.info(f"Extracted {len(insights)} insights from reflection")
        return insights

    def _extract_success_insights(
        self, trajectory: List[Dict[str, Any]], outcome: str
    ) -> List[Dict[str, Any]]:
        """Extract insights from successful trajectories"""
        insights = []

        # Analyze which bullets were most relevant in successful steps
        high_confidence_steps = [s for s in trajectory if s["confidence"] > 0.7]

        if high_confidence_steps:
            # Extract common patterns from successful steps
            insight = {
                "text": f"High-confidence bullets ({len(high_confidence_steps)} steps) led to success: {outcome[:100]}",
                "bullet_type": BulletType.LEARNED.value,
                "confidence": 0.7,  # Start with moderate confidence
                "source": "reflection_success",
                "reasoning": "Identified successful pattern in trajectory",
                "metadata": {
                    "trajectory_length": len(trajectory),
                    "high_confidence_steps": len(high_confidence_steps),
                },
            }
            insights.append(insight)

        return insights

    def _extract_failure_insights(
        self, trajectory: List[Dict[str, Any]], outcome: str
    ) -> List[Dict[str, Any]]:
        """Extract insights from failed trajectories"""
        insights = []

        # Analyze which bullets may have led to failure
        low_confidence_steps = [s for s in trajectory if s["confidence"] < 0.4]

        if low_confidence_steps:
            insight = {
                "text": f"Low-confidence bullets ({len(low_confidence_steps)} steps) may have caused failure: {outcome[:100]}",
                "bullet_type": BulletType.LEARNED.value,
                "confidence": 0.4,  # Lower confidence for failure insights
                "source": "reflection_failure",
                "reasoning": "Identified potential failure pattern in trajectory",
                "metadata": {
                    "trajectory_length": len(trajectory),
                    "low_confidence_steps": len(low_confidence_steps),
                },
            }
            insights.append(insight)

        return insights


class ACECurator:
    """
    Integrates insights into playbook via delta updates with semantic deduplication.

    Implements the Curator component from Section 3 of the ACE paper.
    Core responsibilities:
    - Delta Updates: Apply incremental changes (add/update/remove bullets)
    - Semantic Deduplication: Merge similar bullets to prevent redundancy
    - Grow-and-Refine: Balance expansion and consolidation

    Args:
        embedding_manager: Shared embedding manager for semantic operations
        deduplication_threshold: Cosine similarity threshold for merging (default: 0.85)
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        deduplication_threshold: float = 0.85,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.text_model = self.embedding_manager.get_text_embedder()
        self.deduplication_threshold = deduplication_threshold

    def curate_insights(
        self,
        context: ACEContext,
        insights: List[Dict[str, Any]],
        max_bullets: Optional[int] = None,
    ) -> ACEContext:
        """
        Integrate insights into context via delta updates and deduplication.

        Args:
            context: Current ACE context
            insights: List of insights from reflection
            max_bullets: Optional limit on total bullets (triggers pruning)

        Returns:
            Updated ACEContext
        """
        logger.info(f"Curating {len(insights)} insights into context (v{context.version})")

        for insight in insights:
            self._integrate_insight(context, insight)

        # Deduplicate after all insights are added
        self._deduplicate_bullets(context)

        # Prune if over limit
        if max_bullets and len(context.bullets) > max_bullets:
            self._prune_low_confidence_bullets(context, target_count=max_bullets)

        logger.info(
            f"Curation complete: context now at v{context.version} with {len(context.bullets)} bullets"
        )
        return context

    def _integrate_insight(self, context: ACEContext, insight: Dict[str, Any]) -> None:
        """Integrate a single insight into the context"""
        text = insight["text"]
        bullet_type = BulletType(insight["bullet_type"])
        confidence = insight["confidence"]
        source = insight["source"]

        # Embed the insight
        embedding = self.text_model.encode(text, convert_to_numpy=True)

        # Check if similar bullet already exists
        similar_bullet, similarity = self._find_similar_bullet(context, embedding)

        if similar_bullet and similarity > self.deduplication_threshold:
            # Update existing bullet (refine operation)
            logger.info(f"Updating existing bullet (similarity: {similarity:.3f})")
            updated_bullet = ACEBullet(
                text=text,  # Use new text
                bullet_type=bullet_type,
                embedding=embedding,
                confidence=max(similar_bullet.confidence, confidence),  # Keep higher confidence
                success_count=similar_bullet.success_count,
                failure_count=similar_bullet.failure_count,
                created_at=similar_bullet.created_at,
                updated_at=time.time(),
                source=source,
                bullet_id=similar_bullet.bullet_id,
                metadata=insight.get("metadata", {}),
            )
            context.update_bullet(
                similar_bullet.bullet_id,
                updated_bullet,
                delta_description=f"Refined bullet based on {source}",
            )
        else:
            # Add new bullet (grow operation)
            logger.info("Adding new bullet to context")
            new_bullet = ACEBullet(
                text=text,
                bullet_type=bullet_type,
                embedding=embedding,
                confidence=confidence,
                source=source,
                metadata=insight.get("metadata", {}),
            )
            context.add_bullet(new_bullet, delta_description=f"Added insight from {source}")

    def _find_similar_bullet(
        self, context: ACEContext, embedding: np.ndarray
    ) -> Tuple[Optional[ACEBullet], float]:
        """Find most similar existing bullet"""
        if not context.bullets:
            return None, 0.0

        max_similarity = 0.0
        most_similar = None

        for bullet in context.bullets.values():
            sim = cosine_similarity(embedding.reshape(1, -1), bullet.embedding.reshape(1, -1))[0, 0]
            if sim > max_similarity:
                max_similarity = sim
                most_similar = bullet

        return most_similar, max_similarity

    def _deduplicate_bullets(self, context: ACEContext) -> None:
        """
        Remove duplicate bullets based on semantic similarity.

        Implements semantic deduplication from Section 3.2 of the paper.
        Merges bullets that are semantically similar (above threshold).
        """
        if len(context.bullets) < 2:
            return

        bullets_list = list(context.bullets.values())
        to_remove: Set[str] = set()

        # Compare all pairs
        for i in range(len(bullets_list)):
            if bullets_list[i].bullet_id in to_remove:
                continue

            for j in range(i + 1, len(bullets_list)):
                if bullets_list[j].bullet_id in to_remove:
                    continue

                # Calculate similarity
                sim = cosine_similarity(
                    bullets_list[i].embedding.reshape(1, -1),
                    bullets_list[j].embedding.reshape(1, -1),
                )[0, 0]

                if sim > self.deduplication_threshold:
                    # Merge: keep higher confidence bullet, remove the other
                    if bullets_list[i].confidence >= bullets_list[j].confidence:
                        to_remove.add(bullets_list[j].bullet_id)
                        logger.info(f"Deduplicating: removing similar bullet (sim={sim:.3f})")
                    else:
                        to_remove.add(bullets_list[i].bullet_id)
                        logger.info(f"Deduplicating: removing similar bullet (sim={sim:.3f})")
                        break  # Move to next i

        # Remove marked bullets
        for bullet_id in to_remove:
            context.remove_bullet(bullet_id, delta_description="Removed duplicate bullet")

    def _prune_low_confidence_bullets(self, context: ACEContext, target_count: int) -> None:
        """Prune lowest confidence bullets to reach target count"""
        if len(context.bullets) <= target_count:
            return

        # Sort by confidence ascending
        bullets_sorted = sorted(context.bullets.values(), key=lambda b: b.confidence)
        num_to_remove = len(context.bullets) - target_count

        for bullet in bullets_sorted[:num_to_remove]:
            context.remove_bullet(
                bullet.bullet_id,
                delta_description=f"Pruned low-confidence bullet ({bullet.confidence:.2f})",
            )


# ============================================================================
# ACE Framework Facade
# ============================================================================


class ACEFramework:
    """
    Facade coordinating Generator, Reflector, and Curator.

    Provides high-level interface for the complete ACE cycle:
    1. Generate reasoning trajectory for a task
    2. Reflect on trajectory to extract insights
    3. Curate insights into playbook via delta updates

    Args:
        embedding_manager: Shared embedding manager
        deduplication_threshold: Similarity threshold for merging bullets
        max_bullets: Maximum bullets in context (triggers pruning)
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        deduplication_threshold: float = 0.85,
        max_bullets: int = 100,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.text_model = self.embedding_manager.get_text_embedder()
        self.generator = ACEGenerator(self.embedding_manager)
        self.reflector = ACEReflector(self.embedding_manager)
        self.curator = ACECurator(self.embedding_manager, deduplication_threshold)
        self.max_bullets = max_bullets

    def execute_ace_cycle(
        self,
        task: str,
        context: ACEContext,
        outcome: str,
        success: bool,
        max_trajectory_steps: int = 5,
    ) -> Tuple[ACEContext, List[Dict[str, Any]]]:
        """
        Execute complete ACE cycle: Generate → Reflect → Curate.

        Args:
            task: The task or query
            context: Current ACE playbook context
            outcome: What actually happened
            success: Whether the task succeeded
            max_trajectory_steps: Maximum steps in trajectory

        Returns:
            Tuple of (updated_context, trajectory)
        """
        logger.info(f"Executing ACE cycle for task: {task[:100]}...")

        # Generate trajectory
        trajectory = self.generator.generate_trajectory(
            task=task, context=context, max_steps=max_trajectory_steps
        )

        # Reflect on trajectory
        insights = self.reflector.reflect_on_trajectory(
            trajectory=trajectory, outcome=outcome, success=success
        )

        # Curate insights into context
        updated_context = self.curator.curate_insights(
            context=context, insights=insights, max_bullets=self.max_bullets
        )

        logger.info(f"ACE cycle complete: v{context.version} -> v{updated_context.version}")
        return updated_context, trajectory

    def create_initial_context(
        self, initial_bullets: Optional[List[Tuple[str, BulletType]]] = None
    ) -> ACEContext:
        """
        Create a new ACE context with optional seed bullets.

        Args:
            initial_bullets: List of (text, bullet_type) tuples for seed bullets

        Returns:
            New ACEContext
        """
        context = ACEContext()

        if initial_bullets:
            for text, bullet_type in initial_bullets:
                embedding = self.text_model.encode(text, convert_to_numpy=True)
                bullet = ACEBullet(
                    text=text,
                    bullet_type=bullet_type,
                    embedding=embedding,
                    confidence=0.5,  # Neutral starting confidence
                    source="seed",
                )
                context.add_bullet(bullet, delta_description="Added seed bullet")

        logger.info(f"Created new ACE context with {len(context.bullets)} seed bullets")
        return context
