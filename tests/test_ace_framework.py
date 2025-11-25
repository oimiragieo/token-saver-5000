"""
Unit tests for ACE Framework

Tests the Generator/Reflector/Curator architecture for evolving contexts.

Test Coverage:
- ACEBullet: Performance tracking, serialization
- ACEContext: Bullet management, versioning, delta history
- ACEGenerator: Trajectory generation, bullet relevance
- ACEReflector: Insight extraction from successes/failures
- ACECurator: Delta updates, semantic deduplication, pruning
- ACEFramework: Full ACE cycle execution
"""

import pytest

from src.ace_framework import (
    BulletType,
    ACEBullet,
    ACEContext,
    ACEGenerator,
    ACEReflector,
    ACECurator,
    ACEFramework,
)
from src.embeddings import EmbeddingManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def embedding_manager():
    """Shared embedding manager for tests"""
    return EmbeddingManager()


@pytest.fixture
def sample_bullet(embedding_manager):
    """Create a sample ACEBullet for testing"""
    text = "Be concise and direct in responses"
    text_model = embedding_manager.get_text_embedder()
    embedding = text_model.encode(text, convert_to_numpy=True)
    return ACEBullet(
        text=text,
        bullet_type=BulletType.PRINCIPLE,
        embedding=embedding,
        confidence=0.6,
    )


@pytest.fixture
def sample_context(embedding_manager):
    """Create a sample ACEContext with a few bullets"""
    context = ACEContext()
    text_model = embedding_manager.get_text_embedder()

    bullets_data = [
        ("Be concise and direct", BulletType.PRINCIPLE, 0.7),
        ("Use examples when explaining", BulletType.STRATEGY, 0.6),
        ("Check for edge cases", BulletType.TACTIC, 0.5),
        ("No hallucinations allowed", BulletType.CONSTRAINT, 0.9),
    ]

    for text, bullet_type, confidence in bullets_data:
        embedding = text_model.encode(text, convert_to_numpy=True)
        bullet = ACEBullet(
            text=text,
            bullet_type=bullet_type,
            embedding=embedding,
            confidence=confidence,
        )
        context.add_bullet(bullet, delta_description=f"Added: {text}")

    return context


# ============================================================================
# Test ACEBullet
# ============================================================================


def test_bullet_creation(sample_bullet):
    """Test ACEBullet initialization"""
    assert sample_bullet.text == "Be concise and direct in responses"
    assert sample_bullet.bullet_type == BulletType.PRINCIPLE
    assert sample_bullet.confidence == 0.6
    assert sample_bullet.success_count == 0
    assert sample_bullet.failure_count == 0
    assert sample_bullet.source == "manual"
    assert len(sample_bullet.bullet_id) > 0
    assert sample_bullet.embedding.shape[0] > 0


def test_bullet_update_performance_success(sample_bullet):
    """Test bullet performance tracking on success"""
    initial_confidence = sample_bullet.confidence
    sample_bullet.update_performance(success=True, confidence_boost=0.1)

    assert sample_bullet.success_count == 1
    assert sample_bullet.failure_count == 0
    assert sample_bullet.confidence == initial_confidence + 0.1
    assert sample_bullet.success_rate == 1.0


def test_bullet_update_performance_failure(sample_bullet):
    """Test bullet performance tracking on failure"""
    initial_confidence = sample_bullet.confidence
    sample_bullet.update_performance(success=False, confidence_boost=0.1)

    assert sample_bullet.success_count == 0
    assert sample_bullet.failure_count == 1
    assert sample_bullet.confidence == initial_confidence - 0.1
    assert sample_bullet.success_rate == 0.0


def test_bullet_success_rate_calculation(sample_bullet):
    """Test success rate calculation"""
    # Initial state: no usage
    assert sample_bullet.success_rate == 0.5

    # After successes and failures
    sample_bullet.update_performance(success=True)
    sample_bullet.update_performance(success=True)
    sample_bullet.update_performance(success=False)

    assert sample_bullet.total_usage == 3
    assert sample_bullet.success_count == 2
    assert sample_bullet.failure_count == 1
    assert abs(sample_bullet.success_rate - 2 / 3) < 0.01


def test_bullet_confidence_bounds(sample_bullet):
    """Test confidence stays within [0.0, 1.0] bounds"""
    # Max out confidence
    for _ in range(20):
        sample_bullet.update_performance(success=True, confidence_boost=0.1)
    assert sample_bullet.confidence <= 1.0

    # Min out confidence
    for _ in range(40):
        sample_bullet.update_performance(success=False, confidence_boost=0.1)
    assert sample_bullet.confidence >= 0.0


def test_bullet_serialization(sample_bullet):
    """Test bullet to_dict serialization"""
    bullet_dict = sample_bullet.to_dict()

    assert "bullet_id" in bullet_dict
    assert bullet_dict["text"] == sample_bullet.text
    assert bullet_dict["bullet_type"] == BulletType.PRINCIPLE.value
    assert bullet_dict["confidence"] == sample_bullet.confidence
    assert "success_rate" in bullet_dict
    assert "total_usage" in bullet_dict


# ============================================================================
# Test ACEContext
# ============================================================================


def test_context_creation():
    """Test ACEContext initialization"""
    context = ACEContext()

    assert len(context.bullets) == 0
    assert context.version == 1
    assert len(context.delta_history) == 0
    assert len(context.context_id) > 0


def test_context_add_bullet(embedding_manager):
    """Test adding bullets to context"""
    context = ACEContext()
    text = "Test bullet"
    text_model = embedding_manager.get_text_embedder()
    embedding = text_model.encode(text, convert_to_numpy=True)
    bullet = ACEBullet(text=text, bullet_type=BulletType.TACTIC, embedding=embedding)

    context.add_bullet(bullet, delta_description="Test add")

    assert len(context.bullets) == 1
    assert context.version == 2  # Incremented from 1
    assert len(context.delta_history) == 1
    assert context.delta_history[0]["operation"] == "add"
    assert context.delta_history[0]["bullet_id"] == bullet.bullet_id


def test_context_update_bullet(sample_context):
    """Test updating existing bullet"""
    bullet_id = list(sample_context.bullets.keys())[0]
    old_bullet = sample_context.bullets[bullet_id]
    old_version = sample_context.version

    # Create updated version
    new_text = "Updated text"
    embedding_manager = EmbeddingManager()
    text_model = embedding_manager.get_text_embedder()
    embedding = text_model.encode(new_text, convert_to_numpy=True)
    updated_bullet = ACEBullet(
        text=new_text,
        bullet_type=old_bullet.bullet_type,
        embedding=embedding,
        confidence=0.8,
        bullet_id=bullet_id,
    )

    sample_context.update_bullet(bullet_id, updated_bullet, delta_description="Test update")

    assert sample_context.bullets[bullet_id].text == new_text
    assert sample_context.version == old_version + 1
    assert sample_context.delta_history[-1]["operation"] == "update"
    assert sample_context.delta_history[-1]["old_text"] == old_bullet.text


def test_context_remove_bullet(sample_context):
    """Test removing bullet from context"""
    bullet_id = list(sample_context.bullets.keys())[0]
    old_count = len(sample_context.bullets)
    old_version = sample_context.version

    sample_context.remove_bullet(bullet_id, delta_description="Test remove")

    assert len(sample_context.bullets) == old_count - 1
    assert bullet_id not in sample_context.bullets
    assert sample_context.version == old_version + 1
    assert sample_context.delta_history[-1]["operation"] == "remove"


def test_context_get_bullets_by_type(sample_context):
    """Test filtering bullets by type"""
    principles = sample_context.get_bullets_by_type(BulletType.PRINCIPLE)
    constraints = sample_context.get_bullets_by_type(BulletType.CONSTRAINT)

    assert len(principles) == 1
    assert len(constraints) == 1
    assert principles[0].bullet_type == BulletType.PRINCIPLE
    assert constraints[0].bullet_type == BulletType.CONSTRAINT


def test_context_get_top_bullets(sample_context):
    """Test getting top bullets by confidence"""
    top_bullets = sample_context.get_top_bullets(k=2, min_confidence=0.5)

    assert len(top_bullets) <= 2
    assert all(b.confidence >= 0.5 for b in top_bullets)
    # Should be sorted by confidence descending
    if len(top_bullets) == 2:
        assert top_bullets[0].confidence >= top_bullets[1].confidence


def test_context_performance_stats(sample_context):
    """Test aggregate performance statistics"""
    stats = sample_context.get_performance_stats()

    assert stats["total_bullets"] == len(sample_context.bullets)
    assert "avg_confidence" in stats
    assert "avg_success_rate" in stats
    assert "by_type" in stats
    assert stats["by_type"][BulletType.PRINCIPLE.value] >= 0


def test_context_serialization(sample_context):
    """Test context to_dict serialization"""
    context_dict = sample_context.to_dict()

    assert "context_id" in context_dict
    assert context_dict["version"] == sample_context.version
    assert "bullets" in context_dict
    assert len(context_dict["bullets"]) == len(sample_context.bullets)
    assert "delta_history" in context_dict
    assert "stats" in context_dict


# ============================================================================
# Test ACEGenerator
# ============================================================================


def test_generator_creation(embedding_manager):
    """Test ACEGenerator initialization"""
    generator = ACEGenerator(embedding_manager)
    assert generator.embedding_manager is not None


def test_generator_trajectory_generation(sample_context, embedding_manager):
    """Test trajectory generation"""
    generator = ACEGenerator(embedding_manager)
    task = "Explain quantum computing to a beginner"

    trajectory = generator.generate_trajectory(
        task=task, context=sample_context, max_steps=3, top_k_bullets=2
    )

    assert len(trajectory) <= 3
    assert len(trajectory) > 0

    # Check trajectory structure
    for step in trajectory:
        assert "step_number" in step
        assert "relevant_bullets" in step
        assert "reasoning" in step
        assert "confidence" in step
        assert 0.0 <= step["confidence"] <= 1.0


def test_generator_empty_context(embedding_manager):
    """Test trajectory generation with empty context"""
    generator = ACEGenerator(embedding_manager)
    empty_context = ACEContext()
    task = "Test task"

    trajectory = generator.generate_trajectory(task=task, context=empty_context, max_steps=3)

    # Should return empty trajectory since no bullets available
    assert len(trajectory) == 0


def test_generator_finds_relevant_bullets(sample_context, embedding_manager):
    """Test that generator finds relevant bullets for task"""
    generator = ACEGenerator(embedding_manager)
    task = "Check for edge cases in the implementation"

    trajectory = generator.generate_trajectory(
        task=task, context=sample_context, max_steps=1, top_k_bullets=3
    )

    assert len(trajectory) > 0
    # Should find the "Check for edge cases" bullet as relevant
    step = trajectory[0]
    bullet_texts = [b["text"] for b in step["relevant_bullets"]]
    # At least one bullet should be somewhat relevant
    assert len(bullet_texts) > 0


# ============================================================================
# Test ACEReflector
# ============================================================================


def test_reflector_creation(embedding_manager):
    """Test ACEReflector initialization"""
    reflector = ACEReflector(embedding_manager)
    assert reflector.embedding_manager is not None


def test_reflector_success_insights(embedding_manager):
    """Test insight extraction from successful trajectory"""
    reflector = ACEReflector(embedding_manager)

    trajectory = [
        {"step_number": 1, "reasoning": "Step 1", "confidence": 0.8},
        {"step_number": 2, "reasoning": "Step 2", "confidence": 0.9},
    ]
    outcome = "Successfully completed the task"

    insights = reflector.reflect_on_trajectory(trajectory=trajectory, outcome=outcome, success=True)

    assert len(insights) > 0
    # Success insights should have reasonable confidence
    for insight in insights:
        assert "text" in insight
        assert "bullet_type" in insight
        assert "confidence" in insight
        assert insight["source"] == "reflection_success"
        assert insight["confidence"] >= 0.5


def test_reflector_failure_insights(embedding_manager):
    """Test insight extraction from failed trajectory"""
    reflector = ACEReflector(embedding_manager)

    trajectory = [
        {"step_number": 1, "reasoning": "Step 1", "confidence": 0.3},
        {"step_number": 2, "reasoning": "Step 2", "confidence": 0.2},
    ]
    outcome = "Task failed due to incorrect approach"

    insights = reflector.reflect_on_trajectory(
        trajectory=trajectory, outcome=outcome, success=False
    )

    assert len(insights) > 0
    # Failure insights should have lower confidence
    for insight in insights:
        assert "text" in insight
        assert insight["source"] == "reflection_failure"
        assert insight["confidence"] < 0.7  # Lower confidence for failures


# ============================================================================
# Test ACECurator
# ============================================================================


def test_curator_creation(embedding_manager):
    """Test ACECurator initialization"""
    curator = ACECurator(embedding_manager, deduplication_threshold=0.85)
    assert curator.embedding_manager is not None
    assert curator.deduplication_threshold == 0.85


def test_curator_integrate_new_insight(sample_context, embedding_manager):
    """Test integrating a new insight (grow operation)"""
    curator = ACECurator(embedding_manager)
    initial_count = len(sample_context.bullets)

    insight = {
        "text": "Always validate user input",
        "bullet_type": BulletType.STRATEGY.value,
        "confidence": 0.7,
        "source": "reflection_success",
        "metadata": {},
    }

    curator.curate_insights(context=sample_context, insights=[insight])

    # Should add new bullet since it's not similar to existing ones
    assert len(sample_context.bullets) >= initial_count


def test_curator_update_similar_insight(sample_context, embedding_manager):
    """Test updating similar insight (refine operation)"""
    curator = ACECurator(embedding_manager, deduplication_threshold=0.90)

    # Create insight very similar to existing bullet
    similar_insight = {
        "text": "Be concise and direct in all responses",  # Similar to existing
        "bullet_type": BulletType.PRINCIPLE.value,
        "confidence": 0.8,
        "source": "reflection_success",
        "metadata": {},
    }

    initial_count = len(sample_context.bullets)
    curator.curate_insights(context=sample_context, insights=[similar_insight])

    # Should not significantly increase bullet count due to similarity
    # (might add then deduplicate, or directly update)
    assert len(sample_context.bullets) <= initial_count + 1


def test_curator_deduplication(embedding_manager):
    """Test semantic deduplication"""
    curator = ACECurator(embedding_manager, deduplication_threshold=0.80)
    context = ACEContext()
    text_model = embedding_manager.get_text_embedder()

    # Add two nearly identical bullets
    similar_bullets = [
        ("Be concise", BulletType.PRINCIPLE, 0.6),
        ("Be concise", BulletType.PRINCIPLE, 0.7),  # Exact duplicate
    ]

    for text, bullet_type, confidence in similar_bullets:
        embedding = text_model.encode(text, convert_to_numpy=True)
        bullet = ACEBullet(
            text=text, bullet_type=bullet_type, embedding=embedding, confidence=confidence
        )
        context.add_bullet(bullet)

    initial_count = len(context.bullets)
    curator._deduplicate_bullets(context)

    # Should remove one due to being identical
    assert len(context.bullets) < initial_count


def test_curator_pruning(sample_context, embedding_manager):
    """Test pruning low-confidence bullets"""
    curator = ACECurator(embedding_manager)
    text_model = embedding_manager.get_text_embedder()

    # Add many low-confidence bullets
    for i in range(10):
        embedding = text_model.encode(f"Low confidence bullet {i}", convert_to_numpy=True)
        bullet = ACEBullet(
            text=f"Low confidence bullet {i}",
            bullet_type=BulletType.LEARNED,
            embedding=embedding,
            confidence=0.2,
        )
        sample_context.add_bullet(bullet)

    target_count = 5

    curator._prune_low_confidence_bullets(sample_context, target_count=target_count)

    assert len(sample_context.bullets) == target_count
    # Remaining bullets should be higher confidence
    remaining_confidences = [b.confidence for b in sample_context.bullets.values()]
    assert all(c >= 0.2 for c in remaining_confidences)


def test_curator_max_bullets_limit(sample_context, embedding_manager):
    """Test enforcing max bullets limit"""
    curator = ACECurator(embedding_manager)

    # Create many insights
    insights = []
    for i in range(20):
        insights.append(
            {
                "text": f"Unique insight number {i}",
                "bullet_type": BulletType.LEARNED.value,
                "confidence": 0.5 + (i * 0.01),  # Varying confidence
                "source": "reflection_success",
                "metadata": {},
            }
        )

    curator.curate_insights(context=sample_context, insights=insights, max_bullets=10)

    # Should not exceed max_bullets
    assert len(sample_context.bullets) <= 10


# ============================================================================
# Test ACEFramework
# ============================================================================


def test_framework_creation(embedding_manager):
    """Test ACEFramework initialization"""
    framework = ACEFramework(
        embedding_manager=embedding_manager,
        deduplication_threshold=0.85,
        max_bullets=100,
    )

    assert framework.generator is not None
    assert framework.reflector is not None
    assert framework.curator is not None
    assert framework.max_bullets == 100


def test_framework_create_initial_context(embedding_manager):
    """Test creating initial context with seed bullets"""
    framework = ACEFramework(embedding_manager)

    seed_bullets = [
        ("Be concise", BulletType.PRINCIPLE),
        ("Use examples", BulletType.STRATEGY),
        ("Check edge cases", BulletType.TACTIC),
    ]

    context = framework.create_initial_context(initial_bullets=seed_bullets)

    assert len(context.bullets) == 3
    assert context.version > 1  # Incremented for each addition


def test_framework_execute_ace_cycle_success(sample_context, embedding_manager):
    """Test full ACE cycle with successful outcome"""
    framework = ACEFramework(embedding_manager, max_bullets=20)

    task = "Explain quantum computing simply"
    outcome = "Successfully explained with clear examples"
    initial_version = sample_context.version
    initial_bullet_count = len(sample_context.bullets)

    updated_context, trajectory = framework.execute_ace_cycle(
        task=task,
        context=sample_context,
        outcome=outcome,
        success=True,
        max_trajectory_steps=3,
    )

    # Check trajectory was generated
    assert len(trajectory) > 0
    assert all("step_number" in step for step in trajectory)

    # Context version may or may not increase depending on insights
    # (reflector might not generate new insights, or they might be deduplicated)
    assert updated_context.version >= initial_version
    assert len(updated_context.bullets) >= initial_bullet_count - 2  # Allow for deduplication


def test_framework_execute_ace_cycle_failure(sample_context, embedding_manager):
    """Test full ACE cycle with failed outcome"""
    framework = ACEFramework(embedding_manager, max_bullets=20)

    task = "Solve unsolvable problem"
    outcome = "Failed due to impossible constraints"
    initial_version = sample_context.version

    updated_context, trajectory = framework.execute_ace_cycle(
        task=task,
        context=sample_context,
        outcome=outcome,
        success=False,
        max_trajectory_steps=3,
    )

    # Check trajectory was generated
    assert len(trajectory) > 0

    # Context version may or may not increase depending on insights
    assert updated_context.version >= initial_version


def test_framework_iterative_improvement(embedding_manager):
    """Test that ACE framework can evolve context over iterations"""
    framework = ACEFramework(embedding_manager, max_bullets=50)

    # Start with minimal context
    context = framework.create_initial_context(
        initial_bullets=[("Be helpful", BulletType.PRINCIPLE)]
    )

    tasks_and_outcomes = [
        ("Task 1", "Success", True),
        ("Task 2", "Success", True),
        ("Task 3", "Failure", False),
        ("Task 4", "Success", True),
    ]

    initial_version = context.version

    for task, outcome, success in tasks_and_outcomes:
        context, _ = framework.execute_ace_cycle(
            task=task, context=context, outcome=outcome, success=success
        )

    # Version should be same or higher (may not always increase if no new insights)
    assert context.version >= initial_version

    # Context should have at least the initial bullet
    assert len(context.bullets) >= 1


# ============================================================================
# Test Integration Scenarios
# ============================================================================


def test_full_ace_workflow(embedding_manager):
    """Test complete ACE workflow from seed to evolved playbook"""
    framework = ACEFramework(embedding_manager, max_bullets=30)

    # 1. Create initial context
    context = framework.create_initial_context(
        initial_bullets=[
            ("Be clear and concise", BulletType.PRINCIPLE),
            ("Provide examples", BulletType.STRATEGY),
        ]
    )

    assert len(context.bullets) == 2
    initial_version = context.version

    # 2. Execute several ACE cycles
    cycles = [
        ("Explain machine learning", "Explained with examples", True),
        ("Debug complex code", "Found the bug successfully", True),
        ("Predict stock prices", "Failed - too uncertain", False),
        ("Review code quality", "Identified improvements", True),
    ]

    for task, outcome, success in cycles:
        context, trajectory = framework.execute_ace_cycle(
            task=task, context=context, outcome=outcome, success=success
        )

        # Each cycle should produce trajectory
        assert len(trajectory) > 0

    # 3. Verify evolution (version may or may not increase)
    assert context.version >= initial_version
    assert len(context.bullets) >= 1  # At least some bullets remain

    # 4. Check performance stats
    stats = context.get_performance_stats()
    assert stats["total_bullets"] > 0
    assert "avg_confidence" in stats


def test_deduplication_prevents_bloat(embedding_manager):
    """Test that deduplication prevents context bloat"""
    framework = ACEFramework(embedding_manager, max_bullets=20, deduplication_threshold=0.80)

    context = framework.create_initial_context(
        initial_bullets=[("Be concise", BulletType.PRINCIPLE)]
    )

    # Add many similar insights
    similar_tasks = [
        f"Explain {topic} concisely"
        for topic in ["topic A", "topic B", "topic C", "topic D", "topic E"]
    ]

    for task in similar_tasks:
        context, _ = framework.execute_ace_cycle(
            task=task, context=context, outcome="Success", success=True
        )

    # Despite many cycles, deduplication should prevent excessive growth
    assert len(context.bullets) < 20  # Well below max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
