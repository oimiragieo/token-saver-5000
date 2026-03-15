"""
Comprehensive tests for ace_handlers.py

Coverage target: 80%+ (currently 37%)
Tests ACE (Agentic Context Engineering) MCP tool handlers.
"""

import json
import pytest
from unittest.mock import Mock
import numpy as np

from src.handlers.ace_handlers import (
    handle_ace_generate,
    handle_ace_reflect,
    handle_ace_curate,
    handle_ace_grow_context,
    handle_ace_refine_context,
    handle_ace_get_playbook,
    handle_ace_execute_cycle,
    _get_or_create_context,
    _calculate_avg_confidence,
    _build_success_response,
    _build_error_response,
    _add_bullet_to_context,
    _update_bullets_performance,
    _filter_and_serialize_bullets,
)
from src.ace_framework import BulletType


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_ace_context():
    """Create mock ACE context"""
    context = Mock()
    context.version = 1
    context.bullets = {}
    context.delta_history = []
    context.add_bullet = Mock()
    context.get_performance_stats = Mock(
        return_value={"avg_success_rate": 0.75, "total_bullets": 5}
    )
    return context


@pytest.fixture
def mock_ace_framework(mock_ace_context):
    """Create mock ACE framework"""
    framework = Mock()
    framework.create_initial_context = Mock(return_value=mock_ace_context)

    # Mock generator
    framework.generator = Mock()
    framework.generator.generate_trajectory = Mock(
        return_value=[
            {"step": 1, "thought": "First step", "confidence": 0.8},
            {"step": 2, "thought": "Second step", "confidence": 0.9},
        ]
    )

    # Mock reflector
    framework.reflector = Mock()
    framework.reflector.reflect_on_trajectory = Mock(
        return_value=[
            {"insight": "Insight 1", "confidence": 0.7},
            {"insight": "Insight 2", "confidence": 0.6},
        ]
    )

    # Mock curator
    framework.curator = Mock()

    # Create updated context for curator
    updated_context = Mock()
    updated_context.version = 3
    updated_context.bullets = {"bullet1": Mock(), "bullet2": Mock()}
    updated_context.get_performance_stats = Mock(
        return_value={"avg_success_rate": 0.8, "total_bullets": 2}
    )
    framework.curator.curate_insights = Mock(return_value=updated_context)

    # Mock text model
    framework.text_model = Mock()
    framework.text_model.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))

    # Mock execute_ace_cycle
    framework.execute_ace_cycle = Mock(
        return_value=(
            updated_context,
            [
                {"step": 1, "thought": "Cycle step 1", "confidence": 0.85},
                {"step": 2, "thought": "Cycle step 2", "confidence": 0.90},
            ],
        )
    )

    return framework


@pytest.fixture
def mock_context(mock_ace_framework, mock_ace_context):
    """Create mock handler context"""
    return {
        "ace_framework": mock_ace_framework,
        "ace_contexts": {"existing_context": mock_ace_context},
    }


# ============================================================================
# Test Helper Functions
# ============================================================================


class TestHelperFunctions:
    """Test ACE handler helper functions"""

    def test_get_or_create_context_existing(self, mock_ace_framework, mock_ace_context):
        """Test getting existing context"""
        ace_contexts = {"test_id": mock_ace_context}

        result = _get_or_create_context(ace_contexts, mock_ace_framework, "test_id")

        assert result == mock_ace_context
        mock_ace_framework.create_initial_context.assert_not_called()

    def test_get_or_create_context_new(self, mock_ace_framework, mock_ace_context):
        """Test creating new context"""
        ace_contexts = {}

        result = _get_or_create_context(ace_contexts, mock_ace_framework, "new_id")

        assert result == mock_ace_context
        assert "new_id" in ace_contexts
        mock_ace_framework.create_initial_context.assert_called_once()

    def test_calculate_avg_confidence_normal(self):
        """Test average confidence calculation"""
        items = [{"confidence": 0.8}, {"confidence": 0.6}, {"confidence": 0.9}]

        avg = _calculate_avg_confidence(items)

        assert avg == pytest.approx(0.7666, rel=0.01)

    def test_calculate_avg_confidence_empty(self):
        """Test average confidence with empty list"""
        assert _calculate_avg_confidence([]) == 0.0

    def test_calculate_avg_confidence_single(self):
        """Test average confidence with single item"""
        assert _calculate_avg_confidence([{"confidence": 0.5}]) == 0.5

    def test_build_success_response(self):
        """Test building success response"""
        data = {"result": "test", "count": 42}

        response = _build_success_response(data)

        parsed = json.loads(response)
        assert parsed["status"] == "success"
        assert parsed["result"] == "test"
        assert parsed["count"] == 42

    def test_build_error_response(self):
        """Test building error response"""
        response = _build_error_response("Test error", "test operation")

        parsed = json.loads(response)
        assert parsed["status"] == "error"
        assert parsed["message"] == "Test error"

    def test_add_bullet_to_context(self, mock_ace_context):
        """Test adding bullet to context"""
        bullet_data = {
            "text": "Test bullet content",
            "bullet_type": "strategy",
            "confidence": 0.7,
        }
        text_model = Mock()
        text_model.encode = Mock(return_value=np.array([0.1, 0.2]))

        _add_bullet_to_context(mock_ace_context, bullet_data, text_model)

        # Verify bullet was added
        mock_ace_context.add_bullet.assert_called_once()
        call_args = mock_ace_context.add_bullet.call_args
        bullet = call_args[0][0]
        assert bullet.text == "Test bullet content"
        assert bullet.bullet_type == BulletType.STRATEGY
        assert bullet.confidence == 0.7

    def test_add_bullet_default_confidence(self, mock_ace_context):
        """Test adding bullet with default confidence"""
        bullet_data = {"text": "Test", "bullet_type": "principle"}
        text_model = Mock()
        text_model.encode = Mock(return_value=np.array([0.1]))

        _add_bullet_to_context(mock_ace_context, bullet_data, text_model)

        call_args = mock_ace_context.add_bullet.call_args
        bullet = call_args[0][0]
        assert bullet.confidence == 0.5  # Default

    def test_update_bullets_performance_success(self, mock_ace_context):
        """Test updating bullet performance for success"""
        # Setup mock bullets
        bullet1 = Mock()
        bullet1.confidence = 0.5
        bullet1.update_performance = Mock()
        bullet2 = Mock()
        bullet2.confidence = 0.6
        bullet2.update_performance = Mock()

        mock_ace_context.bullets = {"b1": bullet1, "b2": bullet2}

        updated_count, confidences = _update_bullets_performance(
            mock_ace_context, ["b1", "b2"], success=True, confidence_boost=0.1, context_id="test"
        )

        assert updated_count == 2
        assert len(confidences) == 2
        bullet1.update_performance.assert_called_once_with(success=True, confidence_boost=0.1)
        bullet2.update_performance.assert_called_once_with(success=True, confidence_boost=0.1)

    def test_update_bullets_performance_missing_bullet(self, mock_ace_context):
        """Test updating with non-existent bullet ID"""
        bullet1 = Mock()
        bullet1.confidence = 0.5
        bullet1.update_performance = Mock()
        mock_ace_context.bullets = {"b1": bullet1}

        updated_count, confidences = _update_bullets_performance(
            mock_ace_context,
            ["b1", "b_missing"],
            success=True,
            confidence_boost=0.1,
            context_id="test",
        )

        # Only b1 should be updated
        assert updated_count == 1
        assert len(confidences) == 1

    def test_filter_and_serialize_bullets_no_filters(self, mock_ace_context):
        """Test filtering bullets without filters"""
        # Setup mock bullets
        bullet1 = Mock()
        bullet1.confidence = 0.7
        bullet1.bullet_type = BulletType.STRATEGY
        bullet1.to_display_dict = Mock(return_value={"text": "Test 1"})

        bullet2 = Mock()
        bullet2.confidence = 0.5
        bullet2.bullet_type = BulletType.PRINCIPLE
        bullet2.to_display_dict = Mock(return_value={"text": "Test 2"})

        mock_ace_context.bullets = {"b1": bullet1, "b2": bullet2}

        result = _filter_and_serialize_bullets(
            mock_ace_context, include_embeddings=False, min_confidence=None, bullet_type_filter=None
        )

        assert len(result) == 2
        # Embeddings should not be present when include_embeddings=False
        assert "embedding" not in result[0]
        assert "embedding" not in result[1]

    def test_filter_and_serialize_bullets_with_confidence_filter(self, mock_ace_context):
        """Test filtering bullets by minimum confidence"""
        bullet1 = Mock()
        bullet1.confidence = 0.8
        bullet1.bullet_type = BulletType.STRATEGY
        bullet1.to_display_dict = Mock(return_value={"text": "High conf"})

        bullet2 = Mock()
        bullet2.confidence = 0.4
        bullet2.bullet_type = BulletType.TACTIC
        bullet2.to_display_dict = Mock(return_value={"text": "Low conf"})

        mock_ace_context.bullets = {"b1": bullet1, "b2": bullet2}

        result = _filter_and_serialize_bullets(
            mock_ace_context, include_embeddings=False, min_confidence=0.6, bullet_type_filter=None
        )

        # Only bullet1 should pass (0.8 >= 0.6)
        assert len(result) == 1

    def test_filter_and_serialize_bullets_with_type_filter(self, mock_ace_context):
        """Test filtering bullets by type"""
        bullet1 = Mock()
        bullet1.confidence = 0.7
        bullet1.bullet_type = BulletType.STRATEGY
        bullet1.to_display_dict = Mock(return_value={"text": "Strategy"})

        bullet2 = Mock()
        bullet2.confidence = 0.6
        bullet2.bullet_type = BulletType.PRINCIPLE
        bullet2.to_display_dict = Mock(return_value={"text": "Principle"})

        mock_ace_context.bullets = {"b1": bullet1, "b2": bullet2}

        result = _filter_and_serialize_bullets(
            mock_ace_context,
            include_embeddings=False,
            min_confidence=None,
            bullet_type_filter="strategy",
        )

        # Only strategy bullet should pass
        assert len(result) == 1

    def test_filter_and_serialize_bullets_include_embeddings(self, mock_ace_context):
        """Test including embeddings in serialized bullets"""
        bullet1 = Mock()
        bullet1.confidence = 0.7
        bullet1.bullet_type = BulletType.STRATEGY
        bullet1.to_display_dict = Mock(return_value={"text": "Test"})
        bullet1.embedding = Mock()
        bullet1.embedding.tolist = Mock(return_value=[0.1, 0.2])

        mock_ace_context.bullets = {"b1": bullet1}

        result = _filter_and_serialize_bullets(
            mock_ace_context, include_embeddings=True, min_confidence=None, bullet_type_filter=None
        )

        # Embedding should be included
        assert "embedding" in result[0]


# ============================================================================
# Test ACE Generate Handler
# ============================================================================


class TestHandleAceGenerate:
    """Test ACE trajectory generation handler"""

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_generate_success(self, mock_context):
        """Test successful trajectory generation"""
        args = {
            "task": "Explain quantum entanglement",
            "context_id": "quantum_domain",
            "max_steps": 5,
            "top_k_bullets": 5,
        }

        result = await handle_ace_generate(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "trajectory" in parsed
        assert parsed["context_id"] == "quantum_domain"
        assert parsed["stats"]["total_steps"] == 2
        assert "avg_confidence" in parsed["stats"]

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_generate_with_defaults(self, mock_context):
        """Test generation with default parameters"""
        args = {"task": "Test task"}

        result = await handle_ace_generate(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["context_id"] == "default"

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, mock_context):
        """Test error handling in generation"""
        mock_context["ace_framework"].generator.generate_trajectory.side_effect = Exception(
            "Generation failed"
        )

        args = {"task": "Test task"}
        result = await handle_ace_generate(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Generation failed" in parsed["message"]


# ============================================================================
# Test ACE Reflect Handler
# ============================================================================


class TestHandleAceReflect:
    """Test ACE trajectory reflection handler"""

    @pytest.mark.asyncio
    async def test_reflect_success(self, mock_context):
        """Test successful reflection"""
        args = {
            "trajectory": [{"step": 1, "thought": "Test"}],
            "outcome": "Task completed successfully",
            "success": True,
            "context_id": "test_domain",
        }

        result = await handle_ace_reflect(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "insights" in parsed
        assert parsed["context_id"] == "test_domain"
        assert parsed["stats"]["total_insights"] == 2

    @pytest.mark.asyncio
    async def test_reflect_failure_outcome(self, mock_context):
        """Test reflection on failed trajectory"""
        args = {
            "trajectory": [{"step": 1, "thought": "Test"}],
            "outcome": "Task failed",
            "success": False,
        }

        result = await handle_ace_reflect(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Verify reflector was called with success=False
        mock_context["ace_framework"].reflector.reflect_on_trajectory.assert_called_once()
        call_args = mock_context["ace_framework"].reflector.reflect_on_trajectory.call_args[1]
        assert call_args["success"] is False

    @pytest.mark.asyncio
    async def test_reflect_error_handling(self, mock_context):
        """Test error handling in reflection"""
        mock_context["ace_framework"].reflector.reflect_on_trajectory.side_effect = Exception(
            "Reflection failed"
        )

        args = {"trajectory": [], "outcome": "Test", "success": True}
        result = await handle_ace_reflect(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Reflection failed" in parsed["message"]


# ============================================================================
# Test ACE Curate Handler
# ============================================================================


class TestHandleAceCurate:
    """Test ACE insight curation handler"""

    @pytest.mark.asyncio
    async def test_curate_success(self, mock_context):
        """Test successful curation"""
        args = {"insights": [{"insight": "Test insight", "confidence": 0.7}], "context_id": "test"}

        result = await handle_ace_curate(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["context_id"] == "test"
        assert parsed["version"] == 3
        assert parsed["total_bullets"] == 2
        assert parsed["deltas_applied"] > 0

    @pytest.mark.asyncio
    async def test_curate_with_max_bullets(self, mock_context):
        """Test curation with max bullets limit"""
        args = {"insights": [{"insight": "Test"}], "max_bullets": 10}

        result = await handle_ace_curate(mock_context, args)

        # Verify result is success and max_bullets was passed to curator
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_context["ace_framework"].curator.curate_insights.assert_called_once()
        call_args = mock_context["ace_framework"].curator.curate_insights.call_args[1]
        assert call_args["max_bullets"] == 10

    @pytest.mark.asyncio
    async def test_curate_error_handling(self, mock_context):
        """Test error handling in curation"""
        mock_context["ace_framework"].curator.curate_insights.side_effect = Exception(
            "Curation failed"
        )

        args = {"insights": []}
        result = await handle_ace_curate(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Curation failed" in parsed["message"]


# ============================================================================
# Test ACE Grow Context Handler
# ============================================================================


class TestHandleAceGrowContext:
    """Test ACE context growth handler"""

    @pytest.mark.asyncio
    async def test_grow_context_single_bullet(self, mock_context):
        """Test adding single bullet to context"""
        args = {
            "bullets": [{"text": "Test principle", "bullet_type": "principle", "confidence": 0.8}],
            "context_id": "test",
        }

        result = await handle_ace_grow_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["bullets_added"] == 1
        assert parsed["context_id"] == "test"

    @pytest.mark.asyncio
    async def test_grow_context_multiple_bullets(self, mock_context):
        """Test adding multiple bullets"""
        args = {
            "bullets": [
                {"text": "Strategy 1", "bullet_type": "strategy"},
                {"text": "Tactic 1", "bullet_type": "tactic", "confidence": 0.6},
                {"text": "Constraint 1", "bullet_type": "constraint"},
            ]
        }

        result = await handle_ace_grow_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["bullets_added"] == 3

    @pytest.mark.asyncio
    async def test_grow_context_error_handling(self, mock_context):
        """Test error handling in context growth"""
        # Force error by making text_model.encode fail
        mock_context["ace_framework"].text_model.encode.side_effect = Exception("Encoding failed")

        args = {"bullets": [{"text": "Test", "bullet_type": "strategy"}]}
        result = await handle_ace_grow_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Encoding failed" in parsed["message"]


# ============================================================================
# Test ACE Refine Context Handler
# ============================================================================


class TestHandleAceRefineContext:
    """Test ACE context refinement handler"""

    @pytest.mark.asyncio
    async def test_refine_context_success(self, mock_context):
        """Test successful context refinement"""
        # Setup mock bullet
        bullet = Mock()
        bullet.confidence = 0.7
        bullet.update_performance = Mock()
        mock_context["ace_contexts"]["existing_context"].bullets = {"b1": bullet}

        args = {
            "bullet_ids": ["b1"],
            "success": True,
            "confidence_boost": 0.1,
            "context_id": "existing_context",
        }

        result = await handle_ace_refine_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["bullets_updated"] == 1
        assert "avg_confidence_after" in parsed
        bullet.update_performance.assert_called_once_with(success=True, confidence_boost=0.1)

    @pytest.mark.asyncio
    async def test_refine_context_failure(self, mock_context):
        """Test refinement with failure feedback"""
        bullet = Mock()
        bullet.confidence = 0.5
        bullet.update_performance = Mock()
        mock_context["ace_contexts"]["existing_context"].bullets = {"b1": bullet}

        args = {"bullet_ids": ["b1"], "success": False, "context_id": "existing_context"}

        result = await handle_ace_refine_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        bullet.update_performance.assert_called_once()
        call_args = bullet.update_performance.call_args[1]
        assert call_args["success"] is False

    @pytest.mark.asyncio
    async def test_refine_context_not_found(self, mock_context):
        """Test refinement with non-existent context"""
        args = {"bullet_ids": ["b1"], "success": True, "context_id": "nonexistent"}

        result = await handle_ace_refine_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "not found" in parsed["message"]

    @pytest.mark.asyncio
    async def test_refine_context_error_handling(self, mock_context):
        """Test error handling in refinement"""
        # Force error by making bullets dict not iterable
        mock_context["ace_contexts"]["existing_context"].bullets = None

        args = {"bullet_ids": ["b1"], "success": True, "context_id": "existing_context"}
        result = await handle_ace_refine_context(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"


# ============================================================================
# Test ACE Get Playbook Handler
# ============================================================================


class TestHandleAceGetPlaybook:
    """Test ACE playbook retrieval handler"""

    @pytest.mark.asyncio
    async def test_get_playbook_basic(self, mock_context):
        """Test basic playbook retrieval"""
        # Setup mock bullets
        bullet1 = Mock()
        bullet1.confidence = 0.8
        bullet1.bullet_type = BulletType.STRATEGY
        bullet1.to_display_dict = Mock(return_value={"text": "Test strategy"})

        mock_context["ace_contexts"]["existing_context"].bullets = {"b1": bullet1}
        mock_context["ace_contexts"]["existing_context"].delta_history = [
            "Delta 1",
            "Delta 2",
            "Delta 3",
        ]

        args = {"context_id": "existing_context"}

        result = await handle_ace_get_playbook(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["context_id"] == "existing_context"
        assert parsed["total_bullets"] == 1
        assert parsed["filtered_bullets"] == 1

    @pytest.mark.asyncio
    async def test_get_playbook_with_filters(self, mock_context):
        """Test playbook retrieval with filters"""
        args = {"context_id": "existing_context", "min_confidence": 0.7, "bullet_type": "strategy"}

        result = await handle_ace_get_playbook(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_playbook_include_embeddings(self, mock_context):
        """Test playbook retrieval with embeddings"""
        bullet1 = Mock()
        bullet1.confidence = 0.8
        bullet1.bullet_type = BulletType.PRINCIPLE
        bullet1.to_display_dict = Mock(return_value={"text": "Test"})
        bullet1.embedding = Mock()
        bullet1.embedding.tolist = Mock(return_value=[0.1, 0.2])

        mock_context["ace_contexts"]["existing_context"].bullets = {"b1": bullet1}

        args = {"context_id": "existing_context", "include_embeddings": True}

        result = await handle_ace_get_playbook(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Verify embedding is included in bullets
        assert len(parsed["bullets"]) > 0

    @pytest.mark.asyncio
    async def test_get_playbook_error_handling(self, mock_context):
        """Test error handling in playbook retrieval"""
        mock_context["ace_contexts"]["existing_context"].bullets = None  # Force error

        args = {}
        result = await handle_ace_get_playbook(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"


# ============================================================================
# Test ACE Execute Cycle Handler
# ============================================================================


class TestHandleAceExecuteCycle:
    """Test full ACE cycle execution handler"""

    @pytest.mark.asyncio
    async def test_execute_cycle_success(self, mock_context):
        """Test successful ACE cycle execution"""
        args = {
            "task": "Solve problem X",
            "outcome": "Problem solved",
            "success": True,
            "context_id": "test",
        }

        result = await handle_ace_execute_cycle(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert "trajectory" in parsed
        assert parsed["version_before"] == 1
        assert parsed["version_after"] == 3
        assert parsed["deltas_applied"] == 2

    @pytest.mark.asyncio
    async def test_execute_cycle_with_max_steps(self, mock_context):
        """Test cycle execution with custom max steps"""
        args = {
            "task": "Test task",
            "outcome": "Done",
            "success": True,
            "max_trajectory_steps": 10,
        }

        result = await handle_ace_execute_cycle(mock_context, args)

        # Verify result is success and max_trajectory_steps was passed
        parsed = json.loads(result)
        assert parsed["status"] == "success"
        mock_context["ace_framework"].execute_ace_cycle.assert_called_once()
        call_args = mock_context["ace_framework"].execute_ace_cycle.call_args[1]
        assert call_args["max_trajectory_steps"] == 10

    @pytest.mark.asyncio
    async def test_execute_cycle_failure_scenario(self, mock_context):
        """Test cycle execution with failure"""
        args = {"task": "Test task", "outcome": "Failed", "success": False}

        result = await handle_ace_execute_cycle(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        # Verify success=False was passed to framework
        call_args = mock_context["ace_framework"].execute_ace_cycle.call_args[1]
        assert call_args["success"] is False

    @pytest.mark.asyncio
    async def test_execute_cycle_error_handling(self, mock_context):
        """Test error handling in cycle execution"""
        mock_context["ace_framework"].execute_ace_cycle.side_effect = Exception("Cycle failed")

        args = {"task": "Test", "outcome": "Test", "success": True}
        result = await handle_ace_execute_cycle(mock_context, args)

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "Cycle failed" in parsed["message"]
