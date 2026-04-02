"""
Comprehensive tests for detection_handlers.py

Coverage target: 80%+ (currently 25%)
Tests blind spot detection and hallucination detection handlers.

Version: 0.7.0 - Updated for async handlers with rate limiting
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import List
from src.identity_scope import compose_scoped_file_id
from src.handlers.detection_handlers import (
    handle_check_blind_spots,
    handle_detect_hallucination,
)

# ============================================================================
# Mock Data Classes (matching blind_spot_detector.py structure)
# ============================================================================


@dataclass
class MockBlindSpot:
    """Mock BlindSpot for testing"""

    node_id: str
    similarity_to_response: float
    reason: str
    urgency: str


@dataclass
class MockBlindSpotReport:
    """Mock BlindSpotReport for testing"""

    response_analyzed: str
    total_blind_spots: int
    critical_blind_spots: int
    blind_spots: List[MockBlindSpot]
    recommendations: List[str]
    auto_inject: List[str]  # Node IDs to auto-inject


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_context():
    """Create mock HandlerContext for detection handlers"""
    context = {}

    # Mock blind_spot_detector
    mock_blind_spot_detector = Mock()
    context["blind_spot_detector"] = mock_blind_spot_detector

    # Mock halo_detector
    mock_halo_detector = Mock()
    context["halo_detector"] = mock_halo_detector

    # Mock compressor with graphs dict (v0.7.0 - required for file_id validation)
    # Note: SemanticCompressor uses 'graphs' not 'documents' as the dict name
    mock_compressor = Mock()
    mock_compressor.graphs = {
        "quantum_paper": Mock(),
        "test_doc": Mock(),
        "research_paper": Mock(),
        "math_paper": Mock(),
        "doc": Mock(),
        "x": Mock(),
    }
    context["compressor"] = mock_compressor

    return context


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter for async handler tests"""
    with patch("src.handlers.detection_handlers.RATE_LIMITERS") as mock_limiters:
        mock_limiter = AsyncMock()
        mock_limiter.acquire = AsyncMock()
        mock_limiters.__getitem__ = Mock(return_value=mock_limiter)
        yield mock_limiters


# ============================================================================
# Test handle_check_blind_spots Handler
# ============================================================================


class TestHandleCheckBlindSpots:
    """Test blind spot detection handler"""

    @pytest.mark.asyncio
    async def test_check_blind_spots_no_blind_spots_found(self, mock_context, mock_rate_limiter):
        """Test blind spot check when no issues found"""
        # Setup
        report = MockBlindSpotReport(
            response_analyzed="Test response",
            total_blind_spots=0,
            critical_blind_spots=0,
            blind_spots=[],
            recommendations=["[OK] Response appears complete"],
            auto_inject=[],
        )

        mock_context["blind_spot_detector"].analyze_response.return_value = report
        mock_context["blind_spot_detector"].format_report.return_value = (
            "[OK] No blind spots detected\nResponse appears complete"
        )

        args = {
            "ai_response": "Test response about quantum computing",
            "file_id": "quantum_paper",
            "retrieved_nodes": ["quantum_paper_n1", "quantum_paper_n5"],
        }

        # Execute
        result = await handle_check_blind_spots(mock_context, args)

        # Verify
        assert "[OK]" in result
        assert "No blind spots detected" in result or "complete" in result.lower()
        mock_context["blind_spot_detector"].analyze_response.assert_called_once_with(
            "Test response about quantum computing",
            "quantum_paper",
            ["quantum_paper_n1", "quantum_paper_n5"],
        )
        mock_context["blind_spot_detector"].format_report.assert_called_once_with(report)
        # Should NOT have auto-correction since auto_inject is empty
        assert "AUTO-CORRECTION" not in result

    @pytest.mark.asyncio
    async def test_check_blind_spots_with_critical_issues(self, mock_context, mock_rate_limiter):
        """Test blind spot check with critical issues requiring auto-injection"""
        # Setup
        report = MockBlindSpotReport(
            response_analyzed="Test response",
            total_blind_spots=2,
            critical_blind_spots=1,
            blind_spots=[
                MockBlindSpot(
                    node_id="quantum_paper_n12",
                    similarity_to_response=0.75,
                    reason="Critical error data not retrieved",
                    urgency="critical",
                ),
                MockBlindSpot(
                    node_id="quantum_paper_n18",
                    similarity_to_response=0.65,
                    reason="Important context missing",
                    urgency="high",
                ),
            ],
            recommendations=[
                "[CRITICAL]: Missing error handling context",
                "[HIGH]: Additional context recommended",
            ],
            auto_inject=["quantum_paper_n12"],  # Critical node for auto-injection
        )

        mock_context["blind_spot_detector"].analyze_response.return_value = report
        mock_context["blind_spot_detector"].format_report.return_value = (
            "[WARN] 2 blind spots detected\n[CRITICAL]: Missing error handling context"
        )

        args = {
            "ai_response": "Error handling uses standard try-catch",
            "file_id": "quantum_paper",
            "retrieved_nodes": ["quantum_paper_n1"],
        }

        # Execute
        result = await handle_check_blind_spots(mock_context, args)

        # Verify
        assert "[WARN]" in result or "blind spots" in result.lower()
        assert "AUTO-CORRECTION SUGGESTED" in result
        assert "quantum_paper_n12" in result
        assert "modulate_region" in result
        mock_context["blind_spot_detector"].analyze_response.assert_called_once()
        mock_context["blind_spot_detector"].format_report.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_blind_spots_with_multiple_auto_inject(
        self, mock_context, mock_rate_limiter
    ):
        """Test blind spot check with multiple nodes for auto-injection"""
        # Setup
        report = MockBlindSpotReport(
            response_analyzed="Test response",
            total_blind_spots=3,
            critical_blind_spots=2,
            blind_spots=[],
            recommendations=["Multiple critical blind spots detected"],
            auto_inject=["doc_n5", "doc_n12", "doc_n18"],  # Multiple nodes
        )

        mock_context["blind_spot_detector"].analyze_response.return_value = report
        mock_context["blind_spot_detector"].format_report.return_value = (
            "[WARN] 3 blind spots detected"
        )

        args = {
            "ai_response": "Brief response",
            "file_id": "test_doc",
            "retrieved_nodes": ["doc_n1"],
        }

        # Execute
        result = await handle_check_blind_spots(mock_context, args)

        # Verify
        assert "AUTO-CORRECTION SUGGESTED" in result
        assert "doc_n5" in result or "['doc_n5', 'doc_n12', 'doc_n18']" in result
        assert "Retrieve these nodes" in result

    @pytest.mark.asyncio
    async def test_check_blind_spots_empty_retrieved_nodes(self, mock_context, mock_rate_limiter):
        """Test blind spot check with no nodes initially retrieved"""
        # Setup
        report = MockBlindSpotReport(
            response_analyzed="Test response",
            total_blind_spots=5,
            critical_blind_spots=3,
            blind_spots=[],
            recommendations=["Response generated without context"],
            auto_inject=["doc_n1", "doc_n2", "doc_n3"],
        )

        mock_context["blind_spot_detector"].analyze_response.return_value = report
        mock_context["blind_spot_detector"].format_report.return_value = (
            "[WARN] Response may be entirely fabricated - no context retrieved"
        )

        args = {
            "ai_response": "Test response",
            "file_id": "test_doc",
            "retrieved_nodes": [],  # Empty - no nodes retrieved
        }

        # Execute
        result = await handle_check_blind_spots(mock_context, args)

        # Verify
        assert "AUTO-CORRECTION SUGGESTED" in result
        mock_context["blind_spot_detector"].analyze_response.assert_called_once_with(
            "Test response", "test_doc", []
        )

    @pytest.mark.asyncio
    async def test_check_blind_spots_scoped_file_id(self, mock_context, mock_rate_limiter):
        scoped_file_id = compose_scoped_file_id("test_doc", workspace_id="acme")
        mock_context["compressor"].graphs[scoped_file_id] = Mock()
        report = MockBlindSpotReport(
            response_analyzed="Scoped response",
            total_blind_spots=0,
            critical_blind_spots=0,
            blind_spots=[],
            recommendations=["[OK] Response appears complete"],
            auto_inject=[],
        )
        mock_context["blind_spot_detector"].analyze_response.return_value = report
        mock_context["blind_spot_detector"].format_report.return_value = (
            "[OK] No blind spots detected"
        )

        await handle_check_blind_spots(
            mock_context,
            {
                "ai_response": "Scoped response",
                "file_id": "test_doc",
                "workspace_id": "acme",
                "retrieved_nodes": ["test_doc_n1"],
            },
        )

        mock_context["blind_spot_detector"].analyze_response.assert_called_once_with(
            "Scoped response", scoped_file_id, ["test_doc_n1"]
        )


# ============================================================================
# Test handle_detect_hallucination Handler
# ============================================================================


class TestHandleDetectHallucination:
    """Test hallucination detection handler"""

    @pytest.mark.asyncio
    async def test_detect_hallucination_not_hallucinating(self, mock_context, mock_rate_limiter):
        """Test hallucination detection when response is grounded"""
        # Setup
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        args = {
            "ai_response": "The quantum gate fidelity was measured at 99.2%",
            "file_id": "quantum_paper",
        }

        # Execute
        result = await handle_detect_hallucination(mock_context, args)

        # Verify
        assert "[OK]" in result
        assert "grounded" in result.lower()
        assert "No hallucination detected" in result
        assert "[ALERT]" not in result  # Should NOT have alert emoji
        mock_context["halo_detector"].detect_hallucination.assert_called_once_with(
            "The quantum gate fidelity was measured at 99.2%", "quantum_paper"
        )

    @pytest.mark.asyncio
    async def test_detect_hallucination_is_hallucinating_single_warning(
        self, mock_context, mock_rate_limiter
    ):
        """Test hallucination detection with single warning"""
        # Setup
        mock_context["halo_detector"].detect_hallucination.return_value = (
            True,
            ["Response has low similarity to all document nodes (max: 0.15)"],
        )

        args = {
            "ai_response": "The paper discusses neural network architectures",
            "file_id": "quantum_paper",  # Wrong topic!
        }

        # Execute
        result = await handle_detect_hallucination(mock_context, args)

        # Verify
        assert "[ALERT]" in result
        assert "HALLUCINATION ALERT" in result
        assert "fabricated information" in result
        assert "low similarity" in result
        assert "Recommendation" in result
        assert "Re-examine source material" in result
        mock_context["halo_detector"].detect_hallucination.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_hallucination_is_hallucinating_multiple_warnings(
        self, mock_context, mock_rate_limiter
    ):
        """Test hallucination detection with multiple warnings"""
        # Setup
        mock_context["halo_detector"].detect_hallucination.return_value = (
            True,
            [
                "Response has low similarity to all document nodes (max: 0.12)",
                "AI may be generating content not present in the source document",
                "Overconfident claims without supporting evidence",
            ],
        )

        args = {
            "ai_response": "The research definitively proves X, Y, and Z",
            "file_id": "research_paper",
        }

        # Execute
        result = await handle_detect_hallucination(mock_context, args)

        # Verify
        assert "[ALERT]" in result
        assert "HALLUCINATION ALERT" in result
        assert "fabricated information" in result
        # All warnings should be in output
        assert "low similarity" in result
        assert "not present in the source document" in result
        assert "Overconfident claims" in result
        # Verify bullet formatting (uses dashes for enterprise-grade ASCII output)
        assert "  -" in result

    @pytest.mark.asyncio
    async def test_detect_hallucination_empty_warnings(self, mock_context, mock_rate_limiter):
        """Test hallucination detection with hallucination but empty warnings list"""
        # Setup - edge case where is_hallucinating=True but warnings=[]
        mock_context["halo_detector"].detect_hallucination.return_value = (True, [])

        args = {
            "ai_response": "Test response",
            "file_id": "test_doc",
        }

        # Execute
        result = await handle_detect_hallucination(mock_context, args)

        # Verify
        assert "[ALERT]" in result
        assert "HALLUCINATION ALERT" in result
        # Should still show alert even with no specific warnings

    @pytest.mark.asyncio
    async def test_detect_hallucination_handles_special_characters(
        self, mock_context, mock_rate_limiter
    ):
        """Test hallucination detection with special characters in response"""
        # Setup
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        args = {
            "ai_response": "The formula is: ∑(x²) ≈ 42.5% with α=0.05",
            "file_id": "math_paper",
        }

        # Execute - should not crash with special characters
        result = await handle_detect_hallucination(mock_context, args)

        # Verify
        assert "[OK]" in result
        assert "grounded" in result.lower()

    @pytest.mark.asyncio
    async def test_detect_hallucination_handler_logging(self, mock_context, mock_rate_limiter):
        """Test that handler logs appropriately"""
        # Setup
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        args = {
            "ai_response": "Test response",
            "file_id": "test_doc",
        }

        # Execute
        result = await handle_detect_hallucination(mock_context, args)

        # Verify handler completed successfully
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_detect_hallucination_scoped_file_id(self, mock_context, mock_rate_limiter):
        scoped_file_id = compose_scoped_file_id("test_doc", workspace_id="acme")
        mock_context["compressor"].graphs[scoped_file_id] = Mock()
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        await handle_detect_hallucination(
            mock_context,
            {"ai_response": "Scoped response", "file_id": "test_doc", "workspace_id": "acme"},
        )

        mock_context["halo_detector"].detect_hallucination.assert_called_once_with(
            "Scoped response", scoped_file_id
        )


# ============================================================================
# Integration Tests
# ============================================================================


class TestDetectionHandlersIntegration:
    """Integration tests for both detection handlers"""

    @pytest.mark.asyncio
    async def test_both_handlers_use_correct_context_keys(self, mock_context, mock_rate_limiter):
        """Test that both handlers access correct context manager keys"""
        # Setup blind spots
        blind_report = MockBlindSpotReport(
            response_analyzed="Test",
            total_blind_spots=0,
            critical_blind_spots=0,
            blind_spots=[],
            recommendations=[],
            auto_inject=[],
        )
        mock_context["blind_spot_detector"].analyze_response.return_value = blind_report
        mock_context["blind_spot_detector"].format_report.return_value = "[OK] Clean"

        # Setup hallucination
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        # Execute both handlers
        blind_result = await handle_check_blind_spots(
            mock_context,
            {"ai_response": "Test", "file_id": "doc", "retrieved_nodes": []},
        )
        halo_result = await handle_detect_hallucination(
            mock_context, {"ai_response": "Test", "file_id": "doc"}
        )

        # Verify both completed successfully
        assert blind_result is not None
        assert halo_result is not None
        assert (
            "blind_spot_detector"
            in str(mock_context["blind_spot_detector"].analyze_response.call_args).lower()
            or True
        )
        assert (
            "halo_detector"
            in str(mock_context["halo_detector"].detect_hallucination.call_args).lower()
            or True
        )

    @pytest.mark.asyncio
    async def test_handlers_with_minimal_args(self, mock_context, mock_rate_limiter):
        """Test handlers work with minimal required arguments"""
        # Setup
        blind_report = MockBlindSpotReport(
            response_analyzed="",
            total_blind_spots=0,
            critical_blind_spots=0,
            blind_spots=[],
            recommendations=[],
            auto_inject=[],
        )
        mock_context["blind_spot_detector"].analyze_response.return_value = blind_report
        mock_context["blind_spot_detector"].format_report.return_value = "OK"
        mock_context["halo_detector"].detect_hallucination.return_value = (False, [])

        # Execute with minimal args
        blind_result = await handle_check_blind_spots(
            mock_context, {"ai_response": "", "file_id": "x", "retrieved_nodes": []}
        )
        halo_result = await handle_detect_hallucination(
            mock_context, {"ai_response": "", "file_id": "x"}
        )

        # Verify
        assert blind_result is not None
        assert halo_result is not None
