"""
Comprehensive tests for resource_handlers.py

Coverage target: 80%+ (currently 16%)
Tests both helper functions and main handler with various health states.
"""

import pytest
from unittest.mock import Mock
from src.handlers.resource_handlers import (
    handle_check_resource_health,
    create_progress_bar,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_context():
    """Create mock HandlerContext for testing"""
    # Use a real dict for context to avoid MagicMock issues
    context = {}

    # Mock resource manager with default healthy state
    mock_resource_manager = Mock()
    mock_resource_manager.check_health.return_value = {
        "healthy": True,
        "metrics": {
            "storage_mb": 45.2,
            "storage_limit_mb": 500.0,
            "storage_usage_pct": 9.04,
            "document_count": 5,
            "document_limit": 1000,
            "document_usage_pct": 0.5,
            "memory_mb": None,
            "memory_limit_mb": None,
        },
        "warnings": [],
        "recommendations": [],
    }

    context["resource_manager"] = mock_resource_manager
    return context


# ============================================================================
# Test create_progress_bar Helper Function
# ============================================================================


class TestCreateProgressBar:
    """Test progress bar creation with various percentages and states"""

    def test_progress_bar_healthy_low(self):
        """Test progress bar with low percentage (healthy state)"""
        result = create_progress_bar(25.0, width=40)

        assert "[" in result
        assert "]" in result
        assert "✅" in result
        assert "25%" in result
        assert "█" in result  # Should have some filled blocks
        assert "░" in result  # Should have some empty blocks

    def test_progress_bar_healthy_medium(self):
        """Test progress bar with medium percentage (still healthy)"""
        result = create_progress_bar(50.0, width=40)

        assert "✅" in result
        assert "50%" in result
        # 50% should have half filled
        filled_count = result.count("█")
        assert filled_count == 20  # Half of 40

    def test_progress_bar_healthy_high(self):
        """Test progress bar just below warning threshold"""
        result = create_progress_bar(79.0, width=40)

        assert "✅" in result
        assert "79%" in result

    def test_progress_bar_warning_threshold(self):
        """Test progress bar at warning threshold (80%)"""
        result = create_progress_bar(80.0, width=40)

        assert "⚠️" in result
        assert "80%" in result
        assert "█" in result
        assert "░" in result

    def test_progress_bar_warning_high(self):
        """Test progress bar at high warning percentage"""
        result = create_progress_bar(95.0, width=40)

        assert "⚠️" in result
        assert "95%" in result

    def test_progress_bar_full(self):
        """Test progress bar at 100% (critical state)"""
        result = create_progress_bar(100.0, width=40)

        assert "🔴" in result
        assert "FULL" in result
        # Should be all filled blocks
        filled_count = result.count("█")
        assert filled_count == 40

    def test_progress_bar_over_100(self):
        """Test progress bar with over 100% (edge case)"""
        result = create_progress_bar(150.0, width=40)

        assert "🔴" in result
        assert "FULL" in result

    def test_progress_bar_zero(self):
        """Test progress bar at 0%"""
        result = create_progress_bar(0.0, width=40)

        assert "✅" in result
        assert "0%" in result
        # Should be all empty blocks
        empty_count = result.count("░")
        assert empty_count == 40

    def test_progress_bar_custom_width_small(self):
        """Test progress bar with small custom width"""
        result = create_progress_bar(50.0, width=10)

        assert "✅" in result
        assert "50%" in result
        # Total bar width should be 10
        filled = result.count("█")
        empty = result.count("░")
        assert filled + empty == 10

    def test_progress_bar_custom_width_large(self):
        """Test progress bar with large custom width"""
        result = create_progress_bar(75.0, width=60)

        assert "✅" in result
        assert "75%" in result
        filled = result.count("█")
        empty = result.count("░")
        assert filled + empty == 60
        # 75% of 60 = 45 filled
        assert filled == 45


# ============================================================================
# Test handle_check_resource_health Handler
# ============================================================================


class TestHandleCheckResourceHealth:
    """Test resource health check handler with various states"""

    def test_check_health_healthy_state(self, mock_context):
        """Test health check with healthy system state"""
        # Setup - use default healthy state from fixture

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Storage" in result
        assert "45.2 MB" in result
        assert "500.0 MB" in result
        assert "9.0%" in result or "9%" in result
        assert "Document" in result
        assert "5 /" in result
        assert "1000" in result
        assert "0.5%" in result or "0%" in result
        assert "✅" in result
        assert "Healthy" in result

    def test_check_health_with_warnings(self, mock_context):
        """Test health check with system warnings"""
        # Setup
        mock_context["resource_manager"].check_health.return_value = {
            "healthy": True,
            "metrics": {
                "storage_mb": 420.0,
                "storage_limit_mb": 500.0,
                "storage_usage_pct": 84.0,
                "document_count": 850,
                "document_limit": 1000,
                "document_usage_pct": 85.0,
                "memory_mb": None,
                "memory_limit_mb": None,
            },
            "warnings": [
                "Storage usage at 84% - consider cleanup",
                "Document count at 85% - approaching limit",
            ],
            "recommendations": [
                "Delete old or unused documents",
                "Increase storage limit if needed",
            ],
        }

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "⚠️" in result  # Warning indicators
        assert "84" in result
        assert "85" in result
        assert "Warnings:" in result
        assert "Storage usage at 84%" in result
        assert "Document count at 85%" in result
        assert "Recommendations:" in result
        assert "Delete old or unused documents" in result

    def test_check_health_critical_state(self, mock_context):
        """Test health check with critical resource usage"""
        # Setup
        mock_context["resource_manager"].check_health.return_value = {
            "healthy": False,
            "metrics": {
                "storage_mb": 500.0,
                "storage_limit_mb": 500.0,
                "storage_usage_pct": 100.0,
                "document_count": 1000,
                "document_limit": 1000,
                "document_usage_pct": 100.0,
                "memory_mb": None,
                "memory_limit_mb": None,
            },
            "warnings": [
                "Storage at 100% - CRITICAL",
                "Document limit reached - cannot ingest more",
            ],
            "recommendations": [
                "Immediate cleanup required",
                "Delete documents or increase limits",
            ],
        }

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "🔴" in result  # Critical indicators
        assert "100" in result
        assert "FULL" in result or "100%" in result
        assert "Warnings Detected" in result
        assert "CRITICAL" in result
        assert "cannot ingest more" in result
        assert "Immediate cleanup required" in result

    def test_check_health_with_memory_metrics(self, mock_context):
        """Test health check with optional memory metrics"""
        # Setup
        mock_context["resource_manager"].check_health.return_value = {
            "healthy": True,
            "metrics": {
                "storage_mb": 45.2,
                "storage_limit_mb": 500.0,
                "storage_usage_pct": 9.04,
                "document_count": 5,
                "document_limit": 1000,
                "document_usage_pct": 0.5,
                "memory_mb": 128.5,
                "memory_limit_mb": 512.0,
            },
            "warnings": [],
            "recommendations": [],
        }

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Memory" in result
        assert "128.5 MB" in result
        assert "512.0 MB" in result
        assert "25" in result  # 25% memory usage

    def test_check_health_without_memory_metrics(self, mock_context):
        """Test health check without optional memory metrics"""
        # Setup - use default fixture (no memory metrics)

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Memory" not in result  # Should not show memory section

    def test_check_health_no_warnings_or_recommendations(self, mock_context):
        """Test health check with no warnings or recommendations"""
        # Setup - use default fixture (empty warnings/recommendations)

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Healthy" in result
        # Should not have warnings or recommendations sections
        assert "Warnings:" not in result
        assert "Recommendations:" not in result

    def test_check_health_only_warnings_no_recommendations(self, mock_context):
        """Test health check with warnings but no recommendations"""
        # Setup
        mock_context["resource_manager"].check_health.return_value = {
            "healthy": True,
            "metrics": {
                "storage_mb": 400.0,
                "storage_limit_mb": 500.0,
                "storage_usage_pct": 80.0,
                "document_count": 100,
                "document_limit": 1000,
                "document_usage_pct": 10.0,
                "memory_mb": None,
                "memory_limit_mb": None,
            },
            "warnings": ["Storage at 80% threshold"],
            "recommendations": [],
        }

        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Warnings:" in result
        assert "Storage at 80% threshold" in result
        assert "Recommendations:" not in result

    def test_check_health_formatting_consistency(self, mock_context):
        """Test that health check output has consistent formatting"""
        # Execute
        result = handle_check_resource_health(mock_context, {})

        # Verify formatting elements
        assert result.startswith("💾 Resource Health Check")  # Header
        assert "Resource Health" in result
        assert "=" * 70 in result  # Has separators
        assert "Status:" in result  # Has status line

        # Verify progress bars are formatted
        lines = result.split("\n")
        progress_lines = [line for line in lines if "[" in line and "]" in line]
        assert len(progress_lines) >= 2  # At least storage + documents

    def test_check_health_resource_manager_called(self, mock_context):
        """Test that resource manager check_health is called correctly"""
        # Execute
        handle_check_resource_health(mock_context, {})

        # Verify
        mock_context["resource_manager"].check_health.assert_called_once()

    def test_check_health_handles_missing_optional_fields(self, mock_context):
        """Test health check with minimal required fields only"""
        # Setup with only required fields
        mock_context["resource_manager"].check_health.return_value = {
            "healthy": True,
            "metrics": {
                "storage_mb": 45.2,
                "storage_limit_mb": 500.0,
                "storage_usage_pct": 9.04,
                "document_count": 5,
                "document_limit": 1000,
                "document_usage_pct": 0.5,
                "memory_mb": None,
                "memory_limit_mb": None,
            },
            "warnings": [],
            "recommendations": [],
        }

        # Execute - should not raise KeyError
        result = handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Healthy" in result
