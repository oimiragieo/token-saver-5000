"""
Comprehensive tests for resource_handlers.py

Coverage target: 80%+ (currently 16%)
Tests both helper functions and main handler with various health states.
v0.9.1: Added tests for should_compress token estimation tool.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import Mock, AsyncMock
from src.handlers.resource_handlers import (
    handle_check_resource_health,
    handle_should_compress,
    create_progress_bar,
    CHARS_PER_TOKEN_PROSE,
    CHARS_PER_TOKEN_CODE,
    CHARS_PER_TOKEN_DEFAULT,
    COMPRESS_THRESHOLD_TOKENS,
    STRONG_COMPRESS_THRESHOLD,
    MUST_COMPRESS_THRESHOLD,
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
    # v0.8.0: Handler now calls check_health_async() instead of check_health()
    mock_resource_manager = Mock()
    mock_resource_manager.check_health_async = AsyncMock(
        return_value={
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
    )

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
        assert "[OK]" in result
        assert "25%" in result
        assert "█" in result  # Should have some filled blocks
        assert "░" in result  # Should have some empty blocks

    def test_progress_bar_healthy_medium(self):
        """Test progress bar with medium percentage (still healthy)"""
        result = create_progress_bar(50.0, width=40)

        assert "[OK]" in result
        assert "50%" in result
        # 50% should have half filled
        filled_count = result.count("█")
        assert filled_count == 20  # Half of 40

    def test_progress_bar_healthy_high(self):
        """Test progress bar just below warning threshold"""
        result = create_progress_bar(79.0, width=40)

        assert "[OK]" in result
        assert "79%" in result

    def test_progress_bar_warning_threshold(self):
        """Test progress bar at warning threshold (80%)"""
        result = create_progress_bar(80.0, width=40)

        assert "[WARN]" in result
        assert "80%" in result
        assert "█" in result
        assert "░" in result

    def test_progress_bar_warning_high(self):
        """Test progress bar at high warning percentage"""
        result = create_progress_bar(95.0, width=40)

        assert "[WARN]" in result
        assert "95%" in result

    def test_progress_bar_full(self):
        """Test progress bar at 100% (critical state)"""
        result = create_progress_bar(100.0, width=40)

        assert "[CRIT]" in result
        assert "FULL" in result
        # Should be all filled blocks
        filled_count = result.count("█")
        assert filled_count == 40

    def test_progress_bar_over_100(self):
        """Test progress bar with over 100% (edge case)"""
        result = create_progress_bar(150.0, width=40)

        assert "[CRIT]" in result
        assert "FULL" in result

    def test_progress_bar_zero(self):
        """Test progress bar at 0%"""
        result = create_progress_bar(0.0, width=40)

        assert "[OK]" in result
        assert "0%" in result
        # Should be all empty blocks
        empty_count = result.count("░")
        assert empty_count == 40

    def test_progress_bar_custom_width_small(self):
        """Test progress bar with small custom width"""
        result = create_progress_bar(50.0, width=10)

        assert "[OK]" in result
        assert "50%" in result
        # Total bar width should be 10
        filled = result.count("█")
        empty = result.count("░")
        assert filled + empty == 10

    def test_progress_bar_custom_width_large(self):
        """Test progress bar with large custom width"""
        result = create_progress_bar(75.0, width=60)

        assert "[OK]" in result
        assert "75%" in result
        filled = result.count("█")
        empty = result.count("░")
        assert filled + empty == 60
        # 75% of 60 = 45 filled
        assert filled == 45


# ============================================================================
# Test handle_check_resource_health Handler
# ============================================================================


@pytest.mark.asyncio
class TestHandleCheckResourceHealth:
    """Test resource health check handler with various states"""

    async def test_check_health_healthy_state(self, mock_context):
        """Test health check with healthy system state"""
        # Setup - use default healthy state from fixture

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

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
        assert "[OK]" in result
        assert "Healthy" in result

    async def test_check_health_with_warnings(self, mock_context):
        """Test health check with system warnings"""
        # Setup (v0.8.0: use check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.return_value = {
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

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "[WARN]" in result  # Warning indicators
        assert "84" in result
        assert "85" in result
        assert "Warnings:" in result
        assert "Storage usage at 84%" in result
        assert "Document count at 85%" in result
        assert "Recommendations:" in result
        assert "Delete old or unused documents" in result

    async def test_check_health_critical_state(self, mock_context):
        """Test health check with critical resource usage"""
        # Setup (v0.8.0: use check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.return_value = {
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

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "[CRIT]" in result  # Critical indicators
        assert "100" in result
        assert "FULL" in result or "100%" in result
        assert "Warnings Detected" in result
        assert "CRITICAL" in result
        assert "cannot ingest more" in result
        assert "Immediate cleanup required" in result

    async def test_check_health_with_memory_metrics(self, mock_context):
        """Test health check with optional memory metrics"""
        # Setup (v0.8.0: use check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.return_value = {
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

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Memory" in result
        assert "128.5 MB" in result
        assert "512.0 MB" in result
        assert "25" in result  # 25% memory usage

    async def test_check_health_without_memory_metrics(self, mock_context):
        """Test health check without optional memory metrics"""
        # Setup - use default fixture (no memory metrics)

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Memory" not in result  # Should not show memory section

    async def test_check_health_no_warnings_or_recommendations(self, mock_context):
        """Test health check with no warnings or recommendations"""
        # Setup - use default fixture (empty warnings/recommendations)

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Healthy" in result
        # Should not have warnings or recommendations sections
        assert "Warnings:" not in result
        assert "Recommendations:" not in result

    async def test_check_health_only_warnings_no_recommendations(self, mock_context):
        """Test health check with warnings but no recommendations"""
        # Setup (v0.8.0: use check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.return_value = {
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

        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Warnings:" in result
        assert "Storage at 80% threshold" in result
        assert "Recommendations:" not in result

    async def test_check_health_formatting_consistency(self, mock_context):
        """Test that health check output has consistent formatting"""
        # Execute (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify formatting elements
        assert result.startswith("[STORAGE] Resource Health Check")  # Header
        assert "Resource Health" in result
        assert "=" * 70 in result  # Has separators
        assert "Status:" in result  # Has status line

        # Verify progress bars are formatted
        lines = result.split("\n")
        progress_lines = [line for line in lines if "[" in line and "]" in line]
        assert len(progress_lines) >= 2  # At least storage + documents

    async def test_check_health_resource_manager_called(self, mock_context):
        """Test that resource manager check_health_async is called correctly"""
        # Execute (v0.8.0: handler is now async)
        await handle_check_resource_health(mock_context, {})

        # Verify (v0.8.0: check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.assert_called_once()

    async def test_check_health_handles_missing_optional_fields(self, mock_context):
        """Test health check with minimal required fields only"""
        # Setup with only required fields (v0.8.0: use check_health_async)
        mock_context["resource_manager"].check_health_async.return_value = {
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

        # Execute - should not raise KeyError (v0.8.0: handler is now async)
        result = await handle_check_resource_health(mock_context, {})

        # Verify
        assert "Resource Health" in result
        assert "Healthy" in result


# ============================================================================
# Test handle_should_compress Handler (v0.9.1)
# ============================================================================


@pytest.mark.asyncio
class TestHandleShouldCompress:
    """Test should_compress token estimation handler"""

    async def test_should_compress_missing_file_path(self):
        """Test with missing file_path argument"""
        result = await handle_should_compress({}, {})
        data = json.loads(result)

        assert data["error"] == "file_path is required"
        assert data["recommendation"] == "UNKNOWN"

    async def test_should_compress_file_not_found(self):
        """Test with non-existent file"""
        result = await handle_should_compress({}, {"file_path": "/nonexistent/file.txt"})
        data = json.loads(result)

        assert "error" in data
        assert "not found" in data["error"].lower()
        assert data["recommendation"] == "UNKNOWN"

    async def test_should_compress_small_file_no_compress(self):
        """Test small file returns NO_COMPRESS recommendation"""
        # Create a small temp file (~500 bytes = ~130 tokens)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 500)  # 500 bytes
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            assert data["recommendation"] == "NO_COMPRESS"
            assert data["estimated_tokens"] < COMPRESS_THRESHOLD_TOKENS
            assert "small" in data["reason"].lower()
            assert data["file_size_bytes"] == 500
        finally:
            os.unlink(temp_path)

    async def test_should_compress_medium_file_recommend(self):
        """Test medium file returns RECOMMEND_COMPRESS"""
        # Create medium file (~3000 bytes = ~790 tokens at 3.8 chars/token)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 3000)
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            assert data["recommendation"] == "RECOMMEND_COMPRESS"
            assert COMPRESS_THRESHOLD_TOKENS <= data["estimated_tokens"] < STRONG_COMPRESS_THRESHOLD
            assert "medium" in data["reason"].lower()
        finally:
            os.unlink(temp_path)

    async def test_should_compress_large_file_strongly_recommend(self):
        """Test large file returns STRONGLY_RECOMMEND"""
        # Create large file (~10000 bytes = ~2632 tokens)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 10000)
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            assert data["recommendation"] == "STRONGLY_RECOMMEND"
            assert STRONG_COMPRESS_THRESHOLD <= data["estimated_tokens"] < MUST_COMPRESS_THRESHOLD
            assert "large" in data["reason"].lower()
        finally:
            os.unlink(temp_path)

    async def test_should_compress_huge_file_must_compress(self):
        """Test very large file returns MUST_COMPRESS"""
        # Create huge file (~50000 bytes = ~13158 tokens)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 50000)
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            assert data["recommendation"] == "MUST_COMPRESS"
            assert data["estimated_tokens"] >= MUST_COMPRESS_THRESHOLD
            assert "very large" in data["reason"].lower()
        finally:
            os.unlink(temp_path)

    async def test_should_compress_code_file_auto_detect(self):
        """Test auto-detection of code file types"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x" * 1000)  # 1000 bytes
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            # Should detect as code and use CHARS_PER_TOKEN_CODE (3.5)
            assert data["content_type_detected"] == "code"
            # 1000 bytes / 3.5 = ~286 tokens
            expected_tokens = int(1000 / CHARS_PER_TOKEN_CODE)
            assert abs(data["estimated_tokens"] - expected_tokens) < 5
        finally:
            os.unlink(temp_path)

    async def test_should_compress_prose_file_auto_detect(self):
        """Test auto-detection of prose file types"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("x" * 1000)  # 1000 bytes
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            # Should detect as prose/mixed
            assert data["content_type_detected"] == "prose/mixed"
        finally:
            os.unlink(temp_path)

    async def test_should_compress_explicit_code_type(self):
        """Test explicit code content type"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                {}, {"file_path": temp_path, "content_type": "code"}
            )
            data = json.loads(result)

            # Should use code ratio regardless of extension
            expected_tokens = int(1000 / CHARS_PER_TOKEN_CODE)
            assert abs(data["estimated_tokens"] - expected_tokens) < 5
        finally:
            os.unlink(temp_path)

    async def test_should_compress_explicit_prose_type(self):
        """Test explicit prose content type"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x" * 1000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                {}, {"file_path": temp_path, "content_type": "prose"}
            )
            data = json.loads(result)

            # Should use prose ratio despite .py extension
            expected_tokens = int(1000 / CHARS_PER_TOKEN_PROSE)
            assert abs(data["estimated_tokens"] - expected_tokens) < 5
        finally:
            os.unlink(temp_path)

    async def test_should_compress_potential_savings(self):
        """Test potential token savings calculation"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 20000)  # ~5263 tokens
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            # Should have calculated potential savings
            assert "potential_token_savings" in data
            assert data["potential_token_savings"] > 0
            assert data["estimated_compression_ratio"] in ["5.0x", "10.0x", "15.0x", "2.0x"]
        finally:
            os.unlink(temp_path)

    async def test_should_compress_json_format(self):
        """Test output is valid JSON with all expected fields"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 5000)
            temp_path = f.name

        try:
            result = await handle_should_compress({}, {"file_path": temp_path})
            data = json.loads(result)

            # Verify all expected fields
            assert "file_path" in data
            assert "file_size_bytes" in data
            assert "estimated_tokens" in data
            assert "content_type_detected" in data
            assert "recommendation" in data
            assert "reason" in data
            assert "potential_token_savings" in data
            assert "estimated_compression_ratio" in data
        finally:
            os.unlink(temp_path)

    async def test_should_compress_various_code_extensions(self):
        """Test auto-detection for various code extensions"""
        code_extensions = [".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".go", ".rs"]

        for ext in code_extensions:
            with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
                f.write("x" * 1000)
                temp_path = f.name

            try:
                result = await handle_should_compress({}, {"file_path": temp_path})
                data = json.loads(result)
                assert data["content_type_detected"] == "code", f"Failed for extension {ext}"
            finally:
                os.unlink(temp_path)
