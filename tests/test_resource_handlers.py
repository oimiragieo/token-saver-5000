"""
Comprehensive tests for resource_handlers.py

Coverage target: 80%+ (currently 16%)
Tests both helper functions and main handler with various health states.
v0.9.1: Added tests for should_compress token estimation tool.
v0.9.2: Added binary detection tests and updated recommendation names.
"""

import json
import os
import pytest
import tempfile
from unittest.mock import Mock, AsyncMock
from types import SimpleNamespace
from src.handlers.resource_handlers import (
    handle_check_resource_health,
    handle_check_environment,
    get_check_environment_output_fields,
    handle_should_compress,
    create_progress_bar,
    is_binary_content,
    CHARS_PER_TOKEN_PROSE,
    CHARS_PER_TOKEN_CODE,
    SKIP_THRESHOLD_TOKENS,
    DIRECT_READ_THRESHOLD_TOKENS,
    # COMPRESS_THRESHOLD_TOKENS removed in v0.9.3 (use DIRECT_READ_THRESHOLD_TOKENS)
    STRONG_COMPRESS_THRESHOLD,
    MUST_COMPRESS_THRESHOLD,  # v0.9.3: new module-level constant
)
from src.path_validator import PathValidator

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


@pytest.fixture
def mock_context_with_validator(tmp_path):
    """
    Create mock HandlerContext with PathValidator for should_compress tests (v0.9.3).

    v0.9.3 requires PathValidator for handle_should_compress. This fixture provides
    a context with a PathValidator configured to allow the tmp_path directory.
    """
    context = {}
    # Create PathValidator allowing the pytest tmp_path directory
    context["path_validator"] = PathValidator(allowed_base_dirs=[str(tmp_path)])
    return context, tmp_path


def get_context_for_tempfile():
    """
    Create a context with PathValidator allowing system temp directory.

    Used for tests that create temp files via tempfile.NamedTemporaryFile.
    v0.9.3 requires PathValidator, so we create one allowing the temp dir.
    """
    import tempfile as tf

    temp_dir = tf.gettempdir()
    return {"path_validator": PathValidator(allowed_base_dirs=[temp_dir])}


# ============================================================================
# Test create_progress_bar Helper Function
# ============================================================================


class TestCreateProgressBar:
    """Test progress bar creation with various percentages and states (v0.9.3: ASCII chars)"""

    def test_progress_bar_healthy_low(self):
        """Test progress bar with low percentage (healthy state)"""
        result = create_progress_bar(25.0, width=40)

        assert "[" in result
        assert "]" in result
        assert "[OK]" in result
        assert "25%" in result
        assert "#" in result  # v0.9.3: ASCII filled blocks
        assert "-" in result  # v0.9.3: ASCII empty blocks

    def test_progress_bar_healthy_medium(self):
        """Test progress bar with medium percentage (still healthy)"""
        result = create_progress_bar(50.0, width=40)

        assert "[OK]" in result
        assert "50%" in result
        # 50% should have half filled (v0.9.3: "#" instead of "█")
        filled_count = result.count("#")
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
        assert "#" in result  # v0.9.3: ASCII chars
        assert "-" in result

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
        # Should be all filled blocks (v0.9.3: "#" instead of "█")
        filled_count = result.count("#")
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
        # Should be all empty blocks (v0.9.3: "-" instead of "░")
        empty_count = result.count("-")
        assert empty_count == 40

    def test_progress_bar_custom_width_small(self):
        """Test progress bar with small custom width"""
        result = create_progress_bar(50.0, width=10)

        assert "[OK]" in result
        assert "50%" in result
        # Total bar width should be 10 (v0.9.3: ASCII chars)
        filled = result.count("#")
        empty = result.count("-")
        assert filled + empty == 10

    def test_progress_bar_custom_width_large(self):
        """Test progress bar with large custom width"""
        result = create_progress_bar(75.0, width=60)

        assert "[OK]" in result
        assert "75%" in result
        # v0.9.3: ASCII chars
        filled = result.count("#")
        empty = result.count("-")
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
        result = await handle_check_resource_health(mock_context, {})
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
        result = await handle_check_resource_health(mock_context, {})
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
        result = await handle_check_resource_health(mock_context, {})
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
        result = await handle_check_resource_health(mock_context, {})
        assert "Resource Health" in result
        assert "Memory" in result
        assert "128.5 MB" in result
        assert "512.0 MB" in result
        assert "25" in result  # 25% memory usage

    async def test_check_health_without_memory_metrics(self, mock_context):
        """Test health check without optional memory metrics"""
        result = await handle_check_resource_health(mock_context, {})
        assert "Resource Health" in result
        assert "Memory" not in result  # Should not show memory section

    async def test_check_health_no_warnings_or_recommendations(self, mock_context):
        """Test health check with no warnings or recommendations"""
        result = await handle_check_resource_health(mock_context, {})
        assert "Resource Health" in result
        assert "Healthy" in result
        # Should not have warnings or recommendations sections
        assert "Warnings:" not in result
        assert "Recommendations:" not in result

    async def test_check_health_only_warnings_no_recommendations(self, mock_context):
        """Test health check with warnings but no recommendations"""
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
        result = await handle_check_resource_health(mock_context, {})
        assert "Warnings:" in result
        assert "Storage at 80% threshold" in result
        assert "Recommendations:" not in result

    async def test_check_health_formatting_consistency(self, mock_context):
        """Test that health check output has consistent formatting"""
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
        await handle_check_resource_health(mock_context, {})

        # Verify (v0.8.0: check_health_async instead of check_health)
        mock_context["resource_manager"].check_health_async.assert_called_once()

    async def test_check_health_handles_missing_optional_fields(self, mock_context):
        """Test health check with minimal required fields only"""
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
        result = await handle_check_resource_health(mock_context, {})
        assert "Resource Health" in result
        assert "Healthy" in result


@pytest.mark.asyncio
class TestHandleCheckEnvironment:
    """Test check_environment handler profile diagnostics."""

    async def test_check_environment_includes_tool_profile_metadata(self):
        compressor = SimpleNamespace(graphs={}, chunks={})
        sync_manager = Mock()
        sync_manager.export_metadata.return_value = {}

        context = {
            "compressor": compressor,
            "sync_manager": sync_manager,
            "tool_profile": "core_stable",
            "enabled_tool_names": [
                "ingest_context",
                "read_skeleton",
                "search_semantic",
                "modulate_region",
                "get_stats",
                "list_documents",
                "delete_document",
            ],
        }

        result = await handle_check_environment(context, {})
        payload = json.loads(result)

        assert "tool_profile" in payload
        assert payload["tool_profile"]["profile"] == "core_stable"
        assert payload["tool_profile"]["enabled_tool_count"] == 7
        assert payload["tool_profile"]["enabled_tools"] == context["enabled_tool_names"]

    async def test_check_environment_defaults_tool_profile_metadata(self):
        compressor = SimpleNamespace(graphs={}, chunks={})
        sync_manager = Mock()
        sync_manager.export_metadata.return_value = {}

        context = {
            "compressor": compressor,
            "sync_manager": sync_manager,
        }

        result = await handle_check_environment(context, {})
        payload = json.loads(result)

        assert payload["tool_profile"]["profile"] == "full"
        assert payload["tool_profile"]["enabled_tool_count"] == 0
        assert payload["tool_profile"]["enabled_tools"] == []

    async def test_check_environment_output_fields_contains_profile_keys(self):
        output_fields = get_check_environment_output_fields()
        assert "tool_profile.profile" in output_fields
        assert "tool_profile.enabled_tool_count" in output_fields
        assert "tool_profile.enabled_tools" in output_fields


# ============================================================================
# Test handle_should_compress Handler (v0.9.1)
# ============================================================================


@pytest.mark.asyncio
class TestHandleShouldCompress:
    """Test should_compress token estimation handler (v0.9.3: PathValidator required)"""

    # v0.9.3: PathValidator requirement tests
    async def test_should_compress_no_path_validator_returns_error(self):
        """Test that missing PathValidator returns error (v0.9.3 security requirement)"""
        result = await handle_should_compress({}, {"file_path": "/some/file.txt"})
        data = json.loads(result)

        assert "error" in data
        assert "Path validation unavailable" in data["error"]
        assert data["recommendation"] == "UNKNOWN"
        assert "suggestion" in data

    async def test_should_compress_missing_file_path(self):
        """Test with missing file_path argument"""
        result = await handle_should_compress(get_context_for_tempfile(), {})
        data = json.loads(result)

        assert data["error"] == "file_path is required"
        assert data["recommendation"] == "UNKNOWN"

    async def test_should_compress_file_not_found(self):
        """Test with non-existent file"""
        result = await handle_should_compress(
            get_context_for_tempfile(), {"file_path": "/nonexistent/file.txt"}
        )
        data = json.loads(result)

        assert "error" in data
        # v0.9.3: Will now fail on path validation (outside allowed dirs)
        assert data["recommendation"] == "UNKNOWN"

    async def test_should_compress_small_file_direct_read(self):
        """Test small file returns DIRECT_READ recommendation (v0.9.2)"""
        # Create a small temp file (~500 bytes = ~130 tokens, between 100-500 threshold)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 500)  # 500 bytes
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "DIRECT_READ"
            assert data["estimated_tokens"] < DIRECT_READ_THRESHOLD_TOKENS
            assert "small" in data["reason"].lower()
            assert data["file_size_bytes"] == 500
            # v0.9.2: New fields
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    async def test_should_compress_medium_file_compress(self):
        """Test medium file returns COMPRESS recommendation (v0.9.2)"""
        # Create medium file (~3000 bytes = ~790 tokens at 3.8 chars/token)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 3000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "COMPRESS"
            # v0.9.3: Use DIRECT_READ_THRESHOLD_TOKENS (COMPRESS_THRESHOLD_TOKENS removed)
            assert (
                DIRECT_READ_THRESHOLD_TOKENS <= data["estimated_tokens"] < STRONG_COMPRESS_THRESHOLD
            )
            assert "medium" in data["reason"].lower()
            # v0.9.2: New fields
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    async def test_should_compress_large_file_compress(self):
        """Test large file returns COMPRESS (v0.9.2: unified compress recommendation)"""
        # Create large file (~10000 bytes = ~2632 tokens)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 10000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "COMPRESS"
            assert STRONG_COMPRESS_THRESHOLD <= data["estimated_tokens"] < MUST_COMPRESS_THRESHOLD
            assert "large" in data["reason"].lower()
            # v0.9.2: New fields
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    async def test_should_compress_huge_file_compress(self):
        """Test very large file returns COMPRESS (v0.9.2: unified compress recommendation)"""
        # Create huge file (~50000 bytes = ~13158 tokens)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 50000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "COMPRESS"
            assert data["estimated_tokens"] >= MUST_COMPRESS_THRESHOLD
            assert "very large" in data["reason"].lower()
            # v0.9.2: New fields
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    async def test_should_compress_code_file_auto_detect(self):
        """Test auto-detection of code file types"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("x" * 1000)  # 1000 bytes
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
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
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
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
                get_context_for_tempfile(), {"file_path": temp_path, "content_type": "code"}
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
                get_context_for_tempfile(), {"file_path": temp_path, "content_type": "prose"}
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
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
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
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
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
                result = await handle_should_compress(
                    get_context_for_tempfile(), {"file_path": temp_path}
                )
                data = json.loads(result)
                assert data["content_type_detected"] == "code", f"Failed for extension {ext}"
            finally:
                os.unlink(temp_path)


# ============================================================================
# Test Binary Detection (v0.9.2)
# ============================================================================


class TestIsBinaryContent:
    """Test is_binary_content helper function (v0.9.2 Hardening: returns tuple)"""

    def test_binary_content_with_null_bytes(self):
        """Test detection of binary content via null byte ratio"""
        # Create file with >1% null bytes
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".dat", delete=False) as f:
            # Write 100 bytes: 5 null bytes + 95 regular bytes = 5% null
            f.write(b"\x00" * 5 + b"x" * 95)
            temp_path = f.name

        try:
            is_binary, error = is_binary_content(temp_path)
            assert is_binary is True
            assert error is None
        finally:
            os.unlink(temp_path)

    def test_text_content_no_null_bytes(self):
        """Test text content detection (no null bytes)"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is plain text content with no null bytes.")
            temp_path = f.name

        try:
            is_binary, error = is_binary_content(temp_path)
            assert is_binary is False
            assert error is None
        finally:
            os.unlink(temp_path)

    def test_empty_file_not_binary(self):
        """Test empty file is not considered binary"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name

        try:
            is_binary, error = is_binary_content(temp_path)
            assert is_binary is False
            assert error is None
        finally:
            os.unlink(temp_path)

    def test_nonexistent_file_returns_error(self):
        """Test nonexistent file returns error message (v0.9.2 Hardening)"""
        is_binary, error = is_binary_content("/nonexistent/path/file.xyz")
        assert is_binary is False
        assert error is not None
        assert "Cannot read file" in error


@pytest.mark.asyncio
class TestBinaryFileDetection:
    """Test binary file detection in should_compress (v0.9.2)"""

    # Extension-based detection tests

    async def test_binary_pdf_extension(self):
        """Test PDF file returns CONVERT_THEN_COMPRESS"""
        # Create a dummy PDF-like file (binary content)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\x00" + b"binary content")
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["needs_conversion"] is True
            assert data["is_text_readable"] is False
            assert data["conversion_tool"] == "MarkItDown"
            assert data["content_type_detected"] == "document"
            assert "detected_by" in data
        finally:
            os.unlink(temp_path)

    async def test_binary_docx_extension(self):
        """Test DOCX file returns CONVERT_THEN_COMPRESS"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".docx", delete=False) as f:
            f.write(b"PK\x03\x04" + b"\x00" * 50)  # ZIP signature (DOCX is a ZIP)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["needs_conversion"] is True
            assert data["is_text_readable"] is False
            assert data["content_type_detected"] == "document"
        finally:
            os.unlink(temp_path)

    async def test_binary_image_extension(self):
        """Test image file (PNG) returns CONVERT_THEN_COMPRESS"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".png", delete=False) as f:
            # PNG header
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["needs_conversion"] is True
            assert data["content_type_detected"] == "image"
        finally:
            os.unlink(temp_path)

    # Content-based detection tests (CRITICAL)

    async def test_unknown_extension_binary_content(self):
        """Test unknown extension with binary content detected as binary"""
        # .xyz is not in TEXT_EXTENSIONS, so content will be sniffed
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".xyz", delete=False) as f:
            # Write content with >1% null bytes
            f.write(b"\x00" * 100 + b"some data" * 100)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["needs_conversion"] is True
            assert data["is_text_readable"] is False
            assert data["detected_by"] == "content"  # Detected by content, not extension
        finally:
            os.unlink(temp_path)

    async def test_unknown_extension_text_content(self):
        """Test unknown extension with text content detected as text"""
        # .xyz is not in TEXT_EXTENSIONS, but content has no null bytes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
            f.write("This is plain text in an unknown extension file.\n" * 50)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            # Should NOT be detected as binary since content is text
            assert data["recommendation"] != "CONVERT_THEN_COMPRESS"
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    async def test_text_extension_skips_content_sniff(self):
        """Test known text extension skips content sniffing for efficiency"""
        # .txt is in TEXT_EXTENSIONS, so content sniffing is skipped
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Normal text content")
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
        finally:
            os.unlink(temp_path)

    # Threshold boundary tests

    async def test_threshold_boundary_skip(self):
        """Test file at exactly SKIP threshold boundary"""
        # Create file with exactly ~99 tokens (should be SKIP)
        # 99 tokens * 3.8 chars/token = 376 bytes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 376)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "SKIP"
            assert data["estimated_tokens"] < SKIP_THRESHOLD_TOKENS
        finally:
            os.unlink(temp_path)

    async def test_threshold_boundary_direct_read(self):
        """Test file just above SKIP threshold returns DIRECT_READ"""
        # Create file with ~101 tokens (should be DIRECT_READ)
        # 101 tokens * 3.8 chars/token = 384 bytes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 400)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "DIRECT_READ"
            assert SKIP_THRESHOLD_TOKENS <= data["estimated_tokens"] < DIRECT_READ_THRESHOLD_TOKENS
        finally:
            os.unlink(temp_path)

    async def test_threshold_boundary_compress(self):
        """Test file just above DIRECT_READ threshold returns COMPRESS"""
        # Create file with ~501 tokens (should be COMPRESS)
        # 501 tokens * 3.8 chars/token = 1904 bytes
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1910)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "COMPRESS"
            # v0.9.3: Use DIRECT_READ_THRESHOLD_TOKENS (COMPRESS_THRESHOLD_TOKENS removed)
            assert data["estimated_tokens"] >= DIRECT_READ_THRESHOLD_TOKENS
        finally:
            os.unlink(temp_path)

    async def test_empty_file_skip(self):
        """Test empty file returns SKIP"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            temp_path = f.name  # Don't write anything

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "SKIP"
            assert data["file_size_bytes"] == 0
            assert "empty" in data["reason"].lower()
        finally:
            os.unlink(temp_path)

    # Backward compatibility tests

    async def test_existing_fields_unchanged(self):
        """Test all v0.9.1 response fields still present (backward compat)"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            # v0.9.1 fields must be present
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

    async def test_new_fields_added(self):
        """Test v0.9.2 fields are present"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("x" * 1000)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            # v0.9.2 new fields must be present
            assert "needs_conversion" in data
            assert "is_text_readable" in data
            assert "conversion_tool" in data

            # For text files, values should be:
            assert data["needs_conversion"] is False
            assert data["is_text_readable"] is True
            assert data["conversion_tool"] is None
        finally:
            os.unlink(temp_path)

    async def test_binary_new_fields(self):
        """Test v0.9.2 fields for binary files"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4\x00binary")
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            # For binary files, values should be:
            assert data["needs_conversion"] is True
            assert data["is_text_readable"] is False
            assert data["conversion_tool"] == "MarkItDown"
        finally:
            os.unlink(temp_path)

    # Media and archive file tests

    async def test_media_file_detection(self):
        """Test media file (mp4) returns CONVERT_THEN_COMPRESS"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50)
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["content_type_detected"] == "media"
        finally:
            os.unlink(temp_path)

    async def test_archive_file_detection(self):
        """Test archive file (zip) returns CONVERT_THEN_COMPRESS"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".zip", delete=False) as f:
            f.write(b"PK\x03\x04" + b"\x00" * 50)  # ZIP signature
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["content_type_detected"] == "archive"
        finally:
            os.unlink(temp_path)

    async def test_executable_file_detection(self):
        """Test executable file (exe) returns CONVERT_THEN_COMPRESS"""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".exe", delete=False) as f:
            f.write(b"MZ" + b"\x00" * 50)  # DOS header
            temp_path = f.name

        try:
            result = await handle_should_compress(
                get_context_for_tempfile(), {"file_path": temp_path}
            )
            data = json.loads(result)

            assert data["recommendation"] == "CONVERT_THEN_COMPRESS"
            assert data["content_type_detected"] == "executable"
        finally:
            os.unlink(temp_path)

    # Security tests (v0.9.2 Hardening)

    async def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked (CWE-22 prevention)"""
        # Import PathValidator for the test
        from src.path_validator import PathValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create PathValidator with only tmpdir allowed
            validator = PathValidator(allowed_base_dirs=[tmpdir])
            context = {"path_validator": validator}

            # Attempt path traversal attack
            result = await handle_should_compress(context, {"file_path": "../../etc/passwd"})
            data = json.loads(result)

            # Should return error with UNKNOWN recommendation
            assert "error" in data
            assert "path" in data["error"].lower() or "validation" in data["error"].lower()
            assert data["recommendation"] == "UNKNOWN"
            assert "suggestion" in data

    async def test_path_traversal_with_absolute_path(self):
        """Test that absolute path outside allowed dirs is blocked"""
        from src.path_validator import PathValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            validator = PathValidator(allowed_base_dirs=[tmpdir])
            context = {"path_validator": validator}

            # Attempt to access /etc/passwd (absolute path)
            result = await handle_should_compress(context, {"file_path": "/etc/passwd"})
            data = json.loads(result)

            assert "error" in data
            assert data["recommendation"] == "UNKNOWN"

    async def test_valid_path_with_validator(self):
        """Test that valid paths within allowed dirs work with PathValidator"""
        from src.path_validator import PathValidator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file within the allowed directory
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, "w") as f:
                f.write("x" * 1000)

            validator = PathValidator(allowed_base_dirs=[tmpdir])
            context = {"path_validator": validator}

            # Valid path should work
            result = await handle_should_compress(context, {"file_path": test_file})
            data = json.loads(result)

            # Should succeed, not return error
            assert "error" not in data
            assert data["recommendation"] in ["SKIP", "DIRECT_READ", "COMPRESS"]
            assert data["needs_conversion"] is False
