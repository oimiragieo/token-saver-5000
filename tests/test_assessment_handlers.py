"""
Tests for should_compress assessment handler (v0.9.2+).

Covers:
- Extension-based binary detection
- Content sniffing for unknown extensions
- Token estimation heuristics
- Path security validation (CWE-22)
- All recommendation paths: SKIP, DIRECT_READ, COMPRESS, CONVERT_THEN_COMPRESS
"""

import json
import os
import pytest
from unittest.mock import MagicMock


class TestShouldCompressExtensionDetection:
    """Tests for extension-based binary/text detection."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context with PathValidator."""
        ctx = {
            "path_validator": MagicMock(),
        }
        # PathValidator.validate() returns absolute path
        ctx["path_validator"].validate.side_effect = lambda p: os.path.abspath(p)
        return ctx

    @pytest.mark.asyncio
    async def test_binary_extension_detected(self, mock_context, tmp_path):
        """Test that known binary extensions are detected without content sniffing."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create a fake PDF file (doesn't need real content for extension check)
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b"Fake PDF content")

        result_str = await handle_should_compress(mock_context, {"file_path": str(pdf_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "CONVERT_THEN_COMPRESS"
        assert result["needs_conversion"] is True
        assert result["is_text_readable"] is False
        assert result["content_type_detected"] == "document"
        assert "MarkItDown" in result["conversion_tool"]

    @pytest.mark.asyncio
    async def test_text_extension_detected(self, mock_context, tmp_path):
        """Test that known text extensions skip content sniffing."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create a Python file with enough content to trigger COMPRESS
        py_file = tmp_path / "large_file.py"
        py_file.write_text("x = 1\n" * 500)  # ~3000 chars, ~850 tokens

        result_str = await handle_should_compress(mock_context, {"file_path": str(py_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "COMPRESS"
        assert result["needs_conversion"] is False
        assert result["is_text_readable"] is True
        assert result["content_type_detected"] == "code"


class TestShouldCompressContentSniffing:
    """Tests for content-based binary detection."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context with PathValidator."""
        ctx = {
            "path_validator": MagicMock(),
        }
        ctx["path_validator"].validate.side_effect = lambda p: os.path.abspath(p)
        return ctx

    @pytest.mark.asyncio
    async def test_binary_content_sniffing(self, mock_context, tmp_path):
        """Test that files with null bytes are detected as binary."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create file with unknown extension but binary content (>1% null bytes)
        binary_file = tmp_path / "unknown.data"
        content = b"text" + (b"\x00" * 100) + b"more text"  # ~50% null bytes
        binary_file.write_bytes(content)

        result_str = await handle_should_compress(mock_context, {"file_path": str(binary_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "CONVERT_THEN_COMPRESS"
        assert result["needs_conversion"] is True

    @pytest.mark.asyncio
    async def test_text_content_sniffing(self, mock_context, tmp_path):
        """Test that text files with unknown extensions are detected correctly."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create file with unknown extension but text content
        text_file = tmp_path / "readme.unknown"
        text_file.write_text("This is plain text content.\n" * 100)

        result_str = await handle_should_compress(mock_context, {"file_path": str(text_file)})
        result = json.loads(result_str)

        assert result["recommendation"] in ("DIRECT_READ", "COMPRESS")
        assert result["needs_conversion"] is False
        assert result["is_text_readable"] is True


class TestShouldCompressTokenEstimation:
    """Tests for token count estimation and thresholds."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context with PathValidator."""
        ctx = {
            "path_validator": MagicMock(),
        }
        ctx["path_validator"].validate.side_effect = lambda p: os.path.abspath(p)
        return ctx

    @pytest.mark.asyncio
    async def test_skip_threshold(self, mock_context, tmp_path):
        """Test SKIP recommendation for tiny files (<100 tokens)."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create tiny file (~50 tokens)
        tiny_file = tmp_path / "tiny.txt"
        tiny_file.write_text("x" * 200)  # 200 chars / 4 chars_per_token = 50 tokens

        result_str = await handle_should_compress(mock_context, {"file_path": str(tiny_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "SKIP"
        assert result["estimated_tokens"] < 100

    @pytest.mark.asyncio
    async def test_direct_read_threshold(self, mock_context, tmp_path):
        """Test DIRECT_READ recommendation for small files (100-500 tokens)."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create small file (~300 tokens)
        small_file = tmp_path / "small.txt"
        small_file.write_text("word " * 300)  # ~1500 chars / 4 = 375 tokens

        result_str = await handle_should_compress(mock_context, {"file_path": str(small_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "DIRECT_READ"
        assert 100 <= result["estimated_tokens"] < 500

    @pytest.mark.asyncio
    async def test_compress_threshold(self, mock_context, tmp_path):
        """Test COMPRESS recommendation for medium files (>500 tokens)."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create medium file (~1000 tokens)
        medium_file = tmp_path / "medium.txt"
        medium_file.write_text("longer sentence with more words " * 200)  # ~6400 chars

        result_str = await handle_should_compress(mock_context, {"file_path": str(medium_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "COMPRESS"
        assert result["estimated_tokens"] >= 500
        assert result["potential_token_savings"] > 0


class TestShouldCompressPathSecurity:
    """Tests for path validation security (CWE-22 prevention)."""

    @pytest.fixture
    def mock_context_with_validator(self):
        """Create mock context with real-style PathValidator behavior."""
        ctx = {
            "path_validator": MagicMock(),
        }
        return ctx

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, mock_context_with_validator):
        """Test that path traversal attacks are blocked."""
        from src.handlers.resource_handlers import handle_should_compress

        # Configure PathValidator to reject traversal paths
        mock_context_with_validator["path_validator"].validate.side_effect = ValueError(
            "Path traversal detected"
        )

        result_str = await handle_should_compress(
            mock_context_with_validator, {"file_path": "../../etc/passwd"}
        )
        result = json.loads(result_str)

        assert result["recommendation"] == "UNKNOWN"
        assert "error" in result
        assert "Path validation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_path_validator(self):
        """Test error when PathValidator is not configured."""
        from src.handlers.resource_handlers import handle_should_compress

        # Context without path_validator
        ctx = {}

        result_str = await handle_should_compress(ctx, {"file_path": "/some/file.txt"})
        result = json.loads(result_str)

        assert result["recommendation"] == "UNKNOWN"
        assert "error" in result
        assert "Path validation unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_file_path_argument(self, mock_context_with_validator):
        """Test error when file_path is not provided."""
        from src.handlers.resource_handlers import handle_should_compress

        result_str = await handle_should_compress(mock_context_with_validator, {})
        result = json.loads(result_str)

        assert result["recommendation"] == "UNKNOWN"
        assert "error" in result
        assert "file_path is required" in result["error"]


class TestShouldCompressEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def mock_context(self):
        """Create mock context with PathValidator."""
        ctx = {
            "path_validator": MagicMock(),
        }
        ctx["path_validator"].validate.side_effect = lambda p: os.path.abspath(p)
        return ctx

    @pytest.mark.asyncio
    async def test_empty_file(self, mock_context, tmp_path):
        """Test handling of empty files."""
        from src.handlers.resource_handlers import handle_should_compress

        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        result_str = await handle_should_compress(mock_context, {"file_path": str(empty_file)})
        result = json.loads(result_str)

        assert result["recommendation"] == "SKIP"
        assert result["estimated_tokens"] == 0

    @pytest.mark.asyncio
    async def test_file_not_found(self, mock_context, tmp_path):
        """Test handling of non-existent files."""
        from src.handlers.resource_handlers import handle_should_compress

        result_str = await handle_should_compress(
            mock_context, {"file_path": str(tmp_path / "nonexistent.txt")}
        )
        result = json.loads(result_str)

        assert result["recommendation"] == "UNKNOWN"
        assert "error" in result
        assert "File not found" in result["error"]

    @pytest.mark.asyncio
    async def test_code_content_type_override(self, mock_context, tmp_path):
        """Test explicit content_type='code' uses code token ratio."""
        from src.handlers.resource_handlers import handle_should_compress

        # Create file with .txt extension but code content
        code_file = tmp_path / "script.txt"
        code_file.write_text("def foo():\n    pass\n" * 100)

        result_str = await handle_should_compress(
            mock_context, {"file_path": str(code_file), "content_type": "code"}
        )
        result = json.loads(result_str)

        assert result["content_type_detected"] == "code"
        # Code uses 3.5 chars/token vs prose 4.0, so estimate should be higher
        assert result["estimated_tokens"] > 0
