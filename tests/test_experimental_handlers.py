"""
Tests for experimental handlers.

These tests verify:
1. All experimental tools return "experimental": true flag
2. Graceful degradation when dependencies are missing
3. Proper error handling with informative messages
4. PathValidator integration for file-accepting tools

Tests are designed to skip when optional dependencies are unavailable.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.identity_scope import compose_scoped_file_id


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler_context():
    """Create a mock HandlerContext with all required attributes (dict-style)."""
    # HandlerContext is a TypedDict - use dict-style access
    ctx = {
        "compressor": MagicMock(),
        "path_validator": MagicMock(),
    }

    # Mock compressor with graphs and chunks
    ctx["compressor"].graphs = {}
    ctx["compressor"].chunks = {}

    # Mock path validator with validate() method (not validate_path())
    ctx["path_validator"].validate = MagicMock(side_effect=lambda p: f"/validated/{p}")

    return ctx


@pytest.fixture
def mock_handler_context_no_validator():
    """Create a mock HandlerContext without PathValidator."""
    ctx = {
        "compressor": MagicMock(),
        "path_validator": None,
    }
    ctx["compressor"].graphs = {}
    ctx["compressor"].chunks = {}
    return ctx


# =============================================================================
# TOON Handler Tests (Pure Python - Always Available)
# =============================================================================


class TestTOONEncode:
    """Tests for toon_encode handler."""

    @pytest.mark.asyncio
    async def test_encode_returns_experimental_flag(self, mock_handler_context):
        """Verify toon_encode returns experimental: true."""
        from src.handlers.experimental_handlers import handle_toon_encode

        result_str = await handle_toon_encode(mock_handler_context, {"data": {"key": "value"}})
        result = json.loads(result_str)

        assert result.get("experimental") is True, "Must return experimental flag"

    @pytest.mark.asyncio
    async def test_encode_missing_data(self, mock_handler_context):
        """Test error handling when data is missing."""
        from src.handlers.experimental_handlers import handle_toon_encode

        result_str = await handle_toon_encode(mock_handler_context, {})
        result = json.loads(result_str)

        assert "error" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_encode_dict_data(self, mock_handler_context):
        """Test encoding a simple dictionary."""
        from src.handlers.experimental_handlers import handle_toon_encode

        result_str = await handle_toon_encode(
            mock_handler_context, {"data": {"name": "test", "value": 42}}
        )
        result = json.loads(result_str)

        assert "toon_output" in result
        assert "original_chars" in result
        assert "toon_chars" in result
        assert "savings_percent" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_encode_list_data(self, mock_handler_context):
        """Test encoding a list of items."""
        from src.handlers.experimental_handlers import handle_toon_encode

        result_str = await handle_toon_encode(mock_handler_context, {"data": [{"a": 1}, {"b": 2}]})
        result = json.loads(result_str)

        assert "toon_output" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_encode_results_format(self, mock_handler_context):
        """Test encoding data already in results format."""
        from src.handlers.experimental_handlers import handle_toon_encode

        result_str = await handle_toon_encode(
            mock_handler_context, {"data": {"results": [{"item": "test"}]}}
        )
        result = json.loads(result_str)

        assert "toon_output" in result
        assert result.get("experimental") is True


class TestTOONDecode:
    """Tests for toon_decode handler."""

    @pytest.mark.asyncio
    async def test_decode_returns_experimental_flag(self, mock_handler_context):
        """Verify toon_decode returns experimental: true."""
        from src.handlers.experimental_handlers import handle_toon_decode

        result_str = await handle_toon_decode(mock_handler_context, {"toon_input": "- test item"})
        result = json.loads(result_str)

        assert result.get("experimental") is True, "Must return experimental flag"

    @pytest.mark.asyncio
    async def test_decode_missing_input(self, mock_handler_context):
        """Test error handling when toon_input is missing."""
        from src.handlers.experimental_handlers import handle_toon_decode

        result_str = await handle_toon_decode(mock_handler_context, {})
        result = json.loads(result_str)

        assert "error" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_decode_simple_toon(self, mock_handler_context):
        """Test decoding simple TOON format."""
        from src.handlers.experimental_handlers import handle_toon_decode

        toon_input = """- first item
- second item
"""
        result_str = await handle_toon_decode(mock_handler_context, {"toon_input": toon_input})
        result = json.loads(result_str)

        assert "data" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_decode_with_properties(self, mock_handler_context):
        """Test decoding TOON with key-value properties."""
        from src.handlers.experimental_handlers import handle_toon_decode

        toon_input = """- item
name: test
value: 42
"""
        result_str = await handle_toon_decode(mock_handler_context, {"toon_input": toon_input})
        result = json.loads(result_str)

        assert "data" in result
        assert result.get("experimental") is True


# =============================================================================
# SCAR Handler Tests (Requires PyTorch)
# =============================================================================


class TestSCARCompress:
    """Tests for scar_compress handler."""

    @pytest.mark.asyncio
    async def test_compress_returns_experimental_flag(self, mock_handler_context):
        """Verify scar_compress returns experimental: true."""
        from src.handlers.experimental_handlers import handle_scar_compress

        # Setup mock document in compressor's graphs
        mock_handler_context["compressor"].graphs = {"test_doc": MagicMock()}
        mock_handler_context["compressor"].chunks = {"test_doc_n0": {"embedding": [0.1] * 384}}

        result_str = await handle_scar_compress(mock_handler_context, {"doc_id": "test_doc"})
        result = json.loads(result_str)

        assert result.get("experimental") is True, "Must return experimental flag"

    @pytest.mark.asyncio
    async def test_compress_missing_doc_id(self, mock_handler_context):
        """Test error handling when doc_id is missing."""
        from src.handlers.experimental_handlers import handle_scar_compress

        result_str = await handle_scar_compress(mock_handler_context, {})
        result = json.loads(result_str)

        assert "error" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_compress_doc_not_found(self, mock_handler_context):
        """Test error when document doesn't exist."""
        from src.handlers.experimental_handlers import handle_scar_compress

        mock_handler_context["compressor"].graphs = {}

        result_str = await handle_scar_compress(mock_handler_context, {"doc_id": "nonexistent"})
        result = json.loads(result_str)

        assert "error" in result
        assert "not found" in result["error"].lower()
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_compress_no_embeddings(self, mock_handler_context):
        """Test error when document has no embeddings."""
        from src.handlers.experimental_handlers import handle_scar_compress

        mock_handler_context["compressor"].graphs = {"test_doc": MagicMock()}
        mock_handler_context["compressor"].chunks = {
            "test_doc_n0": {"text": "no embedding here"}  # No embedding key
        }

        result_str = await handle_scar_compress(mock_handler_context, {"doc_id": "test_doc"})
        result = json.loads(result_str)

        # Should fail due to missing PyTorch or no embeddings
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_compress_handles_object_chunks(self, mock_handler_context):
        """Test SCAR handles object-based chunks (SemanticNode-style) with .embedding attribute."""
        from src.handlers.experimental_handlers import handle_scar_compress

        # Create object-style chunks (like SemanticNode) with .embedding attribute
        class MockChunk:
            def __init__(self, embedding):
                self.embedding = embedding
                self.text = "test text"

        mock_handler_context["compressor"].graphs = {"test_doc": MagicMock()}
        mock_handler_context["compressor"].chunks = {
            "test_doc_n0": MockChunk(embedding=[0.1] * 384),
            "test_doc_n1": MockChunk(embedding=[0.2] * 384),
        }

        result_str = await handle_scar_compress(mock_handler_context, {"doc_id": "test_doc"})
        result = json.loads(result_str)

        # Should find embeddings from object-based chunks
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_compress_scoped_doc_id(self, mock_handler_context):
        """SCAR should resolve scoped internal IDs while preserving visible doc_id output."""
        from src.handlers.experimental_handlers import handle_scar_compress
        import numpy as np

        scoped_doc_id = compose_scoped_file_id("test_doc", workspace_id="acme")
        mock_handler_context["compressor"].graphs = {scoped_doc_id: MagicMock()}
        mock_handler_context["compressor"].chunks = {
            f"{scoped_doc_id}_n0": {"embedding": [0.1] * 384}
        }

        with patch("src.handlers.experimental_handlers._get_scar_compressor") as mock_get:
            mock_scar = MagicMock()
            mock_scar.compress_embeddings.return_value = np.zeros((1, 128), dtype=np.float32)
            mock_get.return_value = mock_scar

            result_str = await handle_scar_compress(
                mock_handler_context, {"doc_id": "test_doc", "workspace_id": "acme"}
            )
            result = json.loads(result_str)

        assert result.get("doc_id") == "test_doc"
        assert result.get("experimental") is True


class TestSCARGetStats:
    """Tests for scar_get_stats handler."""

    @pytest.mark.asyncio
    async def test_stats_returns_experimental_flag(self, mock_handler_context):
        """Verify scar_get_stats returns experimental: true."""
        from src.handlers.experimental_handlers import handle_scar_get_stats

        result_str = await handle_scar_get_stats(mock_handler_context, {})
        result = json.loads(result_str)

        assert result.get("experimental") is True, "Must return experimental flag"

    @pytest.mark.asyncio
    async def test_stats_shows_pytorch_availability(self, mock_handler_context):
        """Test that stats indicate PyTorch availability."""
        from src.handlers.experimental_handlers import handle_scar_get_stats

        result_str = await handle_scar_get_stats(mock_handler_context, {})
        result = json.loads(result_str)

        assert "pytorch_available" in result
        assert isinstance(result["pytorch_available"], bool)
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_stats_shows_model_trained_false(self, mock_handler_context):
        """Test that stats indicate model is not trained."""
        from src.handlers.experimental_handlers import handle_scar_get_stats

        result_str = await handle_scar_get_stats(mock_handler_context, {})
        result = json.loads(result_str)

        assert result.get("model_trained") is False
        assert result.get("experimental") is True


# =============================================================================
# Multimodal Handler Tests (Requires Pillow for Images)
# =============================================================================


class TestMultimodalIngest:
    """Tests for multimodal_ingest handler."""

    @pytest.mark.asyncio
    async def test_ingest_returns_experimental_flag(self, mock_handler_context):
        """Verify multimodal_ingest returns experimental: true."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        # Mock the multimodal compressor
        with patch("src.handlers.experimental_handlers._get_multimodal_compressor") as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.ingest_mixed_content.return_value = {"node_count": 1}
            mock_get.return_value = mock_compressor

            result_str = await handle_multimodal_ingest(
                mock_handler_context, {"doc_id": "test", "text_content": "Hello world"}
            )
            result = json.loads(result_str)

        assert result.get("experimental") is True, "Must return experimental flag"

    @pytest.mark.asyncio
    async def test_ingest_missing_doc_id(self, mock_handler_context):
        """Test error handling when doc_id is missing."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        result_str = await handle_multimodal_ingest(mock_handler_context, {})
        result = json.loads(result_str)

        assert "error" in result
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_no_content(self, mock_handler_context):
        """Test error when no content types provided."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        result_str = await handle_multimodal_ingest(mock_handler_context, {"doc_id": "test"})
        result = json.loads(result_str)

        assert "error" in result
        assert "content type required" in result["error"].lower()
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_with_text(self, mock_handler_context):
        """Test ingesting text content."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        with patch("src.handlers.experimental_handlers._get_multimodal_compressor") as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.ingest_mixed_content.return_value = {"node_count": 1}
            mock_get.return_value = mock_compressor

            result_str = await handle_multimodal_ingest(
                mock_handler_context, {"doc_id": "test", "text_content": "Hello world"}
            )
            result = json.loads(result_str)

        assert "text" in result.get("content_types", [])
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_with_code(self, mock_handler_context):
        """Test ingesting code content."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        with patch("src.handlers.experimental_handlers._get_multimodal_compressor") as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.ingest_mixed_content.return_value = {"node_count": 1}
            mock_get.return_value = mock_compressor

            result_str = await handle_multimodal_ingest(
                mock_handler_context,
                {"doc_id": "test", "code_content": "print('hello')", "code_language": "python"},
            )
            result = json.loads(result_str)

        assert "code" in result.get("content_types", [])
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_scoped_doc_id(self, mock_handler_context):
        """Multimodal ingest should pass scoped IDs internally and visible IDs externally."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        with patch("src.handlers.experimental_handlers._get_multimodal_compressor") as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.ingest_mixed_content.return_value = {"node_count": 1}
            mock_get.return_value = mock_compressor

            result_str = await handle_multimodal_ingest(
                mock_handler_context,
                {"doc_id": "test", "workspace_id": "acme", "text_content": "Hello world"},
            )
            result = json.loads(result_str)

        scoped_doc_id = compose_scoped_file_id("test", workspace_id="acme")
        mock_compressor.ingest_mixed_content.assert_called_once_with(
            scoped_doc_id, [{"type": "text", "content": "Hello world"}]
        )
        assert result.get("doc_id") == "test"
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_images_without_validator(self, mock_handler_context_no_validator):
        """Test error when PathValidator not available for image paths."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        result_str = await handle_multimodal_ingest(
            mock_handler_context_no_validator,
            {"doc_id": "test", "image_paths": ["/path/to/image.png"]},
        )
        result = json.loads(result_str)

        assert "error" in result
        assert "PathValidator" in result["error"]
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_images_path_validation(self, mock_handler_context):
        """Test that image paths are validated."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        # Make path validator reject the path (using validate(), not validate_path())
        mock_handler_context["path_validator"].validate.side_effect = ValueError(
            "Path traversal detected"
        )

        result_str = await handle_multimodal_ingest(
            mock_handler_context,
            {"doc_id": "test", "image_paths": ["../../../etc/passwd"]},
        )
        result = json.loads(result_str)

        assert "error" in result
        assert "Invalid image path" in result["error"]
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_ingest_images_pillow_missing(self, mock_handler_context):
        """Test error when image_paths provided but Pillow is not installed."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        # Path validator accepts the path
        mock_handler_context["path_validator"].validate.return_value = "/validated/image.png"

        # Mock PIL import to fail
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            # Force the PIL import check to fail
            original_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "PIL" or name.startswith("PIL."):
                    raise ImportError("No module named 'PIL'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result_str = await handle_multimodal_ingest(
                    mock_handler_context,
                    {"doc_id": "test", "image_paths": ["/path/to/image.png"]},
                )
                result = json.loads(result_str)

        assert "error" in result
        assert "Pillow" in result["error"]
        assert result.get("experimental") is True


# =============================================================================
# Handler Registry Tests
# =============================================================================


class TestExperimentalHandlerRegistry:
    """Tests for EXPERIMENTAL_HANDLERS registry."""

    def test_registry_contains_all_handlers(self):
        """Verify all experimental handlers are in the registry."""
        from src.handlers.experimental_handlers import EXPERIMENTAL_HANDLERS

        expected = [
            "toon_encode",
            "toon_decode",
            "scar_compress",
            "scar_get_stats",
            "multimodal_ingest",
            "verify_compression",
            "calculate_reward",
            "get_evidence_stats",
            "generate_synthetic_tests",
        ]

        for handler_name in expected:
            assert handler_name in EXPERIMENTAL_HANDLERS, f"Missing handler: {handler_name}"

    def test_registry_handlers_are_callable(self):
        """Verify all handlers in registry are async callable."""
        import inspect
        from src.handlers.experimental_handlers import EXPERIMENTAL_HANDLERS

        for name, handler in EXPERIMENTAL_HANDLERS.items():
            assert callable(handler), f"Handler {name} is not callable"
            # Check if it's a coroutine function
            assert inspect.iscoroutinefunction(handler), f"Handler {name} is not async"


# =============================================================================
# MCP Core Integration Tests
# =============================================================================


class TestMCPCoreIntegration:
    """Tests for MCP core integration with experimental handlers."""

    def test_tool_schemas_registered(self):
        """Verify experimental tool schemas are registered in mcp_core."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        tool_names = [t.name for t in tools]

        expected = [
            "toon_encode",
            "toon_decode",
            "scar_compress",
            "scar_get_stats",
            "multimodal_ingest",
            "verify_compression",
            "calculate_reward",
            "get_evidence_stats",
            "generate_synthetic_tests",
        ]

        for name in expected:
            assert name in tool_names, f"Tool schema missing: {name}"

    def test_tool_count_is_48(self):
        """Verify total tool count includes cache telemetry additions."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        assert len(tools) == 109, f"Expected 109 tools, got {len(tools)}"

    def test_experimental_tools_have_experimental_in_description(self):
        """Verify all experimental tool descriptions mention EXPERIMENTAL."""
        from src.handlers.mcp_core import setup_mcp_tools

        experimental_tools = [
            "toon_encode",
            "toon_decode",
            "scar_compress",
            "scar_get_stats",
            "multimodal_ingest",
        ]
        tools = {t.name: t for t in setup_mcp_tools()}

        for name in experimental_tools:
            tool = tools.get(name)
            assert tool is not None, f"Tool not found: {name}"
            assert (
                "EXPERIMENTAL" in tool.description
            ), f"Tool {name} missing EXPERIMENTAL in description"


# =============================================================================
# Optional Dependency Skip Tests
# =============================================================================


class TestOptionalDependencies:
    """Tests that verify graceful handling of missing dependencies."""

    @pytest.mark.asyncio
    async def test_scar_without_pytorch_shows_helpful_error(self, mock_handler_context):
        """Test SCAR shows helpful error when PyTorch is missing."""
        from src.handlers.experimental_handlers import handle_scar_compress

        # Setup mock document in compressor's graphs
        mock_handler_context["compressor"].graphs = {"test_doc": MagicMock()}
        mock_handler_context["compressor"].chunks = {"test_doc_n0": {"embedding": [0.1] * 384}}

        with patch.dict("sys.modules", {"torch": None}):
            with patch("src.handlers.experimental_handlers._get_scar_compressor") as mock_get:
                mock_get.side_effect = ImportError("No module named 'torch'")

                result_str = await handle_scar_compress(
                    mock_handler_context, {"doc_id": "test_doc"}
                )
                result = json.loads(result_str)

        # Should return error with experimental flag
        assert result.get("experimental") is True

    @pytest.mark.asyncio
    async def test_multimodal_without_pillow_shows_helpful_error(self, mock_handler_context):
        """Test multimodal shows helpful error when Pillow is missing."""
        from src.handlers.experimental_handlers import handle_multimodal_ingest

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with patch("src.handlers.experimental_handlers._get_multimodal_compressor") as mock_get:
                mock_get.side_effect = ImportError("No module named 'PIL'")

                result_str = await handle_multimodal_ingest(
                    mock_handler_context, {"doc_id": "test", "text_content": "hello"}
                )
                result = json.loads(result_str)

        # Should return error with experimental flag
        assert result.get("experimental") is True
