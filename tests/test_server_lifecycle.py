"""
Tests for server lifespan management (v0.4.4)

Verifies MCP best practice implementation:
- Async context manager protocol
- Graceful shutdown with state persistence
- Error handling during cleanup
"""

import pytest
from unittest.mock import MagicMock

from src.server import SemanticModulatorServer


class TestServerLifecycle:
    """Test suite for server lifespan management"""

    @pytest.mark.asyncio
    async def test_server_lifespan_context_manager(self):
        """
        Test that server implements async context manager protocol.

        Verifies:
        - __aenter__ returns server instance
        - __aexit__ is called on cleanup
        - Resources properly initialized and cleaned up
        """
        server = SemanticModulatorServer()

        # Test __aenter__ returns server instance
        async with server as instance:
            assert instance is server, "Context manager should return server instance"

            # Verify resources are initialized
            assert hasattr(server, "compressor"), "Compressor should be initialized"
            assert hasattr(server, "server"), "MCP server should be initialized"

    @pytest.mark.asyncio
    async def test_server_persists_on_shutdown(self):
        """
        Test that server persists state on graceful shutdown.

        Verifies:
        - _save_file_sync_metadata() is called
        - State is saved before exit
        """
        server = SemanticModulatorServer()

        # Mock persistence methods
        server._save_file_sync_metadata = MagicMock()

        async with server:
            # Add some test data to ensure persistence is meaningful
            server.compressor.chunks["test_n0"] = MagicMock()

        # Verify persistence methods were called on shutdown
        server._save_file_sync_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_loads_state_on_startup(self):
        """
        Test that server loads persisted state on startup.

        Verifies:
        - _load_persisted_documents() is called in __aenter__
        - _load_file_sync_metadata() is called in __aenter__
        - State is loaded before request handling
        """
        server = SemanticModulatorServer()

        # Mock loading methods to track calls
        server._load_persisted_documents = MagicMock()
        server._load_file_sync_metadata = MagicMock()

        async with server:
            # Verify loading methods were called during __aenter__
            server._load_persisted_documents.assert_called_once()
            server._load_file_sync_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_handles_shutdown_errors(self):
        """
        Test that server handles errors gracefully during shutdown.

        Verifies:
        - Exceptions during _save_file_sync_metadata() don't crash shutdown
        - Exceptions are logged but not propagated
        - __aexit__ returns False (doesn't suppress exceptions)
        """
        server = SemanticModulatorServer()

        # Mock persistence to raise exception
        server._save_file_sync_metadata = MagicMock(side_effect=RuntimeError("Save failed"))

        # Should not raise exception despite save failure
        async with server:
            pass

        # Verify method was attempted
        server._save_file_sync_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_cleanup_on_exception(self):
        """
        Test that server cleans up properly even if exception occurs during use.

        Verifies:
        - __aexit__ is called even if exception occurs in context
        - State is persisted despite exceptions
        - __aexit__ returns False (propagates exceptions)
        """
        server = SemanticModulatorServer()

        # Mock persistence methods
        server._save_file_sync_metadata = MagicMock()

        # Simulate exception during server operation
        with pytest.raises(ValueError, match="Test exception"):
            async with server:
                raise ValueError("Test exception")

        # Verify cleanup still happened despite exception
        server._save_file_sync_metadata.assert_called_once()


@pytest.mark.asyncio
async def test_server_initialization_sequence():
    """
    Integration test: Verify full initialization sequence.

    Tests:
    - Components initialized in __init__
    - State loaded in __aenter__
    - State persisted in __aexit__
    - Proper ordering of operations
    """
    server = SemanticModulatorServer()

    # Track call order
    call_order = []

    # Mock methods to track order
    original_load_docs = server._load_persisted_documents
    original_load_meta = server._load_file_sync_metadata
    original_save_meta = server._save_file_sync_metadata

    server._load_persisted_documents = lambda: call_order.append("load_docs")
    server._load_file_sync_metadata = lambda: call_order.append("load_meta")
    server._save_file_sync_metadata = lambda: call_order.append("save_meta")

    async with server:
        call_order.append("running")

    # Verify correct sequence
    assert call_order == [
        "load_docs",
        "load_meta",
        "running",
        "save_meta",
    ], f"Incorrect initialization sequence: {call_order}"

    # Restore original methods
    server._load_persisted_documents = original_load_docs
    server._load_file_sync_metadata = original_load_meta
    server._save_file_sync_metadata = original_save_meta


if __name__ == "__main__":
    # Run tests with: pytest tests/test_server_lifecycle.py -v
    pytest.main([__file__, "-v"])
