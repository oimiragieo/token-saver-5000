"""
Unit tests for SemanticModulatorServer (src/server.py)

Tests critical server initialization, persistence loading, validation helpers,
and ACEContextManager LRU eviction logic. Focuses on code paths not covered
by integration tests.

Coverage Target: 60%+ of server.py (164 lines)

Run with: pytest tests/test_server_unit.py -v
"""

import pytest
from unittest.mock import Mock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.server import SemanticModulatorServer, ACEContextManager
from src.semantic_compressor import SemanticNode
import numpy as np


class TestACEContextManager:
    """Tests for ACEContextManager LRU eviction (v0.4.2)"""

    def test_initialization_with_default_limit(self):
        """Test ACEContextManager initializes with default max_contexts"""
        manager = ACEContextManager()
        assert manager.max_contexts == 100  # MAX_ACE_CONTEXTS default
        assert len(manager) == 0

    def test_initialization_with_custom_limit(self):
        """Test ACEContextManager initializes with custom max_contexts"""
        manager = ACEContextManager(max_contexts=50)
        assert manager.max_contexts == 50
        assert len(manager) == 0

    def test_initialization_with_unlimited(self):
        """Test ACEContextManager with unlimited contexts (max_contexts=0)"""
        manager = ACEContextManager(max_contexts=0)
        assert manager.max_contexts == 0
        assert len(manager) == 0

    def test_add_context_below_limit(self):
        """Test adding contexts below the limit (no eviction)"""
        manager = ACEContextManager(max_contexts=3)

        manager["ctx1"] = {"bullets": ["bullet1"]}
        manager["ctx2"] = {"bullets": ["bullet2"]}

        assert len(manager) == 2
        assert "ctx1" in manager
        assert "ctx2" in manager

    def test_lru_eviction_when_limit_exceeded(self):
        """Test that oldest context is evicted when limit exceeded"""
        manager = ACEContextManager(max_contexts=3)

        # Add 3 contexts (at limit)
        manager["ctx1"] = {"bullets": ["bullet1"]}
        manager["ctx2"] = {"bullets": ["bullet2"]}
        manager["ctx3"] = {"bullets": ["bullet3"]}

        assert len(manager) == 3

        # Add 4th context - should evict ctx1 (oldest)
        manager["ctx4"] = {"bullets": ["bullet4"]}

        assert len(manager) == 3
        assert "ctx1" not in manager  # Evicted
        assert "ctx2" in manager
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_update_existing_context_moves_to_end(self):
        """Test that updating existing context moves it to end (most recently used)"""
        manager = ACEContextManager(max_contexts=3)

        manager["ctx1"] = {"bullets": ["bullet1"]}
        manager["ctx2"] = {"bullets": ["bullet2"]}
        manager["ctx3"] = {"bullets": ["bullet3"]}

        # Update ctx1 (should move to end)
        manager["ctx1"] = {"bullets": ["updated_bullet1"]}

        # Add ctx4 - should evict ctx2 (now oldest), not ctx1
        manager["ctx4"] = {"bullets": ["bullet4"]}

        assert len(manager) == 3
        assert "ctx1" in manager  # Should still exist (was moved to end)
        assert "ctx2" not in manager  # Evicted (now oldest)
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_get_context_moves_to_end(self):
        """Test that accessing context via __getitem__ moves it to end (LRU)"""
        manager = ACEContextManager(max_contexts=3)

        manager["ctx1"] = {"bullets": ["bullet1"]}
        manager["ctx2"] = {"bullets": ["bullet2"]}
        manager["ctx3"] = {"bullets": ["bullet3"]}

        # Access ctx1 (should move to end)
        _ = manager["ctx1"]

        # Add ctx4 - should evict ctx2 (now oldest), not ctx1
        manager["ctx4"] = {"bullets": ["bullet4"]}

        assert len(manager) == 3
        assert "ctx1" in manager  # Should still exist (was accessed)
        assert "ctx2" not in manager  # Evicted (now oldest)
        assert "ctx3" in manager
        assert "ctx4" in manager

    def test_unlimited_contexts_no_eviction(self):
        """Test that max_contexts=0 disables eviction (unlimited)"""
        manager = ACEContextManager(max_contexts=0)

        # Add many contexts - none should be evicted
        for i in range(10):
            manager[f"ctx{i}"] = {"bullets": [f"bullet{i}"]}

        assert len(manager) == 10
        # All contexts should still exist
        for i in range(10):
            assert f"ctx{i}" in manager

    def test_get_stats_with_contexts(self):
        """Test get_stats returns correct statistics"""
        manager = ACEContextManager(max_contexts=5)

        manager["ctx1"] = {"bullets": ["bullet1"]}
        manager["ctx2"] = {"bullets": ["bullet2"]}

        stats = manager.get_stats()

        assert stats["total_contexts"] == 2
        assert stats["max_contexts_limit"] == 5
        assert "ctx1" in stats["context_ids"]
        assert "ctx2" in stats["context_ids"]
        assert stats["approaching_limit"] is False  # 2/5 = 40%, not > 90%

    def test_get_stats_approaching_limit_warning(self):
        """Test get_stats warns when approaching limit (>90%)"""
        manager = ACEContextManager(max_contexts=10)

        # Add 9 contexts (90% of limit)
        for i in range(9):
            manager[f"ctx{i}"] = {"bullets": [f"bullet{i}"]}

        stats = manager.get_stats()

        assert stats["approaching_limit"] is False  # 9/10 = 90%, not > 90%

        # Add 1 more to exceed 90%
        manager["ctx9"] = {"bullets": ["bullet9"]}
        stats = manager.get_stats()

        # Now at 100%, should warn (but also evict)
        # Actually at limit, so let's test at 91%
        manager = ACEContextManager(max_contexts=100)
        for i in range(91):
            manager[f"ctx{i}"] = {"bullets": [f"bullet{i}"]}

        stats = manager.get_stats()
        assert stats["approaching_limit"] is True  # 91/100 = 91% > 90%

    def test_get_stats_unlimited_no_warning(self):
        """Test get_stats doesn't warn for unlimited contexts"""
        manager = ACEContextManager(max_contexts=0)

        for i in range(100):
            manager[f"ctx{i}"] = {"bullets": [f"bullet{i}"]}

        stats = manager.get_stats()

        assert stats["max_contexts_limit"] == "unlimited"
        assert stats["approaching_limit"] is False  # Can't approach unlimited


class TestSemanticModulatorServerInitialization:
    """Tests for SemanticModulatorServer.__init__"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_success(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        """Test successful server initialization with all components (v0.4.4 - lifespan management)"""
        with patch.object(SemanticModulatorServer, "_load_persisted_documents") as mock_load_docs:
            with patch.object(
                SemanticModulatorServer, "_load_file_sync_metadata"
            ) as mock_load_sync:
                with patch.object(SemanticModulatorServer, "_setup_handlers") as mock_setup:
                    server = SemanticModulatorServer()

                    # Verify all components initialized
                    assert server.compressor is not None
                    assert server.blind_spot_detector is not None
                    assert server.halo_detector is not None
                    assert server.persistence is not None
                    assert server.resource_manager is not None
                    assert server.sync_manager is not None
                    assert server.version_manager is not None
                    assert server.ace_framework is not None

                    # Verify ACEContextManager initialized
                    assert isinstance(server.ace_contexts, ACEContextManager)
                    assert server.ace_contexts.max_contexts == 100

                    # Verify initialization calls (v0.4.4: loading moved to __aenter__)
                    # Loading should NOT happen in __init__ anymore
                    mock_load_docs.assert_not_called()
                    mock_load_sync.assert_not_called()
                    mock_setup.assert_called_once()

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_sets_afm_config(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        """Test that AFM config is properly set"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            with patch("src.server.FocusManager") as mock_focus:
                SemanticModulatorServer()

                # Verify AFM config parameters
                mock_focus.assert_called_once()
                call_args = mock_focus.call_args
                afm_config = call_args[0][0]

                assert afm_config.tau_high == 0.45
                assert afm_config.tau_mid == 0.25
                assert afm_config.half_life == 12
                assert afm_config.use_llm_importance is False
                assert afm_config.use_llm_compression is False

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_sets_resource_limits(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        """Test that resource limits are properly configured"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            SemanticModulatorServer()

            # Verify ResourceManager called with correct limits
            mock_resource.assert_called_once()
            call_args = mock_resource.call_args[0][0]

            assert call_args.max_document_size_mb == 100.0
            assert call_args.max_total_storage_mb == 1024.0
            assert call_args.max_documents == 1000
            assert call_args.max_memory_mb == 2048.0

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_reads_mcp_tool_profile_env(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
            patch("src.server.mcp_core.setup_mcp_tools", return_value=[]),
            patch.dict(os.environ, {"MCP_TOOL_PROFILE": "core_stable"}, clear=False),
        ):
            server = SemanticModulatorServer()
            assert server.tool_profile == "core_stable"

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_falls_back_to_full_on_invalid_mcp_tool_profile(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
            patch(
                "src.server.mcp_core.setup_mcp_tools",
                side_effect=[ValueError("invalid profile"), []],
            ),
            patch.dict(os.environ, {"MCP_TOOL_PROFILE": "broken_profile"}, clear=False),
        ):
            server = SemanticModulatorServer()
            assert server.tool_profile == "full"

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_initialization_logs_active_tool_profile_with_count(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
            patch(
                "src.server.mcp_core.setup_mcp_tools",
                return_value=[Mock() for _ in range(7)],
            ),
            patch("src.server.logger.info") as mock_logger_info,
            patch.dict(os.environ, {"MCP_TOOL_PROFILE": "core_stable"}, clear=False),
        ):
            SemanticModulatorServer()

            assert any(
                call.args
                and call.args[0] == "mcp_tool_profile_active"
                and call.kwargs.get("profile") == "core_stable"
                and call.kwargs.get("enabled_tools") == 7
                for call in mock_logger_info.call_args_list
            )

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.BlindSpotDetector")
    @patch("src.server.HaloEffectDetector")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    @patch("src.server.FileSyncManager")
    @patch("src.server.VersionManager")
    @patch("src.server.ACEFramework")
    def test_invalid_profile_logs_fallback_active_profile(
        self,
        mock_ace,
        mock_version,
        mock_sync,
        mock_resource,
        mock_persistence,
        mock_halo,
        mock_blind_spot,
        mock_compressor,
    ):
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
            patch(
                "src.server.mcp_core.setup_mcp_tools",
                side_effect=[ValueError("invalid profile"), [Mock() for _ in range(48)]],
            ),
            patch("src.server.logger.info") as mock_logger_info,
            patch("src.server.logger.warning") as mock_logger_warning,
            patch.dict(os.environ, {"MCP_TOOL_PROFILE": "bad_profile"}, clear=False),
        ):
            SemanticModulatorServer()

            assert any(
                call.args and call.args[0] == "invalid_tool_profile"
                for call in mock_logger_warning.call_args_list
            )
            assert any(
                call.args
                and call.args[0] == "mcp_tool_profile_active"
                and call.kwargs.get("profile") == "full"
                and call.kwargs.get("enabled_tools") == 48
                for call in mock_logger_info.call_args_list
            )


class TestLoadPersistedDocuments:
    """Tests for _load_persisted_documents method (v0.4.4 - called in __aenter__)"""

    @pytest.mark.asyncio
    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    async def test_load_documents_empty(self, mock_persistence_cls, mock_compressor_cls):
        """Test loading when no persisted documents exist (v0.4.4 - async context manager)"""
        mock_persistence = Mock()
        mock_persistence.list_documents.return_value = []
        mock_persistence_cls.return_value = mock_persistence

        with (patch.object(SemanticModulatorServer, "_setup_handlers"),):
            server = SemanticModulatorServer()

            # Trigger __aenter__ to call _load_persisted_documents
            await server.__aenter__()

            # Should log "No persisted documents found"
            # Verify list_documents was called
            assert mock_persistence.list_documents.called
            assert mock_persistence.load_document.call_count == 0

    @pytest.mark.asyncio
    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.ResourceManager")
    async def test_load_documents_success(
        self, mock_resource_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test successful loading of persisted documents (v0.4.4 - async context manager)"""
        # Setup mocks
        mock_persistence = Mock()
        mock_persistence.list_documents.return_value = ["doc1", "doc2"]

        # Mock document data
        doc1_data = {
            "chunks": {
                "doc1_n0": SemanticNode(
                    node_id="doc1_n0",
                    text="chunk 1",
                    embedding=np.array([0.1] * 384),
                    importance=0.8,
                    metadata={"position": 0},
                )
            },
            "graph_data": {"nodes": ["doc1_n0"], "edges": []},
            "metadata": {"file_id": "doc1", "total_tokens": 100},
        }
        doc2_data = {
            "chunks": {
                "doc2_n0": SemanticNode(
                    node_id="doc2_n0",
                    text="chunk 2",
                    embedding=np.array([0.2] * 384),
                    importance=0.9,
                    metadata={"position": 0},
                )
            },
            "graph_data": {"nodes": ["doc2_n0"], "edges": []},
            "metadata": {"file_id": "doc2", "total_tokens": 150},
        }

        mock_persistence.load_document.side_effect = [doc1_data, doc2_data]
        mock_persistence_cls.return_value = mock_persistence

        # Setup compressor mock
        mock_compressor = Mock()
        mock_compressor.chunks = {}
        mock_compressor.graphs = {}
        mock_compressor.file_metadata = {}
        mock_compressor_cls.return_value = mock_compressor

        # Setup resource manager mock
        mock_resource = Mock()
        mock_resource_cls.return_value = mock_resource

        with (patch.object(SemanticModulatorServer, "_setup_handlers"),):
            server = SemanticModulatorServer()

            # Trigger __aenter__ to call _load_persisted_documents
            await server.__aenter__()

            # Verify documents loaded into compressor
            assert "doc1_n0" in server.compressor.chunks
            assert "doc2_n0" in server.compressor.chunks
            assert "doc1" in server.compressor.graphs
            assert "doc2" in server.compressor.graphs
            assert "doc1" in server.compressor.file_metadata
            assert "doc2" in server.compressor.file_metadata

            # Verify resource manager registration
            assert mock_resource.register_document.call_count == 2

    @pytest.mark.asyncio
    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    async def test_load_documents_partial_failure(self, mock_persistence_cls, mock_compressor_cls):
        """Test loading when some documents fail to load (v0.4.4 - async context manager)"""
        mock_persistence = Mock()
        mock_persistence.list_documents.return_value = ["doc1", "doc2", "doc3"]

        # doc1 succeeds, doc2 fails, doc3 succeeds
        doc1_data = {
            "chunks": {
                "doc1_n0": SemanticNode(
                    node_id="doc1_n0",
                    text="chunk 1",
                    embedding=np.array([0.1] * 384),
                    importance=0.8,
                    metadata={"position": 0},
                )
            },
            "graph_data": {"nodes": ["doc1_n0"], "edges": []},
            "metadata": {"file_id": "doc1", "total_tokens": 100},
        }
        doc3_data = {
            "chunks": {
                "doc3_n0": SemanticNode(
                    node_id="doc3_n0",
                    text="chunk 3",
                    embedding=np.array([0.3] * 384),
                    importance=0.7,
                    metadata={"position": 0},
                )
            },
            "graph_data": {"nodes": ["doc3_n0"], "edges": []},
            "metadata": {"file_id": "doc3", "total_tokens": 200},
        }

        mock_persistence.load_document.side_effect = [
            doc1_data,
            Exception("Corrupted data"),
            doc3_data,
        ]
        mock_persistence_cls.return_value = mock_persistence

        # Setup compressor mock
        mock_compressor = Mock()
        mock_compressor.chunks = {}
        mock_compressor.graphs = {}
        mock_compressor.file_metadata = {}
        mock_compressor_cls.return_value = mock_compressor

        with (patch.object(SemanticModulatorServer, "_setup_handlers"),):
            server = SemanticModulatorServer()

            # Trigger __aenter__ to call _load_persisted_documents
            await server.__aenter__()

            # Verify only successful documents loaded
            assert "doc1_n0" in server.compressor.chunks
            assert "doc3_n0" in server.compressor.chunks
            # doc2 should not be loaded
            assert "doc2_n0" not in server.compressor.chunks

    @pytest.mark.asyncio
    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    async def test_load_documents_exception_handling(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that exceptions during loading are caught and logged (v0.4.4 - async)"""
        mock_persistence = Mock()
        mock_persistence.list_documents.side_effect = Exception("Database error")
        mock_persistence_cls.return_value = mock_persistence

        # Should not raise - exception should be caught
        with (patch.object(SemanticModulatorServer, "_setup_handlers"),):
            server = SemanticModulatorServer()

            # Trigger __aenter__ - should not raise despite exception
            await server.__aenter__()

            # Server should still be initialized
            assert server.persistence is not None


class TestLoadFileSyncMetadata:
    """Tests for _load_file_sync_metadata method"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    def test_load_sync_metadata_empty(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test loading when no file sync metadata exists"""
        mock_persistence = Mock()
        mock_persistence.load_file_sync_metadata.return_value = {}
        mock_persistence_cls.return_value = mock_persistence

        mock_sync = Mock()
        mock_sync_cls.return_value = mock_sync

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            SemanticModulatorServer()

            # import_metadata should not be called for empty metadata
            assert (
                not mock_sync.import_metadata.called
                or mock_sync.import_metadata.call_args[0][0] == {}
            )

    @pytest.mark.asyncio
    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    async def test_load_sync_metadata_success(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test successful loading of file sync metadata (v0.4.4 - async context manager)"""
        metadata = {
            "doc1": {
                "file_path": "/path/to/doc1.txt",
                "last_modified": 1234567890.0,
                "checksum": "abc123",
            },
            "doc2": {
                "file_path": "/path/to/doc2.txt",
                "last_modified": 1234567900.0,
                "checksum": "def456",
            },
        }

        mock_persistence = Mock()
        mock_persistence.load_file_sync_metadata.return_value = metadata
        mock_persistence_cls.return_value = mock_persistence

        mock_sync = Mock()
        mock_sync_cls.return_value = mock_sync

        with (patch.object(SemanticModulatorServer, "_setup_handlers"),):
            server = SemanticModulatorServer()

            # Trigger __aenter__ to call _load_file_sync_metadata
            await server.__aenter__()

            # Verify import_metadata called with correct data
            mock_sync.import_metadata.assert_called_once_with(metadata)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    def test_load_sync_metadata_exception_handling(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test exception handling during sync metadata loading"""
        mock_persistence = Mock()
        mock_persistence.load_file_sync_metadata.side_effect = Exception("IO error")
        mock_persistence_cls.return_value = mock_persistence

        # Should not raise - exception should be caught
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Server should still be initialized
            assert server.sync_manager is not None


class TestSaveFileSyncMetadata:
    """Tests for _save_file_sync_metadata method"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    def test_save_sync_metadata_success(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test successful saving of file sync metadata"""
        metadata = {
            "doc1": {
                "file_path": "/path/to/doc1.txt",
                "last_modified": 1234567890.0,
            }
        }

        mock_sync = Mock()
        mock_sync.export_metadata.return_value = metadata
        mock_sync_cls.return_value = mock_sync

        mock_persistence = Mock()
        mock_persistence.save_file_sync_metadata.return_value = True
        mock_persistence_cls.return_value = mock_persistence

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            server._save_file_sync_metadata()

            # Verify export and save called
            mock_sync.export_metadata.assert_called_once()
            mock_persistence.save_file_sync_metadata.assert_called_once_with(metadata)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    def test_save_sync_metadata_failure(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test handling of save failure"""
        mock_sync = Mock()
        mock_sync.export_metadata.return_value = {}
        mock_sync_cls.return_value = mock_sync

        mock_persistence = Mock()
        mock_persistence.save_file_sync_metadata.return_value = False
        mock_persistence_cls.return_value = mock_persistence

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            server._save_file_sync_metadata()

            # Should log warning but not raise
            mock_persistence.save_file_sync_metadata.assert_called_once()

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    @patch("src.server.FileSyncManager")
    def test_save_sync_metadata_exception_handling(
        self, mock_sync_cls, mock_persistence_cls, mock_compressor_cls
    ):
        """Test exception handling during save"""
        mock_sync = Mock()
        mock_sync.export_metadata.side_effect = Exception("Export error")
        mock_sync_cls.return_value = mock_sync

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise - exception should be caught
            server._save_file_sync_metadata()


class TestBuildContext:
    """Tests for _build_context method"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_build_context_includes_all_components(self, mock_persistence_cls, mock_compressor_cls):
        """Test that _build_context includes all required components"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            context = server._build_context()

            # Verify all required keys present
            required_keys = [
                "compressor",
                "blind_spot_detector",
                "halo_detector",
                "context_window_adapter",
                "multilevel_encoder",
                "focus_manager",
                "persistence",
                "resource_manager",
                "sync_manager",
                "version_manager",
                "ace_framework",
                "ace_contexts",
                "validate_file_id",
                "validate_node_ids",
                "validate_token_count",
                "save_file_sync_metadata",
            ]

            for key in required_keys:
                assert key in context, f"Missing key: {key}"

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_build_context_validators_are_callable(self, mock_persistence_cls, mock_compressor_cls):
        """Test that validation helpers in context are callable"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            context = server._build_context()

            # Verify validators are callable
            assert callable(context["validate_file_id"])
            assert callable(context["validate_node_ids"])
            assert callable(context["validate_token_count"])
            assert callable(context["save_file_sync_metadata"])

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_build_context_includes_tool_profile_metadata(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
            patch("src.server.mcp_core.setup_mcp_tools") as mock_setup_tools,
            patch.dict(os.environ, {"MCP_TOOL_PROFILE": "core_stable"}, clear=False),
        ):
            mock_setup_tools.return_value = [
                Mock(name="ingest_context"),
                Mock(name="read_skeleton"),
            ]
            # Mock objects need explicit name field (unittest mock arg is internal id)
            mock_setup_tools.return_value[0].name = "ingest_context"
            mock_setup_tools.return_value[1].name = "read_skeleton"

            server = SemanticModulatorServer()
            context = server._build_context()

            assert context["tool_profile"] == "core_stable"
            assert context["enabled_tool_names"] == ["ingest_context", "read_skeleton"]


class TestValidateFileId:
    """Tests for _validate_file_id validation helper"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_file_id_empty_string_raises(self, mock_persistence_cls, mock_compressor_cls):
        """Test that empty file_id raises ValueError"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_file_id("", must_exist=False)

            assert "cannot be empty" in str(exc_info.value)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_file_id_exists_no_error(self, mock_persistence_cls, mock_compressor_cls):
        """Test that must_exist=False allows any file_id"""
        mock_compressor = Mock()
        mock_compressor.chunks = {}
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise when must_exist=False
            server._validate_file_id("any_doc", must_exist=False)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_file_id_not_exists_raises(self, mock_persistence_cls, mock_compressor_cls):
        """Test that non-existent file_id raises helpful error"""
        mock_compressor = Mock()
        mock_compressor.chunks = {"other_doc_n0": Mock()}
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_file_id("doc1", must_exist=True)

            error_msg = str(exc_info.value)
            assert "not found" in error_msg
            assert "Available documents" in error_msg
            assert "ingest_context()" in error_msg  # Helpful tip

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_file_id_must_exist_false_allows_new(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that must_exist=False allows new file_id"""
        mock_compressor = Mock()
        mock_compressor.chunks = {}
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise for new file_id when must_exist=False
            server._validate_file_id("new_doc", must_exist=False)


class TestValidateNodeIds:
    """Tests for _validate_node_ids validation helper"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_node_ids_empty_list_raises(self, mock_persistence_cls, mock_compressor_cls):
        """Test that empty node_ids list raises ValueError"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_node_ids([])

            assert "cannot be empty" in str(exc_info.value)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_node_ids_all_valid_no_error(self, mock_persistence_cls, mock_compressor_cls):
        """Test that all valid node IDs do not raise"""
        mock_compressor = Mock()
        mock_compressor.chunks = {
            "doc1_n0": Mock(),
            "doc1_n1": Mock(),
            "doc1_n2": Mock(),
        }
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise
            server._validate_node_ids(["doc1_n0", "doc1_n1"])

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_node_ids_invalid_raises_helpful_error(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that invalid node IDs raise helpful error with suggestions"""
        mock_compressor = Mock()
        mock_compressor.chunks = {
            "doc1_n0": Mock(),
            "doc1_n1": Mock(),
            "doc1_n2": Mock(),
        }
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_node_ids(["doc1_n99", "doc1_n98"])

            error_msg = str(exc_info.value)
            assert "Invalid node IDs" in error_msg
            assert "read_skeleton" in error_msg  # Helpful tip
            assert "Valid nodes for 'doc1'" in error_msg

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_node_ids_mixed_valid_invalid_raises(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that mix of valid and invalid node IDs raises error"""
        mock_compressor = Mock()
        mock_compressor.chunks = {"doc1_n0": Mock(), "doc1_n1": Mock()}
        mock_compressor_cls.return_value = mock_compressor

        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_node_ids(["doc1_n0", "doc1_n99"])  # One valid, one invalid

            error_msg = str(exc_info.value)
            assert "Invalid node IDs" in error_msg
            assert "doc1_n99" in error_msg


class TestValidateTokenCount:
    """Tests for _validate_token_count validation helper"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_token_count_negative_raises(self, mock_persistence_cls, mock_compressor_cls):
        """Test that negative token count raises ValueError"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_token_count(-100)

            assert "must be non-negative" in str(exc_info.value)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_token_count_zero_raises(self, mock_persistence_cls, mock_compressor_cls):
        """Test that zero token count raises helpful error"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_token_count(0)

            error_msg = str(exc_info.value)
            assert "no space for content" in error_msg
            assert "Provide a positive number" in error_msg  # Helpful tip

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_token_count_positive_no_error(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that positive token count does not raise"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise
            server._validate_token_count(1000)
            server._validate_token_count(100000)

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_token_count_exceeds_max_raises(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that available_tokens > max_tokens raises error"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            with pytest.raises(ValueError) as exc_info:
                server._validate_token_count(10000, max_tokens=5000)

            error_msg = str(exc_info.value)
            assert "exceeds max_tokens" in error_msg
            assert "should be ≤ max_tokens" in error_msg  # Helpful tip

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_validate_token_count_within_max_no_error(
        self, mock_persistence_cls, mock_compressor_cls
    ):
        """Test that available_tokens <= max_tokens does not raise"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()

            # Should not raise
            server._validate_token_count(5000, max_tokens=10000)
            server._validate_token_count(5000, max_tokens=5000)  # Equal is OK


class TestCreateProgressBar:
    """Tests for _create_progress_bar helper"""

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_progress_bar_empty(self, mock_persistence_cls, mock_compressor_cls):
        """Test progress bar for 0% usage"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            bar = server._create_progress_bar(0.0)

            assert "░" in bar  # Empty bar
            assert "[OK]" in bar  # Green status

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_progress_bar_half(self, mock_persistence_cls, mock_compressor_cls):
        """Test progress bar for 50% usage"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            bar = server._create_progress_bar(50.0)

            assert "█" in bar  # Filled portion
            assert "░" in bar  # Empty portion
            assert "[OK]" in bar  # Green status
            assert "50%" in bar

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_progress_bar_warning_threshold(self, mock_persistence_cls, mock_compressor_cls):
        """Test progress bar for 80%+ usage (warning)"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            bar = server._create_progress_bar(85.0)

            assert "█" in bar
            assert "[WARN]" in bar  # Warning emoji
            assert "85%" in bar

    @patch("src.server.CodeCompressionAdapter")
    @patch("src.server.PersistenceManager")
    def test_progress_bar_full(self, mock_persistence_cls, mock_compressor_cls):
        """Test progress bar for 100% usage (full)"""
        with (
            patch.object(SemanticModulatorServer, "_load_persisted_documents"),
            patch.object(SemanticModulatorServer, "_load_file_sync_metadata"),
            patch.object(SemanticModulatorServer, "_setup_handlers"),
        ):
            server = SemanticModulatorServer()
            bar = server._create_progress_bar(100.0)

            assert "█" in bar
            assert "[CRIT] FULL" in bar  # Full indicator


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
