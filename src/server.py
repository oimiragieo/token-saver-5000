#!/usr/bin/env python3
"""
Semantic Modulator MCP Server

Exposes adaptive semantic fidelity tools to AI agents via MCP protocol.

Architecture:
- Local semantic compression (no external API calls)
- Graph-based structure preservation
- Adaptive fidelity levels
- Blind spot detection for self-correction
"""

import asyncio
import os
from collections import OrderedDict
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

from .types import HandlerContext  # TypedDict for handler context
from .code_compression_adapter import CodeCompressionAdapter
from .blind_spot_detector import BlindSpotDetector, HaloEffectDetector
from .adaptive_rate_allocator import (
    ContextWindowAdapter,
    MultiLevelSemanticEncoder,
)
from .afm import FocusManager, AFMConfig
from .persistence import PersistenceManager
from .resource_manager import ResourceManager, ResourceLimits
from .file_sync_manager import FileSyncManager
from .version_manager import VersionManager
from .ace_framework import ACEFramework
from .handlers import mcp_core
from .constants import MAX_ACE_CONTEXTS
from .structured_logging import get_logger, configure_structlog


# Configure structured logging
configure_structlog(log_level="INFO")
logger = get_logger("semantic-modulator")


class ACEContextManager(OrderedDict):
    """
    LRU-enabled ACE context storage with automatic eviction (v0.4.2).

    Wraps OrderedDict to provide transparent LRU eviction when max_contexts
    limit is exceeded. Oldest contexts (by creation/access time) are evicted first.

    Args:
        max_contexts: Maximum contexts to retain (0 = unlimited, not recommended)
                      Default: MAX_ACE_CONTEXTS (100 contexts)
    """

    def __init__(self, max_contexts: int = MAX_ACE_CONTEXTS):
        super().__init__()
        self.max_contexts = max_contexts
        logger.info(
            "ace_context_manager_initialized",
            max_contexts=max_contexts if max_contexts > 0 else "unlimited",
        )

    def __setitem__(self, key, value):
        """
        Add or update context with automatic LRU eviction.

        When adding a new context:
        1. If key exists, move it to end (most recently used)
        2. Add the new context
        3. If limit exceeded, evict oldest context (first item)
        """
        # If key exists, move to end (mark as recently used)
        if key in self:
            super().move_to_end(key)

        # Add/update the context
        super().__setitem__(key, value)

        # Automatic LRU eviction
        if self.max_contexts > 0 and len(self) > self.max_contexts:
            # Remove oldest (first) item
            oldest_key = next(iter(self))
            del self[oldest_key]
            logger.info(
                "ace_context_evicted", evicted_context=oldest_key, max_contexts=self.max_contexts
            )

    def __getitem__(self, key):
        """
        Get context and mark as recently used.

        Moves accessed context to end of OrderedDict (LRU policy).
        """
        value = super().__getitem__(key)
        # Move to end (mark as recently accessed)
        super().move_to_end(key)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ACE context storage (v0.4.2).

        Returns:
            Dictionary with stats including max_contexts configuration
        """
        return {
            "total_contexts": len(self),
            "max_contexts_limit": self.max_contexts if self.max_contexts > 0 else "unlimited",
            "context_ids": list(self.keys()),
            "approaching_limit": (
                len(self) / self.max_contexts > 0.9 if self.max_contexts > 0 else False
            ),
        }


class SemanticModulatorServer:
    """MCP Server for Semantic Modulation"""

    def __init__(self):
        self.server = Server("semantic-modulator")
        # CodeCompressionAdapter routes text files to SemanticCompressor
        # and code files (Python, JS, TS, etc.) to CodeSemanticCompressor
        # CodeBERT model (~400MB) loaded lazily on first code file
        # Set PRELOAD_CODE_MODEL=true to prewarm for heavy code workflows
        self.compressor = CodeCompressionAdapter(
            text_model="all-MiniLM-L6-v2",
            code_model="microsoft/codebert-base",
            similarity_threshold=0.75,
            skeleton_ratio=0.2,
            preload_code_model=os.environ.get("PRELOAD_CODE_MODEL", "").lower() == "true",
        )
        self.blind_spot_detector = BlindSpotDetector(self.compressor)
        self.halo_detector = HaloEffectDetector(self.compressor)

        # JSCCM-inspired adaptive components
        self.context_window_adapter = ContextWindowAdapter(self.compressor)
        self.multilevel_encoder = MultiLevelSemanticEncoder(self.compressor)

        # AFM (Adaptive Focus Memory) for dialogue management
        afm_config = AFMConfig(
            tau_high=0.45,
            tau_mid=0.25,
            half_life=12,
            use_llm_importance=False,  # Use heuristic importance
            use_llm_compression=False,  # Use heuristic compression
        )
        self.focus_manager = FocusManager(afm_config)

        # Persistence layer (NEW!)
        self.persistence = PersistenceManager()

        # Resource management (NEW!)
        self.resource_manager = ResourceManager(
            ResourceLimits(
                max_document_size_mb=100.0,
                max_total_storage_mb=1024.0,
                max_documents=1000,
                max_memory_mb=2048.0,
            )
        )

        # File sync and version management (NEW in v0.4.0!)
        self.sync_manager = FileSyncManager()
        self.version_manager = VersionManager()
        logger.info("file_sync_initialized", status="enabled")

        # Path validation for security (NEW in v0.6.1 - fixes CWE-22)
        from .path_validator import PathValidator

        # Allow current directory and user's home directory
        allowed_dirs = [
            os.getcwd(),  # Current working directory
            os.path.expanduser("~"),  # User's home directory
        ]
        self.path_validator = PathValidator(allowed_base_dirs=allowed_dirs)
        logger.info(
            "path_validator_initialized",
            allowed_directories_count=len(allowed_dirs),
            security_feature="CWE-22 path traversal prevention",
        )

        # ACE Framework for evolving contexts (NEW in v0.4.1!)
        self.ace_framework = ACEFramework(
            deduplication_threshold=0.85,
            max_bullets=100,
        )
        # ACE contexts with LRU eviction (v0.4.2)
        self.ace_contexts = ACEContextManager(max_contexts=MAX_ACE_CONTEXTS)
        logger.info(
            "ace_framework_initialized",
            deduplication_threshold=0.85,
            max_bullets=100,
            max_contexts=MAX_ACE_CONTEXTS,
        )

        # Context window monitoring (like SNR in JSCCM)
        self.context_window_monitor = {
            "max_tokens": 100000,  # Typical context window size
            "used_tokens": 0,
            "history": [],
        }

        # Track what the AI has retrieved (for blind spot detection)
        self.retrieval_history: Dict[str, List[str]] = {}

        # NOTE: Auto-load moved to __aenter__ for proper lifespan management
        # This ensures clean startup/shutdown sequencing

        self._setup_handlers()

    def _load_persisted_documents(self):
        """Load previously persisted documents on server start"""
        try:
            file_ids = self.persistence.list_documents()
            if not file_ids:
                logger.info(
                    "persistence_load", status="empty", message="No persisted documents found"
                )
                return

            logger.info("persistence_load_started", document_count=len(file_ids))
            loaded_count = 0

            for file_id in file_ids:
                try:
                    data = self.persistence.load_document(file_id)
                    if data:
                        # Restore to compressor
                        self.compressor.chunks.update(data["chunks"])
                        self.compressor.graphs[file_id] = data["graph_data"]
                        self.compressor.file_metadata[file_id] = data["metadata"]

                        # Register with resource manager
                        # Estimate size from chunks
                        total_size = sum(len(chunk.text) for chunk in data["chunks"].values())
                        self.resource_manager.register_document(file_id, total_size)

                        loaded_count += 1
                        logger.info(
                            "document_loaded", file_id=file_id, node_count=len(data["chunks"])
                        )
                except Exception as e:
                    logger.error("document_load_failed", file_id=file_id, error=str(e))

            logger.info(
                "persistence_load_complete",
                loaded=loaded_count,
                total=len(file_ids),
                success_rate=f"{loaded_count}/{len(file_ids)}",
            )

        except Exception as e:
            logger.error("persistence_load_error", error=str(e), exc_info=True)

    def _load_file_sync_metadata(self):
        """Load file sync metadata on server start"""
        try:
            metadata = self.persistence.load_file_sync_metadata()
            if not metadata:
                logger.info(
                    "file_sync_load",
                    status="empty",
                    message="No file sync metadata found (first run or no tracked files)",
                )
                return

            self.sync_manager.import_metadata(metadata)
            logger.info("file_sync_load_complete", document_count=len(metadata))

        except Exception as e:
            logger.error("file_sync_load_error", error=str(e), exc_info=True)

    def _save_file_sync_metadata(self):
        """Save file sync metadata to persistent storage"""
        try:
            metadata = self.sync_manager.export_metadata()
            success = self.persistence.save_file_sync_metadata(metadata)
            if success:
                logger.info("file_sync_save_complete", document_count=len(metadata))
            else:
                logger.warning("file_sync_save_warning", message="No file sync metadata to save")
        except Exception as e:
            logger.error("file_sync_save_error", error=str(e))

    def _build_context(self) -> HandlerContext:
        """
        Build context dictionary for handlers.

        Returns:
            HandlerContext TypedDict containing all components needed by handlers
        """
        return {
            "compressor": self.compressor,
            "blind_spot_detector": self.blind_spot_detector,
            "halo_detector": self.halo_detector,
            "context_window_adapter": self.context_window_adapter,
            "multilevel_encoder": self.multilevel_encoder,
            "focus_manager": self.focus_manager,
            "persistence": self.persistence,
            "resource_manager": self.resource_manager,
            "sync_manager": self.sync_manager,
            "version_manager": self.version_manager,
            "path_validator": self.path_validator,  # Security: CWE-22 path traversal prevention
            "ace_framework": self.ace_framework,
            "ace_contexts": self.ace_contexts,
            "validate_file_id": self._validate_file_id,
            "validate_node_ids": self._validate_node_ids,
            "validate_token_count": self._validate_token_count,
            "save_file_sync_metadata": self._save_file_sync_metadata,
        }

    def _setup_handlers(self):
        """Register MCP tool handlers using centralized routing"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available semantic modulation tools"""
            return mcp_core.setup_mcp_tools()

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls via centralized router"""
            try:
                context = self._build_context()
                result = await mcp_core.route_tool_call(name, arguments, context)
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                logger.error("tool_handler_error", tool_name=name, error=str(e), exc_info=True)
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def _extract_file_id_from_node(self, node_id: str) -> str:
        """
        Extract file_id from a node_id, handling both patterns:
        - Text pattern: 'doc_n0', 'doc_n1' -> 'doc'
        - Code pattern: 'file.py::ClassName', 'file.py::func_name' -> 'file.py'
        """
        if "::" in node_id:
            # Code-style semantic node ID (e.g., "file.py::ClassName")
            return node_id.split("::")[0]
        elif "_n" in node_id:
            # Text-style numeric node ID (e.g., "doc_n0")
            return node_id.rsplit("_n", 1)[0]
        else:
            # Fallback: assume entire node_id is file_id
            return node_id

    def _validate_file_id(self, file_id: str, must_exist: bool = True) -> None:
        """Validate file_id and provide helpful error messages"""
        if not file_id:
            raise ValueError("file_id cannot be empty")

        if must_exist:
            if file_id not in self.compressor.chunks:
                # Extract file IDs from all node IDs (handle both _n and :: patterns)
                available = list(
                    set(
                        [
                            self._extract_file_id_from_node(nid)
                            for nid in self.compressor.chunks.keys()
                        ]
                    )
                )
                raise ValueError(
                    f"Document '{file_id}' not found. "
                    f"Available documents: {available if available else '(none)'}\n"
                    f"Tip: Use ingest_context() to add documents first."
                )

    def _validate_node_ids(self, node_ids: List[str]) -> None:
        """Validate node_ids and provide helpful suggestions"""
        if not node_ids:
            raise ValueError("node_ids list cannot be empty")

        invalid_nodes = [nid for nid in node_ids if nid not in self.compressor.chunks]
        if invalid_nodes:
            # Extract file_id from first node to give better error message
            # Handle both _n and :: patterns
            file_id = self._extract_file_id_from_node(node_ids[0])
            valid_nodes = [
                nid
                for nid in self.compressor.chunks.keys()
                if self._extract_file_id_from_node(nid) == file_id
            ]

            raise ValueError(
                f"Invalid node IDs: {invalid_nodes[:3]}\n"
                f"Tip: Use read_skeleton('{file_id}') to see valid node IDs.\n"
                f"   Valid nodes for '{file_id}': {valid_nodes[:5]}..."
                if valid_nodes
                else f"   No nodes found for '{file_id}'. Document may not be ingested."
            )

    def _validate_token_count(self, available_tokens: int, max_tokens: int = None) -> None:
        """Validate token counts"""
        if available_tokens < 0:
            raise ValueError(f"available_tokens must be non-negative, got {available_tokens}")

        if available_tokens == 0:
            raise ValueError(
                "available_tokens is 0 - no space for content!\n"
                "Tip: Provide a positive number (e.g., 10000 for 10k tokens available)"
            )

        if max_tokens is not None and available_tokens > max_tokens:
            raise ValueError(
                f"available_tokens ({available_tokens}) exceeds max_tokens ({max_tokens})\n"
                "Tip: available_tokens should be ≤ max_tokens"
            )

    def _create_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Create a text progress bar"""
        filled = int((percentage / 100) * width)
        empty = width - filled

        if percentage >= 100:
            bar = "█" * width
            return f"[{bar}] [CRIT] FULL"
        elif percentage >= 80:
            bar = "█" * filled + "░" * empty
            return f"[{bar}] [WARN] {percentage:.0f}%"
        else:
            bar = "█" * filled + "░" * empty
            return f"[{bar}] [OK] {percentage:.0f}%"

    async def __aenter__(self):
        """
        Async context manager entry - initialize resources.

        This implements MCP best practice for lifespan management, ensuring
        proper resource initialization before the server starts handling requests.
        """
        logger.info("server_startup", phase="initializing")

        # Auto-load persisted state
        self._load_persisted_documents()
        self._load_file_sync_metadata()

        logger.info("server_startup", phase="complete")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit - cleanup resources.

        This implements MCP best practice for graceful shutdown, ensuring
        all state is persisted before the server terminates.

        Returns:
            False to not suppress exceptions
        """
        logger.info("server_shutdown", phase="started")

        try:
            # Persist file sync metadata
            self._save_file_sync_metadata()
            logger.info("server_shutdown", phase="state_persisted")
        except Exception as e:
            logger.error("server_shutdown_error", error=str(e), exc_info=True)

        logger.info("server_shutdown", phase="complete")
        return False  # Don't suppress exceptions

    async def run(self):
        """Run the MCP server"""
        logger.info(
            "mcp_server_starting",
            server_name="Semantic Modulator",
            features=["Semantic Communication", "Fidelity-Preserving Encoding"],
            model="all-MiniLM-L6-v2",
            mode="Adaptive Semantic Fidelity",
        )

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, write_stream, self.server.create_initialization_options()
            )


async def async_main():
    """Async entry point with lifespan management"""
    server = SemanticModulatorServer()
    async with server:
        await server.run()


def main():
    """Entry point"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
