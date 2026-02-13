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
from .semantic_modulator.app.context_service import ServerContextService
from .semantic_modulator.app.lifecycle_service import ServerLifecycleService
from .semantic_modulator.app.persistence_orchestration_service import (
    PersistenceOrchestrationService,
)
from .semantic_modulator.app.progress_service import ProgressRenderService
from .semantic_modulator.app.tooling import MCPToolingGateway
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
        configured_profile = os.environ.get("MCP_TOOL_PROFILE", "full")
        self.tooling = MCPToolingGateway()
        self.context_service = ServerContextService()
        self.lifecycle_service = ServerLifecycleService()
        self.progress_service = ProgressRenderService()
        self.persistence_service = PersistenceOrchestrationService()
        self.tool_profile, enabled_tools, used_fallback = self.tooling.resolve_tools_for_profile(
            configured_profile
        )
        if used_fallback:
            logger.warning(
                "invalid_tool_profile",
                configured_profile=configured_profile,
                fallback_profile="full",
            )
        logger.info(
            "mcp_tool_profile_active",
            profile=self.tool_profile,
            enabled_tools=len(enabled_tools),
            supported_profiles=sorted(self.tooling.supported_profiles),
        )
        self.enabled_tool_names = [tool.name for tool in enabled_tools]

        # Track what the AI has retrieved (for blind spot detection)
        self.retrieval_history: Dict[str, List[str]] = {}

        # NOTE: Auto-load moved to __aenter__ for proper lifespan management
        # This ensures clean startup/shutdown sequencing

        self._setup_handlers()

    def _load_persisted_documents(self):
        """Load previously persisted documents on server start"""
        self.persistence_service.load_persisted_documents(
            compressor=self.compressor,
            persistence=self.persistence,
            resource_manager=self.resource_manager,
            logger=logger,
        )

    def _load_file_sync_metadata(self):
        """Load file sync metadata on server start"""
        self.persistence_service.load_file_sync_metadata(
            persistence=self.persistence,
            sync_manager=self.sync_manager,
            logger=logger,
        )

    def _save_file_sync_metadata(self):
        """Save file sync metadata to persistent storage"""
        self.persistence_service.save_file_sync_metadata(
            sync_manager=self.sync_manager,
            persistence=self.persistence,
            logger=logger,
        )

    def _build_context(self) -> HandlerContext:
        """
        Build context dictionary for handlers.

        Returns:
            HandlerContext TypedDict containing all components needed by handlers
        """
        return self.context_service.build_context(
            compressor=self.compressor,
            blind_spot_detector=self.blind_spot_detector,
            halo_detector=self.halo_detector,
            context_window_adapter=self.context_window_adapter,
            multilevel_encoder=self.multilevel_encoder,
            focus_manager=self.focus_manager,
            persistence=self.persistence,
            resource_manager=self.resource_manager,
            sync_manager=self.sync_manager,
            version_manager=self.version_manager,
            path_validator=self.path_validator,
            ace_framework=self.ace_framework,
            ace_contexts=self.ace_contexts,
            validate_file_id=self._validate_file_id,
            validate_node_ids=self._validate_node_ids,
            validate_token_count=self._validate_token_count,
            save_file_sync_metadata=self._save_file_sync_metadata,
            tool_profile=self.tool_profile,
            enabled_tool_names=self.enabled_tool_names,
        )

    def _setup_handlers(self):
        """Register MCP tool handlers using centralized routing"""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available semantic modulation tools"""
            return self.tooling.list_tools(profile=self.tool_profile)

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Any) -> List[TextContent]:
            """Handle tool calls via centralized router"""
            try:
                context = self._build_context()
                result = await self.tooling.route_tool_call(
                    name, arguments, context, tool_profile=self.tool_profile
                )
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
        return self.context_service.extract_file_id_from_node(node_id)

    def _validate_file_id(self, file_id: str, must_exist: bool = True) -> None:
        """Validate file_id and provide helpful error messages"""
        self.context_service.validate_file_id(
            compressor=self.compressor,
            file_id=file_id,
            must_exist=must_exist,
        )

    def _validate_node_ids(self, node_ids: List[str]) -> None:
        """Validate node_ids and provide helpful suggestions"""
        self.context_service.validate_node_ids(compressor=self.compressor, node_ids=node_ids)

    def _validate_token_count(self, available_tokens: int, max_tokens: int = None) -> None:
        """Validate token counts"""
        self.context_service.validate_token_count(
            available_tokens=available_tokens,
            max_tokens=max_tokens,
        )

    def _create_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Create a text progress bar"""
        return self.progress_service.create_progress_bar(percentage=percentage, width=width)

    async def __aenter__(self):
        """
        Async context manager entry - initialize resources.

        This implements MCP best practice for lifespan management, ensuring
        proper resource initialization before the server starts handling requests.
        """
        self.lifecycle_service.startup(
            load_persisted_documents=self._load_persisted_documents,
            load_file_sync_metadata=self._load_file_sync_metadata,
            logger=logger,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit - cleanup resources.

        This implements MCP best practice for graceful shutdown, ensuring
        all state is persisted before the server terminates.

        Returns:
            False to not suppress exceptions
        """
        return self.lifecycle_service.shutdown(
            save_file_sync_metadata=self._save_file_sync_metadata,
            logger=logger,
        )

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
