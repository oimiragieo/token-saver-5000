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
from .semantic_modulator.app.server_factory_service import ServerFactoryService
from .semantic_modulator.app.tool_profile_service import ToolProfileBootstrapService
from .semantic_modulator.app.tooling import MCPToolingGateway
from .constants import MAX_ACE_CONTEXTS
from .structured_logging import get_logger, configure_structlog
from .semantic_modulator.app.bootstrap import async_main as bootstrap_async_main
from .semantic_modulator.app.bootstrap import main as bootstrap_main


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
        from .path_validator import PathValidator

        self.factory_service = ServerFactoryService()
        components = self.factory_service.build(
            preload_code_model=os.environ.get("PRELOAD_CODE_MODEL", "").lower() == "true",
            cwd=os.getcwd(),
            home_dir=os.path.expanduser("~"),
            max_ace_contexts=MAX_ACE_CONTEXTS,
            code_adapter_cls=CodeCompressionAdapter,
            blind_spot_cls=BlindSpotDetector,
            halo_cls=HaloEffectDetector,
            context_window_adapter_cls=ContextWindowAdapter,
            multilevel_encoder_cls=MultiLevelSemanticEncoder,
            afm_config_cls=AFMConfig,
            focus_manager_cls=FocusManager,
            persistence_cls=PersistenceManager,
            resource_limits_cls=ResourceLimits,
            resource_manager_cls=ResourceManager,
            file_sync_cls=FileSyncManager,
            version_manager_cls=VersionManager,
            path_validator_cls=PathValidator,
            ace_framework_cls=ACEFramework,
            ace_context_manager_cls=ACEContextManager,
            tooling_gateway_cls=MCPToolingGateway,
            context_service_cls=ServerContextService,
            lifecycle_service_cls=ServerLifecycleService,
            progress_service_cls=ProgressRenderService,
            persistence_service_cls=PersistenceOrchestrationService,
            tool_profile_service_cls=ToolProfileBootstrapService,
            logger=logger,
        )
        self.__dict__.update(components)
        configured_profile = os.environ.get("MCP_TOOL_PROFILE", "full")
        self.tool_profile, self.enabled_tool_names = self.tool_profile_service.bootstrap(
            configured_profile=configured_profile,
            tooling=self.tooling,
            logger=logger,
        )

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
    """Async entry point with lifespan management."""
    await bootstrap_async_main()


def main():
    """Entry point"""
    bootstrap_main()


if __name__ == "__main__":
    main()
