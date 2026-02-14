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
from typing import List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent

# Compatibility imports for unit-test patch targets.
from .ace_framework import ACEFramework  # noqa: F401
from .adaptive_rate_allocator import ContextWindowAdapter, MultiLevelSemanticEncoder  # noqa: F401
from .afm import AFMConfig, FocusManager  # noqa: F401
from .blind_spot_detector import BlindSpotDetector, HaloEffectDetector  # noqa: F401
from .code_compression_adapter import CodeCompressionAdapter  # noqa: F401
from .constants import MAX_ACE_CONTEXTS
from .file_sync_manager import FileSyncManager  # noqa: F401
from .persistence import PersistenceManager  # noqa: F401
from .resource_manager import ResourceLimits, ResourceManager  # noqa: F401
from .semantic_modulator.app.ace_context_manager import ACEContextManager  # noqa: F401
from .semantic_modulator.app.bootstrap import async_main as bootstrap_async_main
from .semantic_modulator.app.bootstrap import main as bootstrap_main
from .semantic_modulator.app.router_binding import bind_mcp_handlers
from .semantic_modulator.app.server_aliases import build_server_class_overrides
from .semantic_modulator.app.server_factory_service import ServerFactoryService
from .semantic_modulator.app.tooling import MCPToolingGateway  # noqa: F401
from .structured_logging import configure_structlog, get_logger
from .types import HandlerContext  # TypedDict for handler context
from .version_manager import VersionManager  # noqa: F401


# Configure structured logging
configure_structlog(log_level="INFO")
logger = get_logger("semantic-modulator")


class SemanticModulatorServer:
    """MCP Server for Semantic Modulation"""

    def __init__(self):
        self.server = Server("semantic-modulator")

        self.factory_service = ServerFactoryService()
        components = self.factory_service.build_default(
            preload_code_model=os.environ.get("PRELOAD_CODE_MODEL", "").lower() == "true",
            cwd=os.getcwd(),
            home_dir=os.path.expanduser("~"),
            max_ace_contexts=MAX_ACE_CONTEXTS,
            logger=logger,
            class_overrides=build_server_class_overrides(globals()),
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
        self.service_adapter.load_persisted_documents(
            compressor=self.compressor,
            persistence=self.persistence,
            resource_manager=self.resource_manager,
        )

    def _load_file_sync_metadata(self):
        """Load file sync metadata on server start"""
        self.service_adapter.load_file_sync_metadata(
            persistence=self.persistence,
            sync_manager=self.sync_manager,
        )

    def _save_file_sync_metadata(self):
        """Save file sync metadata to persistent storage"""
        self.service_adapter.save_file_sync_metadata(
            sync_manager=self.sync_manager,
            persistence=self.persistence,
        )

    def _build_context(self) -> HandlerContext:
        """
        Build context dictionary for handlers.

        Returns:
            HandlerContext TypedDict containing all components needed by handlers
        """
        return self.service_adapter.build_context(
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
        """Register MCP tool handlers using centralized routing."""
        bind_mcp_handlers(
            server=self.server,
            tooling=self.tooling,
            tool_profile=self.tool_profile,
            build_context=self._build_context,
            logger=logger,
            text_content_cls=TextContent,
        )

    def _extract_file_id_from_node(self, node_id: str) -> str:
        """
        Extract file_id from a node_id, handling both patterns:
        - Text pattern: 'doc_n0', 'doc_n1' -> 'doc'
        - Code pattern: 'file.py::ClassName', 'file.py::func_name' -> 'file.py'
        """
        return self.service_adapter.extract_file_id_from_node(node_id)

    def _validate_file_id(self, file_id: str, must_exist: bool = True) -> None:
        """Validate file_id and provide helpful error messages"""
        self.service_adapter.validate_file_id(
            compressor=self.compressor,
            file_id=file_id,
            must_exist=must_exist,
        )

    def _validate_node_ids(self, node_ids: List[str]) -> None:
        """Validate node_ids and provide helpful suggestions"""
        self.service_adapter.validate_node_ids(compressor=self.compressor, node_ids=node_ids)

    def _validate_token_count(self, available_tokens: int, max_tokens: int = None) -> None:
        """Validate token counts"""
        self.service_adapter.validate_token_count(
            available_tokens=available_tokens,
            max_tokens=max_tokens,
        )

    def _create_progress_bar(self, percentage: float, width: int = 40) -> str:
        """Create a text progress bar"""
        return self.service_adapter.create_progress_bar(percentage=percentage, width=width)

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
        await self.runtime_service.run(
            server=self.server,
            logger=logger,
            stdio_server_fn=stdio_server,
        )


async def async_main():
    """Async entry point with lifespan management."""
    await bootstrap_async_main()


def main():
    """Entry point"""
    bootstrap_main()


if __name__ == "__main__":
    main()
