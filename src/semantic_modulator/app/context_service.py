"""App-layer service for handler context assembly and input validation."""

from __future__ import annotations

from typing import Any, Callable

from src.node_identity import extract_file_id_from_node
from src.types import HandlerContext


class ServerContextService:
    """Builds handler context and performs server-side validation helpers."""

    @staticmethod
    def extract_file_id_from_node(node_id: str) -> str:
        return extract_file_id_from_node(node_id)

    def build_context(
        self,
        *,
        compressor: Any,
        blind_spot_detector: Any,
        halo_detector: Any,
        context_window_adapter: Any,
        multilevel_encoder: Any,
        focus_manager: Any,
        persistence: Any,
        resource_manager: Any,
        sync_manager: Any,
        version_manager: Any,
        path_validator: Any,
        ace_framework: Any,
        ace_contexts: Any,
        validate_file_id: Callable[..., None],
        validate_node_ids: Callable[..., None],
        validate_token_count: Callable[..., None],
        save_file_sync_metadata: Callable[[], None],
        tool_profile: str,
        enabled_tool_names: list[str],
    ) -> HandlerContext:
        return {
            "compressor": compressor,
            "blind_spot_detector": blind_spot_detector,
            "halo_detector": halo_detector,
            "context_window_adapter": context_window_adapter,
            "multilevel_encoder": multilevel_encoder,
            "focus_manager": focus_manager,
            "persistence": persistence,
            "resource_manager": resource_manager,
            "sync_manager": sync_manager,
            "version_manager": version_manager,
            "path_validator": path_validator,
            "ace_framework": ace_framework,
            "ace_contexts": ace_contexts,
            "validate_file_id": validate_file_id,
            "validate_node_ids": validate_node_ids,
            "validate_token_count": validate_token_count,
            "save_file_sync_metadata": save_file_sync_metadata,
            "tool_profile": tool_profile,
            "enabled_tool_names": enabled_tool_names,
        }

    def validate_file_id(self, *, compressor: Any, file_id: str, must_exist: bool = True) -> None:
        if not file_id:
            raise ValueError("file_id cannot be empty")

        if must_exist and file_id not in compressor.chunks:
            available = list(
                {self.extract_file_id_from_node(node_id) for node_id in compressor.chunks.keys()}
            )
            raise ValueError(
                f"Document '{file_id}' not found. "
                f"Available documents: {available if available else '(none)'}\n"
                f"Tip: Use ingest_context() to add documents first."
            )

    def validate_node_ids(self, *, compressor: Any, node_ids: list[str]) -> None:
        if not node_ids:
            raise ValueError("node_ids list cannot be empty")

        invalid_nodes = [node_id for node_id in node_ids if node_id not in compressor.chunks]
        if invalid_nodes:
            file_id = self.extract_file_id_from_node(node_ids[0])
            valid_nodes = [
                node_id
                for node_id in compressor.chunks.keys()
                if self.extract_file_id_from_node(node_id) == file_id
            ]
            raise ValueError(
                f"Invalid node IDs: {invalid_nodes[:3]}\n"
                f"Tip: Use read_skeleton('{file_id}') to see valid node IDs.\n"
                f"   Valid nodes for '{file_id}': {valid_nodes[:5]}..."
                if valid_nodes
                else f"   No nodes found for '{file_id}'. Document may not be ingested."
            )

    @staticmethod
    def validate_token_count(*, available_tokens: int, max_tokens: int | None = None) -> None:
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
