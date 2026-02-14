"""Adapter that centralizes helper delegation for SemanticModulatorServer."""

from __future__ import annotations

from typing import Any


class ServerServiceAdapter:
    """Thin adapter that routes helper calls to app-layer services."""

    def __init__(
        self, *, persistence_service: Any, context_service: Any, progress_service: Any, logger
    ):
        self.persistence_service = persistence_service
        self.context_service = context_service
        self.progress_service = progress_service
        self.logger = logger

    def load_persisted_documents(
        self, *, compressor: Any, persistence: Any, resource_manager: Any
    ) -> None:
        self.persistence_service.load_persisted_documents(
            compressor=compressor,
            persistence=persistence,
            resource_manager=resource_manager,
            logger=self.logger,
        )

    def load_file_sync_metadata(self, *, persistence: Any, sync_manager: Any) -> None:
        self.persistence_service.load_file_sync_metadata(
            persistence=persistence,
            sync_manager=sync_manager,
            logger=self.logger,
        )

    def save_file_sync_metadata(self, *, sync_manager: Any, persistence: Any) -> None:
        self.persistence_service.save_file_sync_metadata(
            sync_manager=sync_manager,
            persistence=persistence,
            logger=self.logger,
        )

    def build_context(self, **kwargs):
        return self.context_service.build_context(**kwargs)

    def extract_file_id_from_node(self, node_id: str) -> str:
        return self.context_service.extract_file_id_from_node(node_id)

    def validate_file_id(self, *, compressor: Any, file_id: str, must_exist: bool = True) -> None:
        self.context_service.validate_file_id(
            compressor=compressor,
            file_id=file_id,
            must_exist=must_exist,
        )

    def validate_node_ids(self, *, compressor: Any, node_ids: list[str]) -> None:
        self.context_service.validate_node_ids(compressor=compressor, node_ids=node_ids)

    def validate_token_count(self, *, available_tokens: int, max_tokens: int | None = None) -> None:
        self.context_service.validate_token_count(
            available_tokens=available_tokens,
            max_tokens=max_tokens,
        )

    def create_progress_bar(self, percentage: float, width: int = 40) -> str:
        return self.progress_service.create_progress_bar(percentage=percentage, width=width)
