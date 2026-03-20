"""Registry for managed connector feeds and available connector types."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import RLock
from typing import Any, Optional

from .connectors.base import ConnectorDocument
from .connectors.github_connector import GitHubConnector
from .connectors.s3_connector import S3Connector
from .connectors.slack_export_connector import SlackExportConnector
from .connectors.web_connector import WebConnector


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class ConnectorFeedRecord:
    """Stored feed definition."""

    name: str
    connector_type: str
    config: dict[str, Any]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)
    last_synced_at: str | None = None
    last_sync_document_ids: list[str] = field(default_factory=list)

    def to_dict(self, include_config: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "connector_type": self.connector_type,
            "created_at": self.created_at,
            "metadata": deepcopy(self.metadata),
            "last_synced_at": self.last_synced_at,
            "last_sync_document_ids": list(self.last_sync_document_ids),
        }
        if include_config:
            payload["config"] = deepcopy(self.config)
        return payload


class ConnectorRegistry:
    """Registry of connector types and managed feed definitions."""

    _instance: Optional["ConnectorRegistry"] = None

    def __init__(self):
        self._lock = RLock()
        self._connectors = {
            "web": WebConnector(),
            "github": GitHubConnector(),
            "s3": S3Connector(),
            "slack_export": SlackExportConnector(),
        }
        self._feeds: dict[str, ConnectorFeedRecord] = {}

    @classmethod
    def get_registry(cls) -> "ConnectorRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def list_connector_types(self) -> list[dict[str, str]]:
        with self._lock:
            return [self._connectors[name].definition() for name in sorted(self._connectors)]

    def create_feed(
        self,
        *,
        name: str,
        connector_type: str,
        config: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if name in self._feeds:
                raise ValueError(f"Connector feed '{name}' already exists")
            connector = self._get_connector(connector_type)
            connector.validate_config(config)
            record = ConnectorFeedRecord(
                name=name,
                connector_type=connector_type,
                config=deepcopy(config),
                created_at=_utc_now(),
                metadata=deepcopy(metadata or {}),
            )
            self._feeds[name] = record
            return record.to_dict(include_config=True)

    def list_feeds(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._feeds[name].to_dict() for name in sorted(self._feeds)]

    def get_feed(self, name: str) -> dict[str, Any]:
        with self._lock:
            if name not in self._feeds:
                raise ValueError(f"Unknown connector feed '{name}'")
            return self._feeds[name].to_dict(include_config=True)

    def collect_documents(self, feed_name: str) -> tuple[dict[str, Any], list[ConnectorDocument]]:
        with self._lock:
            if feed_name not in self._feeds:
                raise ValueError(f"Unknown connector feed '{feed_name}'")
            feed = self._feeds[feed_name]
            connector = self._get_connector(feed.connector_type)
            docs = connector.collect_documents(deepcopy(feed.config))
            return feed.to_dict(include_config=True), docs

    def mark_synced(self, feed_name: str, document_ids: list[str]) -> dict[str, Any]:
        with self._lock:
            if feed_name not in self._feeds:
                raise ValueError(f"Unknown connector feed '{feed_name}'")
            feed = self._feeds[feed_name]
            feed.last_synced_at = _utc_now()
            feed.last_sync_document_ids = list(document_ids)
            return feed.to_dict(include_config=True)

    def _get_connector(self, connector_type: str):
        if connector_type not in self._connectors:
            raise ValueError(
                f"Unknown connector type '{connector_type}'. "
                f"Available connector types: {sorted(self._connectors)}"
            )
        return self._connectors[connector_type]
