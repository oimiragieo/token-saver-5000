"""Base types for managed connector feeds."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


def sanitize_segment(value: str) -> str:
    """Normalize external identifiers into stable file-id-safe segments."""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-")
    return normalized or "item"


@dataclass(frozen=True)
class ConnectorDocument:
    """Normalized document emitted by a connector."""

    source_id: str
    file_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseConnector(ABC):
    """Abstract base class for all connector adapters."""

    connector_type: str
    description: str

    def definition(self) -> dict[str, str]:
        return {
            "connector_type": self.connector_type,
            "description": self.description,
        }

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate feed configuration before collection."""

    @abstractmethod
    def collect_documents(self, config: dict[str, Any]) -> list[ConnectorDocument]:
        """Normalize connector payloads into ingestible documents."""
