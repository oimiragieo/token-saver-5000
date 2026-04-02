"""Scoped registry for durable handoff bundles."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import sys
from datetime import datetime, timezone

if sys.version_info >= (3, 11):
    from datetime import UTC
else:
    UTC = timezone.utc
from threading import RLock
from typing import Any, Optional
from uuid import uuid4

from .evidence_bundle import EvidenceBundle
from .identity_scope import display_file_id, scope_matches
from .toon_serializer import TOONSerializer


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass
class HandoffBundleRecord:
    """Durable handoff artifact with scoped ownership and audit reference."""

    bundle_id: str
    scoped_doc_id: str
    created_at: str
    updated_at: str
    replay_text: str
    artifact: dict[str, Any]
    artifact_toon: str
    audit: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_artifact: bool = True) -> dict[str, Any]:
        payload = {
            "bundle_id": self.bundle_id,
            "doc_id": display_file_id(self.scoped_doc_id),
            "scoped_doc_id": self.scoped_doc_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "audit": deepcopy(self.audit),
            "metadata": deepcopy(self.metadata),
        }
        if include_artifact:
            payload["artifact"] = deepcopy(self.artifact)
            payload["artifact_toon"] = self.artifact_toon
            payload["replay_text"] = self.replay_text
        return payload


class HandoffBundleRegistry:
    """In-memory scoped bundle registry."""

    _instance: Optional["HandoffBundleRegistry"] = None

    def __init__(self):
        self._lock = RLock()
        self._bundles: dict[str, HandoffBundleRecord] = {}
        self._serializer = TOONSerializer()

    @classmethod
    def get_registry(cls) -> "HandoffBundleRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        cls._instance = None

    def create_bundle(
        self,
        *,
        scoped_doc_id: str,
        artifact: dict[str, Any],
        replay_text: str,
        evidence_bundle: EvidenceBundle,
        metadata: Optional[dict[str, Any]] = None,
        bundle_id: Optional[str] = None,
        created_at: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            resolved_bundle_id = bundle_id or str(uuid4())
            timestamp = created_at or _utc_now()
            artifact_payload = deepcopy(artifact)
            artifact_payload["bundle_id"] = resolved_bundle_id
            artifact_payload.setdefault("doc_id", display_file_id(scoped_doc_id))
            artifact_payload.setdefault("scoped_doc_id", scoped_doc_id)
            artifact_payload.setdefault("created_at", timestamp)
            artifact_toon = self._serializer.serialize_handoff_bundle(artifact_payload)

            record = HandoffBundleRecord(
                bundle_id=resolved_bundle_id,
                scoped_doc_id=scoped_doc_id,
                created_at=timestamp,
                updated_at=timestamp,
                replay_text=replay_text,
                artifact=artifact_payload,
                artifact_toon=artifact_toon,
                audit=evidence_bundle.audit_summary(),
                metadata=deepcopy(metadata or {}),
            )
            self._bundles[resolved_bundle_id] = record
            return record.to_dict(include_artifact=True)

    def list_bundles(
        self,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        file_id: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results: list[dict[str, Any]] = []
            for record in sorted(
                self._bundles.values(), key=lambda item: item.created_at, reverse=True
            ):
                if not scope_matches(
                    record.scoped_doc_id,
                    workspace_id=workspace_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                ):
                    continue
                if file_id and display_file_id(record.scoped_doc_id) != file_id:
                    continue
                results.append(record.to_dict(include_artifact=False))
            return results

    def get_bundle(
        self,
        bundle_id: str,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._bundles.get(bundle_id)
            if record is None or not scope_matches(
                record.scoped_doc_id,
                workspace_id=workspace_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            ):
                raise ValueError(f"Handoff bundle '{bundle_id}' not found")
            return record.to_dict(include_artifact=True)

    def replay_bundle(
        self,
        bundle_id: str,
        *,
        workspace_id: str | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        record = self.get_bundle(
            bundle_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )
        return {
            "bundle_id": record["bundle_id"],
            "doc_id": record["doc_id"],
            "replay_text": record["replay_text"],
            "artifact": record["artifact"],
            "artifact_toon": record["artifact_toon"],
            "audit": deepcopy(record["audit"]),
        }
