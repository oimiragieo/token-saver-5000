"""Dynamic Tool Output Control (DTOC) visibility management (Wave 3).

Provides compact handle substitution for hidden tool/context outputs and
exact SHA-256-verified restoration with honest SavingsTracker expansion debits.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from .savings_tracker import SavingsTracker
from .tee_recovery import TEE_MODE_ALWAYS, TeeStore
from .token_estimation import TokenEstimator

logger = logging.getLogger(__name__)

_ScopedKey = Tuple[str, str, str]
_Scope = Tuple[str, str]


@dataclass
class ContextHandle:
    """Metadata handle for a context/tool output managed by DTOC."""

    tool_key: str
    estimated_tokens: int
    visible: bool = True
    sha256: str = ""
    created_at: float = field(default_factory=time.time)
    summary: str = ""
    workspace_id: str = "default"
    session_id: str = "default"


@dataclass
class ManageContextRequest:
    """Request payload for enabling/disabling context handles within a scope."""

    enable: list[str] = field(default_factory=list)
    disable: list[str] = field(default_factory=list)
    workspace_id: str = "default"
    session_id: str = "default"


def format_hidden_handle(handle: ContextHandle) -> str:
    """Render the compact placeholder for a hidden context handle."""
    return (
        f"[CTX_HIDDEN key={handle.tool_key} "
        f"tokens={handle.estimated_tokens} "
        f"summary={handle.summary}]"
    )


class ContextVisibilityManager:
    """Manages DTOC context visibility, SHA-256 integrity, and expansion debits."""

    def __init__(
        self,
        tee_store: Optional[TeeStore] = None,
        token_estimator: Optional[TokenEstimator] = None,
        savings_tracker_factory: Optional[Callable[[str], SavingsTracker]] = None,
    ) -> None:
        self._tee_store = tee_store if tee_store is not None else TeeStore(mode=TEE_MODE_ALWAYS)
        self._estimator = token_estimator or TokenEstimator()
        self._tracker_factory = savings_tracker_factory
        self._trackers: Dict[_Scope, SavingsTracker] = {}
        self._handles: Dict[_ScopedKey, ContextHandle] = {}
        self._originals: Dict[_ScopedKey, str] = {}
        self._tee_ids: Dict[_ScopedKey, str] = {}

    def _get_tracker(self, workspace_id: str, session_id: str) -> SavingsTracker:
        scope = (workspace_id, session_id)
        if scope not in self._trackers:
            if self._tracker_factory is not None:
                self._trackers[scope] = self._tracker_factory(session_id)
            else:
                self._trackers[scope] = SavingsTracker(session_id=session_id)
        return self._trackers[scope]

    def _get_scoped_handle(
        self, tool_key: str, workspace_id: str, session_id: str
    ) -> ContextHandle:
        scoped_key = (workspace_id, session_id, tool_key)
        if scoped_key in self._handles:
            return self._handles[scoped_key]
        if any(k[2] == tool_key for k in self._handles):
            raise PermissionError(
                f"Access denied for key '{tool_key}' from scope ({workspace_id}, {session_id})"
            )
        raise KeyError(f"Unknown context key '{tool_key}'")

    def register(
        self,
        tool_key: str,
        original_text: str,
        summary: str = "",
        workspace_id: str = "default",
        session_id: str = "default",
        visible: bool = True,
    ) -> ContextHandle:
        """Register a tool/context output for DTOC visibility management."""
        if not tool_key or not tool_key.strip():
            raise ValueError("tool_key must be non-empty")

        scoped_key = (workspace_id, session_id, tool_key)
        digest = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        tokens = self._estimator.count_tokens(original_text)
        compact_summary = summary or (original_text[:60].replace("\n", " ").strip())

        handle = ContextHandle(
            tool_key=tool_key,
            estimated_tokens=tokens,
            visible=visible,
            sha256=digest,
            summary=compact_summary,
            workspace_id=workspace_id,
            session_id=session_id,
        )
        self._handles[scoped_key] = handle
        self._originals[scoped_key] = original_text

        compact_placeholder = format_hidden_handle(handle)
        tee_id = self._tee_store.store(
            original_text=original_text,
            compressed_text=compact_placeholder,
            compression_pct=90.0 if original_text else 0.0,
            source="dtoc",
            command_hint=tool_key,
            workspace_id=workspace_id,
            session_id=session_id,
            visible=visible,
        )
        if tee_id:
            self._tee_ids[scoped_key] = tee_id

        return handle

    @staticmethod
    def _dedup_keys(keys: list[str]) -> list[str]:
        return list(dict.fromkeys(keys))

    def _validate_disable_keys(
        self, keys: list[str], workspace_id: str, session_id: str
    ) -> list[tuple[str, _ScopedKey, ContextHandle]]:
        validated: list[tuple[str, _ScopedKey, ContextHandle]] = []
        for key in keys:
            handle = self._get_scoped_handle(key, workspace_id, session_id)
            validated.append((key, (workspace_id, session_id, key), handle))
        return validated

    def _validate_enable_keys(
        self, keys: list[str], workspace_id: str, session_id: str
    ) -> list[tuple[str, _ScopedKey, ContextHandle, str]]:
        validated: list[tuple[str, _ScopedKey, ContextHandle, str]] = []
        for key in keys:
            scoped_key = (workspace_id, session_id, key)
            handle = self._get_scoped_handle(key, workspace_id, session_id)
            original = self._originals[scoped_key]
            actual_sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
            if actual_sha != handle.sha256:
                raise ValueError(
                    f"Integrity check failed for '{key}': expected {handle.sha256}, got {actual_sha}"
                )
            validated.append((key, scoped_key, handle, original))
        return validated

    def _set_tee_visibility_batch(
        self,
        items: list[tuple[_ScopedKey, bool]],
        target_visible: bool,
        workspace_id: str,
        session_id: str,
    ) -> None:
        updated: list[tuple[str, bool]] = []
        try:
            for scoped_key, prev_visible in items:
                tee_id = self._tee_ids.get(scoped_key)
                if tee_id:
                    self._tee_store.set_visibility(
                        tee_id,
                        target_visible,
                        workspace_id=workspace_id,
                        session_id=session_id,
                    )
                    updated.append((tee_id, prev_visible))
        except Exception:
            for tee_id, prev_visible in reversed(updated):
                try:
                    self._tee_store.set_visibility(
                        tee_id,
                        prev_visible,
                        workspace_id=workspace_id,
                        session_id=session_id,
                    )
                except Exception as rollback_exc:
                    logger.warning(
                        "Failed to rollback tee visibility for %s: %s", tee_id, rollback_exc
                    )
            raise

    def disable(
        self,
        keys: list[str],
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Hide full outputs for *keys*, returning compact handle strings."""
        dedup_keys = self._dedup_keys(keys)
        validated = self._validate_disable_keys(dedup_keys, workspace_id, session_id)

        self._set_tee_visibility_batch(
            [(scoped_key, handle.visible) for _, scoped_key, handle in validated],
            False,
            workspace_id,
            session_id,
        )

        disabled: list[str] = []
        handles: Dict[str, str] = {}
        tokens_hidden = 0
        for key, _, handle in validated:
            if handle.visible:
                tokens_hidden += handle.estimated_tokens
            handle.visible = False
            disabled.append(key)
            handles[key] = format_hidden_handle(handle)

        return {
            "disabled": disabled,
            "handles": handles,
            "tokens_hidden": tokens_hidden,
        }

    def enable(
        self,
        keys: list[str],
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Restore exact original outputs for *keys* and debit SavingsTracker."""
        dedup_keys = self._dedup_keys(keys)
        validated = self._validate_enable_keys(dedup_keys, workspace_id, session_id)

        self._set_tee_visibility_batch(
            [(scoped_key, handle.visible) for _, scoped_key, handle, _ in validated],
            True,
            workspace_id,
            session_id,
        )

        tracker = self._get_tracker(workspace_id, session_id)
        enabled: list[str] = []
        restored: Dict[str, str] = {}
        tokens_debited = 0
        for key, _, handle, original in validated:
            was_visible = handle.visible
            handle.visible = True
            token_count = self._estimator.count_tokens(original) if not was_visible else 0
            if token_count > 0:
                tracker.record_expansion_debit(
                    tool_name="manage_context",
                    expanded_tokens=token_count,
                    reason=f"dtoc_enable:{key}",
                )
            enabled.append(key)
            restored[key] = original
            tokens_debited += token_count

        return {
            "enabled": enabled,
            "restored": restored,
            "tokens_debited": tokens_debited,
        }

    def render(
        self,
        tool_key: str,
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> str:
        """Render either the full output or compact hidden handle for *tool_key*."""
        handle = self._get_scoped_handle(tool_key, workspace_id, session_id)
        if handle.visible:
            return self._originals[(workspace_id, session_id, tool_key)]
        return format_hidden_handle(handle)

    def get_handle(
        self,
        tool_key: str,
        workspace_id: str = "default",
        session_id: str = "default",
    ) -> ContextHandle:
        """Return the scoped ContextHandle for *tool_key*."""
        return self._get_scoped_handle(tool_key, workspace_id, session_id)

    def manage(self, request: ManageContextRequest) -> Dict[str, Any]:
        """Execute a ManageContextRequest (disable + enable) atomically."""
        dedup_disable = self._dedup_keys(request.disable)
        dedup_enable = self._dedup_keys(request.enable)
        overlap = sorted(set(dedup_disable) & set(dedup_enable))
        if overlap:
            raise ValueError(f"Keys cannot appear in both enable and disable: {overlap}")

        # Validate both batches before mutating either
        validated_disable = self._validate_disable_keys(
            dedup_disable, request.workspace_id, request.session_id
        )
        self._validate_enable_keys(dedup_enable, request.workspace_id, request.session_id)

        disable_snapshot = [
            (scoped_key, handle, handle.visible) for _, scoped_key, handle in validated_disable
        ]

        disable_res = (
            self.disable(
                dedup_disable,
                workspace_id=request.workspace_id,
                session_id=request.session_id,
            )
            if dedup_disable
            else {"disabled": [], "handles": {}, "tokens_hidden": 0}
        )
        try:
            enable_res = (
                self.enable(
                    dedup_enable,
                    workspace_id=request.workspace_id,
                    session_id=request.session_id,
                )
                if dedup_enable
                else {"enabled": [], "restored": {}, "tokens_debited": 0}
            )
        except Exception:
            for scoped_key, handle, prev_visible in reversed(disable_snapshot):
                tee_id = self._tee_ids.get(scoped_key)
                if tee_id:
                    try:
                        self._tee_store.set_visibility(
                            tee_id,
                            prev_visible,
                            workspace_id=request.workspace_id,
                            session_id=request.session_id,
                        )
                    except Exception:
                        pass
                handle.visible = prev_visible
            raise
        return {
            "workspace_id": request.workspace_id,
            "session_id": request.session_id,
            **disable_res,
            **enable_res,
        }
