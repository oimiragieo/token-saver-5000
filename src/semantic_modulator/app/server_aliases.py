"""Helpers for server class alias mapping and wiring overrides."""

from __future__ import annotations

from typing import Mapping


SERVER_ALIAS_KEYS: tuple[str, ...] = (
    "CodeCompressionAdapter",
    "BlindSpotDetector",
    "HaloEffectDetector",
    "ContextWindowAdapter",
    "MultiLevelSemanticEncoder",
    "AFMConfig",
    "FocusManager",
    "PersistenceManager",
    "ResourceLimits",
    "ResourceManager",
    "FileSyncManager",
    "VersionManager",
    "ACEFramework",
    "ACEContextManager",
    "MCPToolingGateway",
)


def build_server_class_overrides(namespace: Mapping[str, object]) -> dict[str, object]:
    """Build factory class overrides from a module namespace.

    Args:
        namespace: Mapping containing server module alias symbols.

    Returns:
        Dict keyed by factory override names.

    Raises:
        KeyError: If any required alias key is missing.
    """
    missing = [key for key in SERVER_ALIAS_KEYS if key not in namespace]
    if missing:
        missing_csv = ", ".join(missing)
        raise KeyError(f"Missing server alias keys: {missing_csv}")

    return {key: namespace[key] for key in SERVER_ALIAS_KEYS}
