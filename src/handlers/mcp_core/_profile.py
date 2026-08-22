"""Tool-profile filtering helpers. Moved verbatim from mcp_core.py."""

from typing import List, Set

from mcp.types import Tool

from ._constants import CORE_STABLE_TOOL_NAMES, SUPPORTED_TOOL_PROFILES


def _normalize_tool_profile(profile: str) -> str:
    normalized = (profile or "full").strip().lower()
    if normalized not in SUPPORTED_TOOL_PROFILES:
        raise ValueError(
            f"Unknown tool profile '{profile}'. "
            f"Supported profiles: {sorted(SUPPORTED_TOOL_PROFILES)}"
        )
    return normalized


def _enabled_tool_names(all_names: Set[str], profile: str) -> Set[str]:
    normalized = _normalize_tool_profile(profile)
    if normalized == "full":
        return set(all_names)
    return set(all_names) & CORE_STABLE_TOOL_NAMES


def _tools_for_profile(tools: List[Tool], profile: str) -> List[Tool]:
    enabled_names = _enabled_tool_names({tool.name for tool in tools}, profile)
    return [tool for tool in tools if tool.name in enabled_names]
