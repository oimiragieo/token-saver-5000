"""App-layer tool profile bootstrap and diagnostics service."""

from __future__ import annotations

from typing import Any, TypedDict


class BootstrapRequest(TypedDict):
    """Tool profile bootstrap request envelope."""

    configured_profile: str
    tooling: Any
    logger: Any


class ToolProfileBootstrapService:
    """Resolves active MCP tool profile and emits startup diagnostics."""

    BOOTSTRAP_REQUEST_KEYS: frozenset[str] = frozenset(BootstrapRequest.__annotations__.keys())

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
        return f"{contract_name} keys mismatch: missing={missing} extra={extra}"

    @classmethod
    def validate_contract_keys(
        cls,
        *,
        contract_name: str,
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        actual_keys = set(payload.keys())
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing or extra:
            raise ValueError(
                cls.contract_key_mismatch_message(
                    contract_name=contract_name,
                    missing=missing,
                    extra=extra,
                )
            )

    @classmethod
    def validate_bootstrap_request_map(cls, request: dict[str, Any]) -> BootstrapRequest:
        cls.validate_contract_keys(
            contract_name="bootstrap_request_map",
            payload=request,
            expected_keys=cls.BOOTSTRAP_REQUEST_KEYS,
        )
        return request

    @classmethod
    def bootstrap(cls, *, configured_profile: str, tooling: Any, logger) -> tuple[str, list[str]]:
        request = cls.validate_bootstrap_request_map(
            {
                "configured_profile": configured_profile,
                "tooling": tooling,
                "logger": logger,
            }
        )
        configured_profile = request["configured_profile"]
        tooling = request["tooling"]
        logger = request["logger"]
        active_profile, enabled_tools, used_fallback = tooling.resolve_tools_for_profile(
            configured_profile
        )
        if used_fallback:
            logger.warning(
                "invalid_tool_profile",
                configured_profile=configured_profile,
                fallback_profile="full",
            )

        logger.info(
            "mcp_tool_profile_active",
            profile=active_profile,
            enabled_tools=len(enabled_tools),
            supported_profiles=sorted(tooling.supported_profiles),
        )
        return active_profile, [tool.name for tool in enabled_tools]
