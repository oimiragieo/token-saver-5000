"""MCP router binding helper for server protocol registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypedDict


class BindRequest(TypedDict):
    """Router binding request envelope."""

    server: Any
    tooling: Any
    tool_profile: str
    build_context: Callable[[], Any]
    logger: Any
    text_content_cls: Any


BIND_REQUEST_KEYS: frozenset[str] = frozenset(BindRequest.__annotations__.keys())


def contract_key_mismatch_message(
    *,
    contract_name: str,
    missing: list[str],
    extra: list[str],
) -> str:
    return f"{contract_name} keys mismatch: missing={missing} extra={extra}"


def validate_contract_keys(
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
            contract_key_mismatch_message(
                contract_name=contract_name,
                missing=missing,
                extra=extra,
            )
        )


def validate_bind_request_map(request: dict[str, Any]) -> BindRequest:
    validate_contract_keys(
        contract_name="bind_request_map",
        payload=request,
        expected_keys=BIND_REQUEST_KEYS,
    )
    return request


def bind_mcp_handlers(
    *,
    server: Any,
    tooling: Any,
    tool_profile: str,
    build_context: Callable[[], Any],
    logger: Any,
    text_content_cls: Any,
) -> None:
    """Bind list-tools and call-tool handlers to an MCP server instance."""
    request = validate_bind_request_map(
        {
            "server": server,
            "tooling": tooling,
            "tool_profile": tool_profile,
            "build_context": build_context,
            "logger": logger,
            "text_content_cls": text_content_cls,
        }
    )
    server = request["server"]
    tooling = request["tooling"]
    tool_profile = request["tool_profile"]
    build_context = request["build_context"]
    logger = request["logger"]
    text_content_cls = request["text_content_cls"]

    @server.list_tools()
    async def list_tools():
        return tooling.list_tools(profile=tool_profile)

    @server.call_tool()
    async def call_tool(name: str, arguments: Any):
        try:
            context = build_context()
            result = await tooling.route_tool_call(
                name, arguments, context, tool_profile=tool_profile
            )
            return [text_content_cls(type="text", text=str(result))]
        except Exception as e:
            if logger is not None:
                logger.error("tool_handler_error", tool_name=name, error=str(e), exc_info=True)
            return [text_content_cls(type="text", text=f"Error: {str(e)}")]
