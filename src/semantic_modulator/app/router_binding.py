"""MCP router binding helper for server protocol registration."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypedDict

from src.semantic_modulator.app.contract_validation import validate_contract_keys


class BindRequest(TypedDict):
    """Router binding request envelope."""

    server: Any
    tooling: Any
    tool_profile: str
    build_context: Callable[[], Any]
    logger: Any
    text_content_cls: Any


BIND_REQUEST_KEYS: frozenset[str] = frozenset(BindRequest.__annotations__.keys())


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

    @server.list_prompts()
    async def list_prompts():
        return tooling.list_prompts()

    @server.get_prompt()
    async def get_prompt(name: str, arguments: Any | None = None):
        return tooling.get_prompt(name, arguments)

    @server.list_resources()
    async def list_resources():
        return tooling.list_resources(profile=tool_profile)

    @server.list_resource_templates()
    async def list_resource_templates():
        return tooling.list_resource_templates()

    @server.read_resource()
    async def read_resource(uri: str):
        context = build_context()
        return await tooling.read_resource(uri, context, profile=tool_profile)

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
            error_body = json.dumps({"error": type(e).__name__, "message": str(e), "tool": name})
            return [text_content_cls(type="text", text=error_body)]
