"""MCP router binding helper for server protocol registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


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
