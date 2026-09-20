"""Dispatch MCP calls through the typed engine tool registry.

The registry owns the schema/handler join.  Keeping this module limited to
profile checks, validation, logging, and invocation prevents a second router
table from drifting away from ``setup_mcp_tools``.
"""

from typing import Any

from ...structured_logging import get_logger
from ._profile import _normalize_tool_profile
from .registry import get_registered_tool, registered_tool_names

logger = get_logger("semantic-modulator")


async def route_tool_call(
    name: str, args: dict[str, Any], context: dict[str, Any], tool_profile: str = "full"
) -> str:
    """Validate and invoke one registered MCP tool."""
    # ``registered_tool_names`` normalizes the profile before any lookup, so
    # invalid profiles retain the old fail-fast behavior.
    enabled_tools = registered_tool_names(tool_profile)

    try:
        registered = get_registered_tool(name)
    except KeyError:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Unknown tool: '{name}'\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use list_tools() to see all available tools with descriptions"
        ) from None

    if name not in enabled_tools:
        available_tools = ", ".join(sorted(enabled_tools))
        raise ValueError(
            f"Tool '{name}' is not enabled in profile '{_normalize_tool_profile(tool_profile)}'.\n\n"
            f"Available tools ({len(enabled_tools)}):\n{available_tools}\n\n"
            f"[TIP] Tip: Use profile 'full' to enable advanced tools"
        )

    # Pre-execute input validation
    from ...validation_hooks import validate_tool_input

    validation_errors = validate_tool_input(name, args)
    if validation_errors:
        import json

        return json.dumps(
            {
                "error": "Input validation failed",
                "validation_errors": validation_errors,
            },
            indent=2,
        )

    handler = registered.handler
    handler_module = getattr(handler, "__module__", "unknown")
    handler_name = getattr(handler, "__name__", "unknown")
    logger.info(
        "tool_routing", tool_name=name, handler_module=handler_module, handler_function=handler_name
    )

    return await handler(context, args)
