"""App-layer runtime execution service for MCP stdio serving."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RuntimeService:
    """Executes MCP server runtime loop and startup diagnostics."""

    @staticmethod
    async def run(*, server: Any, logger: Any, stdio_server_fn: Callable[[], Any]) -> None:
        if logger is not None:
            logger.info(
                "mcp_server_starting",
                server_name="Semantic Modulator",
                features=["Semantic Communication", "Fidelity-Preserving Encoding"],
                model="all-MiniLM-L6-v2",
                mode="Adaptive Semantic Fidelity",
            )

        async with stdio_server_fn() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
