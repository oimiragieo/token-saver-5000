"""Bootstrap entrypoints for application wiring."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable


class _ServerProtocol:
    async def __aenter__(self): ...

    async def __aexit__(self, exc_type, exc_val, exc_tb): ...

    async def run(self): ...


def create_server():
    """Create and return the canonical MCP server instance."""
    from src.server import SemanticModulatorServer

    return SemanticModulatorServer()


async def async_main(
    create_server_fn: Callable[[], _ServerProtocol] = create_server,
) -> None:
    """Async entry point with lifespan management.

    When HTTP_ENABLED=true, starts the HTTP health/metrics server as a
    background task alongside the MCP stdio server.
    """
    server = create_server_fn()
    http_task: asyncio.Task | None = None

    async with server:
        if os.environ.get("HTTP_ENABLED", "false").lower() == "true":
            from src.http_server import start_http_server

            http_task = asyncio.create_task(start_http_server())

        try:
            await server.run()
        finally:
            if http_task is not None:
                http_task.cancel()
                try:
                    await http_task
                except asyncio.CancelledError:
                    pass


def main(
    async_main_fn: Callable[[], Awaitable[None]] = async_main,
    run_fn: Callable[[Awaitable[None]], None] = asyncio.run,
) -> None:
    """Sync process entrypoint."""
    run_fn(async_main_fn())
