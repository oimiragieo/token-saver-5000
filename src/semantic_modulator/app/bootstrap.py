"""Bootstrap entrypoints for application wiring."""

from __future__ import annotations

import asyncio
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
    """Async entry point with lifespan management."""
    server = create_server_fn()
    async with server:
        await server.run()


def main(
    async_main_fn: Callable[[], Awaitable[None]] = async_main,
    run_fn: Callable[[Awaitable[None]], None] = asyncio.run,
) -> None:
    """Sync process entrypoint."""
    run_fn(async_main_fn())
