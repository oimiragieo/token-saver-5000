"""Upstream MCP server client: connects to any MCP server as a child process.

Uses the ``mcp`` package's client facilities (``ClientSession`` + ``stdio_client``).
Falls back to a helpful error if the client components are unavailable.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

# All client components are available in mcp>=0.9.0 which is a project dependency.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class UpstreamClient:
    """Manages an upstream MCP server subprocess.

    Wraps ``stdio_client`` + ``ClientSession`` so that callers only deal with
    high-level operations (list tools, call tool) rather than raw protocol
    details.

    Args:
        command: The executable to launch (e.g. ``"python"``, ``"npx"``).
        args: Arguments passed to the command.
        env: Optional extra environment variables for the subprocess.
        cwd: Optional working directory for the subprocess.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.command = command
        self.args = args or []
        self.env = env
        self.cwd = cwd
        self._session: ClientSession | None = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator["UpstreamClient", None]:
        """Async context manager that connects to the upstream server.

        Yields self so callers can call :meth:`list_tools` and :meth:`call_tool`
        within the ``async with`` block.

        Example::

            async with client.connect() as c:
                tools = await c.list_tools()
        """
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env,
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                try:
                    yield self
                finally:
                    self._session = None

    # ------------------------------------------------------------------
    # High-level operations
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the upstream server's tool list as plain dicts.

        Each dict has the keys ``name``, ``description``, and ``inputSchema``.

        Raises:
            RuntimeError: If called outside an active :meth:`connect` context.
        """
        if self._session is None:
            raise RuntimeError("UpstreamClient.list_tools() called outside connect() context")
        result = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in result.tools
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Invoke a tool on the upstream server.

        Args:
            name: Tool name to invoke.
            arguments: Tool arguments dict (may be ``None`` or empty).

        Returns:
            List of content dicts.  Text content items have ``{"type": "text",
            "text": "..."}``; other content types are returned as-is.

        Raises:
            RuntimeError: If called outside an active :meth:`connect` context.
        """
        if self._session is None:
            raise RuntimeError("UpstreamClient.call_tool() called outside connect() context")
        result = await self._session.call_tool(name, arguments or {})
        items = []
        for item in result.content:
            if hasattr(item, "text"):
                items.append({"type": "text", "text": item.text})
            elif hasattr(item, "model_dump"):
                items.append(item.model_dump())
            else:
                items.append({"type": "unknown", "raw": str(item)})
        return items
