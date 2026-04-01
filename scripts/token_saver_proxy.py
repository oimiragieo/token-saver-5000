#!/usr/bin/env python
"""Token Saver MCP Proxy: transparent compression for any MCP server.

Usage::

    token-saver-proxy <command> [args...] [options]

Examples::

    token-saver-proxy npx some-mcp-server
    token-saver-proxy python -m my_server --provider anthropic --schema-compression
    token-saver-proxy python -m my_server --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly as a script as well as via the installed entry point.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.proxy.proxy_server import ProxyConfig, ProxyServer


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Token Saver MCP Proxy: compress any MCP server's responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("command", help="Upstream MCP server command")
    parser.add_argument("args", nargs="*", help="Upstream server arguments")
    parser.add_argument(
        "--provider",
        default="unknown",
        help="AI provider hint (anthropic, google, openai)",
    )
    parser.add_argument(
        "--schema-compression",
        action="store_true",
        help="Replace N upstream tools with 3 meta-tools (search/inspect/invoke)",
    )
    parser.add_argument(
        "--refiner-ratio",
        type=float,
        default=0.7,
        help="Token refiner keep ratio (0.0–1.0, default 0.7)",
    )
    parser.add_argument(
        "--no-meta-tokens",
        action="store_true",
        help="Disable lossless meta-token n-gram compression",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit without starting",
    )
    return parser


def _print_dry_run(config: ProxyConfig) -> None:
    """Print configuration summary for --dry-run mode."""
    print("Token Saver MCP Proxy (dry-run)")
    print(f"  Upstream: {config.upstream_command} {' '.join(config.upstream_args)}")
    print(f"  Provider: {config.provider}")
    print(f"  Schema compression: {config.enable_schema_compression}")
    print(f"  Refiner ratio: {config.refiner_ratio}")
    print(f"  Meta-tokens: {config.enable_meta_tokens}")


def main() -> int:
    """Entry point for the ``token-saver-proxy`` CLI command.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    config = ProxyConfig(
        upstream_command=args.command,
        upstream_args=args.args or [],
        provider=args.provider,
        enable_schema_compression=args.schema_compression,
        refiner_ratio=args.refiner_ratio,
        enable_meta_tokens=not args.no_meta_tokens,
    )

    if args.dry_run:
        _print_dry_run(config)
        return 0

    # --- Full async proxy mode -------------------------------------------------
    print("Token Saver MCP Proxy ready.", file=sys.stderr)
    print(
        f"Upstream: {config.upstream_command} {' '.join(config.upstream_args)}",
        file=sys.stderr,
    )

    try:
        import asyncio

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool

        proxy = ProxyServer(config)

        async def run_proxy() -> None:
            """Connect to upstream and serve the proxy MCP server on stdio."""
            server_params = StdioServerParameters(
                command=config.upstream_command,
                args=config.upstream_args,
                env=config.upstream_env,
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # Fetch upstream tool list once at startup.
                    tools_result = await session.list_tools()
                    upstream_tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": t.inputSchema or {},
                        }
                        for t in tools_result.tools
                    ]

                    # Initialise schema compression (no-op if disabled).
                    proxy.setup_schema_compression(upstream_tools)

                    # Build the proxy MCP server.
                    mcp_server = Server("token-saver-proxy")

                    @mcp_server.list_tools()
                    async def handle_list_tools() -> list[Tool]:
                        processed = proxy.get_tools(upstream_tools)
                        return [
                            Tool(
                                name=t["name"],
                                description=t.get("description", ""),
                                inputSchema=t.get("inputSchema", {}),
                            )
                            for t in processed
                        ]

                    @mcp_server.call_tool()
                    async def handle_call_tool(
                        name: str, arguments: dict | None
                    ) -> list[TextContent]:
                        arguments = arguments or {}

                        # Check for meta-tool call first.
                        meta_result = proxy.handle_meta_tool_call(name, arguments)
                        if meta_result is not None:
                            parsed = json.loads(meta_result)
                            if parsed.get("_invoke_upstream"):
                                # Forward to upstream with new name/args.
                                name = parsed["tool_name"]
                                arguments = parsed.get("arguments", {})
                            else:
                                return [TextContent(type="text", text=meta_result)]

                        # Forward call to upstream server.
                        result = await session.call_tool(name, arguments)

                        # Compress text content items.
                        compressed_content: list[TextContent] = []
                        for item in result.content:
                            if hasattr(item, "text") and item.text:
                                compressed_text, _ = proxy.process_tool_result(name, item.text)
                                compressed_content.append(
                                    TextContent(type="text", text=compressed_text)
                                )
                            else:
                                compressed_content.append(item)

                        return compressed_content

                    # Serve the proxy on stdio.
                    async with stdio_server() as (srv_read, srv_write):
                        await mcp_server.run(
                            srv_read,
                            srv_write,
                            mcp_server.create_initialization_options(),
                        )

        asyncio.run(run_proxy())

    except ImportError as exc:
        print(f"MCP client not available: {exc}", file=sys.stderr)
        print("Install with: pip install 'mcp[client]'", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Proxy error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
