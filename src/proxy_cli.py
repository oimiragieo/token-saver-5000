"""Entry point for token-saver-proxy CLI command."""

from __future__ import annotations


def main() -> int:
    """Run the Token Saver MCP Proxy."""
    # Import here to avoid loading proxy dependencies at package import time
    import argparse
    import sys

    from src.proxy.proxy_server import ProxyConfig, ProxyServer

    parser = argparse.ArgumentParser(
        description="Token Saver MCP Proxy: compress any MCP server's responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", help="Upstream MCP server command")
    parser.add_argument("args", nargs="*", help="Upstream server arguments")
    parser.add_argument(
        "--provider", default="unknown", help="AI provider hint (anthropic, google, openai)"
    )
    parser.add_argument(
        "--schema-compression", action="store_true", help="Replace N tools with 3 meta-tools"
    )
    parser.add_argument(
        "--refiner-ratio", type=float, default=0.7, help="Token refiner keep ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--no-meta-tokens", action="store_true", help="Disable lossless meta-token compression"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument(
        "--verbose", action="store_true", help="Log per-call compression stats to stderr"
    )

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
        print("Token Saver MCP Proxy (dry-run)")
        print(f"  Upstream: {config.upstream_command} {' '.join(config.upstream_args)}")
        print(f"  Provider: {config.provider}")
        print(f"  Schema compression: {config.enable_schema_compression}")
        print(f"  Refiner ratio: {config.refiner_ratio}")
        print(f"  Meta-tokens: {config.enable_meta_tokens}")
        return 0

    proxy = ProxyServer(config)
    print(f"Token Saver MCP Proxy ready. Upstream: {config.upstream_command}", file=sys.stderr)

    try:
        import asyncio
        import json

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool

        from src.savings_tracker import SavingsTracker

        tracker = SavingsTracker(session_id="proxy", model=args.provider or "default")
        verbose = args.verbose

        async def run_proxy():
            server_params = StdioServerParameters(
                command=config.upstream_command,
                args=config.upstream_args,
                env=config.upstream_env,
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    upstream_tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": t.inputSchema or {},
                        }
                        for t in tools_result.tools
                    ]
                    proxy.setup_schema_compression(upstream_tools)
                    mcp_server = Server("token-saver-proxy")

                    @mcp_server.list_tools()
                    async def handle_list_tools():
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
                    async def handle_call_tool(name: str, arguments: dict | None):
                        arguments = arguments or {}
                        meta_result = proxy.handle_meta_tool_call(name, arguments)
                        if meta_result:
                            parsed = json.loads(meta_result)
                            if parsed.get("_invoke_upstream"):
                                name = parsed["tool_name"]
                                arguments = parsed["arguments"]
                            else:
                                return [TextContent(type="text", text=meta_result)]
                        result = await session.call_tool(name, arguments)
                        compressed_content = []
                        for item in result.content:
                            if hasattr(item, "text") and item.text:
                                compressed_text, stats = proxy.process_tool_result(name, item.text)
                                orig_tokens = stats.get("original_tokens_estimate") or (
                                    stats.get("tokens_saved_estimate", 0)
                                    + len(compressed_text) // 4
                                )
                                comp_tokens = len(compressed_text) // 4
                                if orig_tokens > comp_tokens:
                                    tracker.record(
                                        tool_name=name,
                                        original_tokens=orig_tokens,
                                        compressed_tokens=comp_tokens,
                                    )
                                if verbose:
                                    pct = stats.get("savings_pct", 0)
                                    saved = stats.get("tokens_saved_estimate", 0)
                                    stages = ",".join(stats.get("pipeline_stages", []))
                                    print(
                                        f"[proxy] {name}: {saved} tokens saved "
                                        f"({pct:.0f}%) [{stages or 'passthrough'}]",
                                        file=sys.stderr,
                                    )
                                compressed_content.append(
                                    TextContent(type="text", text=compressed_text)
                                )
                            else:
                                compressed_content.append(item)
                        return compressed_content

                    async with stdio_server() as (srv_read, srv_write):
                        await mcp_server.run(
                            srv_read, srv_write, mcp_server.create_initialization_options()
                        )

        asyncio.run(run_proxy())
    except ImportError as e:
        print(f"MCP client not available: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Proxy error: {e}", file=sys.stderr)
        return 1
    finally:
        if "tracker" in dir() and "proxy" in dir() and proxy.metrics.total_calls > 0:
            report = tracker.get_report()
            m = proxy.metrics
            print(
                f"\n--- Proxy Session Summary ---\n"
                f"  Calls:         {m.total_calls}\n"
                f"  Tokens saved:  {m.total_tokens_saved:,}\n"
                f"  Compression:   {m.savings_pct:.0f}% average\n"
                f"  Cost saved:    ${report.total_dollars_saved:.4f}",
                file=sys.stderr,
            )
            if m.by_tool:
                print("  By tool:", file=sys.stderr)
                for tool, info in sorted(
                    m.by_tool.items(), key=lambda x: x[1]["tokens_saved"], reverse=True
                ):
                    print(
                        f"    {tool}: {info['calls']} calls, "
                        f"{info['tokens_saved']:,} tokens saved",
                        file=sys.stderr,
                    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
