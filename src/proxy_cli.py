"""Entry point for token-saver-proxy CLI command."""

from __future__ import annotations


def _tool_to_dict(tool) -> dict:
    """Convert an MCP ``Tool`` object or dict, preserving precondition/effect metadata."""
    if isinstance(tool, dict):
        tool_dict = dict(tool)
        input_schema = tool_dict.get("inputSchema") or {}
    else:
        input_schema = getattr(tool, "inputSchema", None) or {}
        tool_dict = {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", None) or "",
            "inputSchema": input_schema,
        }
        model_extra = getattr(tool, "model_extra", None)
        if isinstance(model_extra, dict):
            for ext_key in ("x-preconditions", "x-effects"):
                if ext_key in model_extra and ext_key not in tool_dict:
                    tool_dict[ext_key] = model_extra[ext_key]
    if isinstance(input_schema, dict):
        for ext_key in ("x-preconditions", "x-effects"):
            if ext_key in input_schema and ext_key not in tool_dict:
                tool_dict[ext_key] = input_schema[ext_key]
    return tool_dict


def _dict_to_tool(tool_dict: dict, tool_cls):
    """Build an MCP ``Tool`` instance preserving top-level precondition/effect extensions."""
    extras = {
        ext_key: tool_dict[ext_key]
        for ext_key in ("x-preconditions", "x-effects")
        if ext_key in tool_dict
    }
    return tool_cls(
        name=tool_dict["name"],
        description=tool_dict.get("description", ""),
        inputSchema=tool_dict.get("inputSchema", {}),
        **extras,
    )


async def _dispatch_tool_call(
    proxy,
    session,
    name: str,
    arguments: dict,
    *,
    tracker=None,
    verbose: bool = False,
):
    """Dispatch a proxied tool call and preserve MCP ``CallToolResult`` semantics."""
    import json
    import sys

    from mcp.types import CallToolResult, TextContent

    via_internal_router = False
    meta_result = proxy.handle_meta_tool_call(name, arguments)
    if meta_result:
        parsed = json.loads(meta_result)
        if parsed.get("_invoke_upstream"):
            name = parsed.get("tool_name", "")
            arguments = parsed.get("arguments") or {}
            via_internal_router = True
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=meta_result)],
                isError=parsed.get("status") == "error",
            )

    gate = proxy.validate_call(name, arguments, via_internal_router=via_internal_router)
    if not gate["allowed"]:
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"status": "error", **gate}))],
            isError=True,
        )

    result = await session.call_tool(name, arguments)
    is_error = bool(getattr(result, "isError", False))
    structured = getattr(result, "structuredContent", None)
    result_meta = getattr(result, "meta", None)
    compressed_content = []
    text_processed = False

    for item in result.content:
        if hasattr(item, "text") and item.text:
            text_processed = True
            compressed_text, stats = proxy.process_tool_result(
                name,
                item.text,
                is_error=is_error,
                structured_content=structured,
            )
            orig_tokens = stats.get("original_tokens_estimate") or (
                stats.get("tokens_saved_estimate", 0) + len(compressed_text) // 4
            )
            comp_tokens = len(compressed_text) // 4
            if tracker is not None and orig_tokens > comp_tokens:
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
            if compressed_text == item.text:
                compressed_content.append(item)
            elif hasattr(item, "model_copy"):
                compressed_content.append(item.model_copy(update={"text": compressed_text}))
            else:
                compressed_content.append(
                    TextContent(
                        type="text",
                        text=compressed_text,
                        annotations=getattr(item, "annotations", None),
                        _meta=getattr(item, "meta", None),
                    )
                )
        else:
            compressed_content.append(item)

    if not text_processed:
        proxy.commit_tool_effects(name, succeeded=not is_error)

    if hasattr(result, "model_copy"):
        return result.model_copy(update={"content": compressed_content})
    return CallToolResult(
        content=compressed_content,
        structuredContent=structured,
        isError=is_error,
        _meta=result_meta,
    )


def main() -> int:
    """Run the Token Saver MCP Proxy."""
    # Import here to avoid loading proxy dependencies at package import time
    import argparse
    import sys

    from src.proxy.proxy_server import ProxyConfig, ProxyServer
    from src.proxy.schema_levels import parse_schema_level

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
        "--schema-level",
        default=None,
        help=(
            "Schema compression level: L0_LOSSLESS, L1_SUMMARY, L2_LAZY, "
            "L3_ADAPTIVE_PINNED, or DUAL_PATH_ROUTER"
        ),
    )
    parser.add_argument(
        "--max-schema-tokens",
        type=int,
        default=4000,
        help="Maximum token budget for pinned/filtered tool schemas (default: 4000)",
    )
    parser.add_argument(
        "--pin-tools",
        default="",
        help="Comma-separated tool names to pin in full schema (in addition to core tools)",
    )
    parser.add_argument(
        "--blacklist",
        default="",
        help="Comma-separated tool names to exclude from routing/discovery",
    )
    parser.add_argument(
        "--whitelist",
        default="",
        help="Comma-separated tool names to allow (when set, non-core tools outside list are excluded)",
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

    schema_level = parse_schema_level(args.schema_level) if args.schema_level else None
    pin_tools = [t.strip() for t in args.pin_tools.split(",") if t.strip()]
    blacklist = [t.strip() for t in args.blacklist.split(",") if t.strip()]
    whitelist = [t.strip() for t in args.whitelist.split(",") if t.strip()]

    config = ProxyConfig(
        upstream_command=args.command,
        upstream_args=args.args or [],
        provider=args.provider,
        enable_schema_compression=args.schema_compression,
        schema_level=schema_level,
        max_schema_tokens=args.max_schema_tokens,
        pin_tools=pin_tools,
        blacklist=blacklist,
        whitelist=whitelist,
        refiner_ratio=args.refiner_ratio,
        enable_meta_tokens=not args.no_meta_tokens,
    )

    if args.dry_run:
        effective = config.effective_schema_level()
        print("Token Saver MCP Proxy (dry-run)")
        print(f"  Upstream: {config.upstream_command} {' '.join(config.upstream_args)}")
        print(f"  Provider: {config.provider}")
        print(f"  Schema compression: {config.enable_schema_compression}")
        print(f"  Schema level: {effective.value if effective else 'none'}")
        print(f"  Max schema tokens: {config.max_schema_tokens}")
        print(f"  Pinned tools: {','.join(config.pin_tools) or 'core-only'}")
        print(f"  Blacklist: {','.join(config.blacklist) or 'none'}")
        print(f"  Whitelist: {','.join(config.whitelist) or 'all'}")
        print(f"  Refiner ratio: {config.refiner_ratio}")
        print(f"  Meta-tokens: {config.enable_meta_tokens}")
        return 0

    proxy = ProxyServer(config)
    print(f"Token Saver MCP Proxy ready. Upstream: {config.upstream_command}", file=sys.stderr)

    try:
        import asyncio

        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool

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
                    upstream_tools = [_tool_to_dict(t) for t in tools_result.tools]
                    proxy.setup_schema_compression(upstream_tools)
                    mcp_server = Server("token-saver-proxy")

                    @mcp_server.list_tools()
                    async def handle_list_tools():
                        processed = proxy.get_tools(upstream_tools)
                        return [_dict_to_tool(t, Tool) for t in processed]

                    @mcp_server.call_tool()
                    async def handle_call_tool(name: str, arguments: dict | None):
                        return await _dispatch_tool_call(
                            proxy,
                            session,
                            name,
                            arguments or {},
                            tracker=tracker,
                            verbose=verbose,
                        )

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
