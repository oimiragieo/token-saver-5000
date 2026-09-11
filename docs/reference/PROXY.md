# MCP Proxy

Transparent MCP proxy that compresses upstream tool results before they reach the client.

| Field | Value |
|-------|-------|
| **Package** | `semantic-modulator` |
| **Version** | 0.11.0 |

## Layout

| Module | Role |
|--------|------|
| `src/proxy/proxy_server.py` | `ProxyServer`, `ProxyConfig`, session metrics |
| `src/proxy/upstream_client.py` | Subprocess upstream MCP client |
| `src/proxy/schema_compressor.py` | Optional meta-tool schema compression |
| `src/proxy/response_interceptor.py` | Tool-result refiner / meta-token stage |
| `scripts/token_saver_proxy.py` | stdio run loop entry point |

## ProxyConfig

| Field | Default | Meaning |
|-------|---------|---------|
| `upstream_command` | — | Executable for upstream MCP server |
| `upstream_args` | `[]` | Arguments for upstream |
| `enable_schema_compression` | `False` | Expose 3 meta-tools instead of full upstream list |
| `refiner_ratio` | `0.7` | Fraction of tokens to keep per tool result |
| `enable_meta_tokens` | `True` | Run meta-token compression stage |
| `preserve_identifiers` | `False` | Keep paths, symbols, URLs in compressed results (default OFF) |
| `provider` | `unknown` | Provider hint for future optimisations |

## Flow

```
MCP client → ProxyServer (stdio) → upstream MCP subprocess
                ↓
         ResponseInterceptor compresses tool results
                ↓
         (optional) SchemaCompressor shrinks tools/list
```

`ProxyServer` is unit-testable without a live subprocess. The async stdio loop lives in `scripts/token_saver_proxy.py`.

## Metrics

`ProxySessionMetrics` tracks per-session:

- `total_calls`, `total_original_chars`, `total_compressed_chars`
- `total_tokens_saved`, `by_tool` breakdown
- `savings_pct` property

## Usage

```bash
python scripts/token_saver_proxy.py --help
token-saver-proxy --upstream-command python --upstream-arg -m --upstream-arg src.server
```

See `scripts/token_saver_proxy.py` for full flags and env overrides.

## Related

- [CLI output optimizer](../../src/cli_output_optimizer.py) — command-output filtering (separate from proxy)
- [Filter Rules DSL](../guides/FILTER_RULES_DSL.md)
