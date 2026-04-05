# Folder guide: `src/proxy/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Token Saver MCP Proxy: transparent compression for any MCP server.

_No top-level classes or functions (may re-export only)._

#### `proxy_server.py`

MCP proxy server: wraps an upstream MCP server with transparent compression.

| Kind | Name |
|------|------|
| `class` | `ProxyConfig` |
| `class` | `ProxySessionMetrics` |
| `class` | `ProxyServer` |

#### `response_interceptor.py`

Applies Token Saver compression pipeline to MCP tool responses.

| Kind | Name |
|------|------|
| `class` | `InterceptionStats` |
| `class` | `ResponseInterceptor` |

#### `schema_compressor.py`

Meta-tool pattern: replaces N upstream tools with 3 search/inspect/invoke meta-tools.

| Kind | Name |
|------|------|
| `class` | `ToolEntry` |
| `class` | `ToolIndex` |
| `class` | `SchemaCompressor` |

#### `upstream_client.py`

Upstream MCP server client: connects to any MCP server as a child process.

| Kind | Name |
|------|------|
| `class` | `UpstreamClient` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
