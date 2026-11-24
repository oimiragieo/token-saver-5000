# Folder guide: `src/proxy/`

Breadcrumb for AI navigation. Master index: [`CLAUDE.md`](../../CLAUDE.md).

## Contents

### Python modules

#### `__init__.py`

Token Saver MCP Proxy: transparent compression for any MCP server.

_No top-level classes or functions (may re-export only)._

#### `cache_router.py`

Dual-Path CacheRouter with 4-stage filtering, ISO/BM25 relevance, and state gating.

| Kind | Name |
|------|------|
| `def` | `_tokenize` |
| `def` | `_tool_document_tokens` |
| `class` | `CacheRouter` |

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

Schema compression (L0-L3) and <0.5ms pre-binding argument validation.

| Kind | Name |
|------|------|
| `def` | `_normalize_ws` |
| `def` | `_is_redundant_title` |
| `def` | `_clean_schema_node` |
| `def` | `compact_schema_l0` |
| `def` | `summarize_description_l1` |
| `def` | `compact_schema_l1` |
| `def` | `_estimate_tool_tokens` |
| `def` | `_matches_json_type` |
| `def` | `_enum_contains` |
| `class` | `_ValidationState` |
| `def` | `_validate_node` |
| `def` | `validate_tool_arguments` |
| `class` | `ToolEntry` |
| `class` | `ToolIndex` |
| `class` | `SchemaCompressor` |

#### `schema_levels.py`

Schema compression levels and routing policy contracts for the MCP proxy.

| Kind | Name |
|------|------|
| `class` | `SchemaCompressionLevel` |
| `def` | `parse_schema_level` |
| `def` | `normalize_tool_name` |
| `class` | `RouterPolicy` |

#### `upstream_client.py`

Upstream MCP server client: connects to any MCP server as a child process.

| Kind | Name |
|------|------|
| `class` | `UpstreamClient` |

---

Symbols are **top-level only** (nested methods and inner functions are not listed). Regenerate: `python scripts/generate_claude_folder_guides.py`.
