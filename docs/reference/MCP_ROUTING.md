# MCP Routing

How MCP tool schemas are registered and dispatched to handler modules.

| Field | Value |
|-------|-------|
| **Package** | `semantic-modulator` |
| **Version** | 0.11.0 |
| **Tools (`full`)** | 128 |
| **Tools (`core_stable`)** | 7 |

## Request flow

```
MCP client → src/server.py (stdio or HTTP)
  → SemanticModulatorServer (src/semantic_modulator/app/)
  → route_tool_call() in src/handlers/mcp_core/dispatch.py
  → handler module (src/handlers/*_handlers.py)
  → JSON-serializable dict / string response
```

Bootstrap wiring: `ServerFactoryService.build_default()` → `bind_mcp_handlers()` in `router_binding.py`.

## Package map (`src/handlers/mcp_core/`)

| Module | Responsibility |
|--------|----------------|
| `setup.py` | `setup_mcp_tools(profile)` — merges schema lists, sorts, filters by profile |
| `dispatch.py` | `route_tool_call(name, args, context, tool_profile)` — router dict |
| `_constants.py` | `SCOPE_PROPERTIES`, `CORE_STABLE_TOOL_NAMES` |
| `_profile.py` | Profile normalization and filtering |
| `schemas_*.py` | Category `Tool(...)` literals (compression split into `schemas_compression_core.py` + `schemas_compression_batch.py`) |

## Profiles

| Profile | Env | Tools exposed |
|---------|-----|----------------|
| `full` | `MCP_TOOL_PROFILE=full` (default) | All 128 tools |
| `core_stable` | `MCP_TOOL_PROFILE=core_stable` | 7 essential compression tools |

See [MCP_TOOL_COUNTS.md](./MCP_TOOL_COUNTS.md) for verification commands.

## Handler context

Handlers receive `HandlerContext` (`src/types.py`) — a TypedDict with server components:

- `compressor`, `persistence`, `resource_manager`, validators, detectors, etc.

All handlers are `async def` and return JSON-serializable structures.

## Multi-tenant scoping

Optional args on every tool schema (via `SCOPE_PROPERTIES`):

- `workspace_id`, `user_id`, `agent_id`, `session_id`

Injected in schema modules; handlers use `identity_scope` helpers for file IDs.

## Compression handler split (2026-08-28)

| Module | Role |
|--------|------|
| `compression_handlers.py` | Re-export facade |
| `compression_handlers_common.py` | Shared helpers and constants |
| `compression_handlers_ingest.py` | ingest, read, search, manage |
| `compression_handlers_extended.py` | batch, directory, codebase ops |

## Tests

| Test | Asserts |
|------|---------|
| `tests/test_all_tools_have_handlers.py` | `setup_mcp_tools('full')` names == `dispatch.py` router keys |
| `tests/test_mcp_routing.py` | Routing, profiles, critical tools |
| `tests/test_mcp_scope_properties.py` | Scope fields on ACE/prompt/model schemas |

## Related

- [MCP_TOOL_COUNTS.md](./MCP_TOOL_COUNTS.md)
- [MCP Tools Guide](../guides/MCP_TOOLS_GUIDE.md)
- Design note: `docs/design/2026-08-22-mcp-core-split.md`
