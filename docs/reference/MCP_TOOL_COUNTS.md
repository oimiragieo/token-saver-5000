# MCP Tool Counts

Single source of truth for the Token Saver 5000 MCP tool inventory.

| Field | Value |
|-------|-------|
| **Package** | `semantic-modulator` |
| **Version** | 0.11.0 |
| **Total tools** (`MCP_TOOL_PROFILE=full`) | **128** |
| **Core stable profile** | **7** essential tools |
| **Profiles** | `full` — all tools; `core_stable` — 7 essential tools only |

## Configuration

Set in `.env.local` or the process environment:

```bash
# full (default) — expose all 128 tools
MCP_TOOL_PROFILE=full

# core_stable — expose only the 7 essential compression tools
MCP_TOOL_PROFILE=core_stable
```

## Where tools are defined

| Location | Role |
|----------|------|
| `src/handlers/mcp_core/setup.py` | `setup_mcp_tools()` — concatenates schema modules, sorts, filters by profile |
| `src/handlers/mcp_core/dispatch.py` | `route_tool_call()` — dispatches by tool name |
| `src/handlers/mcp_core/schemas_*.py` | Per-category `Tool(...)` schema literals |
| `src/handlers/mcp_core/_constants.py` | `SCOPE_PROPERTIES`, `CORE_STABLE_TOOL_NAMES` |

## Verification

```bash
python -c "from src.handlers.mcp_core import setup_mcp_tools; print(len(setup_mcp_tools('full')))"
python -c "from src.handlers.mcp_core import setup_mcp_tools; print(len(setup_mcp_tools('core_stable')))"
```

Expected: **128** for `full`, **7** for `core_stable`.

## Related docs

- [MCP Tools Guide](../guides/MCP_TOOLS_GUIDE.md) — usage reference by category
- [Root CLAUDE.md](../../CLAUDE.md) — architecture and handler routing
