"""Count MCP tools in schema modules vs setup_mcp_tools()."""

import re
from pathlib import Path

from src.handlers.mcp_core import setup_mcp_tools
from src.handlers.mcp_core.dispatch import route_tool_call  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
dispatch_text = (ROOT / "src/handlers/mcp_core/dispatch.py").read_text(encoding="utf-8")
router_keys = re.findall(r'"([a-z_][a-z0-9_]*)":\s+\w+\.handle', dispatch_text)

schema_names: list[str] = []
for path in sorted((ROOT / "src/handlers/mcp_core").glob("schemas_*.py")):
    for match in re.finditer(r'name="([a-z_][a-z0-9_]*)"', path.read_text(encoding="utf-8")):
        schema_names.append(match.group(1))

tools = setup_mcp_tools()
setup_names = [t.name for t in tools]

print(f"setup_mcp_tools (full): {len(tools)}")
print(f"router keys: {len(router_keys)}")
print(f"schema name= entries: {len(schema_names)}")
print("schema not in setup:", sorted(set(schema_names) - set(setup_names)))
print("setup not in schema:", sorted(set(setup_names) - set(schema_names)))
print("router not in setup:", sorted(set(router_keys) - set(setup_names)))
