import re
from pathlib import Path

from src.handlers.mcp_core import setup_mcp_tools

tools = setup_mcp_tools()
text = Path("src/handlers/mcp_core/dispatch.py").read_text(encoding="utf-8")
keys = re.findall(r'"([a-z_][a-z0-9_]*)":\s+\w+\.handle', text)
schemas = sum(
    len(re.findall(r"\bTool\s*\(", f.read_text(encoding="utf-8")))
    for f in Path("src/handlers/mcp_core").glob("schemas_*.py")
)
print("setup_mcp_tools:", len(tools))
print("dispatch router keys:", len(keys))
print("Tool( in schemas:", schemas)
tool_names = {t.name for t in tools}
router_names = set(keys)
print("in setup not router:", sorted(tool_names - router_names))
print("in router not setup:", sorted(router_names - tool_names))
