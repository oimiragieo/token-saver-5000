"""MCP tool schemas must advertise tenant scope properties on all tools."""

from src.handlers.mcp_core.schemas_prompts_ace import ACE_TOOLS, PROMPT_TOOLS
from src.handlers.mcp_core.schemas_model_experiment import EXPERIMENT_TOOLS, MODEL_TOOLS
from src.handlers.mcp_core._constants import SCOPE_PROPERTIES

_SCOPE_KEYS = set(SCOPE_PROPERTIES.keys())


def _tool_properties(tool) -> dict:
    schema = tool.inputSchema or {}
    return schema.get("properties", {})


def test_ace_tools_have_scope_properties():
    for tool in ACE_TOOLS:
        props = _tool_properties(tool)
        assert _SCOPE_KEYS <= set(props.keys()), f"{tool.name} missing scope keys"


def test_prompt_tools_have_scope_properties():
    for tool in PROMPT_TOOLS:
        props = _tool_properties(tool)
        assert _SCOPE_KEYS <= set(props.keys()), f"{tool.name} missing scope keys"


def test_model_tools_have_scope_properties():
    for tool in MODEL_TOOLS:
        props = _tool_properties(tool)
        assert _SCOPE_KEYS <= set(props.keys()), f"{tool.name} missing scope keys"


def test_experiment_tools_have_scope_properties():
    for tool in EXPERIMENT_TOOLS:
        props = _tool_properties(tool)
        assert _SCOPE_KEYS <= set(props.keys()), f"{tool.name} missing scope keys"
