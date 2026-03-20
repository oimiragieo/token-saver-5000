"""Contract tests for app-layer MCP router binding helper."""

from __future__ import annotations

import importlib

import pytest


class FakeServer:
    def __init__(self):
        self.list_tools_handler = None
        self.call_tool_handler = None
        self.list_prompts_handler = None
        self.get_prompt_handler = None
        self.list_resources_handler = None
        self.list_resource_templates_handler = None
        self.read_resource_handler = None

    def list_tools(self):
        def decorator(fn):
            self.list_tools_handler = fn
            return fn

        return decorator

    def call_tool(self):
        def decorator(fn):
            self.call_tool_handler = fn
            return fn

        return decorator

    def list_prompts(self):
        def decorator(fn):
            self.list_prompts_handler = fn
            return fn

        return decorator

    def get_prompt(self):
        def decorator(fn):
            self.get_prompt_handler = fn
            return fn

        return decorator

    def list_resources(self):
        def decorator(fn):
            self.list_resources_handler = fn
            return fn

        return decorator

    def list_resource_templates(self):
        def decorator(fn):
            self.list_resource_templates_handler = fn
            return fn

        return decorator

    def read_resource(self):
        def decorator(fn):
            self.read_resource_handler = fn
            return fn

        return decorator


def test_router_binding_registers_list_and_call_handlers():
    module = importlib.import_module("src.semantic_modulator.app.router_binding")
    fake_server = FakeServer()

    tools = [type("T", (), {"name": "ingest_context"})()]

    class Tooling:
        def list_tools(self, profile):
            assert profile == "core_stable"
            return tools

        def list_prompts(self):
            return ["p1"]

        def get_prompt(self, name, arguments):
            return {"name": name, "arguments": arguments}

        def list_resources(self, profile):
            assert profile == "core_stable"
            return ["r1"]

        def list_resource_templates(self):
            return ["rt1"]

        async def read_resource(self, uri, context, profile):
            assert profile == "core_stable"
            return [{"uri": uri, "context": context}]

    module.bind_mcp_handlers(
        server=fake_server,
        tooling=Tooling(),
        tool_profile="core_stable",
        build_context=lambda: {"ok": True},
        logger=None,
        text_content_cls=lambda **kwargs: kwargs,
    )

    assert fake_server.list_tools_handler is not None
    assert fake_server.call_tool_handler is not None
    assert fake_server.list_prompts_handler is not None
    assert fake_server.get_prompt_handler is not None
    assert fake_server.list_resources_handler is not None
    assert fake_server.list_resource_templates_handler is not None
    assert fake_server.read_resource_handler is not None


@pytest.mark.asyncio
async def test_router_binding_call_handler_wraps_result_and_errors():
    module = importlib.import_module("src.semantic_modulator.app.router_binding")
    fake_server = FakeServer()
    events = {"error_logged": False}

    class Tooling:
        def list_tools(self, profile):
            return []

        def list_prompts(self):
            return []

        def get_prompt(self, name, arguments):
            return {"name": name, "arguments": arguments}

        def list_resources(self, profile):
            return []

        def list_resource_templates(self):
            return []

        async def read_resource(self, uri, context, profile):
            return [{"uri": uri}]

        async def route_tool_call(self, name, arguments, context, tool_profile):
            if name == "boom":
                raise RuntimeError("kaboom")
            return {"ok": True, "name": name}

    class Logger:
        def error(self, *args, **kwargs):
            events["error_logged"] = True

    module.bind_mcp_handlers(
        server=fake_server,
        tooling=Tooling(),
        tool_profile="full",
        build_context=lambda: {"ctx": 1},
        logger=Logger(),
        text_content_cls=lambda **kwargs: kwargs,
    )

    ok = await fake_server.call_tool_handler("ingest_context", {})
    err = await fake_server.call_tool_handler("boom", {})
    prompt = await fake_server.get_prompt_handler("document_compression_workflow", {"goal": "x"})
    resource = await fake_server.read_resource_handler("token-saver://catalog/tools")

    assert ok[0]["text"].startswith("{'ok': True")
    assert "Error: kaboom" in err[0]["text"]
    assert prompt["name"] == "document_compression_workflow"
    assert resource[0]["uri"] == "token-saver://catalog/tools"
    assert events["error_logged"] is True


def test_router_binding_request_contract_declared():
    module = importlib.import_module("src.semantic_modulator.app.router_binding")
    assert module.BIND_REQUEST_KEYS == frozenset(
        {"server", "tooling", "tool_profile", "build_context", "logger", "text_content_cls"}
    )


def test_router_binding_validate_bind_request_map_rejects_extra_key():
    module = importlib.import_module("src.semantic_modulator.app.router_binding")
    with pytest.raises(ValueError, match="bind_request_map keys mismatch"):
        module.validate_bind_request_map(
            {
                "server": FakeServer(),
                "tooling": object(),
                "tool_profile": "full",
                "build_context": lambda: {},
                "logger": None,
                "text_content_cls": lambda **kwargs: kwargs,
                "extra": True,
            }
        )
