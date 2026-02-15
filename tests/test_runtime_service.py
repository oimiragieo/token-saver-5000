"""Contract tests for app-layer runtime execution service."""

from __future__ import annotations

import importlib

import pytest


class _FakeCtx:
    def __init__(self, streams):
        self._streams = streams

    async def __aenter__(self):
        return self._streams

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


@pytest.mark.asyncio
async def test_runtime_service_runs_server_with_stdio_streams():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")
    service = module.RuntimeService()

    events = {}

    class FakeServer:
        def create_initialization_options(self):
            return {"init": True}

        async def run(self, read_stream, write_stream, options):
            events["args"] = (read_stream, write_stream, options)

    def fake_stdio_server():
        return _FakeCtx(("r", "w"))

    await service.run(server=FakeServer(), logger=None, stdio_server_fn=fake_stdio_server)
    assert events["args"] == ("r", "w", {"init": True})


@pytest.mark.asyncio
async def test_runtime_service_logs_startup_event():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")
    service = module.RuntimeService()

    class Logger:
        def __init__(self):
            self.called = False

        def info(self, *args, **kwargs):
            if args and args[0] == "mcp_server_starting":
                self.called = True

    class FakeServer:
        def create_initialization_options(self):
            return {}

        async def run(self, read_stream, write_stream, options):
            return None

    def fake_stdio_server():
        return _FakeCtx((None, None))

    logger = Logger()
    await service.run(server=FakeServer(), logger=logger, stdio_server_fn=fake_stdio_server)
    assert logger.called is True


def test_runtime_service_run_request_contract_declared():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")

    assert hasattr(module, "RunRequest")
    assert set(module.RunRequest.__annotations__.keys()) == {
        "server",
        "logger",
        "stdio_server_fn",
    }


def test_runtime_service_contract_key_mismatch_message_format():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")

    message = module.RuntimeService.contract_key_mismatch_message(
        contract_name="run_request_map",
        missing=["server"],
        extra=["extra"],
    )
    assert message == "run_request_map keys mismatch: missing=['server'] extra=['extra']"


def test_runtime_service_validate_run_request_map_rejects_extra_key():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")

    bad = {
        "server": object(),
        "logger": object(),
        "stdio_server_fn": lambda: None,
        "extra": True,
    }

    with pytest.raises(ValueError) as exc_info:
        module.RuntimeService.validate_run_request_map(bad)
    assert str(exc_info.value) == "run_request_map keys mismatch: missing=[] extra=['extra']"


@pytest.mark.asyncio
async def test_runtime_service_run_uses_validate_run_request_map_class_dispatch():
    module = importlib.import_module("src.semantic_modulator.app.runtime_service")

    calls = []

    class DerivedService(module.RuntimeService):
        @classmethod
        def validate_run_request_map(cls, request):
            calls.append("validate_run_request_map")
            return request

    class FakeServer:
        def create_initialization_options(self):
            return {}

        async def run(self, read_stream, write_stream, options):
            return None

    def fake_stdio_server():
        return _FakeCtx((None, None))

    await DerivedService.run(
        server=FakeServer(),
        logger=None,
        stdio_server_fn=fake_stdio_server,
    )

    assert calls == ["validate_run_request_map"]
