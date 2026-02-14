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
