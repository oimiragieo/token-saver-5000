"""Contract tests for app bootstrap entrypoints."""

from __future__ import annotations

import importlib

import pytest


def test_bootstrap_create_server_returns_server_instance():
    bootstrap = importlib.import_module("src.semantic_modulator.app.bootstrap")
    server = bootstrap.create_server()
    assert server.__class__.__name__ == "SemanticModulatorServer"


@pytest.mark.asyncio
async def test_bootstrap_async_main_uses_lifespan_and_run():
    bootstrap = importlib.import_module("src.semantic_modulator.app.bootstrap")
    calls: list[str] = []

    class FakeServer:
        async def __aenter__(self):
            calls.append("enter")
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            calls.append("exit")
            return False

        async def run(self):
            calls.append("run")

    await bootstrap.async_main(create_server_fn=lambda: FakeServer())
    assert calls == ["enter", "run", "exit"]


def test_bootstrap_main_delegates_to_run_function():
    bootstrap = importlib.import_module("src.semantic_modulator.app.bootstrap")
    captured = {"called": False}

    async def fake_async_main():
        return None

    def fake_run(coro):
        captured["called"] = True
        assert hasattr(coro, "__await__")
        coro.close()

    bootstrap.main(async_main_fn=fake_async_main, run_fn=fake_run)
    assert captured["called"] is True
