"""Compatibility tests for server runtime delegation wrapper."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from src.server import SemanticModulatorServer


@pytest.mark.asyncio
async def test_server_run_delegates_to_runtime_service():
    with patch.object(SemanticModulatorServer, "_setup_handlers"):
        server = SemanticModulatorServer()

    called = {}

    async def fake_run(*, server: object, logger: object, stdio_server_fn: object):
        called["server"] = server
        called["logger"] = logger
        called["stdio_server_fn"] = stdio_server_fn

    server.runtime_service.run = fake_run
    await server.run()

    assert called["server"] is server.server
    assert called["logger"] is not None
    assert callable(called["stdio_server_fn"])
