"""Contract tests for ACE context manager module extraction."""

from __future__ import annotations

import importlib


def test_ace_context_manager_module_exports_class():
    module = importlib.import_module("src.semantic_modulator.app.ace_context_manager")
    assert hasattr(module, "ACEContextManager")


def test_server_reexports_same_ace_context_manager_class():
    server_module = importlib.import_module("src.server")
    ace_module = importlib.import_module("src.semantic_modulator.app.ace_context_manager")
    assert server_module.ACEContextManager is ace_module.ACEContextManager
