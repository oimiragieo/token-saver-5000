"""Contract tests for server alias mapping helpers."""

from __future__ import annotations

import importlib

import pytest


def test_build_server_class_overrides_collects_required_aliases():
    module = importlib.import_module("src.semantic_modulator.app.server_aliases")

    namespace = {name: object() for name in module.SERVER_ALIAS_KEYS}
    overrides = module.build_server_class_overrides(namespace)

    assert set(overrides.keys()) == set(module.SERVER_ALIAS_KEYS)
    for key in module.SERVER_ALIAS_KEYS:
        assert overrides[key] is namespace[key]


def test_build_server_class_overrides_missing_alias_raises_helpful_error():
    module = importlib.import_module("src.semantic_modulator.app.server_aliases")

    namespace = {
        name: object() for name in module.SERVER_ALIAS_KEYS if name != "CodeCompressionAdapter"
    }

    with pytest.raises(KeyError) as exc_info:
        module.build_server_class_overrides(namespace)

    assert "CodeCompressionAdapter" in str(exc_info.value)
