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


def test_allowed_factory_override_keys_include_server_and_app_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_aliases")

    assert set(module.SERVER_ALIAS_KEYS).issubset(module.ALLOWED_FACTORY_OVERRIDE_KEYS)
    assert set(module.APP_FACTORY_ONLY_KEYS).issubset(module.ALLOWED_FACTORY_OVERRIDE_KEYS)


def test_validate_override_keys_rejects_unknown_entries():
    module = importlib.import_module("src.semantic_modulator.app.server_aliases")

    with pytest.raises(ValueError) as exc_info:
        module.validate_override_keys(
            overrides={"unknown_key": object()},
            allowed_keys=module.ALLOWED_FACTORY_OVERRIDE_KEYS,
        )

    assert "unknown_key" in str(exc_info.value)


def test_validate_override_keys_accepts_none_and_known_entries():
    module = importlib.import_module("src.semantic_modulator.app.server_aliases")

    module.validate_override_keys(
        overrides=None,
        allowed_keys=module.ALLOWED_FACTORY_OVERRIDE_KEYS,
    )
    module.validate_override_keys(
        overrides={"CodeCompressionAdapter": object()},
        allowed_keys=module.ALLOWED_FACTORY_OVERRIDE_KEYS,
    )
