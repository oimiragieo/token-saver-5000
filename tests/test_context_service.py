"""Contract tests for app-layer server context service."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


class _Node:
    def __init__(self, text: str):
        self.text = text


def test_context_service_build_context_includes_contract_keys():
    module = importlib.import_module("src.semantic_modulator.app.context_service")
    service = module.ServerContextService()

    context = service.build_context(
        compressor=SimpleNamespace(chunks={}),
        blind_spot_detector=object(),
        halo_detector=object(),
        context_window_adapter=object(),
        multilevel_encoder=object(),
        focus_manager=object(),
        persistence=object(),
        resource_manager=object(),
        sync_manager=object(),
        version_manager=object(),
        path_validator=object(),
        ace_framework=object(),
        ace_contexts=object(),
        validate_file_id=lambda *_: None,
        validate_node_ids=lambda *_: None,
        validate_token_count=lambda *_: None,
        save_file_sync_metadata=lambda: None,
        tool_profile="core_stable",
        enabled_tool_names=["ingest_context"],
    )

    assert context["tool_profile"] == "core_stable"
    assert context["enabled_tool_names"] == ["ingest_context"]
    assert callable(context["validate_file_id"])


def test_context_service_contract_key_constants_align_with_expected_context_map():
    module = importlib.import_module("src.semantic_modulator.app.context_service")

    expected = {
        "compressor",
        "blind_spot_detector",
        "halo_detector",
        "context_window_adapter",
        "multilevel_encoder",
        "focus_manager",
        "persistence",
        "resource_manager",
        "sync_manager",
        "version_manager",
        "path_validator",
        "ace_framework",
        "ace_contexts",
        "validate_file_id",
        "validate_node_ids",
        "validate_token_count",
        "save_file_sync_metadata",
        "tool_profile",
        "enabled_tool_names",
    }
    assert module.ServerContextService.CONTEXT_MAP_KEYS == frozenset(expected)


def test_context_service_contract_key_mismatch_message_format():
    module = importlib.import_module("src.semantic_modulator.app.context_service")

    message = module.ServerContextService.contract_key_mismatch_message(
        contract_name="context_map",
        missing=["compressor"],
        extra=["extra"],
    )

    assert message == "context_map keys mismatch: missing=['compressor'] extra=['extra']"


def test_context_service_validate_context_map_rejects_extra_keys_with_exact_message():
    module = importlib.import_module("src.semantic_modulator.app.context_service")

    service = module.ServerContextService()
    payload = {
        "compressor": object(),
        "blind_spot_detector": object(),
        "halo_detector": object(),
        "context_window_adapter": object(),
        "multilevel_encoder": object(),
        "focus_manager": object(),
        "persistence": object(),
        "resource_manager": object(),
        "sync_manager": object(),
        "version_manager": object(),
        "path_validator": object(),
        "ace_framework": object(),
        "ace_contexts": object(),
        "validate_file_id": lambda *_: None,
        "validate_node_ids": lambda *_: None,
        "validate_token_count": lambda *_: None,
        "save_file_sync_metadata": lambda: None,
        "tool_profile": "core_stable",
        "enabled_tool_names": [],
        "extra": True,
    }

    with pytest.raises(ValueError) as exc_info:
        service.validate_context_map(payload)

    assert str(exc_info.value) == "context_map keys mismatch: missing=[] extra=['extra']"


def test_context_service_build_context_uses_validate_context_map_class_dispatch():
    module = importlib.import_module("src.semantic_modulator.app.context_service")

    calls = []

    class DerivedService(module.ServerContextService):
        @classmethod
        def validate_context_map(cls, context):
            calls.append("validate_context_map")
            return context

    service = DerivedService()
    context = service.build_context(
        compressor=SimpleNamespace(chunks={}),
        blind_spot_detector=object(),
        halo_detector=object(),
        context_window_adapter=object(),
        multilevel_encoder=object(),
        focus_manager=object(),
        persistence=object(),
        resource_manager=object(),
        sync_manager=object(),
        version_manager=object(),
        path_validator=object(),
        ace_framework=object(),
        ace_contexts=object(),
        validate_file_id=lambda *_: None,
        validate_node_ids=lambda *_: None,
        validate_token_count=lambda *_: None,
        save_file_sync_metadata=lambda: None,
        tool_profile="core_stable",
        enabled_tool_names=["ingest_context"],
    )

    assert calls == ["validate_context_map"]
    assert context["tool_profile"] == "core_stable"


def test_context_service_validate_file_id_not_found_has_helpful_message():
    module = importlib.import_module("src.semantic_modulator.app.context_service")
    service = module.ServerContextService()
    compressor = SimpleNamespace(chunks={"doc_a_n0": _Node("x")})

    with pytest.raises(ValueError, match="Document 'missing' not found"):
        service.validate_file_id(compressor=compressor, file_id="missing", must_exist=True)


@pytest.mark.parametrize(
    "node_id,expected",
    [
        ("doc_n1", "doc"),
        ("src/file.py::ClassName", "src/file.py"),
    ],
)
def test_context_service_extract_file_id(node_id: str, expected: str):
    module = importlib.import_module("src.semantic_modulator.app.context_service")
    service = module.ServerContextService()

    assert service.extract_file_id_from_node(node_id) == expected
