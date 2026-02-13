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
