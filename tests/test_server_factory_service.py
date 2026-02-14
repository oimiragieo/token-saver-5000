"""Contract tests for app-layer server factory service."""

from __future__ import annotations

import importlib
import pytest
from unittest.mock import Mock


def test_factory_build_creates_core_components_and_services():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")
    service = module.ServerFactoryService()

    def make(name):
        def _factory(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return _factory

    components = service.build(
        preload_code_model=False,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=100,
        code_adapter_cls=make("compressor"),
        blind_spot_cls=make("blind"),
        halo_cls=make("halo"),
        context_window_adapter_cls=make("cwa"),
        multilevel_encoder_cls=make("mle"),
        afm_config_cls=make("afm_config"),
        focus_manager_cls=make("focus"),
        persistence_cls=make("persistence"),
        resource_limits_cls=make("resource_limits"),
        resource_manager_cls=make("resource_manager"),
        file_sync_cls=make("sync"),
        version_manager_cls=make("version"),
        path_validator_cls=make("path_validator"),
        ace_framework_cls=make("ace"),
        ace_context_manager_cls=make("ace_contexts"),
        tooling_gateway_cls=make("tooling"),
        context_service_cls=make("context_service"),
        lifecycle_service_cls=make("lifecycle_service"),
        progress_service_cls=make("progress_service"),
        persistence_service_cls=make("persistence_service"),
        tool_profile_service_cls=make("tool_profile_service"),
        runtime_service_cls=make("runtime_service"),
        server_service_adapter_cls=make("service_adapter"),
        logger=Mock(),
    )

    assert "compressor" in components
    assert "resource_manager" in components
    assert "path_validator" in components
    assert "tooling" in components
    assert "runtime_service" in components
    assert "service_adapter" in components
    assert "context_window_monitor" in components
    assert components["retrieval_history"] == {}


def test_factory_build_uses_cwd_and_home_for_path_validator():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")
    service = module.ServerFactoryService()

    captured = {}

    def path_validator_cls(*, allowed_base_dirs):
        captured["dirs"] = allowed_base_dirs
        return {"ok": True}

    def noop(*args, **kwargs):
        return Mock()

    service.build(
        preload_code_model=True,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=100,
        code_adapter_cls=noop,
        blind_spot_cls=noop,
        halo_cls=noop,
        context_window_adapter_cls=noop,
        multilevel_encoder_cls=noop,
        afm_config_cls=noop,
        focus_manager_cls=noop,
        persistence_cls=noop,
        resource_limits_cls=noop,
        resource_manager_cls=noop,
        file_sync_cls=noop,
        version_manager_cls=noop,
        path_validator_cls=path_validator_cls,
        ace_framework_cls=noop,
        ace_context_manager_cls=noop,
        tooling_gateway_cls=noop,
        context_service_cls=noop,
        lifecycle_service_cls=noop,
        progress_service_cls=noop,
        persistence_service_cls=noop,
        tool_profile_service_cls=noop,
        runtime_service_cls=noop,
        server_service_adapter_cls=noop,
        logger=Mock(),
    )

    assert captured["dirs"] == ["C:/repo", "C:/Users/test"]


def test_factory_build_default_delegates_to_build_with_production_wiring():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")
    service = module.ServerFactoryService()

    sentinel = {"ok": True}
    original_build = module.ServerFactoryService.build
    call_kwargs = {}

    def fake_build(**kwargs):
        call_kwargs.update(kwargs)
        return sentinel

    module.ServerFactoryService.build = staticmethod(fake_build)
    try:
        result = service.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=77,
            logger=Mock(),
        )
    finally:
        module.ServerFactoryService.build = original_build

    assert result is sentinel
    assert call_kwargs["preload_code_model"] is True
    assert call_kwargs["cwd"] == "C:/repo"
    assert call_kwargs["home_dir"] == "C:/Users/test"
    assert call_kwargs["max_ace_contexts"] == 77
    assert callable(call_kwargs["tooling_gateway_cls"])
    assert callable(call_kwargs["runtime_service_cls"])
    assert callable(call_kwargs["server_service_adapter_cls"])


def test_factory_build_default_rejects_unknown_override_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")
    service = module.ServerFactoryService()

    with pytest.raises(ValueError) as exc_info:
        service.build_default(
            preload_code_model=False,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=5,
            logger=Mock(),
            class_overrides={"NotARealKey": object()},
        )

    assert "NotARealKey" in str(exc_info.value)
