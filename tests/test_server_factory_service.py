"""Contract tests for app-layer server factory service."""

from __future__ import annotations

import importlib
import pytest
from unittest.mock import Mock


def test_factory_build_creates_core_components_and_services():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    def make(name):
        def _factory(*args, **kwargs):
            return {"name": name, "args": args, "kwargs": kwargs}

        return _factory

    components = module.ServerFactoryService.build(
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

    captured = {}

    def path_validator_cls(*, allowed_base_dirs):
        captured["dirs"] = allowed_base_dirs
        return {"ok": True}

    def noop(*args, **kwargs):
        return Mock()

    module.ServerFactoryService.build(
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

    sentinel = {"ok": True}
    original_build = module.ServerFactoryService.__dict__["build"]
    call_kwargs = {}

    def fake_build(_cls, **kwargs):
        call_kwargs.update(kwargs)
        return sentinel

    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default(
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

    with pytest.raises(ValueError) as exc_info:
        module.ServerFactoryService.build_default(
            preload_code_model=False,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=5,
            logger=Mock(),
            class_overrides={"NotARealKey": object()},
        )

    assert "NotARealKey" in str(exc_info.value)


def test_resolve_class_overrides_merges_known_keys_and_defaults():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    defaults = {
        "CodeCompressionAdapter": object(),
        "FocusManager": object(),
    }
    override_focus = object()

    resolved = module.ServerFactoryService.resolve_class_overrides(
        defaults=defaults,
        overrides={"FocusManager": override_focus},
    )

    assert resolved["CodeCompressionAdapter"] is defaults["CodeCompressionAdapter"]
    assert resolved["FocusManager"] is override_focus


def test_default_class_map_covers_allowed_override_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    default_map = module.ServerFactoryService.default_class_map()

    assert set(default_map.keys()) == set(module.ALLOWED_OVERRIDE_KEYS)


def test_build_default_uses_resolve_class_overrides_result_for_build_wiring():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel_result = {"ok": True}
    resolved = {
        "CodeCompressionAdapter": object(),
        "BlindSpotDetector": object(),
        "HaloEffectDetector": object(),
        "ContextWindowAdapter": object(),
        "MultiLevelSemanticEncoder": object(),
        "AFMConfig": object(),
        "FocusManager": object(),
        "PersistenceManager": object(),
        "ResourceLimits": object(),
        "ResourceManager": object(),
        "FileSyncManager": object(),
        "VersionManager": object(),
        "PathValidator": object(),
        "ACEFramework": object(),
        "ACEContextManager": object(),
        "MCPToolingGateway": object(),
        "ServerContextService": object(),
        "ServerLifecycleService": object(),
        "ProgressRenderService": object(),
        "PersistenceOrchestrationService": object(),
        "ToolProfileBootstrapService": object(),
        "RuntimeService": object(),
        "ServerServiceAdapter": object(),
    }

    captured_resolve = {}
    captured_build = {}

    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_build = module.ServerFactoryService.__dict__["build"]

    def fake_resolve(*, defaults, overrides):
        captured_resolve["defaults"] = defaults
        captured_resolve["overrides"] = overrides
        return resolved

    def fake_build(_cls, **kwargs):
        captured_build.update(kwargs)
        return sentinel_result

    module.ServerFactoryService.resolve_class_overrides = staticmethod(fake_resolve)
    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=11,
            logger=Mock(),
            class_overrides={"FocusManager": object()},
        )
    finally:
        module.ServerFactoryService.resolve_class_overrides = original_resolve
        module.ServerFactoryService.build = original_build

    assert result is sentinel_result
    assert set(captured_resolve["defaults"].keys()) == set(module.ALLOWED_OVERRIDE_KEYS)
    assert "FocusManager" in captured_resolve["overrides"]
    assert captured_build["focus_manager_cls"] is resolved["FocusManager"]
    assert captured_build["server_service_adapter_cls"] is resolved["ServerServiceAdapter"]


def test_build_kwargs_from_resolved_classes_maps_expected_build_arguments():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    resolved = {
        "CodeCompressionAdapter": object(),
        "BlindSpotDetector": object(),
        "HaloEffectDetector": object(),
        "ContextWindowAdapter": object(),
        "MultiLevelSemanticEncoder": object(),
        "AFMConfig": object(),
        "FocusManager": object(),
        "PersistenceManager": object(),
        "ResourceLimits": object(),
        "ResourceManager": object(),
        "FileSyncManager": object(),
        "VersionManager": object(),
        "PathValidator": object(),
        "ACEFramework": object(),
        "ACEContextManager": object(),
        "MCPToolingGateway": object(),
        "ServerContextService": object(),
        "ServerLifecycleService": object(),
        "ProgressRenderService": object(),
        "PersistenceOrchestrationService": object(),
        "ToolProfileBootstrapService": object(),
        "RuntimeService": object(),
        "ServerServiceAdapter": object(),
    }

    kwargs = module.ServerFactoryService.build_kwargs_from_resolved_classes(resolved)

    assert kwargs["code_adapter_cls"] is resolved["CodeCompressionAdapter"]
    assert kwargs["path_validator_cls"] is resolved["PathValidator"]
    assert kwargs["server_service_adapter_cls"] is resolved["ServerServiceAdapter"]
    assert len(kwargs) == 23


def test_build_default_uses_build_kwargs_helper_for_build_invocation():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel_result = {"ok": True}
    helper_kwargs = {"code_adapter_cls": object()}

    original_default_map = module.ServerFactoryService.default_class_map
    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_helper = module.ServerFactoryService.build_kwargs_from_resolved_classes
    original_build = module.ServerFactoryService.__dict__["build"]

    captured = {}

    def fake_default_map():
        return {"CodeCompressionAdapter": object()}

    def fake_resolve(*, defaults, overrides):
        captured["defaults"] = defaults
        captured["overrides"] = overrides
        return {"CodeCompressionAdapter": object()}

    def fake_helper(resolved):
        captured["resolved"] = resolved
        return helper_kwargs

    def fake_build(_cls, **kwargs):
        captured["build_kwargs"] = kwargs
        return sentinel_result

    module.ServerFactoryService.default_class_map = staticmethod(fake_default_map)
    module.ServerFactoryService.resolve_class_overrides = staticmethod(fake_resolve)
    module.ServerFactoryService.build_kwargs_from_resolved_classes = staticmethod(fake_helper)
    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=9,
            logger=Mock(),
            class_overrides={"CodeCompressionAdapter": object()},
        )
    finally:
        module.ServerFactoryService.default_class_map = original_default_map
        module.ServerFactoryService.resolve_class_overrides = original_resolve
        module.ServerFactoryService.build_kwargs_from_resolved_classes = original_helper
        module.ServerFactoryService.build = original_build

    assert result is sentinel_result
    assert captured["build_kwargs"]["code_adapter_cls"] is helper_kwargs["code_adapter_cls"]
    assert captured["build_kwargs"]["preload_code_model"] is True
    assert captured["build_kwargs"]["cwd"] == "C:/repo"
    assert captured["build_kwargs"]["home_dir"] == "C:/Users/test"
    assert captured["build_kwargs"]["max_ace_contexts"] == 9


def test_build_helpers_expose_expected_default_configs():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    code_cfg = module.ServerFactoryService.code_adapter_config(preload_code_model=True)
    afm_cfg = module.ServerFactoryService.afm_config_kwargs()
    resource_cfg = module.ServerFactoryService.resource_limits_kwargs()
    ace_cfg = module.ServerFactoryService.ace_framework_kwargs()
    monitor = module.ServerFactoryService.default_context_window_monitor()

    assert code_cfg == {
        "text_model": "all-MiniLM-L6-v2",
        "code_model": "microsoft/codebert-base",
        "similarity_threshold": 0.75,
        "skeleton_ratio": 0.2,
        "preload_code_model": True,
    }
    assert afm_cfg == {
        "tau_high": 0.45,
        "tau_mid": 0.25,
        "half_life": 12,
        "use_llm_importance": False,
        "use_llm_compression": False,
    }
    assert resource_cfg == {
        "max_document_size_mb": 100.0,
        "max_total_storage_mb": 1024.0,
        "max_documents": 1000,
        "max_memory_mb": 2048.0,
    }
    assert ace_cfg == {
        "deduplication_threshold": 0.85,
        "max_bullets": 100,
    }
    assert monitor == {"max_tokens": 100000, "used_tokens": 0, "history": []}


def test_build_uses_helper_configs_for_constructor_kwargs():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    captured = {}

    def code_adapter_cls(**kwargs):
        captured["code_adapter_kwargs"] = kwargs
        return Mock()

    def afm_config_cls(**kwargs):
        captured["afm_kwargs"] = kwargs
        return Mock()

    def resource_limits_cls(**kwargs):
        captured["resource_kwargs"] = kwargs
        return Mock()

    def ace_framework_cls(**kwargs):
        captured["ace_kwargs"] = kwargs
        return Mock()

    def noop(*args, **kwargs):
        return Mock()

    module.ServerFactoryService.build(
        preload_code_model=False,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=100,
        code_adapter_cls=code_adapter_cls,
        blind_spot_cls=noop,
        halo_cls=noop,
        context_window_adapter_cls=noop,
        multilevel_encoder_cls=noop,
        afm_config_cls=afm_config_cls,
        focus_manager_cls=noop,
        persistence_cls=noop,
        resource_limits_cls=resource_limits_cls,
        resource_manager_cls=noop,
        file_sync_cls=noop,
        version_manager_cls=noop,
        path_validator_cls=noop,
        ace_framework_cls=ace_framework_cls,
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

    assert captured["code_adapter_kwargs"] == module.ServerFactoryService.code_adapter_config(
        preload_code_model=False
    )
    assert captured["afm_kwargs"] == module.ServerFactoryService.afm_config_kwargs()
    assert captured["resource_kwargs"] == module.ServerFactoryService.resource_limits_kwargs()
    assert captured["ace_kwargs"] == module.ServerFactoryService.ace_framework_kwargs()


def test_build_class_dispatch_uses_subclass_helper_overrides():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    class DerivedFactory(module.ServerFactoryService):
        @staticmethod
        def code_adapter_config(*, preload_code_model: bool):
            return {
                "text_model": "text-x",
                "code_model": "code-y",
                "similarity_threshold": 0.11,
                "skeleton_ratio": 0.33,
                "preload_code_model": preload_code_model,
            }

        @staticmethod
        def afm_config_kwargs():
            return {
                "tau_high": 0.9,
                "tau_mid": 0.4,
                "half_life": 3,
                "use_llm_importance": True,
                "use_llm_compression": True,
            }

        @staticmethod
        def resource_limits_kwargs():
            return {
                "max_document_size_mb": 1.0,
                "max_total_storage_mb": 2.0,
                "max_documents": 3,
                "max_memory_mb": 4.0,
            }

        @staticmethod
        def ace_framework_kwargs():
            return {"deduplication_threshold": 0.1, "max_bullets": 7}

        @staticmethod
        def default_context_window_monitor():
            return {"max_tokens": 42, "used_tokens": 1, "history": ["seed"]}

    captured = {}

    def code_adapter_cls(**kwargs):
        captured["code"] = kwargs
        return Mock()

    def afm_config_cls(**kwargs):
        captured["afm"] = kwargs
        return Mock()

    def resource_limits_cls(**kwargs):
        captured["resource"] = kwargs
        return Mock()

    def ace_framework_cls(**kwargs):
        captured["ace"] = kwargs
        return Mock()

    def noop(*args, **kwargs):
        return Mock()

    components = DerivedFactory.build(
        preload_code_model=True,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=5,
        code_adapter_cls=code_adapter_cls,
        blind_spot_cls=noop,
        halo_cls=noop,
        context_window_adapter_cls=noop,
        multilevel_encoder_cls=noop,
        afm_config_cls=afm_config_cls,
        focus_manager_cls=noop,
        persistence_cls=noop,
        resource_limits_cls=resource_limits_cls,
        resource_manager_cls=noop,
        file_sync_cls=noop,
        version_manager_cls=noop,
        path_validator_cls=noop,
        ace_framework_cls=ace_framework_cls,
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

    assert captured["code"]["text_model"] == "text-x"
    assert captured["afm"]["half_life"] == 3
    assert captured["resource"]["max_documents"] == 3
    assert captured["ace"]["max_bullets"] == 7
    assert components["context_window_monitor"]["max_tokens"] == 42


def test_build_logging_helpers_expose_expected_payloads():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert module.ServerFactoryService.file_sync_log_kwargs() == {"status": "enabled"}
    assert module.ServerFactoryService.path_validator_log_kwargs(
        allowed_base_dirs=["C:/repo", "C:/Users/test"]
    ) == {
        "allowed_directories_count": 2,
        "security_feature": "CWE-22 path traversal prevention",
    }
    assert module.ServerFactoryService.ace_framework_log_kwargs(max_ace_contexts=77) == {
        "deduplication_threshold": 0.85,
        "max_bullets": 100,
        "max_contexts": 77,
    }


def test_build_class_dispatch_uses_subclass_logging_helper_overrides():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    class DerivedFactory(module.ServerFactoryService):
        @staticmethod
        def file_sync_log_kwargs() -> dict[str, object]:
            return {"status": "custom-sync"}

        @staticmethod
        def path_validator_log_kwargs(*, allowed_base_dirs: list[str]) -> dict[str, object]:
            return {
                "allowed_directories_count": len(allowed_base_dirs),
                "security_feature": "custom-policy",
            }

        @staticmethod
        def ace_framework_log_kwargs(*, max_ace_contexts: int) -> dict[str, object]:
            return {
                "deduplication_threshold": 0.42,
                "max_bullets": 9,
                "max_contexts": max_ace_contexts,
            }

    class LoggerProbe:
        def __init__(self):
            self.calls = []

        def info(self, event, **kwargs):
            self.calls.append((event, kwargs))

    logger = LoggerProbe()

    def noop(*args, **kwargs):
        return Mock()

    DerivedFactory.build(
        preload_code_model=False,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=11,
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
        path_validator_cls=noop,
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
        logger=logger,
    )

    events = {name: payload for name, payload in logger.calls}
    assert events["file_sync_initialized"]["status"] == "custom-sync"
    assert events["path_validator_initialized"]["security_feature"] == "custom-policy"
    assert events["ace_framework_initialized"]["deduplication_threshold"] == 0.42
    assert events["ace_framework_initialized"]["max_contexts"] == 11
