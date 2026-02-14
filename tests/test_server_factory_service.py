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
    original_validate = module.ServerFactoryService.validate_default_class_map
    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_validate_build_kwargs = module.ServerFactoryService.validate_build_kwargs_map
    original_helper = module.ServerFactoryService.build_kwargs_from_resolved_classes
    original_build = module.ServerFactoryService.__dict__["build"]

    captured = {}

    def fake_default_map():
        return {"CodeCompressionAdapter": object()}

    def fake_validate(default_map):
        return default_map

    def fake_resolve(*, defaults, overrides):
        captured["defaults"] = defaults
        captured["overrides"] = overrides
        return {"CodeCompressionAdapter": object()}

    def fake_helper(resolved):
        captured["resolved"] = resolved
        return helper_kwargs

    def fake_validate_build_kwargs(build_kwargs):
        return build_kwargs

    def fake_build(_cls, **kwargs):
        captured["build_kwargs"] = kwargs
        return sentinel_result

    module.ServerFactoryService.default_class_map = staticmethod(fake_default_map)
    module.ServerFactoryService.validate_default_class_map = staticmethod(fake_validate)
    module.ServerFactoryService.resolve_class_overrides = staticmethod(fake_resolve)
    module.ServerFactoryService.build_kwargs_from_resolved_classes = staticmethod(fake_helper)
    module.ServerFactoryService.validate_build_kwargs_map = staticmethod(fake_validate_build_kwargs)
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
        module.ServerFactoryService.validate_default_class_map = original_validate
        module.ServerFactoryService.resolve_class_overrides = original_resolve
        module.ServerFactoryService.build_kwargs_from_resolved_classes = original_helper
        module.ServerFactoryService.validate_build_kwargs_map = original_validate_build_kwargs
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


def test_build_service_layer_wires_adapter_dependencies():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    captured = {}

    class ContextService:
        pass

    class LifecycleService:
        pass

    class ProgressService:
        pass

    class PersistenceService:
        pass

    class ToolProfileService:
        pass

    class RuntimeService:
        pass

    def adapter_cls(*, persistence_service, context_service, progress_service, logger):
        captured["persistence_service"] = persistence_service
        captured["context_service"] = context_service
        captured["progress_service"] = progress_service
        captured["logger"] = logger
        return {"adapter": True}

    logger = Mock()

    services = module.ServerFactoryService.build_service_layer(
        context_service_cls=ContextService,
        lifecycle_service_cls=LifecycleService,
        progress_service_cls=ProgressService,
        persistence_service_cls=PersistenceService,
        tool_profile_service_cls=ToolProfileService,
        runtime_service_cls=RuntimeService,
        server_service_adapter_cls=adapter_cls,
        logger=logger,
    )

    assert isinstance(services["context_service"], ContextService)
    assert isinstance(services["lifecycle_service"], LifecycleService)
    assert isinstance(services["progress_service"], ProgressService)
    assert isinstance(services["persistence_service"], PersistenceService)
    assert isinstance(services["tool_profile_service"], ToolProfileService)
    assert isinstance(services["runtime_service"], RuntimeService)
    assert services["service_adapter"] == {"adapter": True}
    assert captured["persistence_service"] is services["persistence_service"]
    assert captured["context_service"] is services["context_service"]
    assert captured["progress_service"] is services["progress_service"]
    assert captured["logger"] is logger


def test_build_delegates_service_layer_construction_through_helper():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel_services = {
        "context_service": object(),
        "lifecycle_service": object(),
        "progress_service": object(),
        "persistence_service": object(),
        "tool_profile_service": object(),
        "runtime_service": object(),
        "service_adapter": object(),
    }
    captured = {}

    original = module.ServerFactoryService.__dict__.get("build_service_layer")

    def fake_service_layer(
        _cls,
        *,
        context_service_cls,
        lifecycle_service_cls,
        progress_service_cls,
        persistence_service_cls,
        tool_profile_service_cls,
        runtime_service_cls,
        server_service_adapter_cls,
        logger,
    ):
        captured["context_service_cls"] = context_service_cls
        captured["lifecycle_service_cls"] = lifecycle_service_cls
        captured["progress_service_cls"] = progress_service_cls
        captured["persistence_service_cls"] = persistence_service_cls
        captured["tool_profile_service_cls"] = tool_profile_service_cls
        captured["runtime_service_cls"] = runtime_service_cls
        captured["server_service_adapter_cls"] = server_service_adapter_cls
        captured["logger"] = logger
        return sentinel_services

    module.ServerFactoryService.build_service_layer = classmethod(fake_service_layer)
    try:
        logger = Mock()

        def marker(*args, **kwargs):
            return Mock()

        components = module.ServerFactoryService.build(
            preload_code_model=False,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=5,
            code_adapter_cls=marker,
            blind_spot_cls=marker,
            halo_cls=marker,
            context_window_adapter_cls=marker,
            multilevel_encoder_cls=marker,
            afm_config_cls=marker,
            focus_manager_cls=marker,
            persistence_cls=marker,
            resource_limits_cls=marker,
            resource_manager_cls=marker,
            file_sync_cls=marker,
            version_manager_cls=marker,
            path_validator_cls=marker,
            ace_framework_cls=marker,
            ace_context_manager_cls=marker,
            tooling_gateway_cls=marker,
            context_service_cls=str,
            lifecycle_service_cls=int,
            progress_service_cls=float,
            persistence_service_cls=list,
            tool_profile_service_cls=dict,
            runtime_service_cls=tuple,
            server_service_adapter_cls=set,
            logger=logger,
        )
    finally:
        if original is None:
            delattr(module.ServerFactoryService, "build_service_layer")
        else:
            module.ServerFactoryService.build_service_layer = original

    assert captured["context_service_cls"] is str
    assert captured["lifecycle_service_cls"] is int
    assert captured["progress_service_cls"] is float
    assert captured["persistence_service_cls"] is list
    assert captured["tool_profile_service_cls"] is dict
    assert captured["runtime_service_cls"] is tuple
    assert captured["server_service_adapter_cls"] is set
    assert captured["logger"] is logger

    assert components["context_service"] is sentinel_services["context_service"]
    assert components["lifecycle_service"] is sentinel_services["lifecycle_service"]
    assert components["progress_service"] is sentinel_services["progress_service"]
    assert components["persistence_service"] is sentinel_services["persistence_service"]
    assert components["tool_profile_service"] is sentinel_services["tool_profile_service"]
    assert components["runtime_service"] is sentinel_services["runtime_service"]
    assert components["service_adapter"] is sentinel_services["service_adapter"]


def test_build_core_runtime_layer_wires_foundational_dependencies():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    captured = {}

    class AFMConfig:
        pass

    class FocusManager:
        def __init__(self, config):
            captured["focus_config"] = config

    class Persistence:
        pass

    class ResourceLimits:
        def __init__(self, **kwargs):
            captured["resource_limits_kwargs"] = kwargs

    class ResourceManager:
        def __init__(self, limits):
            captured["resource_limits_obj"] = limits

    class SyncManager:
        pass

    class VersionManager:
        pass

    class PathValidator:
        def __init__(self, *, allowed_base_dirs):
            captured["allowed_base_dirs"] = allowed_base_dirs

    class ACEFramework:
        def __init__(self, **kwargs):
            captured["ace_kwargs"] = kwargs

    class ACEContexts:
        def __init__(self, *, max_contexts):
            captured["max_contexts"] = max_contexts

    logger = Mock()

    core = module.ServerFactoryService.build_core_runtime_layer(
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=17,
        afm_config=AFMConfig(),
        focus_manager_cls=FocusManager,
        persistence_cls=Persistence,
        resource_limits_cls=ResourceLimits,
        resource_manager_cls=ResourceManager,
        file_sync_cls=SyncManager,
        version_manager_cls=VersionManager,
        path_validator_cls=PathValidator,
        ace_framework_cls=ACEFramework,
        ace_context_manager_cls=ACEContexts,
        logger=logger,
    )

    assert isinstance(core["persistence"], Persistence)
    assert isinstance(core["sync_manager"], SyncManager)
    assert isinstance(core["version_manager"], VersionManager)
    assert isinstance(core["path_validator"], PathValidator)
    assert isinstance(core["ace_framework"], ACEFramework)
    assert isinstance(core["ace_contexts"], ACEContexts)
    assert isinstance(core["focus_manager"], FocusManager)

    assert captured["allowed_base_dirs"] == ["C:/repo", "C:/Users/test"]
    assert (
        captured["resource_limits_kwargs"] == module.ServerFactoryService.resource_limits_kwargs()
    )
    assert captured["ace_kwargs"] == module.ServerFactoryService.ace_framework_kwargs()
    assert captured["max_contexts"] == 17


def test_build_delegates_core_runtime_construction_through_helper():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel_core = {
        "focus_manager": object(),
        "persistence": object(),
        "resource_manager": object(),
        "sync_manager": object(),
        "version_manager": object(),
        "path_validator": object(),
        "ace_framework": object(),
        "ace_contexts": object(),
    }
    captured = {}

    original_core = module.ServerFactoryService.__dict__.get("build_core_runtime_layer")

    def fake_core_layer(
        _cls,
        *,
        cwd,
        home_dir,
        max_ace_contexts,
        afm_config,
        focus_manager_cls,
        persistence_cls,
        resource_limits_cls,
        resource_manager_cls,
        file_sync_cls,
        version_manager_cls,
        path_validator_cls,
        ace_framework_cls,
        ace_context_manager_cls,
        logger,
    ):
        captured["cwd"] = cwd
        captured["home_dir"] = home_dir
        captured["max_ace_contexts"] = max_ace_contexts
        captured["afm_config"] = afm_config
        captured["focus_manager_cls"] = focus_manager_cls
        captured["persistence_cls"] = persistence_cls
        captured["resource_limits_cls"] = resource_limits_cls
        captured["resource_manager_cls"] = resource_manager_cls
        captured["file_sync_cls"] = file_sync_cls
        captured["version_manager_cls"] = version_manager_cls
        captured["path_validator_cls"] = path_validator_cls
        captured["ace_framework_cls"] = ace_framework_cls
        captured["ace_context_manager_cls"] = ace_context_manager_cls
        captured["logger"] = logger
        return sentinel_core

    original_service = module.ServerFactoryService.__dict__.get("build_service_layer")

    def fake_service_layer(
        _cls,
        *,
        context_service_cls,
        lifecycle_service_cls,
        progress_service_cls,
        persistence_service_cls,
        tool_profile_service_cls,
        runtime_service_cls,
        server_service_adapter_cls,
        logger,
    ):
        return {
            "context_service": object(),
            "lifecycle_service": object(),
            "progress_service": object(),
            "persistence_service": object(),
            "tool_profile_service": object(),
            "runtime_service": object(),
            "service_adapter": object(),
        }

    module.ServerFactoryService.build_core_runtime_layer = classmethod(fake_core_layer)
    module.ServerFactoryService.build_service_layer = classmethod(fake_service_layer)
    try:
        logger = Mock()

        def marker(*args, **kwargs):
            return Mock()

        components = module.ServerFactoryService.build(
            preload_code_model=False,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=19,
            code_adapter_cls=marker,
            blind_spot_cls=marker,
            halo_cls=marker,
            context_window_adapter_cls=marker,
            multilevel_encoder_cls=marker,
            afm_config_cls=marker,
            focus_manager_cls=str,
            persistence_cls=int,
            resource_limits_cls=float,
            resource_manager_cls=list,
            file_sync_cls=dict,
            version_manager_cls=tuple,
            path_validator_cls=set,
            ace_framework_cls=bytes,
            ace_context_manager_cls=bytearray,
            tooling_gateway_cls=marker,
            context_service_cls=marker,
            lifecycle_service_cls=marker,
            progress_service_cls=marker,
            persistence_service_cls=marker,
            tool_profile_service_cls=marker,
            runtime_service_cls=marker,
            server_service_adapter_cls=marker,
            logger=logger,
        )
    finally:
        if original_core is None:
            delattr(module.ServerFactoryService, "build_core_runtime_layer")
        else:
            module.ServerFactoryService.build_core_runtime_layer = original_core

        if original_service is None:
            delattr(module.ServerFactoryService, "build_service_layer")
        else:
            module.ServerFactoryService.build_service_layer = original_service

    assert captured["cwd"] == "C:/repo"
    assert captured["home_dir"] == "C:/Users/test"
    assert captured["max_ace_contexts"] == 19
    assert captured["focus_manager_cls"] is str
    assert captured["persistence_cls"] is int
    assert captured["resource_limits_cls"] is float
    assert captured["resource_manager_cls"] is list
    assert captured["file_sync_cls"] is dict
    assert captured["version_manager_cls"] is tuple
    assert captured["path_validator_cls"] is set
    assert captured["ace_framework_cls"] is bytes
    assert captured["ace_context_manager_cls"] is bytearray
    assert captured["logger"] is logger

    assert components["focus_manager"] is sentinel_core["focus_manager"]
    assert components["persistence"] is sentinel_core["persistence"]
    assert components["resource_manager"] is sentinel_core["resource_manager"]
    assert components["sync_manager"] is sentinel_core["sync_manager"]
    assert components["version_manager"] is sentinel_core["version_manager"]
    assert components["path_validator"] is sentinel_core["path_validator"]
    assert components["ace_framework"] is sentinel_core["ace_framework"]
    assert components["ace_contexts"] is sentinel_core["ace_contexts"]


def test_factory_typed_artifact_contracts_are_declared():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module, "CoreRuntimeArtifacts")
    assert hasattr(module, "ServiceLayerArtifacts")
    assert hasattr(module, "BuildArtifacts")

    core_keys = set(module.CoreRuntimeArtifacts.__annotations__.keys())
    service_keys = set(module.ServiceLayerArtifacts.__annotations__.keys())
    build_keys = set(module.BuildArtifacts.__annotations__.keys())

    assert core_keys == {
        "focus_manager",
        "persistence",
        "resource_manager",
        "sync_manager",
        "version_manager",
        "path_validator",
        "ace_framework",
        "ace_contexts",
    }
    assert service_keys == {
        "context_service",
        "lifecycle_service",
        "progress_service",
        "persistence_service",
        "tool_profile_service",
        "runtime_service",
        "service_adapter",
    }
    assert build_keys.issuperset(core_keys | service_keys)
    assert {
        "compressor",
        "blind_spot_detector",
        "halo_detector",
        "context_window_adapter",
        "multilevel_encoder",
        "tooling",
        "context_window_monitor",
        "retrieval_history",
    }.issubset(build_keys)


def test_factory_class_map_contract_declared_and_aligned_with_overrides():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module, "FactoryClassMap")
    assert set(module.FactoryClassMap.__annotations__.keys()) == set(module.ALLOWED_OVERRIDE_KEYS)


def test_build_default_rejects_default_class_map_drift_before_override_merge():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_default_map = module.ServerFactoryService.default_class_map
    original_validate = module.ServerFactoryService.validate_default_class_map
    original_resolve = module.ServerFactoryService.resolve_class_overrides

    called = {"resolve": False}

    def bad_default_map():
        result = original_default_map()
        result.pop("FocusManager")
        return result

    def should_not_run(*, defaults, overrides):
        called["resolve"] = True
        return defaults

    module.ServerFactoryService.default_class_map = staticmethod(bad_default_map)
    module.ServerFactoryService.resolve_class_overrides = staticmethod(should_not_run)
    try:
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_default(
                preload_code_model=False,
                cwd="C:/repo",
                home_dir="C:/Users/test",
                max_ace_contexts=5,
                logger=Mock(),
            )
    finally:
        module.ServerFactoryService.default_class_map = original_default_map
        module.ServerFactoryService.validate_default_class_map = original_validate
        module.ServerFactoryService.resolve_class_overrides = original_resolve

    assert "FocusManager" in str(exc_info.value)
    assert called["resolve"] is False


def test_factory_build_kwargs_contract_declared_and_aligned_with_helper_output():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module, "BuildKwargsMap")
    expected = {
        "code_adapter_cls",
        "blind_spot_cls",
        "halo_cls",
        "context_window_adapter_cls",
        "multilevel_encoder_cls",
        "afm_config_cls",
        "focus_manager_cls",
        "persistence_cls",
        "resource_limits_cls",
        "resource_manager_cls",
        "file_sync_cls",
        "version_manager_cls",
        "path_validator_cls",
        "ace_framework_cls",
        "ace_context_manager_cls",
        "tooling_gateway_cls",
        "context_service_cls",
        "lifecycle_service_cls",
        "progress_service_cls",
        "persistence_service_cls",
        "tool_profile_service_cls",
        "runtime_service_cls",
        "server_service_adapter_cls",
    }

    assert set(module.BuildKwargsMap.__annotations__.keys()) == expected


def test_build_default_rejects_build_kwargs_drift_before_build_call():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_validate_default = module.ServerFactoryService.validate_default_class_map
    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_build_kwargs = module.ServerFactoryService.build_kwargs_from_resolved_classes
    original_build = module.ServerFactoryService.__dict__["build"]

    called = {"build": False}

    resolved = module.ServerFactoryService.default_class_map()

    def fake_validate_default(default_map):
        return default_map

    def fake_resolve(*, defaults, overrides):
        return resolved

    def bad_build_kwargs(resolved_classes):
        kwargs = dict(original_build_kwargs(resolved_classes))
        kwargs.pop("focus_manager_cls")
        return kwargs

    def fake_build(_cls, **kwargs):
        called["build"] = True
        return {}

    module.ServerFactoryService.validate_default_class_map = staticmethod(fake_validate_default)
    module.ServerFactoryService.resolve_class_overrides = staticmethod(fake_resolve)
    module.ServerFactoryService.build_kwargs_from_resolved_classes = staticmethod(bad_build_kwargs)
    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_default(
                preload_code_model=False,
                cwd="C:/repo",
                home_dir="C:/Users/test",
                max_ace_contexts=5,
                logger=Mock(),
            )
    finally:
        module.ServerFactoryService.validate_default_class_map = original_validate_default
        module.ServerFactoryService.resolve_class_overrides = original_resolve
        module.ServerFactoryService.build_kwargs_from_resolved_classes = original_build_kwargs
        module.ServerFactoryService.build = original_build

    assert "focus_manager_cls" in str(exc_info.value)
    assert called["build"] is False


def test_validate_factory_contracts_runs_validation_pipeline_in_order():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []
    defaults_map = {"CodeCompressionAdapter": object()}
    resolved_map = {"CodeCompressionAdapter": object()}
    build_kwargs = {"code_adapter_cls": object()}

    original_default = module.ServerFactoryService.default_class_map
    original_validate_default = module.ServerFactoryService.validate_default_class_map
    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_build_kwargs = module.ServerFactoryService.build_kwargs_from_resolved_classes
    original_validate_build = module.ServerFactoryService.validate_build_kwargs_map

    def fake_default_map():
        calls.append("default_class_map")
        return defaults_map

    def fake_validate_default(default_map):
        calls.append("validate_default_class_map")
        assert default_map is defaults_map
        return default_map

    def fake_resolve(*, defaults, overrides):
        calls.append("resolve_class_overrides")
        assert defaults is defaults_map
        assert overrides == {"FocusManager": object_override}
        return resolved_map

    def fake_build_kwargs(resolved_classes):
        calls.append("build_kwargs_from_resolved_classes")
        assert resolved_classes is resolved_map
        return build_kwargs

    def fake_validate_build_kwargs(kwargs):
        calls.append("validate_build_kwargs_map")
        assert kwargs is build_kwargs
        return kwargs

    object_override = object()

    module.ServerFactoryService.default_class_map = staticmethod(fake_default_map)
    module.ServerFactoryService.validate_default_class_map = staticmethod(fake_validate_default)
    module.ServerFactoryService.resolve_class_overrides = staticmethod(fake_resolve)
    module.ServerFactoryService.build_kwargs_from_resolved_classes = staticmethod(fake_build_kwargs)
    module.ServerFactoryService.validate_build_kwargs_map = staticmethod(fake_validate_build_kwargs)
    try:
        result = module.ServerFactoryService.validate_factory_contracts(
            class_overrides={"FocusManager": object_override}
        )
    finally:
        module.ServerFactoryService.default_class_map = original_default
        module.ServerFactoryService.validate_default_class_map = original_validate_default
        module.ServerFactoryService.resolve_class_overrides = original_resolve
        module.ServerFactoryService.build_kwargs_from_resolved_classes = original_build_kwargs
        module.ServerFactoryService.validate_build_kwargs_map = original_validate_build

    assert calls == [
        "default_class_map",
        "validate_default_class_map",
        "resolve_class_overrides",
        "build_kwargs_from_resolved_classes",
        "validate_build_kwargs_map",
    ]
    assert result["resolved_classes"] is resolved_map
    assert result["build_kwargs"] is build_kwargs


def test_build_default_delegates_to_validate_factory_contracts():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel = {
        "resolved_classes": {"CodeCompressionAdapter": object()},
        "build_kwargs": {"code_adapter_cls": object()},
    }
    captured = {}

    original_validate = module.ServerFactoryService.validate_factory_contracts
    original_build = module.ServerFactoryService.__dict__["build"]

    def fake_validate(_cls, *, class_overrides):
        captured["class_overrides"] = class_overrides
        return sentinel

    def fake_build(_cls, **kwargs):
        captured["build_kwargs"] = kwargs
        return {"ok": True}

    module.ServerFactoryService.validate_factory_contracts = classmethod(fake_validate)
    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=7,
            logger=Mock(),
            class_overrides={"FocusManager": object()},
        )
    finally:
        module.ServerFactoryService.validate_factory_contracts = original_validate
        module.ServerFactoryService.build = original_build

    assert result == {"ok": True}
    assert "FocusManager" in captured["class_overrides"]
    assert (
        captured["build_kwargs"]["code_adapter_cls"] is sentinel["build_kwargs"]["code_adapter_cls"]
    )


def test_build_default_from_validation_delegates_to_build():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    captured = {}
    sentinel_validation = {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}}
    sentinel_request = {
        "preload_code_model": False,
        "cwd": "C:/repo",
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 3,
        "logger": Mock(),
    }

    original_build = module.ServerFactoryService.__dict__["build"]

    def fake_build(_cls, **kwargs):
        captured["kwargs"] = kwargs
        return {"ok": True}

    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default_from_validation(
            request=sentinel_request,
            validation=sentinel_validation,
        )
    finally:
        module.ServerFactoryService.build = original_build

    assert result == {"ok": True}
    assert captured["kwargs"]["preload_code_model"] is sentinel_request["preload_code_model"]
    assert captured["kwargs"]["cwd"] == sentinel_request["cwd"]
    assert captured["kwargs"]["home_dir"] == sentinel_request["home_dir"]
    assert captured["kwargs"]["max_ace_contexts"] == sentinel_request["max_ace_contexts"]
    assert captured["kwargs"]["logger"] is sentinel_request["logger"]
    assert (
        captured["kwargs"]["code_adapter_cls"]
        is sentinel_validation["build_kwargs"]["code_adapter_cls"]
    )


def test_build_default_delegates_to_build_default_from_validation():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel_validation = {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}}
    captured = {}

    original_validate = module.ServerFactoryService.validate_factory_contracts
    original_orchestrate = module.ServerFactoryService.build_default_from_validation

    def fake_validate(_cls, *, class_overrides):
        captured["class_overrides"] = class_overrides
        return sentinel_validation

    def fake_orchestrate(
        _cls,
        *,
        request,
        validation,
    ):
        captured["orchestration"] = {
            "request": request,
            "validation": validation,
        }
        return {"ok": True}

    module.ServerFactoryService.validate_factory_contracts = classmethod(fake_validate)
    module.ServerFactoryService.build_default_from_validation = classmethod(fake_orchestrate)
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=5,
            logger=Mock(),
            class_overrides={"FocusManager": object()},
        )
    finally:
        module.ServerFactoryService.validate_factory_contracts = original_validate
        module.ServerFactoryService.build_default_from_validation = original_orchestrate

    assert result == {"ok": True}
    assert "FocusManager" in captured["class_overrides"]
    assert captured["orchestration"]["validation"] is sentinel_validation
    assert captured["orchestration"]["request"]["preload_code_model"] is True
    assert captured["orchestration"]["request"]["cwd"] == "C:/repo"
    assert captured["orchestration"]["request"]["home_dir"] == "C:/Users/test"
    assert captured["orchestration"]["request"]["max_ace_contexts"] == 5


def test_factory_build_default_request_contract_declared():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module, "BuildDefaultRequest")
    assert set(module.BuildDefaultRequest.__annotations__.keys()) == {
        "preload_code_model",
        "cwd",
        "home_dir",
        "max_ace_contexts",
        "logger",
    }


def test_factory_build_request_contract_declared():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module, "BuildRequest")
    expected_runtime = {
        "preload_code_model",
        "cwd",
        "home_dir",
        "max_ace_contexts",
        "logger",
    }
    expected = expected_runtime | set(module.BuildKwargsMap.__annotations__.keys())
    assert set(module.BuildRequest.__annotations__.keys()) == expected


def test_build_request_from_default_validation_merges_runtime_and_wiring():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    request = {
        "preload_code_model": True,
        "cwd": "C:/repo",
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 9,
        "logger": Mock(),
    }
    validation = {
        "resolved_classes": {"CodeCompressionAdapter": object()},
        "build_kwargs": {"code_adapter_cls": object()},
    }

    merged = module.ServerFactoryService.build_request_from_default_validation(
        request=request,
        validation=validation,
    )

    assert merged["cwd"] == "C:/repo"
    assert merged["home_dir"] == "C:/Users/test"
    assert merged["max_ace_contexts"] == 9
    assert merged["logger"] is request["logger"]
    assert merged["code_adapter_cls"] is validation["build_kwargs"]["code_adapter_cls"]


def test_build_default_from_validation_delegates_to_build_from_request():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    request = {
        "preload_code_model": False,
        "cwd": "C:/repo",
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 2,
        "logger": Mock(),
    }
    validation = {
        "resolved_classes": {"CodeCompressionAdapter": object()},
        "build_kwargs": {"code_adapter_cls": object()},
    }
    captured = {}

    original_merge = module.ServerFactoryService.build_request_from_default_validation
    original_build_from_request = module.ServerFactoryService.build_from_request

    def fake_merge(_cls, *, request, validation):
        captured["request"] = request
        captured["validation"] = validation
        return {"preload_code_model": True, "cwd": "X", "code_adapter_cls": object()}

    def fake_build_from_request(_cls, *, request):
        captured["build_request"] = request
        return {"ok": True}

    module.ServerFactoryService.build_request_from_default_validation = classmethod(fake_merge)
    module.ServerFactoryService.build_from_request = classmethod(fake_build_from_request)
    try:
        result = module.ServerFactoryService.build_default_from_validation(
            request=request,
            validation=validation,
        )
    finally:
        module.ServerFactoryService.build_request_from_default_validation = original_merge
        module.ServerFactoryService.build_from_request = original_build_from_request

    assert result == {"ok": True}
    assert captured["request"] is request
    assert captured["validation"] is validation
    assert captured["build_request"]["cwd"] == "X"


def test_validate_build_request_map_contract_declared_and_aligned():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module.ServerFactoryService, "validate_build_request_map")
    expected_runtime = {
        "preload_code_model",
        "cwd",
        "home_dir",
        "max_ace_contexts",
        "logger",
    }
    expected = expected_runtime | set(module.BuildKwargsMap.__annotations__.keys())
    assert set(module.BuildRequest.__annotations__.keys()) == expected


def test_build_from_request_rejects_request_drift_before_build_call():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_build = module.ServerFactoryService.__dict__["build"]

    called = {"build": False}

    def fake_build(_cls, **kwargs):
        called["build"] = True
        return {"ok": True}

    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        bad_request = dict(
            module.ServerFactoryService.build_request_from_default_validation(
                request=module.ServerFactoryService.build_default_request(
                    preload_code_model=True,
                    cwd="C:/repo",
                    home_dir="C:/Users/test",
                    max_ace_contexts=3,
                    logger=Mock(),
                ),
                validation=module.ServerFactoryService.validate_factory_contracts(
                    class_overrides=None
                ),
            )
        )
        bad_request.pop("cwd")
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_from_request(request=bad_request)
    finally:
        module.ServerFactoryService.build = original_build

    assert "cwd" in str(exc_info.value)
    assert called["build"] is False
