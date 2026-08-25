"""Contract tests for app-layer server factory service, part 2/3.

Split 2026-08-24 out of test_server_factory_service.py (backlog N10, pure
file-size hygiene -- no test logic changed). See test_server_factory_service.py
(part 1) for the docstring covering the original scope.

This part covers: typed-artifact/class-map/build-kwargs contract declarations
and their drift-rejection tests, plus the validate_factory_contracts pipeline
ordering and its delegation chain.

See test_server_factory_service_validation_chain.py (part 3) for the rest.
"""

from __future__ import annotations

import importlib

import pytest
from unittest.mock import Mock


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
    original_validate = module.ServerFactoryService.__dict__["validate_default_class_map"]
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

    original_validate_default = module.ServerFactoryService.__dict__["validate_default_class_map"]
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
    original_validate_default = module.ServerFactoryService.__dict__["validate_default_class_map"]
    original_resolve = module.ServerFactoryService.resolve_class_overrides
    original_build_kwargs = module.ServerFactoryService.build_kwargs_from_resolved_classes
    original_validate_build = module.ServerFactoryService.__dict__["validate_build_kwargs_map"]

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


def test_validate_build_default_request_map_contract_declared_and_aligned():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module.ServerFactoryService, "validate_build_default_request_map")
    assert set(module.BuildDefaultRequest.__annotations__.keys()) == {
        "preload_code_model",
        "cwd",
        "home_dir",
        "max_ace_contexts",
        "logger",
    }


def test_build_default_rejects_default_request_drift_before_factory_validation():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_request = module.ServerFactoryService.build_default_request
    original_validate_factory = module.ServerFactoryService.validate_factory_contracts

    called = {"validate_factory": False}

    def bad_request(*, preload_code_model, cwd, home_dir, max_ace_contexts, logger):
        return {
            "preload_code_model": preload_code_model,
            "home_dir": home_dir,
            "max_ace_contexts": max_ace_contexts,
            "logger": logger,
        }

    def fake_validate_factory(_cls, *, class_overrides):
        called["validate_factory"] = True
        return {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}}

    module.ServerFactoryService.build_default_request = staticmethod(bad_request)
    module.ServerFactoryService.validate_factory_contracts = classmethod(fake_validate_factory)
    try:
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_default(
                preload_code_model=True,
                cwd="C:/repo",
                home_dir="C:/Users/test",
                max_ace_contexts=3,
                logger=Mock(),
            )
    finally:
        module.ServerFactoryService.build_default_request = original_request
        module.ServerFactoryService.validate_factory_contracts = original_validate_factory

    assert "cwd" in str(exc_info.value)
    assert called["validate_factory"] is False


def test_validate_factory_validation_result_contract_declared_and_aligned():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module.ServerFactoryService, "validate_factory_validation_result_map")
    assert set(module.FactoryValidationResult.__annotations__.keys()) == {
        "resolved_classes",
        "build_kwargs",
    }


def test_build_default_from_validation_rejects_validation_drift_before_request_merge():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    request = module.ServerFactoryService.build_default_request(
        preload_code_model=True,
        cwd="C:/repo",
        home_dir="C:/Users/test",
        max_ace_contexts=3,
        logger=Mock(),
    )
    bad_validation = {"build_kwargs": {"code_adapter_cls": object()}}

    original_merge = module.ServerFactoryService.build_request_from_default_validation

    called = {"merge": False}

    def fake_merge(_cls, *, request, validation):
        called["merge"] = True
        return {"preload_code_model": True, "cwd": "X", "code_adapter_cls": object()}

    module.ServerFactoryService.build_request_from_default_validation = classmethod(fake_merge)
    try:
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_default_from_validation(
                request=request,
                validation=bad_validation,
            )
    finally:
        module.ServerFactoryService.build_request_from_default_validation = original_merge

    assert "resolved_classes" in str(exc_info.value)
    assert called["merge"] is False
