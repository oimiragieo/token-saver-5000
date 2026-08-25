"""Contract tests for app-layer server factory service, part 3/3.

Split 2026-08-24 out of test_server_factory_service.py (backlog N10, pure
file-size hygiene -- no test logic changed). See test_server_factory_service.py
(part 1) for the docstring covering the original scope.

This part covers: the default-build-inputs validation chain (request merge,
nested-payload validation, class-dispatch for nested validators), the
contract-key-mismatch message format, the full happy-path validated chain,
and class-dispatch coverage for every contract-key validator.
"""

from __future__ import annotations

import importlib

import pytest
from unittest.mock import Mock


def test_validate_default_build_inputs_runs_request_then_factory_validation():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []
    sentinel_request = {"preload_code_model": True, "cwd": "C:/repo"}
    sentinel_validation = {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}}

    original_request = module.ServerFactoryService.build_default_request
    original_validate_request = module.ServerFactoryService.__dict__[
        "validate_build_default_request_map"
    ]
    original_validate_factory = module.ServerFactoryService.validate_factory_contracts

    def fake_request(*, preload_code_model, cwd, home_dir, max_ace_contexts, logger):
        calls.append("build_default_request")
        return sentinel_request

    def fake_validate_request(request):
        calls.append("validate_build_default_request_map")
        assert request is sentinel_request
        return request

    def fake_validate_factory(_cls, *, class_overrides):
        calls.append("validate_factory_contracts")
        assert class_overrides == {"FocusManager": object_override}
        return sentinel_validation

    object_override = object()

    module.ServerFactoryService.build_default_request = staticmethod(fake_request)
    module.ServerFactoryService.validate_build_default_request_map = staticmethod(
        fake_validate_request
    )
    module.ServerFactoryService.validate_factory_contracts = classmethod(fake_validate_factory)
    try:
        result = module.ServerFactoryService.validate_default_build_inputs(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=3,
            logger=Mock(),
            class_overrides={"FocusManager": object_override},
        )
    finally:
        module.ServerFactoryService.build_default_request = original_request
        module.ServerFactoryService.validate_build_default_request_map = original_validate_request
        module.ServerFactoryService.validate_factory_contracts = original_validate_factory

    assert calls == [
        "build_default_request",
        "validate_build_default_request_map",
        "validate_factory_contracts",
    ]
    assert result["request"] is sentinel_request
    assert result["validation"] is sentinel_validation


def test_build_default_delegates_to_validate_default_build_inputs():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    sentinel = {
        "request": {
            "preload_code_model": True,
            "cwd": "C:/repo",
            "home_dir": "C:/Users/test",
            "max_ace_contexts": 4,
            "logger": Mock(),
        },
        "validation": {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}},
    }
    captured = {}

    original_validate_inputs = module.ServerFactoryService.validate_default_build_inputs
    original_build_default_from_validation = (
        module.ServerFactoryService.build_default_from_validation
    )

    def fake_validate_inputs(
        _cls,
        *,
        preload_code_model,
        cwd,
        home_dir,
        max_ace_contexts,
        logger,
        class_overrides,
    ):
        captured["validate_inputs"] = {
            "preload_code_model": preload_code_model,
            "cwd": cwd,
            "home_dir": home_dir,
            "max_ace_contexts": max_ace_contexts,
            "logger": logger,
            "class_overrides": class_overrides,
        }
        return sentinel

    def fake_build_default_from_validation(_cls, *, request, validation):
        captured["request"] = request
        captured["validation"] = validation
        return {"ok": True}

    module.ServerFactoryService.validate_default_build_inputs = classmethod(fake_validate_inputs)
    module.ServerFactoryService.build_default_from_validation = classmethod(
        fake_build_default_from_validation
    )
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=4,
            logger=Mock(),
            class_overrides={"FocusManager": object()},
        )
    finally:
        module.ServerFactoryService.validate_default_build_inputs = original_validate_inputs
        module.ServerFactoryService.build_default_from_validation = (
            original_build_default_from_validation
        )

    assert result == {"ok": True}
    assert captured["validate_inputs"]["cwd"] == "C:/repo"
    assert "FocusManager" in captured["validate_inputs"]["class_overrides"]
    assert captured["request"] is sentinel["request"]
    assert captured["validation"] is sentinel["validation"]


def test_validate_default_build_inputs_map_contract_declared_and_aligned():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert hasattr(module.ServerFactoryService, "validate_default_build_inputs_map")
    assert set(module.DefaultBuildInputs.__annotations__.keys()) == {
        "request",
        "validation",
    }


def test_build_default_rejects_default_build_inputs_drift_before_dispatch():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_validate_inputs = module.ServerFactoryService.validate_default_build_inputs
    original_dispatch = module.ServerFactoryService.build_default_from_validation

    called = {"dispatch": False}

    def bad_validate_inputs(
        _cls,
        *,
        preload_code_model,
        cwd,
        home_dir,
        max_ace_contexts,
        logger,
        class_overrides,
    ):
        return {
            "request": {
                "preload_code_model": preload_code_model,
                "cwd": cwd,
                "home_dir": home_dir,
                "max_ace_contexts": max_ace_contexts,
                "logger": logger,
            }
        }

    def fake_dispatch(_cls, *, request, validation):
        called["dispatch"] = True
        return {"ok": True}

    module.ServerFactoryService.validate_default_build_inputs = classmethod(bad_validate_inputs)
    module.ServerFactoryService.build_default_from_validation = classmethod(fake_dispatch)
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
        module.ServerFactoryService.validate_default_build_inputs = original_validate_inputs
        module.ServerFactoryService.build_default_from_validation = original_dispatch

    assert "validation" in str(exc_info.value)
    assert called["dispatch"] is False


def test_validate_default_build_inputs_map_validates_nested_payloads_in_order():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []
    sentinel_request = {"preload_code_model": True, "cwd": "C:/repo"}
    sentinel_validation = {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}}

    original_validate_request = module.ServerFactoryService.__dict__[
        "validate_build_default_request_map"
    ]
    original_validate_validation = module.ServerFactoryService.__dict__[
        "validate_factory_validation_result_map"
    ]

    def fake_validate_request(request):
        calls.append("validate_build_default_request_map")
        assert request is sentinel_request
        return {
            "preload_code_model": True,
            "cwd": "C:/repo",
            "home_dir": "X",
            "max_ace_contexts": 1,
            "logger": Mock(),
        }

    def fake_validate_validation(validation):
        calls.append("validate_factory_validation_result_map")
        assert validation is sentinel_validation
        return validation

    module.ServerFactoryService.validate_build_default_request_map = staticmethod(
        fake_validate_request
    )
    module.ServerFactoryService.validate_factory_validation_result_map = staticmethod(
        fake_validate_validation
    )
    try:
        result = module.ServerFactoryService.validate_default_build_inputs_map(
            {"request": sentinel_request, "validation": sentinel_validation}
        )
    finally:
        module.ServerFactoryService.validate_build_default_request_map = original_validate_request
        module.ServerFactoryService.validate_factory_validation_result_map = (
            original_validate_validation
        )

    assert calls == [
        "validate_build_default_request_map",
        "validate_factory_validation_result_map",
    ]
    assert result["validation"] is sentinel_validation
    assert result["request"]["cwd"] == "C:/repo"


def test_build_default_rejects_default_build_inputs_nested_request_drift_before_dispatch():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    original_validate_inputs = module.ServerFactoryService.validate_default_build_inputs
    original_dispatch = module.ServerFactoryService.build_default_from_validation

    called = {"dispatch": False}

    def bad_validate_inputs(
        _cls,
        *,
        preload_code_model,
        cwd,
        home_dir,
        max_ace_contexts,
        logger,
        class_overrides,
    ):
        return {
            "request": {
                "preload_code_model": preload_code_model,
                "home_dir": home_dir,
                "max_ace_contexts": max_ace_contexts,
                "logger": logger,
            },
            "validation": {"resolved_classes": {}, "build_kwargs": {"code_adapter_cls": object()}},
        }

    def fake_dispatch(_cls, *, request, validation):
        called["dispatch"] = True
        return {"ok": True}

    module.ServerFactoryService.validate_default_build_inputs = classmethod(bad_validate_inputs)
    module.ServerFactoryService.build_default_from_validation = classmethod(fake_dispatch)
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
        module.ServerFactoryService.validate_default_build_inputs = original_validate_inputs
        module.ServerFactoryService.build_default_from_validation = original_dispatch

    assert "cwd" in str(exc_info.value)
    assert called["dispatch"] is False


def test_validate_default_build_inputs_map_uses_class_dispatch_for_nested_validators():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []

    class DerivedFactory(module.ServerFactoryService):
        @staticmethod
        def validate_build_default_request_map(request):
            calls.append("validate_build_default_request_map")
            return request

        @staticmethod
        def validate_factory_validation_result_map(validation):
            calls.append("validate_factory_validation_result_map")
            return validation

    result = DerivedFactory.validate_default_build_inputs_map(
        {
            "request": {
                "preload_code_model": True,
                "cwd": "C:/repo",
                "home_dir": "C:/Users/test",
                "max_ace_contexts": 3,
                "logger": Mock(),
            },
            "validation": {
                "resolved_classes": {},
                "build_kwargs": {
                    "code_adapter_cls": object(),
                    "blind_spot_cls": object(),
                    "halo_cls": object(),
                    "context_window_adapter_cls": object(),
                    "multilevel_encoder_cls": object(),
                    "afm_config_cls": object(),
                    "focus_manager_cls": object(),
                    "persistence_cls": object(),
                    "resource_limits_cls": object(),
                    "resource_manager_cls": object(),
                    "file_sync_cls": object(),
                    "version_manager_cls": object(),
                    "path_validator_cls": object(),
                    "ace_framework_cls": object(),
                    "ace_context_manager_cls": object(),
                    "tooling_gateway_cls": object(),
                    "context_service_cls": object(),
                    "lifecycle_service_cls": object(),
                    "progress_service_cls": object(),
                    "persistence_service_cls": object(),
                    "tool_profile_service_cls": object(),
                    "runtime_service_cls": object(),
                    "server_service_adapter_cls": object(),
                },
            },
        }
    )

    assert calls == [
        "validate_build_default_request_map",
        "validate_factory_validation_result_map",
    ]
    assert "request" in result
    assert "validation" in result


def test_build_default_from_validation_rejects_request_drift_before_validation_merge():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    bad_request = {
        "preload_code_model": True,
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 3,
        "logger": Mock(),
    }
    validation = {
        "resolved_classes": {},
        "build_kwargs": {"code_adapter_cls": object()},
    }

    original_merge = module.ServerFactoryService.build_request_from_default_validation
    called = {"merge": False}

    def fake_merge(_cls, *, request, validation):
        called["merge"] = True
        return {"preload_code_model": True, "cwd": "C:/repo", "code_adapter_cls": object()}

    module.ServerFactoryService.build_request_from_default_validation = classmethod(fake_merge)
    try:
        with pytest.raises(ValueError) as exc_info:
            module.ServerFactoryService.build_default_from_validation(
                request=bad_request,
                validation=validation,
            )
    finally:
        module.ServerFactoryService.build_request_from_default_validation = original_merge

    assert "cwd" in str(exc_info.value)
    assert called["merge"] is False


def _full_build_kwargs():
    return {
        "code_adapter_cls": object(),
        "blind_spot_cls": object(),
        "halo_cls": object(),
        "context_window_adapter_cls": object(),
        "multilevel_encoder_cls": object(),
        "afm_config_cls": object(),
        "focus_manager_cls": object(),
        "persistence_cls": object(),
        "resource_limits_cls": object(),
        "resource_manager_cls": object(),
        "file_sync_cls": object(),
        "version_manager_cls": object(),
        "path_validator_cls": object(),
        "ace_framework_cls": object(),
        "ace_context_manager_cls": object(),
        "tooling_gateway_cls": object(),
        "context_service_cls": object(),
        "lifecycle_service_cls": object(),
        "progress_service_cls": object(),
        "persistence_service_cls": object(),
        "tool_profile_service_cls": object(),
        "runtime_service_cls": object(),
        "server_service_adapter_cls": object(),
    }


def test_factory_contract_key_constants_align_with_typeddict_schemas():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    assert module.BUILD_DEFAULT_REQUEST_KEYS == frozenset(
        module.BuildDefaultRequest.__annotations__.keys()
    )
    assert module.FACTORY_VALIDATION_RESULT_KEYS == frozenset(
        module.FactoryValidationResult.__annotations__.keys()
    )
    assert module.DEFAULT_BUILD_INPUTS_KEYS == frozenset(
        module.DefaultBuildInputs.__annotations__.keys()
    )
    assert module.BUILD_REQUEST_KEYS == frozenset(module.BuildRequest.__annotations__.keys())
    assert module.BUILD_KWARGS_KEYS == frozenset(module.BuildKwargsMap.__annotations__.keys())


def test_contract_key_mismatch_message_uses_canonical_format():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    message = module.ServerFactoryService.contract_key_mismatch_message(
        contract_name="build_default_request_map",
        missing=["cwd"],
        extra=["unknown"],
    )

    assert message == "build_default_request_map keys mismatch: missing=['cwd'] extra=['unknown']"


def test_validate_build_default_request_map_rejects_extra_keys_with_exact_message():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    bad_request = {
        "preload_code_model": True,
        "cwd": "C:/repo",
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 3,
        "logger": Mock(),
        "extra": "nope",
    }

    with pytest.raises(ValueError) as exc_info:
        module.ServerFactoryService.validate_build_default_request_map(bad_request)

    assert (
        str(exc_info.value) == "build_default_request_map keys mismatch: missing=[] extra=['extra']"
    )


def test_validate_factory_validation_result_map_rejects_extra_keys_with_exact_message():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    bad_validation = {
        "resolved_classes": {},
        "build_kwargs": {"code_adapter_cls": object()},
        "extra": True,
    }

    with pytest.raises(ValueError) as exc_info:
        module.ServerFactoryService.validate_factory_validation_result_map(bad_validation)

    assert (
        str(exc_info.value)
        == "factory_validation_result_map keys mismatch: missing=[] extra=['extra']"
    )


def test_validate_default_build_inputs_map_rejects_extra_keys_with_exact_message():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    bad_inputs = {
        "request": {
            "preload_code_model": True,
            "cwd": "C:/repo",
            "home_dir": "C:/Users/test",
            "max_ace_contexts": 3,
            "logger": Mock(),
        },
        "validation": {
            "resolved_classes": {},
            "build_kwargs": {"code_adapter_cls": object()},
        },
        "extra": 1,
    }

    with pytest.raises(ValueError) as exc_info:
        module.ServerFactoryService.validate_default_build_inputs_map(bad_inputs)

    assert (
        str(exc_info.value) == "default_build_inputs_map keys mismatch: missing=[] extra=['extra']"
    )


def test_validate_build_request_map_rejects_extra_keys_with_exact_message():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    bad_request = {
        "preload_code_model": True,
        "cwd": "C:/repo",
        "home_dir": "C:/Users/test",
        "max_ace_contexts": 3,
        "logger": Mock(),
        "extra": "x",
    }

    with pytest.raises(ValueError) as exc_info:
        module.ServerFactoryService.validate_build_request_map(bad_request)

    assert str(exc_info.value) == "build_request_map keys mismatch: missing=[] extra=['extra']"


def test_default_build_happy_path_runs_full_validated_chain():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    order = []
    captured = {}

    original_validate_default_inputs = module.ServerFactoryService.validate_default_build_inputs
    original_build_default_from_validation = (
        module.ServerFactoryService.build_default_from_validation
    )
    original_build_from_request = module.ServerFactoryService.build_from_request
    original_build = module.ServerFactoryService.__dict__["build"]

    def wrapped_validate_default_inputs(
        _cls,
        *,
        preload_code_model,
        cwd,
        home_dir,
        max_ace_contexts,
        logger,
        class_overrides,
    ):
        order.append("validate_default_build_inputs")
        return original_validate_default_inputs(
            preload_code_model=preload_code_model,
            cwd=cwd,
            home_dir=home_dir,
            max_ace_contexts=max_ace_contexts,
            logger=logger,
            class_overrides=class_overrides,
        )

    def wrapped_build_default_from_validation(_cls, *, request, validation):
        order.append("build_default_from_validation")
        return original_build_default_from_validation(request=request, validation=validation)

    def wrapped_build_from_request(_cls, *, request):
        order.append("build_from_request")
        return original_build_from_request(request=request)

    def fake_build(_cls, **kwargs):
        order.append("build")
        captured["kwargs"] = kwargs
        return {"ok": True}

    module.ServerFactoryService.validate_default_build_inputs = classmethod(
        wrapped_validate_default_inputs
    )
    module.ServerFactoryService.build_default_from_validation = classmethod(
        wrapped_build_default_from_validation
    )
    module.ServerFactoryService.build_from_request = classmethod(wrapped_build_from_request)
    module.ServerFactoryService.build = classmethod(fake_build)
    try:
        result = module.ServerFactoryService.build_default(
            preload_code_model=True,
            cwd="C:/repo",
            home_dir="C:/Users/test",
            max_ace_contexts=3,
            logger=Mock(),
            class_overrides=None,
        )
    finally:
        module.ServerFactoryService.validate_default_build_inputs = original_validate_default_inputs
        module.ServerFactoryService.build_default_from_validation = (
            original_build_default_from_validation
        )
        module.ServerFactoryService.build_from_request = original_build_from_request
        module.ServerFactoryService.build = original_build

    assert result == {"ok": True}
    assert order == [
        "validate_default_build_inputs",
        "build_default_from_validation",
        "build_from_request",
        "build",
    ]
    assert captured["kwargs"]["cwd"] == "C:/repo"
    assert "code_adapter_cls" in captured["kwargs"]


def test_validate_default_class_map_uses_class_dispatch_for_contract_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []

    class DerivedFactory(module.ServerFactoryService):
        @classmethod
        def validate_contract_keys(cls, *, contract_name, payload, expected_keys):
            calls.append(contract_name)

    bad_map = {"OnlyOneKey": object()}
    result = DerivedFactory.validate_default_class_map(bad_map)

    assert calls == ["default_class_map"]
    assert result is bad_map


def test_validate_build_kwargs_map_uses_class_dispatch_for_contract_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []

    class DerivedFactory(module.ServerFactoryService):
        @classmethod
        def validate_contract_keys(cls, *, contract_name, payload, expected_keys):
            calls.append(contract_name)

    bad_kwargs = {"code_adapter_cls": object()}
    result = DerivedFactory.validate_build_kwargs_map(bad_kwargs)

    assert calls == ["build_kwargs_map"]
    assert result is bad_kwargs


def test_validate_build_default_request_map_uses_class_dispatch_for_contract_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []

    class DerivedFactory(module.ServerFactoryService):
        @classmethod
        def validate_contract_keys(cls, *, contract_name, payload, expected_keys):
            calls.append(contract_name)

    bad_request = {"cwd": "C:/repo"}
    result = DerivedFactory.validate_build_default_request_map(bad_request)

    assert calls == ["build_default_request_map"]
    assert result is bad_request


def test_validate_factory_validation_result_map_uses_class_dispatch_for_contract_keys():
    module = importlib.import_module("src.semantic_modulator.app.server_factory_service")

    calls = []

    class DerivedFactory(module.ServerFactoryService):
        @classmethod
        def validate_contract_keys(cls, *, contract_name, payload, expected_keys):
            calls.append(contract_name)

    bad_validation = {"build_kwargs": {"code_adapter_cls": object()}}
    result = DerivedFactory.validate_factory_validation_result_map(bad_validation)

    assert calls == ["factory_validation_result_map"]
    assert result is bad_validation
