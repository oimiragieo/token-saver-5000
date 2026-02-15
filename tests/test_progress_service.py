"""Contract tests for app-layer progress rendering service."""

from __future__ import annotations

import importlib

import pytest


def test_progress_service_render_empty_bar():
    module = importlib.import_module("src.semantic_modulator.app.progress_service")
    service = module.ProgressRenderService()

    bar = service.create_progress_bar(0.0)
    assert "[OK] 0%" in bar


def test_progress_service_render_warning_and_crit_states():
    module = importlib.import_module("src.semantic_modulator.app.progress_service")
    service = module.ProgressRenderService()

    warn = service.create_progress_bar(85.0)
    crit = service.create_progress_bar(100.0)

    assert "[WARN] 85%" in warn
    assert "[CRIT] FULL" in crit


def test_progress_service_request_contract_declared():
    module = importlib.import_module("src.semantic_modulator.app.progress_service")
    service = module.ProgressRenderService
    assert service.PROGRESS_REQUEST_KEYS == frozenset({"percentage", "width"})


def test_progress_service_validate_progress_request_map_rejects_extra_key():
    module = importlib.import_module("src.semantic_modulator.app.progress_service")
    service = module.ProgressRenderService
    with pytest.raises(ValueError, match="progress_request_map keys mismatch"):
        service.validate_progress_request_map({"percentage": 10.0, "width": 20, "extra": True})
