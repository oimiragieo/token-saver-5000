"""App-layer progress bar rendering service."""

from __future__ import annotations

from typing import Any, TypedDict

from src.semantic_modulator.app.contract_validation import (
    contract_key_mismatch_message as _contract_key_mismatch_message,
    validate_contract_keys as _validate_contract_keys,
)


class ProgressRequest(TypedDict):
    """Progress rendering request envelope."""

    percentage: float
    width: int


class ProgressRenderService:
    """Renders compact terminal-friendly progress bars for server diagnostics."""

    PROGRESS_REQUEST_KEYS: frozenset[str] = frozenset(ProgressRequest.__annotations__.keys())

    @staticmethod
    def contract_key_mismatch_message(
        *,
        contract_name: str,
        missing: list[str],
        extra: list[str],
    ) -> str:
        return _contract_key_mismatch_message(
            contract_name=contract_name, missing=missing, extra=extra
        )

    @classmethod
    def validate_contract_keys(
        cls,
        *,
        contract_name: str,
        payload: dict[str, Any],
        expected_keys: frozenset[str],
    ) -> None:
        _validate_contract_keys(
            contract_name=contract_name, payload=payload, expected_keys=expected_keys
        )

    @classmethod
    def validate_progress_request_map(cls, request: dict[str, Any]) -> ProgressRequest:
        cls.validate_contract_keys(
            contract_name="progress_request_map",
            payload=request,
            expected_keys=cls.PROGRESS_REQUEST_KEYS,
        )
        return request

    @classmethod
    def create_progress_bar(cls, percentage: float, width: int = 40) -> str:
        request = cls.validate_progress_request_map({"percentage": percentage, "width": width})
        percentage = request["percentage"]
        width = request["width"]
        filled = int((percentage / 100) * width)
        empty = width - filled

        if percentage >= 100:
            bar = "█" * width
            return f"[{bar}] [CRIT] FULL"
        if percentage >= 80:
            bar = "█" * filled + "░" * empty
            return f"[{bar}] [WARN] {percentage:.0f}%"

        bar = "█" * filled + "░" * empty
        return f"[{bar}] [OK] {percentage:.0f}%"
