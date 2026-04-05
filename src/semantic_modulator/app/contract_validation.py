"""Shared contract-validation utilities for the app layer.

Provides keyset validation for typed request/context dicts so that
payload drift is caught immediately instead of causing silent bugs.
"""

from typing import Any


def contract_key_mismatch_message(
    *,
    contract_name: str,
    missing: list[str],
    extra: list[str],
) -> str:
    """Build canonical contract drift message."""
    return f"{contract_name} keys mismatch: missing={missing} extra={extra}"


def validate_contract_keys(
    *,
    contract_name: str,
    payload: dict[str, Any],
    expected_keys: frozenset[str],
) -> None:
    """Fail fast when a payload keyset drifts from the expected contract."""
    actual_keys = set(payload.keys())
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(
            contract_key_mismatch_message(
                contract_name=contract_name,
                missing=missing,
                extra=extra,
            )
        )
