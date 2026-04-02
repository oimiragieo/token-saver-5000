"""Application error hierarchy and error handling utilities.

Defines structured error types, HTTP status mapping, and an error_handler
that converts exceptions into JSON-serialisable response dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AppError(Exception):
    """Base class for all application-level errors.

    Attributes:
        message: Human-readable description.
        code: Machine-readable error code string.
        status_code: HTTP status code to return.
        details: Optional dict with additional context.
    """

    message: str
    code: str = "INTERNAL_ERROR"
    status_code: int = 500
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Serialise error to a JSON-safe dict."""
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
            "status": self.status_code,
        }
        if self.details:
            payload["details"] = self.details
        return payload


@dataclass
class NotFoundError(AppError):
    """Resource was not found (HTTP 404)."""

    message: str = "Resource not found"
    code: str = "NOT_FOUND"
    status_code: int = 404
    resource_type: str = ""
    resource_id: str = ""

    def __post_init__(self) -> None:
        if self.resource_type and self.resource_id:
            self.message = f"{self.resource_type} not found: {self.resource_id!r}"
            self.details["resource_type"] = self.resource_type
            self.details["resource_id"] = self.resource_id


@dataclass
class ValidationError(AppError):
    """Request payload failed validation (HTTP 422)."""

    message: str = "Validation failed"
    code: str = "VALIDATION_ERROR"
    status_code: int = 422
    field_errors: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.field_errors:
            self.details["field_errors"] = self.field_errors

    def add_field_error(self, field_name: str, error: str) -> None:
        """Append an error message for a specific field."""
        if field_name not in self.field_errors:
            self.field_errors[field_name] = []
        self.field_errors[field_name].append(error)
        self.details["field_errors"] = self.field_errors

    def has_errors(self) -> bool:
        return bool(self.field_errors)


@dataclass
class AuthenticationError(AppError):
    """Authentication failed (HTTP 401)."""

    message: str = "Authentication required"
    code: str = "AUTHENTICATION_ERROR"
    status_code: int = 401


@dataclass
class AuthorizationError(AppError):
    """Insufficient permissions (HTTP 403)."""

    message: str = "Insufficient permissions"
    code: str = "AUTHORIZATION_ERROR"
    status_code: int = 403
    required_role: str = ""

    def __post_init__(self) -> None:
        if self.required_role:
            self.details["required_role"] = self.required_role


@dataclass
class ConflictError(AppError):
    """Resource conflict, e.g. duplicate email (HTTP 409)."""

    message: str = "Resource conflict"
    code: str = "CONFLICT"
    status_code: int = 409


@dataclass
class RateLimitError(AppError):
    """Client has exceeded rate limits (HTTP 429)."""

    message: str = "Rate limit exceeded"
    code: str = "RATE_LIMIT_EXCEEDED"
    status_code: int = 429
    retry_after_sec: int = 60

    def __post_init__(self) -> None:
        self.details["retry_after_sec"] = self.retry_after_sec


def error_handler(exc: Exception) -> dict[str, Any]:
    """Convert any exception to a JSON-serialisable error response dict.

    Args:
        exc: The exception to handle. AppError subclasses are serialised
             with their structured payload; all others map to 500.

    Returns:
        Dict with at minimum 'error', 'message', and 'status' keys.
    """
    if isinstance(exc, AppError):
        return exc.to_dict()

    if isinstance(exc, PermissionError):
        return AuthorizationError(message=str(exc)).to_dict()

    if isinstance(exc, LookupError):
        return NotFoundError(message=str(exc)).to_dict()

    if isinstance(exc, ValueError):
        ve = ValidationError(message=str(exc))
        return ve.to_dict()

    # Unhandled: generic 500
    return AppError(
        message="An unexpected error occurred",
        code="INTERNAL_ERROR",
        status_code=500,
    ).to_dict()


def raise_if_not_found(
    value: Any,
    resource_type: str,
    resource_id: str,
) -> Any:
    """Raise NotFoundError if value is None or falsy.

    Args:
        value: Value to check.
        resource_type: Human-readable resource type name.
        resource_id: Identifier used in the error message.

    Returns:
        value if truthy.

    Raises:
        NotFoundError: If value is None/falsy.
    """
    if not value:
        raise NotFoundError(resource_type=resource_type, resource_id=resource_id)
    return value
