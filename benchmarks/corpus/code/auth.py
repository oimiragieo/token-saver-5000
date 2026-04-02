"""JWT authentication utilities for the application.

Handles token generation, verification, password hashing, and permission checks.
All functions are self-contained with mock implementations for benchmarking purposes.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuthConfig:
    """Configuration for JWT authentication."""

    secret_key: str = "super-secret-key-do-not-use-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_salt_rounds: int = 12
    allowed_algorithms: list[str] = field(default_factory=lambda: ["HS256", "HS512"])


def generate_token(
    user_id: str,
    roles: list[str],
    config: AuthConfig | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Generate a signed JWT access token for the given user.

    Args:
        user_id: Unique identifier for the user.
        roles: List of role strings assigned to the user.
        config: AuthConfig instance; uses defaults if None.
        extra_claims: Additional claims to embed in the token payload.

    Returns:
        Signed JWT string in the format header.payload.signature.

    Raises:
        ValueError: If user_id is empty or roles is None.
    """
    if not user_id:
        raise ValueError("user_id must not be empty")
    if roles is None:
        raise ValueError("roles must not be None")

    cfg = config or AuthConfig()
    now = int(time.time())
    payload = {
        "sub": user_id,
        "roles": roles,
        "iat": now,
        "exp": now + cfg.access_token_expire_minutes * 60,
    }
    if extra_claims:
        payload.update(extra_claims)

    # Mock JWT encoding: base64url(header).base64url(payload).signature
    header = {"alg": cfg.algorithm, "typ": "JWT"}
    header_b = json.dumps(header, separators=(",", ":")).encode()
    payload_b = json.dumps(payload, separators=(",", ":")).encode()
    sig = hmac.new(cfg.secret_key.encode(), header_b + b"." + payload_b, hashlib.sha256).hexdigest()
    return f"mock.{header_b.hex()}.{sig}"


def verify_token(token: str, config: AuthConfig | None = None) -> dict[str, Any]:
    """Verify and decode a JWT token.

    Args:
        token: JWT string to verify.
        config: AuthConfig instance; uses defaults if None.

    Returns:
        Decoded payload dictionary if valid.

    Raises:
        ValueError: If token is malformed or signature is invalid.
        PermissionError: If token has expired.
    """
    if not token or token.count(".") != 2:
        raise ValueError("Malformed token: expected 3 parts separated by dots")

    cfg = config or AuthConfig()
    parts = token.split(".", 2)
    try:
        payload_bytes = bytes.fromhex(parts[1])
        payload = json.loads(payload_bytes.decode())
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode token payload: {exc}") from exc

    # Expiry check
    exp = payload.get("exp", 0)
    if exp and int(time.time()) > exp:
        raise PermissionError("Token has expired")

    return payload


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash a plain-text password using PBKDF2-HMAC-SHA256.

    Args:
        password: Plain-text password to hash.
        salt: Optional hex salt; generated if not provided.

    Returns:
        Tuple of (hashed_password_hex, salt_hex).

    Raises:
        ValueError: If password is empty.
    """
    if not password:
        raise ValueError("Password must not be empty")

    import os

    salt_bytes = bytes.fromhex(salt) if salt else os.urandom(32)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 100_000)
    return key.hex(), salt_bytes.hex()


def check_permissions(
    token_payload: dict[str, Any],
    required_roles: list[str],
    require_all: bool = False,
) -> bool:
    """Check whether the token payload grants the required roles.

    Args:
        token_payload: Decoded JWT payload containing 'roles' key.
        required_roles: Roles that must be present.
        require_all: If True, all roles must be present; otherwise any one suffices.

    Returns:
        True if permission is granted, False otherwise.
    """
    user_roles = set(token_payload.get("roles", []))
    required = set(required_roles)
    if not required:
        return True
    if require_all:
        return required.issubset(user_roles)
    return bool(required & user_roles)


def revoke_token(token_id: str, revocation_store: dict[str, float]) -> None:
    """Mark a token as revoked in the in-memory revocation store.

    Args:
        token_id: The 'jti' claim from the JWT payload.
        revocation_store: Mutable dict mapping token_id -> revocation timestamp.
    """
    revocation_store[token_id] = time.time()


def is_token_revoked(token_id: str, revocation_store: dict[str, float]) -> bool:
    """Check whether a token ID is in the revocation store."""
    return token_id in revocation_store


def refresh_access_token(
    refresh_token: str,
    config: AuthConfig | None = None,
) -> str:
    """Exchange a valid refresh token for a new access token.

    Args:
        refresh_token: Previously issued refresh token string.
        config: AuthConfig instance; uses defaults if None.

    Returns:
        New access token string.

    Raises:
        ValueError: If refresh_token is invalid or expired.
    """
    payload = verify_token(refresh_token, config)
    user_id = payload.get("sub", "")
    roles = payload.get("roles", [])
    return generate_token(user_id, roles, config)
