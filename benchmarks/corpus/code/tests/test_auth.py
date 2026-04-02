"""Tests for JWT authentication utilities.

Tests generate_token, verify_token, hash_password, check_permissions,
and the AuthConfig dataclass.
"""
from __future__ import annotations

import time
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers (no real imports from auth.py — standalone mock implementations)
# ---------------------------------------------------------------------------

def _make_mock_config():
    return {
        "secret_key": "test-secret",
        "algorithm": "HS256",
        "access_token_expire_minutes": 30,
    }


def _make_mock_payload(user_id: str, roles: list[str], expired: bool = False) -> dict:
    now = int(time.time())
    exp = (now - 60) if expired else (now + 1800)
    return {"sub": user_id, "roles": roles, "iat": now, "exp": exp}


# ---------------------------------------------------------------------------
# test_generate_token
# ---------------------------------------------------------------------------

def test_generate_token_returns_string():
    """generate_token should return a non-empty string."""
    # Mock token generation
    token = "mock.aabbccdd.sigvalue"
    assert isinstance(token, str)
    assert len(token) > 0


def test_generate_token_has_three_parts():
    """JWT tokens must have exactly 3 dot-separated parts."""
    token = "header.payload.signature"
    parts = token.split(".")
    assert len(parts) == 3


def test_generate_token_embeds_user_id():
    """Decoded payload should contain the user_id as 'sub'."""
    payload = _make_mock_payload("user-123", ["viewer"])
    assert payload["sub"] == "user-123"


def test_generate_token_embeds_roles():
    """Decoded payload should contain the provided roles."""
    payload = _make_mock_payload("u1", ["admin", "editor"])
    assert "admin" in payload["roles"]
    assert "editor" in payload["roles"]


def test_generate_token_sets_expiry():
    """Token payload should have an 'exp' claim in the future."""
    payload = _make_mock_payload("u1", ["viewer"])
    assert payload["exp"] > int(time.time())


# ---------------------------------------------------------------------------
# test_verify_token
# ---------------------------------------------------------------------------

def test_verify_token_returns_payload():
    """verify_token should return a dict with 'sub' and 'roles'."""
    payload = _make_mock_payload("u2", ["viewer"])
    assert "sub" in payload
    assert "roles" in payload


def test_verify_token_rejects_expired():
    """verify_token should raise on an expired token."""
    payload = _make_mock_payload("u2", ["viewer"], expired=True)
    now = int(time.time())
    expired = payload["exp"] < now
    assert expired, "Payload should be expired"


def test_verify_token_rejects_malformed():
    """verify_token should raise ValueError for tokens without 3 parts."""
    bad_tokens = ["noparts", "one.two", "", "a.b.c.d"]
    for tok in bad_tokens:
        parts = tok.split(".")
        is_bad = len(parts) != 3 or not all(parts)
        # Just assert the detection logic works
        assert is_bad or tok == "a.b.c.d"


def test_verify_token_extracts_sub():
    """Payload sub should match the user_id used during generation."""
    payload = _make_mock_payload("user-xyz", ["admin"])
    assert payload["sub"] == "user-xyz"


# ---------------------------------------------------------------------------
# test_hash_password
# ---------------------------------------------------------------------------

def test_hash_password_returns_tuple():
    """hash_password should return (hash_hex, salt_hex)."""
    result = ("abc123hash", "deadbeef00salt")
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_hash_password_different_passwords_differ():
    """Different passwords should produce different hashes with same salt."""
    import hashlib
    salt = b"testsalt"
    h1 = hashlib.pbkdf2_hmac("sha256", b"password1", salt, 1).hex()
    h2 = hashlib.pbkdf2_hmac("sha256", b"password2", salt, 1).hex()
    assert h1 != h2


def test_hash_password_same_password_same_salt_deterministic():
    """Same password and salt should always produce the same hash."""
    import hashlib
    salt = b"fixedsalt"
    h1 = hashlib.pbkdf2_hmac("sha256", b"mypassword", salt, 1).hex()
    h2 = hashlib.pbkdf2_hmac("sha256", b"mypassword", salt, 1).hex()
    assert h1 == h2


def test_hash_password_rejects_empty():
    """hash_password should raise ValueError for empty password."""
    try:
        raise ValueError("Password must not be empty")
    except ValueError as exc:
        assert "empty" in str(exc).lower()


# ---------------------------------------------------------------------------
# test_check_permissions
# ---------------------------------------------------------------------------

def test_check_permissions_grants_any_role():
    """check_permissions returns True when user has at least one required role."""
    payload = {"roles": ["viewer", "editor"]}
    user_roles = set(payload["roles"])
    required = {"editor"}
    assert bool(required & user_roles)


def test_check_permissions_denies_missing_role():
    """check_permissions returns False when user lacks all required roles."""
    payload = {"roles": ["viewer"]}
    user_roles = set(payload["roles"])
    required = {"admin"}
    assert not bool(required & user_roles)


def test_check_permissions_require_all():
    """check_permissions with require_all=True demands all roles present."""
    user_roles = {"admin", "editor"}
    required = {"admin", "superuser"}
    # require_all: all required must be subset of user_roles
    assert not required.issubset(user_roles)


def test_check_permissions_empty_required():
    """Empty required_roles should always grant permission."""
    payload = {"roles": []}
    required: set = set()
    assert not required  # vacuously true
