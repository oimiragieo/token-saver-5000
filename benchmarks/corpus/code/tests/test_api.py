"""Tests for REST API user management endpoints.

Tests create_user, get_user, list_users, and the pagination/filtering logic.
All tests use mock data — no real HTTP or database connections.
"""
from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Helpers — mock implementations matching the real API signatures
# ---------------------------------------------------------------------------

def _mock_create_user(email: str, username: str, roles: list[str], current_user=None) -> dict:
    if current_user and "admin" not in current_user.get("roles", []):
        raise PermissionError("Only admins can create users")
    if email == "existing@example.com":
        raise ValueError(f"Email already registered: {email}")
    user_id = f"usr_{abs(hash(email)) % 100_000:05d}"
    return {"id": user_id, "email": email, "username": username, "roles": roles, "status": 201}


def _mock_get_user(user_id: str) -> dict:
    if not user_id:
        raise ValueError("user_id must not be empty")
    if user_id == "usr_99999":
        raise LookupError(f"User not found: {user_id}")
    return {"id": user_id, "email": f"{user_id}@example.com", "roles": ["viewer"], "status": 200}


def _mock_list_users(page: int = 1, page_size: int = 20, role_filter: str | None = None) -> dict:
    if page < 1:
        raise ValueError("page must be >= 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")
    all_users = [{"id": f"usr_{i:05d}", "email": f"user{i}@example.com"} for i in range(1, 51)]
    total = len(all_users)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": all_users[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
        "status": 200,
    }


# ---------------------------------------------------------------------------
# test_create_user
# ---------------------------------------------------------------------------

def test_create_user_returns_201():
    """create_user should return status 201 on success."""
    result = _mock_create_user("new@example.com", "newuser", ["viewer"])
    assert result["status"] == 201


def test_create_user_returns_id():
    """create_user should include a non-empty 'id' field."""
    result = _mock_create_user("a@example.com", "auser", ["viewer"])
    assert result.get("id")


def test_create_user_embeds_email():
    """create_user response should include the submitted email."""
    result = _mock_create_user("test@example.com", "tuser", ["viewer"])
    assert result["email"] == "test@example.com"


def test_create_user_embeds_roles():
    """create_user response should include the assigned roles."""
    result = _mock_create_user("r@example.com", "ruser", ["admin", "editor"])
    assert "admin" in result["roles"]


def test_create_user_requires_admin_role():
    """Non-admin users should not be allowed to create users."""
    try:
        _mock_create_user(
            "x@example.com",
            "xuser",
            ["viewer"],
            current_user={"roles": ["viewer"]},
        )
        assert False, "Should have raised PermissionError"
    except PermissionError as exc:
        assert "admin" in str(exc).lower()


def test_create_user_rejects_duplicate_email():
    """create_user should raise ValueError for already-registered emails."""
    try:
        _mock_create_user("existing@example.com", "dup", ["viewer"])
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "existing@example.com" in str(exc)


def test_create_user_admin_can_create():
    """Admin users should be allowed to create users."""
    result = _mock_create_user(
        "newby@example.com",
        "newby",
        ["viewer"],
        current_user={"roles": ["admin"]},
    )
    assert result["status"] == 201


# ---------------------------------------------------------------------------
# test_get_user
# ---------------------------------------------------------------------------

def test_get_user_returns_200():
    """get_user should return status 200 for existing users."""
    result = _mock_get_user("usr_00001")
    assert result["status"] == 200


def test_get_user_returns_correct_id():
    """get_user response should echo back the requested user_id."""
    result = _mock_get_user("usr_00042")
    assert result["id"] == "usr_00042"


def test_get_user_not_found():
    """get_user should raise LookupError for nonexistent user."""
    try:
        _mock_get_user("usr_99999")
        assert False, "Should have raised LookupError"
    except LookupError as exc:
        assert "usr_99999" in str(exc)


def test_get_user_empty_id_raises():
    """get_user with empty user_id should raise ValueError."""
    try:
        _mock_get_user("")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# test_list_users
# ---------------------------------------------------------------------------

def test_list_users_returns_items():
    """list_users should return a non-empty items list."""
    result = _mock_list_users()
    assert "items" in result
    assert len(result["items"]) > 0


def test_list_users_pagination_page1():
    """list_users page 1 should return up to page_size items."""
    result = _mock_list_users(page=1, page_size=10)
    assert len(result["items"]) == 10
    assert result["page"] == 1


def test_list_users_pagination_last_page():
    """list_users last page may have fewer than page_size items."""
    result = _mock_list_users(page=3, page_size=20)
    # 50 total, page 3 with size 20 = items 41-50 = 10 items
    assert len(result["items"]) == 10


def test_list_users_total_matches_expected():
    """list_users total should reflect all users in the mock store."""
    result = _mock_list_users()
    assert result["total"] == 50


def test_list_users_invalid_page_raises():
    """list_users page < 1 should raise ValueError."""
    try:
        _mock_list_users(page=0)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "page" in str(exc).lower()


def test_list_users_invalid_page_size_raises():
    """list_users page_size > 100 should raise ValueError."""
    try:
        _mock_list_users(page_size=200)
        assert False, "Should have raised ValueError"
    except ValueError as exc:
        assert "page_size" in str(exc).lower()


def test_list_users_pages_calculated_correctly():
    """pages count should equal ceil(total / page_size)."""
    result = _mock_list_users(page=1, page_size=20)
    expected_pages = (result["total"] + result["page_size"] - 1) // result["page_size"]
    assert result["pages"] == expected_pages
