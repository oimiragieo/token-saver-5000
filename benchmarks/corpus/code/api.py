"""REST API endpoints for user resource management.

Implements CRUD operations for users with validation, pagination, and
authentication guards. All handlers are mock implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class APIRouter:
    """Lightweight router that maps method+path to handler functions."""

    prefix: str = "/api/v1"
    routes: dict[str, Any] = field(default_factory=dict)
    middleware: list[Any] = field(default_factory=list)

    def register(self, method: str, path: str, handler: Any) -> None:
        """Register a handler for the given HTTP method and path."""
        key = f"{method.upper()}:{self.prefix}{path}"
        self.routes[key] = handler

    def dispatch(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """Dispatch a request to the appropriate handler."""
        key = f"{method.upper()}:{path}"
        handler = self.routes.get(key)
        if handler is None:
            return {"error": "Not Found", "status": 404}
        return handler(**kwargs)


@dataclass
class CreateUserRequest:
    """Validated request body for creating a user."""

    email: str
    username: str
    password: str
    roles: list[str] = field(default_factory=lambda: ["viewer"])
    display_name: str = ""


@dataclass
class UpdateUserRequest:
    """Validated request body for updating an existing user."""

    display_name: str | None = None
    roles: list[str] | None = None
    is_active: bool | None = None


def create_user(
    request: CreateUserRequest,
    db_pool: Any = None,
    current_user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new user account.

    Args:
        request: Validated user creation payload.
        db_pool: Database connection pool for persistence.
        current_user: Decoded JWT payload of the requesting user.

    Returns:
        Dict with created user data and HTTP status 201.

    Raises:
        PermissionError: If current_user lacks 'admin' role.
        ValueError: If email or username is already taken.
    """
    if current_user and "admin" not in current_user.get("roles", []):
        raise PermissionError("Only admins can create users")

    # Mock uniqueness check
    if request.email == "existing@example.com":
        raise ValueError(f"Email already registered: {request.email}")

    user_id = f"usr_{hash(request.email) % 100_000:05d}"
    return {
        "id": user_id,
        "email": request.email,
        "username": request.username,
        "roles": request.roles,
        "display_name": request.display_name or request.username,
        "is_active": True,
        "status": 201,
    }


def get_user(
    user_id: str,
    db_pool: Any = None,
    current_user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Retrieve a user by ID.

    Args:
        user_id: The unique user identifier.
        db_pool: Database connection pool.
        current_user: Decoded JWT payload of the requesting user.

    Returns:
        User resource dict with status 200.

    Raises:
        LookupError: If the user is not found.
        PermissionError: If requesting user cannot view target user.
    """
    if not user_id:
        raise ValueError("user_id must not be empty")

    # Mock: simulate not found
    if user_id == "usr_99999":
        raise LookupError(f"User not found: {user_id}")

    return {
        "id": user_id,
        "email": f"{user_id}@example.com",
        "username": user_id,
        "roles": ["viewer"],
        "display_name": user_id,
        "is_active": True,
        "status": 200,
    }


def update_user(
    user_id: str,
    request: UpdateUserRequest,
    db_pool: Any = None,
    current_user: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Partially update an existing user.

    Args:
        user_id: Target user identifier.
        request: Fields to update; None fields are left unchanged.
        db_pool: Database connection pool.
        current_user: Decoded JWT payload of the requesting user.

    Returns:
        Updated user resource dict with status 200.

    Raises:
        LookupError: If user not found.
        PermissionError: If non-admin user tries to modify roles.
    """
    if not user_id:
        raise ValueError("user_id must not be empty")

    existing = get_user(user_id, db_pool, current_user)

    if request.roles is not None:
        if current_user and "admin" not in current_user.get("roles", []):
            raise PermissionError("Only admins can change roles")
        existing["roles"] = request.roles

    if request.display_name is not None:
        existing["display_name"] = request.display_name

    if request.is_active is not None:
        existing["is_active"] = request.is_active

    existing["status"] = 200
    return existing


def delete_user(
    user_id: str,
    db_pool: Any = None,
    current_user: dict[str, Any] | None = None,
    hard_delete: bool = False,
) -> dict[str, Any]:
    """Delete (or soft-delete) a user.

    Args:
        user_id: Target user identifier.
        db_pool: Database connection pool.
        current_user: Decoded JWT payload.
        hard_delete: If True, permanently remove; otherwise mark inactive.

    Returns:
        Confirmation dict with status 204.

    Raises:
        PermissionError: If current_user lacks admin role.
        LookupError: If user is not found.
    """
    if current_user and "admin" not in current_user.get("roles", []):
        raise PermissionError("Only admins can delete users")

    _ = get_user(user_id, db_pool, current_user)  # raises LookupError if missing

    return {
        "deleted": user_id,
        "hard_delete": hard_delete,
        "status": 204,
    }


def list_users(
    db_pool: Any = None,
    current_user: dict[str, Any] | None = None,
    page: int = 1,
    page_size: int = 20,
    role_filter: str | None = None,
    is_active: bool | None = None,
) -> dict[str, Any]:
    """List users with optional filtering and pagination.

    Args:
        db_pool: Database connection pool.
        current_user: Decoded JWT payload.
        page: 1-based page number.
        page_size: Number of results per page (max 100).
        role_filter: If set, return only users with this role.
        is_active: If set, filter by active/inactive status.

    Returns:
        Paginated response dict with items, total, page, page_size.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    if not 1 <= page_size <= 100:
        raise ValueError("page_size must be between 1 and 100")

    # Mock data
    mock_users = [
        {"id": f"usr_{i:05d}", "email": f"user{i}@example.com", "roles": ["viewer"]}
        for i in range(1, 51)
    ]

    if role_filter:
        mock_users = [u for u in mock_users if role_filter in u["roles"]]

    total = len(mock_users)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "items": mock_users[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
        "status": 200,
    }
