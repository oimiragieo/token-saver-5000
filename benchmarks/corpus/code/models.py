"""Data models for the application.

Defines core domain entities as dataclasses with basic validation.
No ORM dependencies — pure Python data structures.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class ProjectStatus(Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DRAFT = "draft"


@dataclass
class User:
    """Application user account model."""

    id: str
    email: str
    username: str
    roles: list[str] = field(default_factory=lambda: ["viewer"])
    display_name: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("User.id must not be empty")
        if not self.email or "@" not in self.email:
            raise ValueError(f"User.email is invalid: {self.email!r}")
        if not self.username:
            raise ValueError("User.username must not be empty")
        if not self.display_name:
            self.display_name = self.username

    def has_role(self, role: str) -> bool:
        """Return True if the user holds the given role."""
        return role in self.roles

    def to_public_dict(self) -> dict[str, Any]:
        """Return a safe representation excluding sensitive fields."""
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "display_name": self.display_name,
            "roles": self.roles,
            "is_active": self.is_active,
        }


@dataclass
class Project:
    """A project that groups related tasks."""

    id: str
    name: str
    owner_id: str
    description: str = ""
    status: ProjectStatus = ProjectStatus.ACTIVE
    tags: list[str] = field(default_factory=list)
    member_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Project.id must not be empty")
        if not self.name or len(self.name) > 200:
            raise ValueError("Project.name must be 1-200 characters")
        if not self.owner_id:
            raise ValueError("Project.owner_id must not be empty")
        # Normalise tags
        self.tags = [t.lower().strip() for t in self.tags if t.strip()]

    def add_member(self, user_id: str) -> None:
        """Add a member to the project if not already present."""
        if user_id and user_id not in self.member_ids:
            self.member_ids.append(user_id)

    def remove_member(self, user_id: str) -> None:
        """Remove a member from the project."""
        self.member_ids = [m for m in self.member_ids if m != user_id]

    def is_member(self, user_id: str) -> bool:
        """Return True if user_id is the owner or an explicit member."""
        return user_id == self.owner_id or user_id in self.member_ids


@dataclass
class Task:
    """A unit of work within a project."""

    id: str
    title: str
    project_id: str
    assignee_id: str | None = None
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    priority: int = 3  # 1=highest, 5=lowest
    due_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task.id must not be empty")
        if not self.title or len(self.title) > 500:
            raise ValueError("Task.title must be 1-500 characters")
        if not 1 <= self.priority <= 5:
            raise ValueError("Task.priority must be between 1 and 5")

    def assign_to(self, user_id: str) -> None:
        """Assign this task to a user."""
        self.assignee_id = user_id
        self.updated_at = datetime.utcnow()

    def complete(self) -> None:
        """Mark task as done."""
        self.status = TaskStatus.DONE
        self.updated_at = datetime.utcnow()

    def is_overdue(self) -> bool:
        """Return True if due_date is in the past and task is not done."""
        if self.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return False
        if self.due_date is None:
            return False
        return datetime.utcnow() > self.due_date


@dataclass
class Comment:
    """A comment attached to a task."""

    id: str
    task_id: str
    author_id: str
    body: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    is_deleted: bool = False
    mentions: list[str] = field(default_factory=list)

    _MENTION_RE: Any = field(default=None, repr=False, compare=False, init=False)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Comment.id must not be empty")
        if not self.task_id:
            raise ValueError("Comment.task_id must not be empty")
        if not self.body.strip():
            raise ValueError("Comment.body must not be blank")
        # Extract @mentions from body
        pattern = re.compile(r"@(\w+)")
        self.mentions = pattern.findall(self.body)

    def edit(self, new_body: str) -> None:
        """Update comment body and refresh updated_at."""
        if not new_body.strip():
            raise ValueError("Comment body must not be blank")
        self.body = new_body
        self.updated_at = datetime.utcnow()
        pattern = re.compile(r"@(\w+)")
        self.mentions = pattern.findall(new_body)

    def soft_delete(self) -> None:
        """Mark the comment as deleted without removing it."""
        self.is_deleted = True
        self.updated_at = datetime.utcnow()
