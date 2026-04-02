"""General-purpose utility helpers.

String manipulation, pagination, validation, date formatting, and retry logic.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def slugify(text: str, separator: str = "-", max_length: int = 80) -> str:
    """Convert arbitrary text to a URL-safe slug.

    Args:
        text: Input text to slugify.
        separator: Character used to replace whitespace and special chars.
        max_length: Maximum length of the resulting slug.

    Returns:
        Lowercase slug string.

    Raises:
        ValueError: If text is empty after stripping.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("  My  Project  Name  ", separator="_")
        'my_project_name'
    """
    if not text or not text.strip():
        raise ValueError("Cannot slugify empty or whitespace-only text")

    slug = text.lower().strip()
    # Replace non-alphanumeric runs with separator
    slug = re.sub(r"[^a-z0-9]+", separator, slug)
    # Remove leading/trailing separators
    slug = slug.strip(separator)
    return slug[:max_length]


def paginate(
    items: list[Any],
    page: int,
    page_size: int,
) -> dict[str, Any]:
    """Slice a list into a paginated response envelope.

    Args:
        items: Full list of items to paginate.
        page: 1-based page number.
        page_size: Number of items per page.

    Returns:
        Dict with keys: items, page, page_size, total, pages, has_next, has_prev.

    Raises:
        ValueError: If page < 1 or page_size < 1.
    """
    if page < 1:
        raise ValueError(f"page must be >= 1, got {page}")
    if page_size < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")

    total = len(items)
    total_pages = max(1, (total + page_size - 1) // page_size)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "items": items[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": total_pages,
        "has_next": page < total_pages,
        "has_prev": page > 1,
    }


_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")


def validate_email(email: str) -> bool:
    """Return True if email matches a basic RFC-5321 pattern.

    Args:
        email: Email address string to validate.

    Returns:
        True if valid, False otherwise.
    """
    if not email or len(email) > 254:
        return False
    return bool(_EMAIL_RE.match(email))


def format_date(
    dt: datetime,
    fmt: str = "iso",
    tz: timezone = timezone.utc,
) -> str:
    """Format a datetime object as a string.

    Args:
        dt: Datetime to format.
        fmt: One of 'iso', 'human', 'date', 'timestamp'.
        tz: Timezone to normalise to before formatting.

    Returns:
        Formatted string.

    Raises:
        ValueError: If fmt is not recognised.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_local = dt.astimezone(tz)

    if fmt == "iso":
        return dt_local.isoformat()
    elif fmt == "human":
        return dt_local.strftime("%B %d, %Y at %H:%M UTC")
    elif fmt == "date":
        return dt_local.strftime("%Y-%m-%d")
    elif fmt == "timestamp":
        return str(int(dt_local.timestamp()))
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Expected one of: iso, human, date, timestamp")


def retry_with_backoff(
    func: Callable[..., Any],
    max_retries: int = 3,
    base_delay_sec: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Any:
    """Call func with exponential backoff on failure.

    Args:
        func: Zero-argument callable to invoke.
        max_retries: Maximum number of retry attempts.
        base_delay_sec: Initial delay in seconds before first retry.
        backoff_factor: Multiplier applied to delay after each retry.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Return value of func on success.

    Raises:
        The last exception raised by func after all retries are exhausted.
    """
    last_exc: Exception | None = None
    delay = base_delay_sec

    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(delay)
                delay *= backoff_factor

    raise last_exc  # type: ignore[misc]


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated.

    Args:
        text: Source string.
        max_length: Maximum character length of the result (including suffix).
        suffix: String appended when truncation occurs.

    Returns:
        Original text if short enough; otherwise truncated text + suffix.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary (not mutated).
        override: Dictionary whose values take precedence.

    Returns:
        New merged dictionary.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
