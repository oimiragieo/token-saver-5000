"""Structured logging service with JSON and human-readable formatters.

Provides a Logger class that wraps Python's logging module with context
propagation, request ID tracking, and configurable output formats.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogRecord:
    """Structured log entry."""

    level: str
    message: str
    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    user_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "user_id": self.user_id,
            **self.extra,
        }


def json_formatter(record: LogRecord) -> str:
    """Serialize a LogRecord to a compact JSON string.

    Args:
        record: LogRecord to serialize.

    Returns:
        JSON string with all fields present.
    """
    return json.dumps(record.to_dict(), separators=(",", ":"), default=str)


def human_formatter(record: LogRecord) -> str:
    """Format a LogRecord as a human-readable log line.

    Args:
        record: LogRecord to format.

    Returns:
        String of the form: [LEVEL] HH:MM:SS - message [extra_key=value ...]
    """
    ts = time.strftime("%H:%M:%S", time.localtime(record.timestamp))
    parts = [f"[{record.level.upper():<7}]", ts, "-", record.message]
    if record.request_id:
        parts.append(f"req={record.request_id[:8]}")
    if record.user_id:
        parts.append(f"user={record.user_id}")
    for k, v in record.extra.items():
        parts.append(f"{k}={v}")
    return " ".join(parts)


class Logger:
    """Application logger with context tracking and pluggable formatters.

    Wraps stdlib logging while adding structured fields (request_id, user_id)
    and a swappable formatter for JSON vs human-readable output.
    """

    def __init__(
        self,
        name: str = "app",
        level: str = "INFO",
        fmt: str = "json",
        stream: Any = None,
    ) -> None:
        self._name = name
        self._level = logging.getLevelName(level.upper())
        self._fmt = fmt
        self._stream = stream or sys.stdout
        self._context: dict[str, Any] = {}
        self._stdlib = logging.getLogger(name)

    def set_level(self, level: str) -> None:
        """Change the logging level at runtime.

        Args:
            level: Level string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        self._level = logging.getLevelName(level.upper())
        self._stdlib.setLevel(self._level)

    def set_context(self, **kwargs: Any) -> None:
        """Attach key-value pairs to all subsequent log records."""
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Remove all attached context fields."""
        self._context.clear()

    def _emit(self, level: str, message: str, **extra: Any) -> None:
        level_num = logging.getLevelName(level.upper())
        if level_num < self._level:
            return

        merged_extra = {**self._context, **extra}
        record = LogRecord(
            level=level,
            message=message,
            request_id=merged_extra.pop("request_id", ""),
            user_id=merged_extra.pop("user_id", ""),
            extra=merged_extra,
        )

        if self._fmt == "json":
            line = json_formatter(record)
        else:
            line = human_formatter(record)

        print(line, file=self._stream)

    def debug(self, message: str, **extra: Any) -> None:
        self._emit("debug", message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        self._emit("info", message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        self._emit("warning", message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        self._emit("error", message, **extra)

    def critical(self, message: str, **extra: Any) -> None:
        self._emit("critical", message, **extra)

    def log_request(
        self,
        method: str,
        path: str,
        status: int,
        duration_ms: float,
        request_id: str | None = None,
    ) -> None:
        """Emit a structured HTTP request log entry.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path.
            status: HTTP response status code.
            duration_ms: Request duration in milliseconds.
            request_id: Optional correlation ID.
        """
        rid = request_id or str(uuid.uuid4())
        level = "info" if status < 400 else "error"
        self._emit(
            level,
            f"{method} {path} -> {status}",
            request_id=rid,
            duration_ms=round(duration_ms, 2),
            status=status,
        )


_default_logger: Logger | None = None


def get_logger(name: str = "app", fmt: str = "json") -> Logger:
    """Return or create the shared application logger.

    Args:
        name: Logger name (used as stdlib logger name).
        fmt: Output format ('json' or 'human').

    Returns:
        Logger instance.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = Logger(name=name, fmt=fmt)
    return _default_logger
