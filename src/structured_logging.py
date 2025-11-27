"""
Structured Logging Configuration for MCP Server

This module provides production-ready structured logging using structlog with:
- JSON output for production (machine-parseable)
- Pretty printing for development (human-readable)
- Performance tracking with request context
- Correlation IDs for distributed tracing
- Automatic error context capture

Architecture:
- Uses structlog for structured logging
- orjson for fast JSON serialization
- Canonical log lines (one per request)
- Performance metrics (latency, memory)
- Thread-local context binding

Best Practices (following MCP and structlog recommendations):
1. Log to unbuffered stdout (12-factor app)
2. Use JSON in production for log aggregators (ELK, Datadog)
3. Include correlation IDs for request tracing
4. Capture performance metrics (duration, memory)
5. Minimize log entries (canonical log lines)

Usage:
    from src.structured_logging import get_logger, track_performance

    logger = get_logger("my_module")
    logger.info("operation_started", file_id="doc123", user_id="user456")

    @track_performance
    async def my_handler(tool_name, arguments, context):
        # Automatically logs duration and performance metrics
        return result
"""

import functools
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Optional

import psutil
import structlog

# Check if orjson is available (optional but recommended)
try:
    import orjson

    def orjson_dumps(obj: Any, **kwargs) -> bytes:
        """Wrapper for orjson.dumps that handles structlog's options parameter"""
        # orjson doesn't support 'default' parameter like json.dumps
        # structlog may pass 'default' in kwargs, so we ignore it
        return orjson.dumps(obj)

    HAS_ORJSON = True
    JSON_SERIALIZER = orjson_dumps
except ImportError:
    HAS_ORJSON = False
    import json

    def json_dumps(obj: Any, **kwargs) -> str:
        """Fallback to standard json if orjson not available"""
        return json.dumps(obj, **kwargs)

    JSON_SERIALIZER = json_dumps


# Environment detection for logging mode
def is_production() -> bool:
    """
    Detect if running in production environment.

    Returns True if:
    - ENVIRONMENT env var is "production"
    - stderr is not a TTY (running in Docker/systemd)
    """
    return os.getenv("ENVIRONMENT") == "production" or not sys.stderr.isatty()


def configure_structlog(
    log_level: str = "INFO",
    use_json: Optional[bool] = None,
    cache_loggers: bool = True,
) -> None:
    """
    Configure structlog with production-ready settings.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: Force JSON output (None = auto-detect from environment)
        cache_loggers: Cache loggers on first use for performance

    Features:
    - JSON output in production (or when use_json=True)
    - Pretty printing in development (or when use_json=False)
    - ISO 8601 timestamps in UTC
    - Structured exception tracebacks
    - Log level filtering
    - Context variable support (thread-local)
    """
    # Auto-detect JSON mode if not specified
    if use_json is None:
        use_json = is_production()

    # Shared processors (run for both dev and prod)
    shared_processors = [
        # Merge thread-local context (correlation IDs, session info)
        structlog.contextvars.merge_contextvars,
        # Add log level name
        structlog.processors.add_log_level,
        # Add timestamp (ISO 8601, UTC)
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if use_json:
        # Production: JSON output with structured tracebacks
        processors = shared_processors + [
            # Format exceptions as structured dict
            structlog.processors.dict_tracebacks,
            # Render as JSON
            structlog.processors.JSONRenderer(serializer=JSON_SERIALIZER),
        ]

        # Use BytesLoggerFactory if orjson (returns bytes), else PrintLoggerFactory
        # Note: BytesLogger doesn't support add_logger_name, so we bind logger names manually
        if HAS_ORJSON:
            logger_factory = structlog.BytesLoggerFactory()
        else:
            logger_factory = structlog.PrintLoggerFactory()
    else:
        # Development: Pretty printing with colors
        # Add logger name for stdlib compatibility
        processors = [
            *shared_processors,
            structlog.stdlib.add_logger_name,
            # Format exceptions as strings
            structlog.processors.format_exc_info,
            # Pretty console output (auto-detects rich for better tracebacks)
            structlog.dev.ConsoleRenderer(),
        ]
        logger_factory = structlog.PrintLoggerFactory()

    # Convert log level string to int
    log_level_int = getattr(logging, log_level.upper())

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=logger_factory,
        cache_logger_on_first_use=cache_loggers,
    )

    # Configure standard library logging to match
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_int,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger for a module with logger name bound to context.

    Args:
        name: Logger name (typically __name__ or module name)

    Returns:
        Structured logger with all configured processors and logger name bound

    Example:
        logger = get_logger(__name__)
        logger.info("operation_complete", file_id="doc123", duration_ms=42)
    """
    # Get base logger and bind the logger name to context
    # This works with both BytesLogger and PrintLogger
    return structlog.get_logger().bind(logger=name)


@contextmanager
def log_context(**kwargs):
    """
    Context manager for binding context variables to all logs in a scope.

    Args:
        **kwargs: Key-value pairs to bind to context

    Example:
        with log_context(request_id="req-123", user_id="user-456"):
            logger.info("processing_request")  # Includes request_id and user_id
            do_work()
            logger.info("request_complete")     # Also includes request_id and user_id
    """
    # Bind context variables
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)
    try:
        yield
    finally:
        # Clear context variables after scope
        structlog.contextvars.clear_contextvars()


def generate_request_id() -> str:
    """
    Generate a unique request ID for correlation.

    Returns:
        UUID v4 as string (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    return str(uuid.uuid4())


def track_performance(func: Callable) -> Callable:
    """
    Decorator for tracking performance metrics of MCP tool handlers.

    Automatically logs:
    - Tool name
    - Request ID (correlation)
    - Duration (milliseconds)
    - Memory delta (MB)
    - Success/failure status
    - Error details (if failed)

    Supports both sync and async functions.

    Example:
        @track_performance
        async def handle_ingest(tool_name, arguments, context):
            # Implementation
            return result

    Log Output (JSON):
        {
            "event": "mcp_tool_call",
            "tool_name": "ingest_context",
            "request_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "success",
            "duration_ms": 42.5,
            "memory_delta_mb": 1.2,
            "timestamp": "2025-01-26T12:34:56.789Z",
            "level": "info"
        }
    """
    # Get logger for the module containing the function
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        """Wrapper for async functions"""
        # Extract tool_name from arguments
        tool_name = args[0] if len(args) > 0 else kwargs.get("name", "unknown")

        # Generate request ID for correlation
        request_id = generate_request_id()

        # Capture initial memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.perf_counter()

        # Bind request context to all logs in this scope
        with log_context(request_id=request_id, tool_name=tool_name):
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Calculate performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta_mb = memory_after - memory_before

                # Log successful completion (canonical log line)
                logger.info(
                    "mcp_tool_call",
                    status="success",
                    duration_ms=round(duration_ms, 2),
                    memory_delta_mb=round(memory_delta_mb, 2),
                )

                return result

            except Exception as e:
                # Calculate metrics even on error
                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta_mb = memory_after - memory_before

                # Log error with context (canonical log line)
                logger.error(
                    "mcp_tool_call",
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_ms=round(duration_ms, 2),
                    memory_delta_mb=round(memory_delta_mb, 2),
                    exc_info=True,  # Include structured traceback
                )

                # Re-raise to preserve exception handling
                raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        """Wrapper for sync functions"""
        # Extract tool_name from arguments
        tool_name = args[0] if len(args) > 0 else kwargs.get("name", "unknown")

        # Generate request ID for correlation
        request_id = generate_request_id()

        # Capture initial memory usage
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.perf_counter()

        # Bind request context to all logs in this scope
        with log_context(request_id=request_id, tool_name=tool_name):
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Calculate performance metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta_mb = memory_after - memory_before

                # Log successful completion (canonical log line)
                logger.info(
                    "mcp_tool_call",
                    status="success",
                    duration_ms=round(duration_ms, 2),
                    memory_delta_mb=round(memory_delta_mb, 2),
                )

                return result

            except Exception as e:
                # Calculate metrics even on error
                duration_ms = (time.perf_counter() - start_time) * 1000
                memory_after = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta_mb = memory_after - memory_before

                # Log error with context (canonical log line)
                logger.error(
                    "mcp_tool_call",
                    status="error",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    duration_ms=round(duration_ms, 2),
                    memory_delta_mb=round(memory_delta_mb, 2),
                    exc_info=True,  # Include structured traceback
                )

                # Re-raise to preserve exception handling
                raise

    # Return appropriate wrapper based on function type
    import inspect

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Initialize structlog on module import
# This ensures logging is configured before any loggers are created
configure_structlog()
