"""
Structured logging with async context propagation and OpenTelemetry integration.

Provides JSON-formatted logs with automatic trace correlation, operation tracking,
and async-safe context propagation using contextvars.

Features:
- JSON output format for production parsing
- Human-readable format for development
- Async context propagation via contextvars
- OpenTelemetry trace correlation (optional)
- Operation tracking with context managers
- Request correlation IDs
- Log sampling for high-volume scenarios
- Thread-safe and async-safe

Usage Examples:
    Basic logging:
        logger = get_logger()
        logger.info("Document ingested", doc_id="abc123", token_count=500)
        logger.error("Processing failed", error=exception, doc_id="abc123")

    Operation tracking (auto-propagates context):
        with logger.operation("compress", doc_id="abc123", fidelity="BALANCED"):
            logger.debug("Compression started")  # Inherits doc_id and fidelity
            process_document()
            logger.info("Compression completed", compression_ratio=5.2)

    Async context propagation:
        async def process_batch():
            with logger.operation("batch_compress", batch_id="batch-456"):
                tasks = [compress_doc(doc) for doc in batch]
                await asyncio.gather(*tasks)  # Context propagates to all tasks

    Global configuration:
        configure_structlog(
            log_level="INFO",
            format="json",  # or "human"
            service="token-saver-5000",
            environment="production"
        )

Output Format (JSON):
    {
        "timestamp": "2025-11-26T10:30:45.123456Z",
        "level": "INFO",
        "message": "Document ingested",
        "service": "token-saver-5000",
        "version": "0.7.0",
        "environment": "production",
        "trace_id": "a9938fd7a6313e0f27f3fc87f574bff6",
        "span_id": "ed58f84d8971bf60",
        "operation_name": "compress",
        "doc_id": "abc123",
        "token_count": 500
    }
"""

import contextvars
import json
import logging
import os
import random
import sys
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

from .constants import VERSION

# Context variables for async propagation
_trace_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "trace_context", default={}
)
_operation_stack: contextvars.ContextVar[list] = contextvars.ContextVar(
    "operation_stack", default=[]
)

# PII fields to redact from logs (v0.7.0 security hardening)
REDACTED_FIELDS = {
    "password",
    "api_key",
    "secret",
    "token",
    "credential",
    "email",
    "ssn",
    "credit_card",
    "phone",
    "ip_address",
    "auth",
    "bearer",
    "private_key",
    "access_key",
}


def _redact_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Redact PII fields from context before logging.

    Recursively processes nested dictionaries and redacts any keys
    that contain PII-related substrings.

    Args:
        context: Context dictionary to redact

    Returns:
        Redacted context dictionary
    """
    if not context:
        return context

    redacted = {}
    for key, value in context.items():
        key_lower = key.lower()
        # Check if key contains any PII field name
        if any(field in key_lower for field in REDACTED_FIELDS):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = _redact_context(value)
        elif isinstance(value, list):
            # Redact lists of dicts
            redacted[key] = [
                _redact_context(item) if isinstance(item, dict) else item for item in value
            ]
        else:
            redacted[key] = value
    return redacted


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Converts log records to JSON format with automatic field extraction.
    Handles exceptions, stack traces, and custom context fields.
    """

    def __init__(self, service: str, version: str, environment: str):
        super().__init__()
        self.service = service
        self.version = version
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Base log structure
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "service": self.service,
            "version": self.version,
            "environment": self.environment,
            "logger": record.name,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add custom context fields from extra
        if hasattr(record, "_context"):
            log_data.update(record._context)

        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter for development.

    Provides colorized output with readable timestamps and context.
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as human-readable text."""
        # Colorize level name
        level_color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]
        colored_level = f"{level_color}{record.levelname:8}{reset}"

        # Format timestamp (local time for readability)
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Base message
        parts = [f"{timestamp} {colored_level} {record.getMessage()}"]

        # Add context fields if present
        if hasattr(record, "_context"):
            context = record._context
            if context:
                context_str = " ".join(f"{k}={v}" for k, v in context.items())
                parts.append(f" [{context_str}]")

        # Add exception if present
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            parts.append(f"\n{exc_text}")

        return "".join(parts)


class StructuredLogger:
    """
    Structured logger with async context propagation.

    Provides JSON-formatted logging with automatic trace correlation,
    operation tracking, and async-safe context management.

    Features:
    - Singleton pattern for global access
    - Async context propagation via contextvars
    - OpenTelemetry trace correlation (optional)
    - Operation tracking with context managers
    - Request correlation IDs
    - Log sampling for DEBUG logs
    """

    _instance: Optional["StructuredLogger"] = None
    _initialized: bool = False

    def __init__(
        self,
        name: str = "token-saver-5000",
        service: str = "token-saver-5000",
        version: str = VERSION,
        environment: str = "development",
        log_level: str = "INFO",
        format: str = "json",
        debug_sample_rate: float = 0.01,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            service: Service identifier
            version: Service version
            environment: Deployment environment (development/production)
            log_level: Minimum log level (DEBUG/INFO/WARNING/ERROR)
            format: Output format ("json" or "human")
            debug_sample_rate: Fraction of DEBUG logs to emit (0.0-1.0)
        """
        if StructuredLogger._initialized:
            return

        self.name = name
        self.service = service
        self.version = version
        self.environment = environment
        self.debug_sample_rate = debug_sample_rate

        # Create Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add formatter based on format type
        handler = logging.StreamHandler(sys.stdout)
        if format == "json":
            formatter = JSONFormatter(service, version, environment)
        else:
            formatter = HumanFormatter()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        StructuredLogger._initialized = True

    @classmethod
    def get_logger(
        cls,
        name: str = "token-saver-5000",
        **config,
    ) -> "StructuredLogger":
        """
        Get or create singleton logger instance.

        Args:
            name: Logger name
            **config: Configuration overrides (service, version, environment, etc.)

        Returns:
            Singleton StructuredLogger instance
        """
        if cls._instance is None:
            cls._instance = cls(name=name, **config)
        return cls._instance

    def _get_trace_context(self) -> Dict[str, str]:
        """
        Extract trace context from OpenTelemetry.

        Returns trace_id and span_id if OpenTelemetry is installed and active,
        otherwise returns empty dict. Gracefully handles missing dependency.

        Returns:
            Dict with trace_id and span_id, or empty dict
        """
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                return {
                    "trace_id": format(span_context.trace_id, "032x"),
                    "span_id": format(span_context.span_id, "016x"),
                }
        except ImportError:
            # OpenTelemetry not installed - graceful fallback
            pass
        except Exception:
            # Any other error - graceful fallback
            pass

        return {}

    def _get_current_context(self) -> Dict[str, Any]:
        """
        Get current context from contextvars.

        Merges operation stack context with trace context.

        Returns:
            Combined context dict
        """
        context = {}

        # Add trace context from contextvars
        try:
            trace_ctx = _trace_context.get()
            if trace_ctx:
                context.update(trace_ctx)
        except LookupError:
            pass

        # Add operation stack context
        try:
            op_stack = _operation_stack.get()
            if op_stack:
                # Merge all operation contexts from bottom to top
                for op_ctx in op_stack:
                    context.update(op_ctx)
        except LookupError:
            pass

        return context

    def _format_context(self, **context) -> Dict[str, Any]:
        """
        Merge default context, trace context, and custom context.

        Priority order (highest to lowest):
        1. Custom context (passed as kwargs)
        2. Current operation context
        3. Trace context from contextvars
        4. OpenTelemetry trace context

        PII redaction is applied before returning (v0.7.0 security).

        Args:
            **context: Custom context fields

        Returns:
            Merged and redacted context dict
        """
        merged = {}

        # Add OpenTelemetry trace context (lowest priority)
        merged.update(self._get_trace_context())

        # Add current context from contextvars (medium priority)
        merged.update(self._get_current_context())

        # Add custom context (highest priority)
        merged.update(context)

        # Apply PII redaction (v0.7.0 security hardening)
        return _redact_context(merged)

    def _should_sample_debug(self) -> bool:
        """
        Determine if DEBUG log should be emitted based on sample rate.

        Returns:
            True if log should be emitted, False otherwise
        """
        return random.random() < self.debug_sample_rate

    def debug(self, message: str, **context):
        """
        Log DEBUG level message.

        DEBUG logs are sampled based on debug_sample_rate to reduce volume
        in production environments.

        Args:
            message: Log message
            **context: Additional context fields
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        # Apply sampling
        if not self._should_sample_debug():
            return

        ctx = self._format_context(**context)
        self.logger.debug(message, extra={"_context": ctx})

    def info(self, message: str, **context):
        """
        Log INFO level message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        ctx = self._format_context(**context)
        self.logger.info(message, extra={"_context": ctx})

    def warning(self, message: str, **context):
        """
        Log WARNING level message.

        Args:
            message: Log message
            **context: Additional context fields
        """
        if not self.logger.isEnabledFor(logging.WARNING):
            return

        ctx = self._format_context(**context)
        self.logger.warning(message, extra={"_context": ctx})

    def error(self, message: str, error: Optional[Exception] = None, **context):
        """
        Log ERROR level message with optional exception.

        Args:
            message: Log message
            error: Optional exception to include
            **context: Additional context fields
        """
        if not self.logger.isEnabledFor(logging.ERROR):
            return

        ctx = self._format_context(**context)

        # Add error details if provided
        if error:
            ctx["error_type"] = type(error).__name__
            ctx["error_message"] = str(error)

        # Log with exception info if available
        exc_info = (
            (type(error), error, error.__traceback__)
            if (error and isinstance(error, BaseException))
            else None
        )
        self.logger.error(message, extra={"_context": ctx}, exc_info=exc_info)

    def critical(self, message: str, error: Optional[Exception] = None, **context):
        """
        Log CRITICAL level message with optional exception.

        Args:
            message: Log message
            error: Optional exception to include
            **context: Additional context fields
        """
        ctx = self._format_context(**context)

        # Add error details if provided
        if error:
            ctx["error_type"] = type(error).__name__
            ctx["error_message"] = str(error)

        # Log with exception info if available
        exc_info = (
            (type(error), error, error.__traceback__)
            if (error and isinstance(error, BaseException))
            else None
        )
        self.logger.critical(message, extra={"_context": ctx}, exc_info=exc_info)

    @contextmanager
    def operation(self, operation_name: str, **context) -> Generator[None, None, None]:
        """
        Context manager for operation tracking.

        Automatically adds operation_name and custom context to all logs
        within the context. Context propagates to async tasks.

        Args:
            operation_name: Name of the operation
            **context: Additional context fields

        Yields:
            None

        Example:
            with logger.operation("compress", doc_id="abc123", fidelity="BALANCED"):
                logger.debug("Compression started")  # Inherits doc_id and fidelity
                process_document()
                logger.info("Compression completed", compression_ratio=5.2)
        """
        # Create operation context
        op_context = {"operation_name": operation_name, **context}

        # Add request_id if not present
        if "request_id" not in op_context:
            op_context["request_id"] = str(uuid.uuid4())

        # Push onto operation stack
        try:
            stack = _operation_stack.get().copy()
        except LookupError:
            stack = []

        stack.append(op_context)
        token = _operation_stack.set(stack)

        try:
            yield
        finally:
            # Pop from operation stack
            _operation_stack.reset(token)


# Global configuration function
def configure_structlog(
    log_level: str = "INFO",
    format: str = "json",
    service: str = "token-saver-5000",
    version: str = VERSION,
    environment: Optional[str] = None,
    debug_sample_rate: float = 0.01,
):
    """
    Configure global structured logger.

    Args:
        log_level: Minimum log level (DEBUG/INFO/WARNING/ERROR)
        format: Output format ("json" or "human")
        service: Service identifier
        version: Service version
        environment: Deployment environment (auto-detected if None)
        debug_sample_rate: Fraction of DEBUG logs to emit (0.0-1.0)
    """
    # Auto-detect environment if not specified
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # Reset singleton state for reconfiguration
    StructuredLogger._initialized = False
    StructuredLogger._instance = None

    # Create new configured instance
    StructuredLogger.get_logger(
        service=service,
        version=version,
        environment=environment,
        log_level=log_level,
        format=format,
        debug_sample_rate=debug_sample_rate,
    )


# Global singleton access
def get_logger(name: str = "token-saver-5000") -> StructuredLogger:
    """
    Get global structured logger instance.

    Args:
        name: Logger name

    Returns:
        Singleton StructuredLogger instance
    """
    return StructuredLogger.get_logger(name=name)
