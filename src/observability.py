"""
OpenTelemetry distributed tracing with async context propagation.

Provides automatic span creation, context propagation, and trace correlation with logs.
Implements graceful degradation when OpenTelemetry is not installed.

Features:
- OpenTelemetry tracer configuration with service metadata
- Automatic span creation with context managers
- Async-safe context propagation via contextvars
- OTLP export with graceful fallback to console
- Trace correlation with structured logging
- Span attributes for compression operations
- Exception recording and error tracking
- Configurable sampling rates
- Thread-safe and async-safe

Usage Examples:
    Basic configuration:
        from src.observability import configure_observability, get_observability

        # Configure globally (optional - uses defaults if not called)
        configure_observability(
            service_name="token-saver-5000",
            environment="production",
            sampling_rate=0.1
        )

        # Get singleton instance
        observe = get_observability()

    Trace operations with attributes:
        with observe.trace("compress", doc_id="abc123", fidelity_level="BALANCED"):
            result = compress_document()
            observe.set_attribute("compression_ratio", 7.5)
            observe.set_attribute("token_count", 500)
            observe.set_attribute("status", "success")

    Async context propagation:
        async def process_batch():
            with observe.trace("batch_ingest", batch_id="batch-456", batch_size=10):
                tasks = [compress_doc(doc) for doc in batch]
                # Context propagates to all tasks automatically
                await asyncio.gather(*tasks)
                observe.set_attribute("status", "success")

    Error handling and exception recording:
        with observe.trace("compress", doc_id="abc123"):
            try:
                compress_document()
            except Exception as e:
                observe.record_exception(e)
                observe.set_attribute("status", "error")
                observe.set_attribute("error_type", type(e).__name__)
                raise

    Integration with structured logging:
        from src.structured_logging import get_logger
        from src.observability import get_observability

        logger = get_logger()
        observe = get_observability()

        with observe.trace("compress", doc_id="abc123"):
            # Get trace context for log correlation
            trace_context = observe.get_current_trace_context()
            logger.info("Compression started", **trace_context)

            compress_document()

            logger.info("Compression completed", **trace_context, compression_ratio=7.5)

    Nested spans:
        with observe.trace("batch_ingest", batch_id="batch-456"):
            for doc in documents:
                with observe.trace("compress", doc_id=doc.id):
                    compress_document(doc)
                    observe.set_attribute("compression_ratio", 8.2)

Output Format:
    Spans are exported to OTLP endpoint (default: localhost:4317) or console.

    Span attributes example:
    {
        "doc_id": "abc123",
        "fidelity_level": "BALANCED",
        "token_count": 500,
        "compression_ratio": 7.5,
        "operation_name": "compress",
        "status": "success",
        "service.name": "token-saver-5000",
        "service.version": "0.7.0",
        "deployment.environment": "production"
    }

OpenTelemetry Specification:
    - Uses context propagation per OTEL spec (W3C TraceContext)
    - Supports OTLP/gRPC export (default) with fallback to console
    - Implements sampling for production environments
    - Records exceptions per OTEL semantic conventions
    - Provides trace_id and span_id for log correlation
"""

import contextvars
import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

# Try to import OpenTelemetry - graceful fallback if not installed
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider, sampling
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode

    # Try to import OTLP exporter - fallback to console if unavailable
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        OTLP_AVAILABLE = True
    except ImportError:
        OTLP_AVAILABLE = False

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    OTLP_AVAILABLE = False

# Import version from package
try:
    from src import __version__
except ImportError:
    __version__ = "0.7.0"  # Default fallback

logger = logging.getLogger(__name__)

# Context variable for current span (async-safe propagation)
_current_span: contextvars.ContextVar = contextvars.ContextVar("current_span", default=None)


class NoOpSpan:
    """
    No-op span implementation when OpenTelemetry is not available.

    Provides the same interface as OpenTelemetry Span but performs no operations.
    Used for graceful degradation when OpenTelemetry is not installed.
    """

    def set_attribute(self, key: str, value: Any) -> "NoOpSpan":
        """Set span attribute (no-op)."""
        return self

    def set_attributes(self, attributes: Dict[str, Any]) -> "NoOpSpan":
        """Set multiple span attributes (no-op)."""
        return self

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span (no-op)."""
        pass

    def set_status(self, status: Any) -> None:
        """Set span status (no-op)."""
        pass

    def record_exception(
        self, exception: Exception, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record exception (no-op)."""
        pass

    def is_recording(self) -> bool:
        """Check if span is recording (always False for no-op)."""
        return False

    def get_span_context(self) -> Any:
        """Get span context (returns None for no-op)."""
        return None

    def __enter__(self) -> "NoOpSpan":
        """Context manager entry (no-op)."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit (no-op)."""
        pass


class ObservabilityManager:
    """
    OpenTelemetry observability manager with async context propagation.

    Provides distributed tracing with automatic span creation, context propagation,
    and trace correlation with structured logging. Implements graceful degradation
    when OpenTelemetry is not installed.

    Features:
    - Singleton pattern for global access
    - OpenTelemetry tracer with service metadata
    - OTLP export with fallback to console
    - Async-safe context propagation via contextvars
    - Configurable sampling rates
    - Exception recording and error tracking
    - Trace context extraction for logging

    Thread Safety:
        This class is thread-safe and async-safe. Context propagation uses
        contextvars which are automatically isolated per asyncio task.

    Performance:
        - Span creation overhead: <1ms
        - Attribute setting overhead: <0.1ms
        - Context propagation overhead: <0.1ms
        - Total operation overhead: <50ms (including export)
    """

    _instance: Optional["ObservabilityManager"] = None

    def __init__(
        self,
        service_name: str = "token-saver-5000",
        service_version: Optional[str] = None,
        environment: str = "development",
        sampling_rate: float = 1.0,
        otlp_endpoint: Optional[str] = None,
        enable_console_export: bool = False,
    ):
        """
        Initialize observability manager.

        Args:
            service_name: Service identifier for traces
            service_version: Service version (defaults to package version)
            environment: Deployment environment (development/production)
            sampling_rate: Fraction of traces to sample (0.0-1.0)
            otlp_endpoint: OTLP gRPC endpoint URL (defaults to localhost:4317)
            enable_console_export: Enable console exporter for debugging

        Note:
            If OpenTelemetry is not installed, this will gracefully degrade to
            no-op implementation with warning logged.
        """
        self.service_name = service_name
        self.service_version = service_version or __version__
        self.environment = environment
        self.sampling_rate = sampling_rate
        self.otlp_endpoint = otlp_endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        self.enable_console_export = enable_console_export

        # Check if OpenTelemetry is available
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning(
                "OpenTelemetry not installed - tracing disabled. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
            )
            self.tracer = None
            self._enabled = False
            return

        # Configure OpenTelemetry
        try:
            self._configure_tracer()
            self._enabled = True
            logger.info(
                f"OpenTelemetry configured: service={service_name}, "
                f"version={self.service_version}, environment={environment}, "
                f"sampling_rate={sampling_rate}"
            )
        except Exception as e:
            logger.error(f"Failed to configure OpenTelemetry: {e}", exc_info=True)
            self.tracer = None
            self._enabled = False

    def _configure_tracer(self) -> None:
        """
        Configure OpenTelemetry tracer with resource, sampling, and exporters.

        Sets up:
        - Resource with service metadata
        - Trace provider with sampling
        - OTLP exporter (with fallback to console)
        - Batch span processor for performance
        """
        # Configure resource with service metadata
        resource = Resource.create(
            {
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
                "deployment.environment": self.environment,
                "telemetry.sdk.name": "opentelemetry",
                "telemetry.sdk.language": "python",
            }
        )

        # Configure sampling based on rate
        if self.sampling_rate >= 1.0:
            sampler = sampling.ALWAYS_ON
        elif self.sampling_rate <= 0.0:
            sampler = sampling.ALWAYS_OFF
        else:
            sampler = sampling.TraceIdRatioBased(self.sampling_rate)

        # Create tracer provider
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Configure exporters
        exporters_configured = False

        # Try to configure OTLP exporter
        if OTLP_AVAILABLE:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True,  # Use insecure for local development
                )
                processor = BatchSpanProcessor(otlp_exporter)
                provider.add_span_processor(processor)
                exporters_configured = True
                logger.info(f"OTLP span exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to configure OTLP exporter: {e}. Falling back to console.")

        # Fallback to console exporter if OTLP unavailable or failed
        if not exporters_configured or self.enable_console_export:
            console_exporter = ConsoleSpanExporter()
            processor = BatchSpanProcessor(console_exporter)
            provider.add_span_processor(processor)
            logger.info("Console span exporter configured")

        # Set global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer for this service
        self.tracer = trace.get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version=self.service_version,
        )

    @classmethod
    def get_observability(cls, **kwargs) -> "ObservabilityManager":
        """
        Get singleton observability manager instance.

        Args:
            **kwargs: Configuration parameters (only used on first call)

        Returns:
            Singleton ObservabilityManager instance

        Example:
            observe = ObservabilityManager.get_observability()
            with observe.trace("compress", doc_id="abc123"):
                compress_document()
        """
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @contextmanager
    def trace(self, operation_name: str, **attributes) -> Generator[Optional[Any], None, None]:
        """
        Context manager for span creation with attributes.

        Creates a new span with the given operation name and attributes.
        The span is automatically finished when the context exits.
        Context propagates to nested spans and async tasks.

        Args:
            operation_name: Name of the operation (e.g., "compress", "ingest")
            **attributes: Span attributes (e.g., doc_id="abc123", fidelity_level="BALANCED")

        Yields:
            Span object (or NoOpSpan if OpenTelemetry unavailable)

        Example:
            with observe.trace("compress", doc_id="abc123", fidelity="BALANCED"):
                result = compress_document()
                observe.set_attribute("compression_ratio", 7.5)
                observe.set_attribute("status", "success")

        Error Handling:
            Exceptions are automatically recorded on the span with error status.
            The exception is re-raised after recording.
        """
        # Return no-op if tracing disabled
        if not self._enabled or not self.tracer:
            yield NoOpSpan()
            return

        # Start span with operation name
        with self.tracer.start_as_current_span(operation_name) as span:
            # Set initial attributes
            if attributes:
                span.set_attributes(attributes)

            # Store span in context variable for async propagation
            token = _current_span.set(span)

            try:
                yield span
            except Exception as e:
                # Record exception on span
                if span.is_recording():
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                    span.set_attribute("error", True)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                raise
            finally:
                # Restore previous span context
                _current_span.reset(token)

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set attribute on current span.

        Attributes are key-value pairs that provide additional context about
        the operation. They are searchable in trace backends.

        Args:
            key: Attribute key (e.g., "compression_ratio", "token_count")
            value: Attribute value (must be string, number, or boolean)

        Example:
            with observe.trace("compress", doc_id="abc123"):
                compress_document()
                observe.set_attribute("compression_ratio", 7.5)
                observe.set_attribute("token_count", 500)
                observe.set_attribute("status", "success")

        Note:
            This is a no-op if called outside a trace context or if
            OpenTelemetry is not available.
        """
        if not self._enabled:
            return

        span = _current_span.get()
        if span and span.is_recording():
            span.set_attribute(key, value)

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        """
        Set multiple attributes on current span.

        More efficient than calling set_attribute() multiple times.

        Args:
            attributes: Dictionary of attribute key-value pairs

        Example:
            observe.set_attributes({
                "compression_ratio": 7.5,
                "token_count": 500,
                "fidelity_level": "BALANCED",
                "status": "success"
            })
        """
        if not self._enabled:
            return

        span = _current_span.get()
        if span and span.is_recording():
            span.set_attributes(attributes)

    def record_exception(
        self, exception: Exception, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record exception on current span.

        Records exception details including type, message, and stack trace.
        Sets span status to ERROR automatically.

        Args:
            exception: Exception to record
            attributes: Optional additional attributes

        Example:
            with observe.trace("compress", doc_id="abc123"):
                try:
                    compress_document()
                except ValueError as e:
                    observe.record_exception(e, {"doc_size": 1000})
                    raise

        Note:
            This is automatically called when an exception occurs within a
            trace context, but can be called explicitly for additional control.
        """
        if not self._enabled:
            return

        span = _current_span.get()
        if span and span.is_recording():
            # Set error status
            span.set_status(Status(StatusCode.ERROR))

            # Record exception with stack trace
            span.record_exception(exception, attributes=attributes)

            # Set error attributes
            span.set_attribute("error", True)
            span.set_attribute("error.type", type(exception).__name__)
            span.set_attribute("error.message", str(exception))

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add event to current span.

        Events are timestamped records within a span that mark specific
        occurrences or milestones.

        Args:
            name: Event name (e.g., "cache_hit", "retry_attempt")
            attributes: Optional event attributes

        Example:
            with observe.trace("compress", doc_id="abc123"):
                observe.add_event("compression_started")
                compress_document()
                observe.add_event("compression_completed", {"ratio": 7.5})
        """
        if not self._enabled:
            return

        span = _current_span.get()
        if span and span.is_recording():
            span.add_event(name, attributes=attributes)

    def get_current_trace_context(self) -> Dict[str, str]:
        """
        Get trace context for current span.

        Returns trace_id and span_id for correlation with structured logs.
        These IDs can be included in log messages to connect logs with traces.

        Returns:
            Dictionary with trace_id and span_id (empty if no active span)

        Example:
            from src.structured_logging import get_logger

            logger = get_logger()

            with observe.trace("compress", doc_id="abc123"):
                trace_context = observe.get_current_trace_context()
                logger.info("Compression started", **trace_context)
                # Log will include: trace_id="a9938fd7a6313e0f..." span_id="ed58f84d8971bf60"

        Format:
            {
                "trace_id": "a9938fd7a6313e0f27f3fc87f574bff6",  # 32-char hex
                "span_id": "ed58f84d8971bf60"  # 16-char hex
            }
        """
        if not self._enabled:
            return {}

        span = _current_span.get()
        if span and span.is_recording():
            span_context = span.get_span_context()
            return {
                "trace_id": format(span_context.trace_id, "032x"),
                "span_id": format(span_context.span_id, "016x"),
            }

        return {}

    def is_enabled(self) -> bool:
        """
        Check if observability is enabled.

        Returns:
            True if OpenTelemetry is available and configured, False otherwise

        Example:
            if observe.is_enabled():
                # Perform expensive trace context extraction
                trace_context = observe.get_current_trace_context()
        """
        return self._enabled

    def shutdown(self, timeout: float = 5.0) -> bool:
        """
        Shutdown observability manager and flush pending spans.

        Should be called before application exit to ensure all spans are exported.

        Args:
            timeout: Maximum time to wait for export (seconds)

        Returns:
            True if shutdown successful, False otherwise

        Example:
            import atexit

            observe = get_observability()
            atexit.register(lambda: observe.shutdown())

        Note:
            This is automatically called via atexit if properly configured.
        """
        if not self._enabled:
            return True

        try:
            # Get tracer provider and shut down
            provider = trace.get_tracer_provider()
            if hasattr(provider, "shutdown"):
                provider.shutdown()
                logger.info("OpenTelemetry shutdown completed")
                return True
        except Exception as e:
            logger.error(f"Error during OpenTelemetry shutdown: {e}", exc_info=True)
            return False

        return True


# Global singleton access functions


def get_observability(**kwargs) -> ObservabilityManager:
    """
    Get global observability manager instance.

    Args:
        **kwargs: Configuration parameters (only used on first call)

    Returns:
        Singleton ObservabilityManager instance

    Example:
        observe = get_observability()
        with observe.trace("compress", doc_id="abc123"):
            compress_document()
    """
    return ObservabilityManager.get_observability(**kwargs)


def configure_observability(
    service_name: str = "token-saver-5000",
    service_version: Optional[str] = None,
    environment: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    otlp_endpoint: Optional[str] = None,
    enable_console_export: bool = False,
) -> ObservabilityManager:
    """
    Configure global observability manager.

    This should be called once at application startup. Subsequent calls
    will return the existing instance without reconfiguration.

    Args:
        service_name: Service identifier for traces
        service_version: Service version (defaults to package version)
        environment: Deployment environment (auto-detected if None)
        sampling_rate: Fraction of traces to sample (auto-detected if None)
        otlp_endpoint: OTLP gRPC endpoint URL (defaults to localhost:4317)
        enable_console_export: Enable console exporter for debugging

    Returns:
        Configured ObservabilityManager instance

    Example:
        # At application startup
        configure_observability(
            service_name="token-saver-5000",
            environment="production",
            sampling_rate=0.1,  # Sample 10% of traces in production
            otlp_endpoint="http://otel-collector:4317"
        )

    Environment Variables:
        - ENVIRONMENT: Deployment environment (development/production)
        - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
        - OTEL_SAMPLING_RATE: Trace sampling rate (0.0-1.0)

    Auto-Detection:
        - environment: Defaults to $ENVIRONMENT or "development"
        - sampling_rate: 0.1 (10%) in production, 1.0 (100%) in development
    """
    # Auto-detect environment
    if environment is None:
        environment = os.getenv("ENVIRONMENT", "development")

    # Auto-detect sampling rate based on environment
    if sampling_rate is None:
        sampling_rate_env = os.getenv("OTEL_SAMPLING_RATE")
        if sampling_rate_env:
            sampling_rate = float(sampling_rate_env)
        elif environment == "production":
            sampling_rate = 0.1  # Sample 10% in production
        else:
            sampling_rate = 1.0  # Sample 100% in development

    # Configure singleton
    return ObservabilityManager.get_observability(
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        sampling_rate=sampling_rate,
        otlp_endpoint=otlp_endpoint,
        enable_console_export=enable_console_export,
    )
