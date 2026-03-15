"""
Prometheus metrics collection with cardinality control.

This module provides comprehensive metrics tracking for the Token Saver 5000 MCP server,
including compression ratios, processing latency, throughput, cache performance, errors,
and batch processing sizes. Implements graceful degradation when prometheus_client is
not installed.

Features:
- Singleton pattern for global access
- Cardinality control to prevent label explosion
- Thread-safe metric operations (handled by prometheus_client)
- Graceful fallback when prometheus_client unavailable
- Comprehensive metrics for observability and performance tuning

Metrics Tracked:
- compression_ratio: Histogram of compression effectiveness by fidelity level
- processing_latency_seconds: Histogram of operation latency by operation and fidelity
- documents_processed_total: Counter of processed documents by operation, fidelity, status
- cache_hit_ratio: Gauge of current cache hit rate (0-1)
- active_documents: Gauge of currently loaded documents
- errors_total: Counter of errors by error_type and operation
- batch_size: Histogram of batch processing sizes

Usage Examples:
    Basic usage:
        >>> from src.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_compression_ratio(7.5, "BALANCED")
        >>> metrics.record_latency("compress", 0.35, "BALANCED")
        >>> metrics.increment_documents_processed("ingest", "HIGH", "success")

    With context manager for automatic latency measurement:
        >>> import time
        >>> from contextlib import contextmanager
        >>>
        >>> @contextmanager
        >>> def measure_latency(operation: str, fidelity: str = None):
        ...     start = time.perf_counter()
        ...     try:
        ...         yield
        ...     finally:
        ...         elapsed = time.perf_counter() - start
        ...         metrics.record_latency(operation, elapsed, fidelity)
        >>>
        >>> with measure_latency("compress", "BALANCED"):
        ...     compress_document()

    Prometheus scraping:
        >>> metrics_text = metrics.generate_metrics_text()
        >>> print(metrics_text)
        # HELP compression_ratio Compression ratio distribution
        # TYPE compression_ratio histogram
        compression_ratio_bucket{fidelity_level="BALANCED",le="1.0"} 0
        compression_ratio_bucket{fidelity_level="BALANCED",le="7.0"} 0
        compression_ratio_bucket{fidelity_level="BALANCED",le="10.0"} 1
        compression_ratio_sum{fidelity_level="BALANCED"} 7.5
        compression_ratio_count{fidelity_level="BALANCED"} 1
"""

import logging
from dataclasses import dataclass, asdict
from typing import Optional

# Try to import prometheus_client, fall back to NoOp if not available
try:
    from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Allowed label values for cardinality control
ALLOWED_FIDELITY_LEVELS = {"LOW", "BALANCED", "HIGH", "EXTREME", "NONE"}
ALLOWED_OPERATIONS = {"ingest", "compress", "expand", "batch_ingest", "refresh"}
ALLOWED_STATUSES = {"success", "failure"}

# Model pricing: cost per million input tokens (USD)
MODEL_PRICING = {
    "claude-opus-4": 15.0,
    "claude-opus-4.5": 15.0,
    "claude-opus-4.6": 15.0,
    "claude-sonnet-4": 3.0,
    "claude-sonnet-4.5": 3.0,
    "claude-sonnet-4.6": 3.0,
    "claude-haiku-3.5": 0.80,
    "claude-haiku-4": 0.80,
}
DEFAULT_COST_PER_MILLION = 3.0  # Default to Sonnet pricing


@dataclass
class TokenSavingsTelemetry:
    """Telemetry data for a single compression operation's cost savings."""

    original_tokens: int
    compressed_tokens: int
    saved_tokens: int
    model: str
    cost_per_million: float
    cost_savings_usd: float
    savings_percent: float

    def to_dict(self) -> dict:
        """Serialize to dict for JSON responses."""
        return asdict(self)


def compute_cost_savings(
    original_tokens: int,
    compressed_tokens: int,
    model: str = None,
) -> TokenSavingsTelemetry:
    """Calculate dollar savings from compression for a given model.

    Args:
        original_tokens: Token count before compression
        compressed_tokens: Token count after compression
        model: Model identifier (e.g. "claude-sonnet-4"). Defaults to Sonnet pricing.

    Returns:
        TokenSavingsTelemetry with cost savings breakdown
    """
    saved = original_tokens - compressed_tokens
    cost_per_million = MODEL_PRICING.get(model, DEFAULT_COST_PER_MILLION) if model else DEFAULT_COST_PER_MILLION
    cost_savings = (saved / 1_000_000) * cost_per_million
    savings_pct = (saved / original_tokens * 100) if original_tokens > 0 else 0.0

    return TokenSavingsTelemetry(
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        saved_tokens=saved,
        model=model or "default",
        cost_per_million=cost_per_million,
        cost_savings_usd=round(cost_savings, 6),
        savings_percent=round(savings_pct, 1),
    )


class MetricsCollector:
    """
    Prometheus metrics collector with cardinality control.

    Implements singleton pattern to ensure global access to a single metrics instance.
    Provides methods to record compression ratios, latency, throughput, cache performance,
    errors, and batch sizes with label validation to prevent cardinality explosion.

    Attributes:
        registry: Prometheus CollectorRegistry for metric storage
        compression_ratio: Histogram tracking compression effectiveness
        processing_latency: Histogram tracking operation latency
        documents_processed: Counter tracking processed documents
        cache_hit_ratio: Gauge tracking cache hit rate
        active_documents: Gauge tracking currently loaded documents
        errors_total: Counter tracking errors by type
        batch_size: Histogram tracking batch processing sizes
    """

    _instance: Optional["MetricsCollector"] = None  # Singleton instance

    def __init__(self):
        """
        Initialize MetricsCollector with Prometheus metrics.

        Private constructor - use get_metrics() to access singleton instance.
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning(
                "prometheus_client not installed, metrics collection disabled. "
                "Install with: pip install prometheus-client"
            )
            self._enabled = False
            return

        self._enabled = True
        self.registry = CollectorRegistry()

        # Compression ratio histogram with fidelity level labels
        # Buckets optimized for typical compression ratios (1x to 50x)
        self.compression_ratio = Histogram(
            "compression_ratio",
            "Compression ratio distribution (original_tokens / compressed_tokens)",
            labelnames=["fidelity_level"],
            buckets=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50],
            registry=self.registry,
        )

        # Processing latency histogram with operation and fidelity labels
        # Buckets optimized for sub-second to multi-second operations
        self.processing_latency = Histogram(
            "processing_latency_seconds",
            "Processing latency distribution in seconds",
            labelnames=["operation", "fidelity_level"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
            registry=self.registry,
        )

        # Documents processed counter with operation, fidelity, and status labels
        self.documents_processed = Counter(
            "documents_processed_total",
            "Total number of documents processed",
            labelnames=["operation", "fidelity_level", "status"],
            registry=self.registry,
        )

        # Cache hit ratio gauge (simple gauge, no labels)
        self.cache_hit_ratio = Gauge(
            "cache_hit_ratio",
            "Current cache hit ratio (0.0 to 1.0)",
            registry=self.registry,
        )

        # Active documents gauge (simple gauge, no labels)
        self.active_documents = Gauge(
            "active_documents",
            "Number of currently loaded documents",
            registry=self.registry,
        )

        # Errors counter with error_type and operation labels
        self.errors_total = Counter(
            "errors_total",
            "Total number of errors encountered",
            labelnames=["error_type", "operation"],
            registry=self.registry,
        )

        # Batch size histogram with operation label
        # Buckets optimized for batch processing (1 to 500 documents)
        self.batch_size = Histogram(
            "batch_size",
            "Batch processing size distribution",
            labelnames=["operation"],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry,
        )

        logger.info("MetricsCollector initialized with Prometheus metrics")

    @classmethod
    def get_metrics(cls) -> "MetricsCollector":
        """
        Get singleton MetricsCollector instance.

        Returns:
            Singleton MetricsCollector instance (real or NoOp)
        """
        if cls._instance is None:
            if PROMETHEUS_AVAILABLE:
                cls._instance = MetricsCollector()
            else:
                cls._instance = NoOpMetricsCollector()
        return cls._instance

    @classmethod
    def reset_singleton(cls):
        """
        Reset singleton instance (for testing only).

        WARNING: This is intended for testing only. Do not use in production code.
        """
        cls._instance = None

    def _validate_fidelity(self, fidelity: Optional[str]) -> bool:
        """
        Validate fidelity level against allowed values.

        Args:
            fidelity: Fidelity level to validate (or None)

        Returns:
            True if valid or None, False if invalid
        """
        if fidelity is None:
            return True

        if fidelity not in ALLOWED_FIDELITY_LEVELS:
            logger.warning(
                f"Invalid fidelity level '{fidelity}', must be one of {ALLOWED_FIDELITY_LEVELS}. "
                f"Metric not recorded to prevent cardinality explosion."
            )
            return False
        return True

    def _validate_operation(self, operation: str) -> bool:
        """
        Validate operation against allowed values.

        Args:
            operation: Operation to validate

        Returns:
            True if valid, False if invalid
        """
        if operation not in ALLOWED_OPERATIONS:
            logger.warning(
                f"Invalid operation '{operation}', must be one of {ALLOWED_OPERATIONS}. "
                f"Metric not recorded to prevent cardinality explosion."
            )
            return False
        return True

    def _validate_status(self, status: str) -> bool:
        """
        Validate status against allowed values.

        Args:
            status: Status to validate

        Returns:
            True if valid, False if invalid
        """
        if status not in ALLOWED_STATUSES:
            logger.warning(
                f"Invalid status '{status}', must be one of {ALLOWED_STATUSES}. "
                f"Metric not recorded to prevent cardinality explosion."
            )
            return False
        return True

    def record_compression_ratio(self, ratio: float, fidelity: str):
        """
        Record compression ratio metric.

        Args:
            ratio: Compression ratio (original_tokens / compressed_tokens)
            fidelity: Fidelity level ("LOW", "BALANCED", "HIGH", "EXTREME", "NONE")

        Example:
            >>> metrics.record_compression_ratio(7.5, "BALANCED")
        """
        if not self._enabled:
            return

        if not self._validate_fidelity(fidelity):
            return

        self.compression_ratio.labels(fidelity_level=fidelity).observe(ratio)

    def record_latency(self, operation: str, latency_seconds: float, fidelity: str = None):
        """
        Record processing latency metric.

        Args:
            operation: Operation name ("ingest", "compress", "expand", "batch_ingest", "refresh")
            latency_seconds: Latency in seconds
            fidelity: Optional fidelity level (defaults to "NONE" if not provided)

        Example:
            >>> metrics.record_latency("compress", 0.35, "BALANCED")
            >>> metrics.record_latency("batch_ingest", 5.2)  # No fidelity
        """
        if not self._enabled:
            return

        if not self._validate_operation(operation):
            return

        # Use "NONE" for operations without fidelity context
        fidelity_label = fidelity if fidelity is not None else "NONE"

        if not self._validate_fidelity(fidelity_label):
            return

        self.processing_latency.labels(operation=operation, fidelity_level=fidelity_label).observe(
            latency_seconds
        )

    def increment_documents_processed(self, operation: str, fidelity: str, status: str = "success"):
        """
        Increment documents processed counter.

        Args:
            operation: Operation name ("ingest", "compress", "expand", "batch_ingest", "refresh")
            fidelity: Fidelity level ("LOW", "BALANCED", "HIGH", "EXTREME", "NONE")
            status: Processing status ("success" or "failure", defaults to "success")

        Example:
            >>> metrics.increment_documents_processed("ingest", "HIGH", "success")
            >>> metrics.increment_documents_processed("compress", "BALANCED", "failure")
        """
        if not self._enabled:
            return

        if not self._validate_operation(operation):
            return

        if not self._validate_fidelity(fidelity):
            return

        if not self._validate_status(status):
            return

        self.documents_processed.labels(
            operation=operation, fidelity_level=fidelity, status=status
        ).inc()

    def set_cache_hit_ratio(self, ratio: float):
        """
        Set cache hit ratio gauge.

        Args:
            ratio: Cache hit ratio (0.0 to 1.0)

        Example:
            >>> metrics.set_cache_hit_ratio(0.75)  # 75% hit rate
        """
        if not self._enabled:
            return

        # Clamp ratio to [0.0, 1.0] range
        ratio = max(0.0, min(1.0, ratio))
        self.cache_hit_ratio.set(ratio)

    def set_active_documents(self, count: int):
        """
        Set active documents gauge.

        Args:
            count: Number of currently loaded documents

        Example:
            >>> metrics.set_active_documents(42)
        """
        if not self._enabled:
            return

        self.active_documents.set(count)

    def increment_errors(self, error_type: str, operation: str):
        """
        Increment error counter.

        Args:
            error_type: Type of error (e.g., "ValueError", "FileNotFoundError", "TimeoutError")
            operation: Operation during which error occurred

        Example:
            >>> metrics.increment_errors("ValueError", "compress")
            >>> metrics.increment_errors("FileNotFoundError", "ingest")

        Note:
            error_type is NOT validated for cardinality control because error types
            are generally bounded by code structure (exception classes) and are
            valuable for debugging. However, avoid using unbounded strings like
            file paths or user input as error_type.
        """
        if not self._enabled:
            return

        if not self._validate_operation(operation):
            return

        self.errors_total.labels(error_type=error_type, operation=operation).inc()

    def record_batch_size(self, size: int, operation: str = "batch_ingest"):
        """
        Record batch processing size.

        Args:
            size: Number of documents in batch
            operation: Batch operation name (defaults to "batch_ingest")

        Example:
            >>> metrics.record_batch_size(25, "batch_ingest")
        """
        if not self._enabled:
            return

        if not self._validate_operation(operation):
            return

        self.batch_size.labels(operation=operation).observe(size)

    def reset_all_metrics(self):
        """
        Reset all metrics to initial state.

        WARNING: This is intended for testing only. Do not use in production code.
        Metrics will be recreated with new registries to ensure clean state.
        """
        if not self._enabled:
            return

        # Recreate registry and all metrics to ensure clean state
        self.__init__()
        logger.info("All metrics reset to initial state")

    def generate_metrics_text(self) -> str:
        """
        Generate Prometheus text format for scraping.

        Returns:
            Prometheus text format string with all metrics

        Example:
            >>> metrics_text = metrics.generate_metrics_text()
            >>> print(metrics_text)
            # HELP compression_ratio Compression ratio distribution
            # TYPE compression_ratio histogram
            compression_ratio_bucket{fidelity_level="BALANCED",le="1.0"} 0
            ...
        """
        if not self._enabled:
            return "# Prometheus metrics unavailable (prometheus_client not installed)\n"

        return generate_latest(self.registry).decode("utf-8")


class NoOpMetricsCollector:
    """
    No-op metrics collector when prometheus_client is not available.

    Provides the same interface as MetricsCollector but does nothing, allowing
    code to use metrics without conditional checks for availability.
    """

    def __init__(self):
        """Initialize NoOp collector."""
        self._enabled = False

    @classmethod
    def get_metrics(cls) -> "NoOpMetricsCollector":
        """Get NoOp instance."""
        return cls()

    @classmethod
    def reset_singleton(cls):
        """NoOp reset."""
        pass

    def record_compression_ratio(self, ratio: float, fidelity: str):
        """NoOp compression ratio recording."""
        pass

    def record_latency(self, operation: str, latency_seconds: float, fidelity: str = None):
        """NoOp latency recording."""
        pass

    def increment_documents_processed(self, operation: str, fidelity: str, status: str = "success"):
        """NoOp documents processed increment."""
        pass

    def set_cache_hit_ratio(self, ratio: float):
        """NoOp cache hit ratio setting."""
        pass

    def set_active_documents(self, count: int):
        """NoOp active documents setting."""
        pass

    def increment_errors(self, error_type: str, operation: str):
        """NoOp error increment."""
        pass

    def record_batch_size(self, size: int, operation: str = "batch_ingest"):
        """NoOp batch size recording."""
        pass

    def reset_all_metrics(self):
        """NoOp reset."""
        pass

    def generate_metrics_text(self) -> str:
        """NoOp metrics text generation."""
        return "# Prometheus metrics unavailable (prometheus_client not installed)\n"


# Global singleton access function
def get_metrics() -> MetricsCollector:
    """
    Get singleton MetricsCollector instance.

    Returns:
        Singleton MetricsCollector instance (real or NoOp depending on availability)

    Example:
        >>> from src.metrics import get_metrics
        >>> metrics = get_metrics()
        >>> metrics.record_compression_ratio(7.5, "BALANCED")
    """
    return MetricsCollector.get_metrics()
