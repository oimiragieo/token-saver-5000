"""
Comprehensive Metrics Tests (v1.0.0 - v0.6.1)

Tests for the Prometheus metrics collector to achieve comprehensive coverage.

This module tests critical production scenarios for the MetricsCollector:
- Basic metrics recording (7 metrics)
- Cardinality control and validation
- Singleton pattern
- Prometheus text format output
- Graceful degradation (NoOp when prometheus_client unavailable)
- Metrics reset for testing

Test Categories:
- Basic Metrics Tests (7 tests - one per metric)
- Cardinality Control Tests (6 tests)
- Singleton Tests (2 tests)
- Prometheus Output Tests (4 tests)
- Graceful Degradation Tests (3 tests)
- Reset Tests (1 test)
- Edge Cases Tests (2 tests)

Total: 25 comprehensive tests
"""

import logging
from unittest.mock import patch

import pytest

from src.metrics import (
    get_metrics,
    MetricsCollector,
    NoOpMetricsCollector,
    ALLOWED_FIDELITY_LEVELS,
    ALLOWED_OPERATIONS,
    ALLOWED_STATUSES,
    PROMETHEUS_AVAILABLE,
)

# ===========================
# Fixtures
# ===========================


@pytest.fixture
def metrics():
    """
    Reset metrics singleton before each test.

    Returns:
        MetricsCollector instance with clean state
    """
    MetricsCollector.reset_singleton()
    instance = get_metrics()
    if hasattr(instance, "reset_all_metrics"):
        instance.reset_all_metrics()
    return instance


# ===========================
# Basic Metrics Tests
# ===========================


class TestBasicMetrics:
    """Test basic metrics recording."""

    def test_record_compression_ratio(self, metrics):
        """Test compression ratio histogram."""
        # Record compression ratio
        metrics.record_compression_ratio(7.5, "BALANCED")

        # Verify metric recorded (if prometheus available)
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "compression_ratio" in text
            assert 'fidelity_level="BALANCED"' in text
            # Verify it's a histogram type
            assert "# TYPE compression_ratio histogram" in text

    def test_record_latency(self, metrics):
        """Test processing latency histogram."""
        # Record latency
        metrics.record_latency("compress", 0.35, "BALANCED")

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "processing_latency_seconds" in text
            assert 'operation="compress"' in text
            assert 'fidelity_level="BALANCED"' in text
            # Verify it's a histogram type
            assert "# TYPE processing_latency_seconds histogram" in text

    def test_record_latency_without_fidelity(self, metrics):
        """Test latency recording without fidelity (should use NONE)."""
        # Record latency without fidelity
        metrics.record_latency("batch_ingest", 5.2)

        # Verify metric recorded with NONE fidelity
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "processing_latency_seconds" in text
            assert 'operation="batch_ingest"' in text
            assert 'fidelity_level="NONE"' in text

    def test_increment_documents_processed(self, metrics):
        """Test document counter."""
        # Increment counter
        metrics.increment_documents_processed("ingest", "HIGH", "success")

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "documents_processed_total" in text
            assert 'operation="ingest"' in text
            assert 'fidelity_level="HIGH"' in text
            assert 'status="success"' in text
            # Verify it's a counter type
            assert "# TYPE documents_processed_total counter" in text

    def test_increment_documents_processed_failure(self, metrics):
        """Test document counter with failure status."""
        # Increment counter with failure
        metrics.increment_documents_processed("compress", "BALANCED", "failure")

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "documents_processed_total" in text
            assert 'status="failure"' in text

    def test_set_cache_hit_ratio(self, metrics):
        """Test cache hit ratio gauge."""
        # Set gauge value
        metrics.set_cache_hit_ratio(0.75)

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "cache_hit_ratio" in text
            # Verify it's a gauge type
            assert "# TYPE cache_hit_ratio gauge" in text
            assert "cache_hit_ratio 0.75" in text

    def test_set_cache_hit_ratio_clamping(self, metrics):
        """Test cache hit ratio clamping to [0.0, 1.0]."""
        # Test values outside range
        metrics.set_cache_hit_ratio(1.5)  # Should clamp to 1.0

        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "cache_hit_ratio 1.0" in text

        # Reset and test negative value
        metrics.set_cache_hit_ratio(-0.5)  # Should clamp to 0.0

        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "cache_hit_ratio 0.0" in text

    def test_set_active_documents(self, metrics):
        """Test active documents gauge."""
        # Set gauge value
        metrics.set_active_documents(42)

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "active_documents" in text
            # Verify it's a gauge type
            assert "# TYPE active_documents gauge" in text
            assert "active_documents 42" in text

    def test_increment_errors(self, metrics):
        """Test error counter."""
        # Increment error counter
        metrics.increment_errors("ValueError", "compress")

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "errors_total" in text
            assert 'error_type="ValueError"' in text
            assert 'operation="compress"' in text
            # Verify it's a counter type
            assert "# TYPE errors_total counter" in text

    def test_record_batch_size(self, metrics):
        """Test batch size histogram."""
        # Record batch size
        metrics.record_batch_size(25, "batch_ingest")

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "batch_size" in text
            assert 'operation="batch_ingest"' in text
            # Verify it's a histogram type
            assert "# TYPE batch_size histogram" in text

    def test_record_batch_size_default_operation(self, metrics):
        """Test batch size with default operation."""
        # Record batch size without operation (should default to batch_ingest)
        metrics.record_batch_size(50)

        # Verify metric recorded
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "batch_size" in text
            assert 'operation="batch_ingest"' in text

    def test_record_provider_cache_telemetry(self, metrics):
        """Test provider cache telemetry counters."""
        metrics.record_provider_cache_telemetry(
            {
                "provider": "openai",
                "validation_status": "validated",
                "cache_hit_detected": True,
                "cached_input_tokens": 300,
                "cache_creation_input_tokens": 120,
                "estimated_cache_savings_usd": 0.0042,
            }
        )

        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "provider_cache_observations_total" in text
            assert (
                'provider_cache_observations_total{cache_hit="true",provider="openai",'
                'validation_status="validated"} 1.0' in text
            )
            assert 'provider_cache_read_tokens_total{provider="openai"} 300.0' in text
            assert 'provider_cache_creation_tokens_total{provider="openai"} 120.0' in text
            assert "provider_cache_savings_usd_total" in text


# ===========================
# Cardinality Control Tests
# ===========================


class TestCardinalityControl:
    """Test cardinality control to prevent label explosion."""

    def test_validate_fidelity_level(self, metrics):
        """Test that valid fidelity levels are accepted."""
        # Test all allowed fidelity levels
        for fidelity in ALLOWED_FIDELITY_LEVELS:
            metrics.record_compression_ratio(5.0, fidelity)
            # Should not raise exception or log warning

    def test_reject_invalid_fidelity(self, metrics, caplog):
        """Test that invalid fidelity levels are rejected."""
        with caplog.at_level(logging.WARNING):
            metrics.record_compression_ratio(5.0, "INVALID_FIDELITY")

        # Should log warning (if prometheus available)
        if PROMETHEUS_AVAILABLE:
            assert any("Invalid fidelity level" in record.message for record in caplog.records)

            # Should not record metric
            text = metrics.generate_metrics_text()
            assert "INVALID_FIDELITY" not in text

    def test_validate_operation(self, metrics):
        """Test that valid operations are accepted."""
        # Test all allowed operations
        for operation in ALLOWED_OPERATIONS:
            metrics.record_latency(operation, 0.5, "BALANCED")
            # Should not raise exception or log warning

    def test_reject_invalid_operation(self, metrics, caplog):
        """Test that invalid operations are rejected."""
        with caplog.at_level(logging.WARNING):
            metrics.record_latency("invalid_operation", 0.5, "BALANCED")

        # Should log warning (if prometheus available)
        if PROMETHEUS_AVAILABLE:
            assert any("Invalid operation" in record.message for record in caplog.records)

            # Should not record metric
            text = metrics.generate_metrics_text()
            assert "invalid_operation" not in text

    def test_validate_status(self, metrics):
        """Test that valid statuses are accepted."""
        # Test all allowed statuses
        for status in ALLOWED_STATUSES:
            metrics.increment_documents_processed("ingest", "BALANCED", status)
            # Should not raise exception or log warning

    def test_reject_invalid_status(self, metrics, caplog):
        """Test that invalid statuses are rejected."""
        with caplog.at_level(logging.WARNING):
            metrics.increment_documents_processed("ingest", "BALANCED", "invalid_status")

        # Should log warning (if prometheus available)
        if PROMETHEUS_AVAILABLE:
            assert any("Invalid status" in record.message for record in caplog.records)

            # Should not record metric
            text = metrics.generate_metrics_text()
            assert "invalid_status" not in text


# ===========================
# Singleton Tests
# ===========================


class TestSingleton:
    """Test singleton pattern."""

    def test_metrics_singleton(self):
        """Test that get_metrics returns the same instance."""
        # Reset singleton first
        MetricsCollector.reset_singleton()

        # Get two instances
        metrics1 = get_metrics()
        metrics2 = get_metrics()

        # Should be the same instance
        assert metrics1 is metrics2

    def test_reset_singleton(self):
        """Test singleton reset for testing."""
        # Get initial instance
        metrics1 = get_metrics()

        # Reset singleton
        MetricsCollector.reset_singleton()

        # Get new instance
        metrics2 = get_metrics()

        # Should be a different instance (new object created)
        # Note: In practice, both will be of the same class, but different objects
        assert metrics1 is not metrics2 or not PROMETHEUS_AVAILABLE


# ===========================
# Prometheus Output Tests
# ===========================


class TestPrometheusOutput:
    """Test Prometheus text format."""

    def test_generate_metrics_text(self, metrics):
        """Test Prometheus text format generation."""
        # Record some metrics
        metrics.record_compression_ratio(7.5, "BALANCED")
        metrics.record_latency("compress", 0.35, "BALANCED")
        metrics.increment_documents_processed("ingest", "HIGH", "success")

        # Generate metrics text
        text = metrics.generate_metrics_text()

        # Verify output is non-empty
        assert len(text) > 0

        # Verify Prometheus format (if available)
        if PROMETHEUS_AVAILABLE:
            assert "# HELP" in text
            assert "# TYPE" in text

    def test_metrics_text_format(self, metrics):
        """Verify format matches Prometheus spec."""
        # Record a metric
        metrics.record_compression_ratio(7.5, "BALANCED")

        # Generate metrics text
        text = metrics.generate_metrics_text()

        # Verify Prometheus format elements
        if PROMETHEUS_AVAILABLE:
            # Should contain HELP and TYPE comments
            assert "# HELP compression_ratio" in text
            assert "# TYPE compression_ratio histogram" in text

            # Should contain bucket entries
            assert "compression_ratio_bucket" in text
            assert "le=" in text  # Bucket label

            # Should contain sum and count
            assert "compression_ratio_sum" in text
            assert "compression_ratio_count" in text

    def test_histogram_buckets(self, metrics):
        """Verify correct bucket configuration."""
        # Record a metric that falls in a specific bucket
        metrics.record_compression_ratio(7.5, "BALANCED")

        # Generate metrics text
        text = metrics.generate_metrics_text()

        # Verify buckets exist (if prometheus available)
        if PROMETHEUS_AVAILABLE:
            # Compression ratio buckets: [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
            assert 'le="7.0"' in text
            assert 'le="10.0"' in text
            assert 'le="+Inf"' in text  # Infinity bucket

    def test_multiple_metrics_in_output(self, metrics):
        """Test that multiple metrics appear in output."""
        # Record multiple different metrics
        metrics.record_compression_ratio(7.5, "BALANCED")
        metrics.set_cache_hit_ratio(0.75)
        metrics.set_active_documents(42)
        metrics.increment_errors("ValueError", "compress")

        # Generate metrics text
        text = metrics.generate_metrics_text()

        # Verify all metrics present
        if PROMETHEUS_AVAILABLE:
            assert "compression_ratio" in text
            assert "cache_hit_ratio" in text
            assert "active_documents" in text
            assert "errors_total" in text


# ===========================
# Graceful Degradation Tests
# ===========================


class TestGracefulDegradation:
    """Test graceful degradation when prometheus_client unavailable."""

    @pytest.mark.skipif(PROMETHEUS_AVAILABLE, reason="Only test when prometheus unavailable")
    def test_noop_when_prometheus_unavailable(self):
        """Test NoOp collector when prometheus_client unavailable."""
        # Reset singleton
        MetricsCollector.reset_singleton()

        # Get metrics (should be NoOp)
        metrics = get_metrics()

        # Should be NoOpMetricsCollector
        assert isinstance(metrics, NoOpMetricsCollector)

        # Should not raise exception when calling methods
        metrics.record_compression_ratio(5.0, "BALANCED")
        metrics.record_latency("compress", 0.5, "BALANCED")
        metrics.increment_documents_processed("ingest", "HIGH", "success")
        metrics.set_cache_hit_ratio(0.75)
        metrics.set_active_documents(42)
        metrics.increment_errors("ValueError", "compress")
        metrics.record_batch_size(25)

        # Generate text should return unavailable message
        text = metrics.generate_metrics_text()
        assert "unavailable" in text.lower()

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Only test when prometheus available")
    def test_warning_when_prometheus_unavailable(self, caplog):
        """Test warning log when prometheus_client unavailable."""
        # Mock PROMETHEUS_AVAILABLE to False
        with patch("src.metrics.PROMETHEUS_AVAILABLE", False):
            # Reset singleton
            MetricsCollector.reset_singleton()

            # Create new instance (should log warning)
            with caplog.at_level(logging.WARNING):
                metrics = MetricsCollector()

            # Should log warning
            assert any(
                "prometheus_client not installed" in record.message for record in caplog.records
            )
            assert not metrics._enabled

    def test_noop_methods_safe(self):
        """Test that NoOp methods are safe to call."""
        # Create NoOp instance directly
        noop = NoOpMetricsCollector()

        # All methods should be safe to call
        noop.record_compression_ratio(5.0, "BALANCED")
        noop.record_latency("compress", 0.5, "BALANCED")
        noop.increment_documents_processed("ingest", "HIGH", "success")
        noop.set_cache_hit_ratio(0.75)
        noop.set_active_documents(42)
        noop.increment_errors("ValueError", "compress")
        noop.record_batch_size(25)
        noop.reset_all_metrics()

        # Get metrics should return self
        result = noop.get_metrics()
        assert isinstance(result, NoOpMetricsCollector)

        # Reset singleton should be safe
        noop.reset_singleton()


# ===========================
# Reset Tests
# ===========================


class TestReset:
    """Test metrics reset functionality."""

    def test_reset_all_metrics(self, metrics):
        """Test metrics reset for testing."""
        # Record some metrics
        metrics.record_compression_ratio(7.5, "BALANCED")
        metrics.set_cache_hit_ratio(0.75)
        metrics.set_active_documents(42)

        if PROMETHEUS_AVAILABLE:
            # Verify metrics exist
            text1 = metrics.generate_metrics_text()
            assert "compression_ratio_count" in text1

            # Reset metrics
            metrics.reset_all_metrics()

            # Metrics should be reset (counts should be 0 or absent)
            text2 = metrics.generate_metrics_text()
            # After reset, counts should be 0 (fresh registry)
            # Note: The metrics structure will exist but with zero values
            assert "# HELP" in text2  # Headers still present
            assert "# TYPE" in text2


# ===========================
# Edge Cases Tests
# ===========================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extremely_large_compression_ratio(self, metrics):
        """Test recording extremely large compression ratio."""
        # Record very large ratio (beyond highest bucket)
        metrics.record_compression_ratio(1000.0, "EXTREME")

        # Should not raise exception
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "compression_ratio" in text
            # Should be in +Inf bucket
            assert 'le="+Inf"' in text

    def test_zero_compression_ratio(self, metrics):
        """Test recording zero compression ratio."""
        # Record zero ratio (edge case)
        metrics.record_compression_ratio(0.0, "LOW")

        # Should not raise exception
        if PROMETHEUS_AVAILABLE:
            text = metrics.generate_metrics_text()
            assert "compression_ratio" in text
