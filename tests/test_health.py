"""Tests for health checks and diagnostics."""

import os
import pytest
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from src.health import (
    ComponentHealth,
    HealthStatus,
    get_health,
)


class TestBasicHealthChecks:
    """Test basic health check functionality."""

    def test_health_singleton(self):
        """Test singleton pattern."""
        health1 = get_health()
        health2 = get_health()
        assert health1 is health2

    def test_liveness_check(self):
        """Test liveness check always returns healthy."""
        health = get_health()
        health.reset_metrics()  # Clear any cached data

        result = health.check_liveness()

        assert result["status"] == HealthStatus.HEALTHY.value
        assert "timestamp" in result
        # Verify timestamp format
        assert result["timestamp"].endswith("Z")

    def test_readiness_check(self):
        """Test readiness check validates components."""
        health = get_health()
        health.reset_metrics()  # Clear any cached data

        result = health.check_readiness()

        assert "status" in result
        assert result["status"] in [
            HealthStatus.HEALTHY.value,
            HealthStatus.DEGRADED.value,
            HealthStatus.UNHEALTHY.value,
        ]
        assert "timestamp" in result
        assert "components" in result
        assert "embedding_manager" in result["components"]
        assert "persistence" in result["components"]
        assert "cache" in result["components"]
        assert "disk_space" in result["components"]

        # Verify component structure
        for component_name, component_data in result["components"].items():
            assert "status" in component_data
            assert "message" in component_data
            assert "details" in component_data

    def test_check_component_specific(self):
        """Test checking specific component."""
        health = get_health()
        health.reset_metrics()

        # Check embedding manager specifically
        component = health.check_component("embedding_manager")
        assert component is not None
        assert component.name == "embedding_manager"
        assert isinstance(component.status, HealthStatus)

        # Check invalid component
        invalid = health.check_component("nonexistent")
        assert invalid is None


class TestComponentHealth:
    """Test individual component health checks."""

    def test_check_embedding_manager(self):
        """Test embedding manager health check."""
        health = get_health()
        health.reset_metrics()

        component = health._check_embedding_manager()

        assert component.name == "embedding_manager"
        assert component.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
        assert component.message
        assert isinstance(component.last_check, datetime)

        # If healthy, should have embedding_dim in details
        if component.status == HealthStatus.HEALTHY:
            assert "embedding_dim" in component.details
            assert component.details["embedding_dim"] > 0

    def test_check_embedding_manager_failure(self):
        """Test embedding manager health check with import failure."""
        health = get_health()
        health.reset_metrics()

        with patch("src.embeddings.EmbeddingManager", side_effect=ImportError("Model not found")):
            component = health._check_embedding_manager()

            assert component.name == "embedding_manager"
            assert component.status == HealthStatus.UNHEALTHY
            assert "unavailable" in component.message.lower()
            assert "error" in component.details

    def test_check_persistence(self):
        """Test persistence layer health check."""
        health = get_health()
        health.reset_metrics()

        component = health._check_persistence()

        assert component.name == "persistence"
        assert component.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert isinstance(component.last_check, datetime)

        # If healthy, test file should have been cleaned up
        test_file = ".semantic_modulator_data/health_check.tmp"
        assert not os.path.exists(test_file)

    def test_check_persistence_failure(self):
        """Test persistence health check with write failure."""
        health = get_health()
        health.reset_metrics()

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            component = health._check_persistence()

            assert component.name == "persistence"
            assert component.status == HealthStatus.UNHEALTHY
            assert "unavailable" in component.message.lower()
            assert "error" in component.details

    def test_check_cache(self):
        """Test cache health check."""
        health = get_health()
        health.reset_metrics()

        component = health._check_cache()

        assert component.name == "cache"
        assert component.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert isinstance(component.last_check, datetime)

        # Should have cache metrics in details
        if component.status != HealthStatus.DEGRADED or "error" not in component.details:
            assert "entries" in component.details
            assert "capacity" in component.details
            assert "usage_ratio" in component.details
            assert "hit_rate" in component.details

    def test_check_cache_near_capacity(self):
        """Test cache health when nearing capacity."""
        health = get_health()
        health.reset_metrics()

        # Mock cache stats to show high usage
        mock_stats = Mock()
        mock_stats.entries = 850
        mock_stats.capacity = 1000
        mock_stats.hit_rate = 0.75

        # Create mock cache instance
        mock_cache_instance = Mock()
        mock_cache_instance.get_stats.return_value = mock_stats

        # Mock the LRUEmbeddingCache class at import location
        with patch("src.embedding_cache.LRUEmbeddingCache") as MockCache:
            MockCache.get_cache.return_value = mock_cache_instance

            component = health._check_cache()

            assert component.status == HealthStatus.DEGRADED
            assert "nearing capacity" in component.message.lower()

    def test_check_cache_at_capacity(self):
        """Test cache health when at capacity."""
        health = get_health()
        health.reset_metrics()

        # Mock cache stats to show critical usage
        mock_stats = Mock()
        mock_stats.entries = 970
        mock_stats.capacity = 1000
        mock_stats.hit_rate = 0.80

        # Create mock cache instance
        mock_cache_instance = Mock()
        mock_cache_instance.get_stats.return_value = mock_stats

        # Mock the LRUEmbeddingCache class at import location
        with patch("src.embedding_cache.LRUEmbeddingCache") as MockCache:
            MockCache.get_cache.return_value = mock_cache_instance

            component = health._check_cache()

            assert component.status == HealthStatus.UNHEALTHY
            assert "at capacity" in component.message.lower()

    def test_check_disk_space(self):
        """Test disk space health check."""
        health = get_health()
        health.reset_metrics()

        component = health._check_disk_space()

        assert component.name == "disk_space"
        assert component.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert isinstance(component.last_check, datetime)

        # Should have disk metrics
        if component.status != HealthStatus.DEGRADED or "error" not in component.details:
            assert "free_gb" in component.details
            assert "used_gb" in component.details
            assert "total_gb" in component.details

    def test_check_disk_space_low(self):
        """Test disk space health with low space."""
        health = get_health()
        health.reset_metrics()

        # Mock disk usage to show low space
        mock_usage = Mock()
        mock_usage.free = int(0.5 * 1024**3)  # 0.5GB free
        mock_usage.used = int(99.5 * 1024**3)
        mock_usage.total = int(100 * 1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            component = health._check_disk_space()

            assert component.status == HealthStatus.DEGRADED
            assert "low disk space" in component.message.lower()

    def test_check_disk_space_critical(self):
        """Test disk space health with critical space."""
        health = get_health()
        health.reset_metrics()

        # Mock disk usage to show critical space
        mock_usage = Mock()
        mock_usage.free = int(0.05 * 1024**3)  # 0.05GB free
        mock_usage.used = int(99.95 * 1024**3)
        mock_usage.total = int(100 * 1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            component = health._check_disk_space()

            assert component.status == HealthStatus.UNHEALTHY
            assert "critical disk space" in component.message.lower()


class TestHealthStatus:
    """Test overall health status determination."""

    def test_overall_status_healthy(self):
        """Test all components healthy → overall healthy."""
        health = get_health()
        health.reset_metrics()

        with (
            patch.object(health, "_check_embedding_manager") as mock_emb,
            patch.object(health, "_check_persistence") as mock_pers,
            patch.object(health, "_check_cache") as mock_cache,
            patch.object(health, "_check_disk_space") as mock_disk,
        ):

            # All components healthy
            for mock in [mock_emb, mock_pers, mock_cache, mock_disk]:
                mock.return_value = ComponentHealth(
                    name="test",
                    status=HealthStatus.HEALTHY,
                    message="OK",
                    last_check=datetime.now(timezone.utc),
                    details={},
                )

            result = health.check_readiness()
            assert result["status"] == HealthStatus.HEALTHY.value

    def test_overall_status_degraded(self):
        """Test some degraded → overall degraded."""
        health = get_health()
        health.reset_metrics()

        with (
            patch.object(health, "_check_embedding_manager") as mock_emb,
            patch.object(health, "_check_persistence") as mock_pers,
            patch.object(health, "_check_cache") as mock_cache,
            patch.object(health, "_check_disk_space") as mock_disk,
        ):

            # Some degraded
            mock_emb.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_pers.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.DEGRADED,
                message="Warning",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_cache.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_disk.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )

            result = health.check_readiness()
            assert result["status"] == HealthStatus.DEGRADED.value

    def test_overall_status_unhealthy(self):
        """Test any unhealthy → overall unhealthy."""
        health = get_health()
        health.reset_metrics()

        with (
            patch.object(health, "_check_embedding_manager") as mock_emb,
            patch.object(health, "_check_persistence") as mock_pers,
            patch.object(health, "_check_cache") as mock_cache,
            patch.object(health, "_check_disk_space") as mock_disk,
        ):

            # One unhealthy
            mock_emb.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.UNHEALTHY,
                message="Error",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_pers.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_cache.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_disk.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )

            result = health.check_readiness()
            assert result["status"] == HealthStatus.UNHEALTHY.value

    def test_overall_status_mixed_degraded(self):
        """Test multiple degraded → overall degraded (no unhealthy)."""
        health = get_health()
        health.reset_metrics()

        with (
            patch.object(health, "_check_embedding_manager") as mock_emb,
            patch.object(health, "_check_persistence") as mock_pers,
            patch.object(health, "_check_cache") as mock_cache,
            patch.object(health, "_check_disk_space") as mock_disk,
        ):

            # Multiple degraded
            mock_emb.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.DEGRADED,
                message="Warning",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_pers.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.DEGRADED,
                message="Warning",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_cache.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )
            mock_disk.return_value = ComponentHealth(
                name="test",
                status=HealthStatus.HEALTHY,
                message="OK",
                last_check=datetime.now(timezone.utc),
                details={},
            )

            result = health.check_readiness()
            assert result["status"] == HealthStatus.DEGRADED.value


class TestCaching:
    """Test health check caching."""

    def test_readiness_caching(self):
        """Test readiness results cached for 10 seconds."""
        health = get_health()
        health.reset_metrics()  # Clear cache

        # First call should execute checks
        result1 = health.check_readiness()
        timestamp1 = result1["timestamp"]

        # Immediate second call should return cached result
        time.sleep(0.1)  # Small delay
        result2 = health.check_readiness()
        timestamp2 = result2["timestamp"]

        # Timestamps should be identical (cached)
        assert timestamp1 == timestamp2
        assert result1 == result2

    def test_readiness_cache_expiry(self):
        """Test readiness cache expires after 10 seconds."""
        health = get_health()
        health.reset_metrics()

        # Override cache duration for testing
        original_duration = health.cache_duration
        health.cache_duration = timedelta(milliseconds=100)

        try:
            result1 = health.check_readiness()
            timestamp1 = result1["timestamp"]

            # Wait for cache to expire
            time.sleep(0.15)

            result2 = health.check_readiness()
            timestamp2 = result2["timestamp"]

            # Timestamps should be different (cache expired)
            assert timestamp1 != timestamp2
        finally:
            health.cache_duration = original_duration

    def test_diagnostics_caching(self):
        """Test diagnostics results cached for 10 seconds."""
        health = get_health()
        health.reset_metrics()  # Clear cache

        result1 = health.get_diagnostics()
        timestamp1 = result1["timestamp"]

        time.sleep(0.1)
        result2 = health.get_diagnostics()
        timestamp2 = result2["timestamp"]

        # Should return same cached result (within cache window)
        assert timestamp1 == timestamp2

    def test_diagnostics_cache_expiry(self):
        """Test diagnostics cache expires after 10 seconds."""
        health = get_health()
        health.reset_metrics()

        # Override cache duration for testing
        original_duration = health.cache_duration
        health.cache_duration = timedelta(milliseconds=100)

        try:
            result1 = health.get_diagnostics()
            timestamp1 = result1["timestamp"]

            # Wait for cache to expire
            time.sleep(0.15)

            result2 = health.get_diagnostics()
            timestamp2 = result2["timestamp"]

            # Timestamps should be different (cache expired)
            assert timestamp1 != timestamp2
        finally:
            health.cache_duration = original_duration

    def test_reset_metrics_clears_cache(self):
        """Test reset_metrics clears cache."""
        health = get_health()

        # Populate cache
        health.check_readiness()
        health.get_diagnostics()

        assert health._cached_readiness is not None
        assert health._cached_diagnostics is not None

        # Reset should clear cache
        health.reset_metrics()

        assert health._cached_readiness is None
        assert health._cached_diagnostics is None


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_record_operation_latency(self):
        """Test latency recording."""
        health = get_health()
        health.reset_metrics()  # Clear metrics

        health.record_operation_latency("compress", 0.123)
        health.record_operation_latency("compress", 0.456)
        health.record_operation_latency("compress", 0.789)

        percentiles = health._calculate_latency_percentiles()
        assert "compress" in percentiles
        assert percentiles["compress"]["count"] == 3

    def test_calculate_latency_percentiles(self):
        """Test p50/p95/p99 calculation."""
        health = get_health()
        health.reset_metrics()

        # Record 100 latencies
        for i in range(100):
            health.record_operation_latency("test", i / 1000.0)

        percentiles = health._calculate_latency_percentiles()
        assert "test" in percentiles
        assert "p50" in percentiles["test"]
        assert "p95" in percentiles["test"]
        assert "p99" in percentiles["test"]
        assert percentiles["test"]["count"] == 100

        # Verify percentiles are in expected range
        assert 0.04 <= percentiles["test"]["p50"] <= 0.06
        assert 0.09 <= percentiles["test"]["p95"] <= 0.10
        assert 0.098 <= percentiles["test"]["p99"] <= 0.10

    def test_latency_percentiles_edge_cases(self):
        """Test latency percentiles with edge cases."""
        health = get_health()
        health.reset_metrics()

        # Single value
        health.record_operation_latency("single", 0.5)
        percentiles = health._calculate_latency_percentiles()
        assert percentiles["single"]["p50"] == 0.5
        assert percentiles["single"]["p95"] == 0.5
        assert percentiles["single"]["p99"] == 0.5

    def test_record_operation_success(self):
        """Test recording successful operations."""
        health = get_health()
        health.reset_metrics()

        health.record_operation_success("compress")
        health.record_operation_success("compress")
        health.record_operation_success("expand")

        error_rates = health._calculate_error_rates()
        assert "compress" in error_rates
        assert error_rates["compress"]["successes"] == 2
        assert error_rates["compress"]["errors"] == 0
        assert error_rates["compress"]["error_rate"] == 0.0

    def test_record_operation_error(self):
        """Test recording failed operations."""
        health = get_health()
        health.reset_metrics()

        health.record_operation_error("compress")
        health.record_operation_success("compress")
        health.record_operation_success("compress")

        error_rates = health._calculate_error_rates()
        assert "compress" in error_rates
        assert error_rates["compress"]["errors"] == 1
        assert error_rates["compress"]["successes"] == 2
        assert error_rates["compress"]["total"] == 3
        assert error_rates["compress"]["error_rate"] == pytest.approx(0.3333, abs=0.01)

    def test_calculate_error_rates_multiple_operations(self):
        """Test error rate calculation across multiple operations."""
        health = get_health()
        health.reset_metrics()

        # Operation 1: High success rate
        for _ in range(9):
            health.record_operation_success("compress")
        health.record_operation_error("compress")

        # Operation 2: High error rate
        for _ in range(7):
            health.record_operation_error("expand")
        for _ in range(3):
            health.record_operation_success("expand")

        error_rates = health._calculate_error_rates()

        assert error_rates["compress"]["error_rate"] == 0.1
        assert error_rates["expand"]["error_rate"] == 0.7

    def test_latency_pruning(self):
        """Test latency list pruning to last 1000 measurements."""
        health = get_health()
        health.reset_metrics()

        # Record 1500 latencies
        for i in range(1500):
            health.record_operation_latency("test", i / 10000.0)

        # Should only keep last 1000
        assert len(health._operation_latencies["test"]) == 1000

        # Should have pruned the oldest ones (0-499)
        percentiles = health._calculate_latency_percentiles()
        assert percentiles["test"]["count"] == 1000


class TestDiagnostics:
    """Test full diagnostics."""

    def test_get_diagnostics(self):
        """Test full diagnostics output."""
        health = get_health()
        health.reset_metrics()

        diagnostics = health.get_diagnostics()

        assert "status" in diagnostics
        assert "timestamp" in diagnostics
        assert "components" in diagnostics
        assert "performance" in diagnostics
        assert "resources" in diagnostics

    def test_diagnostics_performance_section(self):
        """Test diagnostics performance metrics."""
        health = get_health()
        health.reset_metrics()

        # Record some metrics
        health.record_operation_latency("compress", 0.123)
        health.record_operation_success("compress")

        diagnostics = health.get_diagnostics()

        assert "latencies" in diagnostics["performance"]
        assert "error_rates" in diagnostics["performance"]

    def test_diagnostics_resources_section(self):
        """Test diagnostics resource metrics."""
        health = get_health()
        health.reset_metrics()

        diagnostics = health.get_diagnostics()

        assert "memory" in diagnostics["resources"]
        assert "disk" in diagnostics["resources"]
        assert "cache" in diagnostics["resources"]

    def test_diagnostics_memory_usage(self):
        """Test memory usage metrics."""
        health = get_health()
        health.reset_metrics()

        memory = health._get_memory_usage()

        assert "available" in memory
        # If psutil available, check for detailed metrics
        if memory["available"]:
            assert "rss_mb" in memory
            assert "vms_mb" in memory

    def test_diagnostics_disk_usage(self):
        """Test disk usage metrics."""
        health = get_health()
        health.reset_metrics()

        disk = health._get_disk_usage()

        # Should have disk metrics or error
        if "available" in disk and not disk["available"]:
            assert "error" in disk
        else:
            assert "free_gb" in disk
            assert "used_gb" in disk
            assert "total_gb" in disk
            assert "percent_used" in disk

    def test_diagnostics_cache_usage(self):
        """Test cache usage metrics."""
        health = get_health()
        health.reset_metrics()

        cache = health._get_cache_usage()

        # Should have cache metrics or error
        if "available" in cache and not cache["available"]:
            assert "error" in cache
        else:
            assert "entries" in cache
            assert "capacity" in cache
            assert "hit_rate" in cache
            assert "hits" in cache
            assert "misses" in cache


class TestComponentHealthDataclass:
    """Test ComponentHealth dataclass."""

    def test_component_health_creation(self):
        """Test creating ComponentHealth objects."""
        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            last_check=datetime.now(timezone.utc),
            details={"key": "value"},
        )

        assert component.name == "test_component"
        assert component.status == HealthStatus.HEALTHY
        assert component.message == "All good"
        assert isinstance(component.last_check, datetime)
        assert component.details == {"key": "value"}

    def test_component_health_default_details(self):
        """Test ComponentHealth with default empty details."""
        component = ComponentHealth(
            name="test",
            status=HealthStatus.DEGRADED,
            message="Warning",
            last_check=datetime.now(timezone.utc),
        )

        assert component.details == {}


class TestHealthStatusEnum:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"

    def test_health_status_string_comparison(self):
        """Test comparing HealthStatus with strings."""
        status = HealthStatus.HEALTHY
        assert status.value == "healthy"
        assert status == HealthStatus.HEALTHY


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_metrics(self):
        """Test diagnostics with no recorded metrics."""
        health = get_health()
        health.reset_metrics()

        diagnostics = health.get_diagnostics()

        # Should handle empty metrics gracefully
        assert diagnostics["performance"]["latencies"] == {}
        assert diagnostics["performance"]["error_rates"] == {}

    def test_concurrent_metric_recording(self):
        """Test recording metrics doesn't interfere with health checks."""
        health = get_health()
        health.reset_metrics()

        # Record metrics while checking health
        health.record_operation_latency("compress", 0.1)
        result1 = health.check_readiness()
        health.record_operation_success("compress")
        result2 = health.check_readiness()

        # Both should succeed
        assert "status" in result1
        assert "status" in result2

    def test_health_check_during_cache_expiry(self):
        """Test health check during cache expiry window."""
        health = get_health()
        health.reset_metrics()

        # Override cache duration
        original_duration = health.cache_duration
        health.cache_duration = timedelta(milliseconds=50)

        try:
            result1 = health.check_readiness()

            # Sleep exactly at cache boundary
            time.sleep(0.05)

            result2 = health.check_readiness()

            # Should handle boundary condition gracefully
            assert "status" in result1
            assert "status" in result2
        finally:
            health.cache_duration = original_duration
