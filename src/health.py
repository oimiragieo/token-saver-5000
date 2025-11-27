"""
Health check and diagnostics for production monitoring.

Provides liveness, readiness, and detailed diagnostics endpoints for monitoring
the Token Saver 5000 MCP server in production environments.

Features:
- Liveness check (server running)
- Readiness check (dependencies available)
- Detailed diagnostics (model loaded, cache operational, disk space)
- Component health status (embedding manager, persistence, cache)
- Performance metrics (p50/p95/p99 latencies)
- Caching: Health check results cached for 10 seconds

Usage Examples:
    # Basic usage
    from src.health import get_health

    health = get_health()

    # Liveness check (always healthy if server running)
    liveness = health.check_liveness()
    print(liveness)  # {"status": "healthy", "timestamp": "2025-11-27T..."}

    # Readiness check (components must be ready)
    readiness = health.check_readiness()
    print(readiness["status"])  # "healthy" or "degraded" or "unhealthy"
    print(readiness["components"]["embedding_manager"]["status"])

    # Full diagnostics with performance metrics
    diagnostics = health.get_diagnostics()
    print(diagnostics["performance"]["latencies"])  # p50/p95/p99 latencies
    print(diagnostics["resources"]["memory"])  # Memory usage

    # Record operation latency for metrics
    import time
    start = time.perf_counter()
    compress_document()
    elapsed = time.perf_counter() - start
    health.record_operation_latency("compress", elapsed)

Architecture:
- Singleton pattern for global access
- 10-second result caching to avoid expensive checks
- Thread-safe health checks
- Graceful degradation (no hard dependencies)
- Performance target: <50ms for full health check
"""

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional dependency for detailed resource metrics
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - resource metrics will be limited")


class HealthStatus(str, Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a component."""

    name: str
    status: HealthStatus
    message: str
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health checker with component status and diagnostics.

    Provides three levels of health checks:
    1. Liveness: Server is running (always returns healthy)
    2. Readiness: All components are ready (checks dependencies)
    3. Diagnostics: Detailed health + performance metrics

    Results are cached for 10 seconds to avoid expensive checks on every call.
    """

    _instance: Optional["HealthChecker"] = None  # Singleton

    def __init__(self):
        """Initialize health checker with caching and metrics tracking."""
        self.cache_duration = timedelta(seconds=10)
        self._cached_readiness: Optional[Dict[str, Any]] = None
        self._cached_diagnostics: Optional[Dict[str, Any]] = None
        self._last_readiness_check: Optional[datetime] = None
        self._last_diagnostics_check: Optional[datetime] = None

        # Performance metrics
        self._operation_latencies: Dict[str, List[float]] = {}  # operation -> list of latencies
        self._operation_errors: Dict[str, int] = {}  # operation -> error count
        self._operation_successes: Dict[str, int] = {}  # operation -> success count

        logger.debug("HealthChecker initialized with 10s cache duration")

    @classmethod
    def get_health(cls) -> "HealthChecker":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def check_liveness(self) -> Dict[str, Any]:
        """
        Liveness check - always returns healthy if server running.

        This is a lightweight check used by load balancers to determine if the
        server process is alive. It does NOT check component health.

        Returns:
            Dict with status="healthy" and current timestamp
        """
        return {
            "status": HealthStatus.HEALTHY.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def check_readiness(self) -> Dict[str, Any]:
        """
        Readiness check - returns healthy only if all components ready.

        This check verifies that all critical components are operational:
        - Embedding manager: Model loaded and functional
        - Persistence: Disk writable and space available
        - Cache: Memory within limits
        - Disk space: Sufficient space available

        Results are cached for 10 seconds to avoid expensive checks.

        Returns:
            Dict with overall status and component details
        """
        # Check cache
        now = datetime.utcnow()
        if self._last_readiness_check and (now - self._last_readiness_check) < self.cache_duration:
            logger.debug("Returning cached readiness check")
            return self._cached_readiness

        # Perform checks
        logger.debug("Performing readiness checks")
        components = {
            "embedding_manager": self._check_embedding_manager(),
            "persistence": self._check_persistence(),
            "cache": self._check_cache(),
            "disk_space": self._check_disk_space(),
        }

        # Determine overall status
        statuses = [comp.status for comp in components.values()]
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        result = {
            "status": overall_status.value,
            "timestamp": now.isoformat() + "Z",
            "components": {
                name: {
                    "status": comp.status.value,
                    "message": comp.message,
                    "details": comp.details,
                }
                for name, comp in components.items()
            },
        }

        # Cache result
        self._cached_readiness = result
        self._last_readiness_check = now

        logger.debug(f"Readiness check complete: {overall_status.value}")
        return result

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get detailed diagnostics with performance metrics.

        Includes everything from readiness check plus:
        - Performance metrics: p50/p95/p99 latencies by operation
        - Error rates by operation
        - Resource usage: memory, disk, cache

        Results are cached for 10 seconds.

        Returns:
            Dict with readiness info + performance + resources
        """
        # Check cache
        now = datetime.utcnow()
        if (
            self._last_diagnostics_check
            and (now - self._last_diagnostics_check) < self.cache_duration
        ):
            logger.debug("Returning cached diagnostics")
            return self._cached_diagnostics

        # Get readiness (will use cache if available)
        logger.debug("Performing diagnostics check")
        readiness = self.check_readiness()

        # Add performance metrics
        diagnostics = {
            **readiness,
            "performance": {
                "latencies": self._calculate_latency_percentiles(),
                "error_rates": self._calculate_error_rates(),
            },
            "resources": {
                "memory": self._get_memory_usage(),
                "disk": self._get_disk_usage(),
                "cache": self._get_cache_usage(),
            },
        }

        # Cache result
        self._cached_diagnostics = diagnostics
        self._last_diagnostics_check = now

        logger.debug("Diagnostics check complete")
        return diagnostics

    def check_component(self, component_name: str) -> Optional[ComponentHealth]:
        """
        Check specific component health.

        Args:
            component_name: Name of component (embedding_manager, persistence, cache, disk_space)

        Returns:
            ComponentHealth object or None if component not found
        """
        check_methods = {
            "embedding_manager": self._check_embedding_manager,
            "persistence": self._check_persistence,
            "cache": self._check_cache,
            "disk_space": self._check_disk_space,
        }

        check_method = check_methods.get(component_name)
        if check_method:
            return check_method()
        return None

    def record_operation_latency(self, operation: str, latency_seconds: float):
        """
        Record operation latency for metrics.

        Args:
            operation: Operation name (e.g., "compress", "ingest", "expand")
            latency_seconds: Duration in seconds
        """
        if operation not in self._operation_latencies:
            self._operation_latencies[operation] = []
        self._operation_latencies[operation].append(latency_seconds)

        # Keep last 1000 measurements per operation
        if len(self._operation_latencies[operation]) > 1000:
            self._operation_latencies[operation] = self._operation_latencies[operation][-1000:]

    def record_operation_success(self, operation: str):
        """Record successful operation for error rate calculation."""
        self._operation_successes[operation] = self._operation_successes.get(operation, 0) + 1

    def record_operation_error(self, operation: str):
        """Record failed operation for error rate calculation."""
        self._operation_errors[operation] = self._operation_errors.get(operation, 0) + 1

    def reset_metrics(self):
        """Reset performance metrics (for testing)."""
        self._operation_latencies.clear()
        self._operation_errors.clear()
        self._operation_successes.clear()
        self._cached_readiness = None
        self._cached_diagnostics = None
        self._last_readiness_check = None
        self._last_diagnostics_check = None
        logger.debug("Health metrics reset")

    # Component check methods

    def _check_embedding_manager(self) -> ComponentHealth:
        """Check embedding manager health."""
        try:
            # Try to import and test encode
            from src.embeddings import EmbeddingManager

            manager = EmbeddingManager()

            # Test encoding a simple string
            test_embedding = manager.encode("health check test", normalize=True)

            if test_embedding is not None and len(test_embedding) > 0:
                return ComponentHealth(
                    name="embedding_manager",
                    status=HealthStatus.HEALTHY,
                    message="Embedding model loaded and functional",
                    last_check=datetime.utcnow(),
                    details={"embedding_dim": len(test_embedding)},
                )
            else:
                return ComponentHealth(
                    name="embedding_manager",
                    status=HealthStatus.UNHEALTHY,
                    message="Embedding model returned invalid result",
                    last_check=datetime.utcnow(),
                    details={},
                )
        except Exception as e:
            logger.warning(f"Embedding manager health check failed: {e}")
            return ComponentHealth(
                name="embedding_manager",
                status=HealthStatus.UNHEALTHY,
                message=f"Embedding model unavailable: {str(e)[:100]}",
                last_check=datetime.utcnow(),
                details={"error": str(e)[:200]},
            )

    def _check_persistence(self) -> ComponentHealth:
        """Check persistence layer health."""
        try:
            # Try to write a small test file
            test_file = ".semantic_modulator_data/health_check.tmp"
            os.makedirs(os.path.dirname(test_file), exist_ok=True)

            # Write test data
            test_data = f"health_check_{time.time()}"
            with open(test_file, "w") as f:
                f.write(test_data)

            # Read back to verify
            with open(test_file, "r") as f:
                read_data = f.read()

            # Clean up
            os.remove(test_file)

            if read_data == test_data:
                return ComponentHealth(
                    name="persistence",
                    status=HealthStatus.HEALTHY,
                    message="Persistence layer operational",
                    last_check=datetime.utcnow(),
                    details={},
                )
            else:
                return ComponentHealth(
                    name="persistence",
                    status=HealthStatus.DEGRADED,
                    message="Persistence layer returned unexpected data",
                    last_check=datetime.utcnow(),
                    details={},
                )
        except Exception as e:
            logger.warning(f"Persistence health check failed: {e}")
            return ComponentHealth(
                name="persistence",
                status=HealthStatus.UNHEALTHY,
                message=f"Persistence layer unavailable: {str(e)[:100]}",
                last_check=datetime.utcnow(),
                details={"error": str(e)[:200]},
            )

    def _check_cache(self) -> ComponentHealth:
        """Check cache health."""
        try:
            from src.embedding_cache import LRUEmbeddingCache

            # Get cache stats (cache is singleton, so this gets the shared instance)
            cache = LRUEmbeddingCache.get_cache()
            stats = cache.get_stats()

            # Calculate usage ratio
            usage_ratio = stats.entries / stats.capacity if stats.capacity > 0 else 0

            # Determine status based on usage
            if usage_ratio < 0.8:
                status = HealthStatus.HEALTHY
                message = "Cache operational"
            elif usage_ratio < 0.95:
                status = HealthStatus.DEGRADED
                message = "Cache nearing capacity"
            else:
                status = HealthStatus.UNHEALTHY
                message = "Cache at capacity"

            return ComponentHealth(
                name="cache",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "entries": stats.entries,
                    "capacity": stats.capacity,
                    "usage_ratio": round(usage_ratio, 3),
                    "hit_rate": round(stats.hit_rate, 3),
                },
            )
        except Exception as e:
            logger.warning(f"Cache health check failed: {e}")
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache unavailable: {str(e)[:100]}",
                last_check=datetime.utcnow(),
                details={"error": str(e)[:200]},
            )

    def _check_disk_space(self) -> ComponentHealth:
        """Check disk space availability."""
        try:
            usage = shutil.disk_usage(".")
            free_gb = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)

            # Determine status based on free space
            if free_gb > 1.0:
                status = HealthStatus.HEALTHY
                message = f"{free_gb:.2f}GB free"
            elif free_gb > 0.1:
                status = HealthStatus.DEGRADED
                message = f"Low disk space: {free_gb:.2f}GB free"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Critical disk space: {free_gb:.2f}GB free"

            return ComponentHealth(
                name="disk_space",
                status=status,
                message=message,
                last_check=datetime.utcnow(),
                details={
                    "free_gb": round(free_gb, 2),
                    "used_gb": round(used_gb, 2),
                    "total_gb": round(total_gb, 2),
                },
            )
        except Exception as e:
            logger.warning(f"Disk space health check failed: {e}")
            return ComponentHealth(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Disk check failed: {str(e)[:100]}",
                last_check=datetime.utcnow(),
                details={"error": str(e)[:200]},
            )

    # Performance metrics methods

    def _calculate_latency_percentiles(self) -> Dict[str, Dict[str, float]]:
        """Calculate latency percentiles for all operations."""
        percentiles = {}
        for operation, latencies in self._operation_latencies.items():
            if latencies:
                sorted_latencies = sorted(latencies)
                n = len(sorted_latencies)
                percentiles[operation] = {
                    "p50": round(sorted_latencies[int(n * 0.50)], 4),
                    "p95": round(sorted_latencies[int(n * 0.95)], 4),
                    "p99": round(sorted_latencies[int(n * 0.99)], 4),
                    "count": n,
                }
        return percentiles

    def _calculate_error_rates(self) -> Dict[str, Dict[str, Any]]:
        """Calculate error rates for all operations."""
        error_rates = {}

        # Combine all operations
        all_operations = set(self._operation_errors.keys()) | set(self._operation_successes.keys())

        for operation in all_operations:
            errors = self._operation_errors.get(operation, 0)
            successes = self._operation_successes.get(operation, 0)
            total = errors + successes

            if total > 0:
                error_rate = errors / total
                error_rates[operation] = {
                    "errors": errors,
                    "successes": successes,
                    "total": total,
                    "error_rate": round(error_rate, 4),
                }

        return error_rates

    # Resource usage methods

    def _get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage metrics."""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                mem_info = process.memory_info()

                return {
                    "rss_mb": round(mem_info.rss / (1024**2), 2),  # Resident Set Size
                    "vms_mb": round(mem_info.vms / (1024**2), 2),  # Virtual Memory Size
                    "available": True,
                }
            except Exception as e:
                logger.warning(f"Memory usage check failed: {e}")
                return {"available": False, "error": str(e)[:100]}
        else:
            return {"available": False, "reason": "psutil not installed"}

    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage metrics."""
        try:
            usage = shutil.disk_usage(".")
            return {
                "free_gb": round(usage.free / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "total_gb": round(usage.total / (1024**3), 2),
                "percent_used": round((usage.used / usage.total) * 100, 1),
            }
        except Exception as e:
            logger.warning(f"Disk usage check failed: {e}")
            return {"available": False, "error": str(e)[:100]}

    def _get_cache_usage(self) -> Dict[str, Any]:
        """Get cache usage metrics."""
        try:
            from src.embedding_cache import LRUEmbeddingCache

            cache = LRUEmbeddingCache.get_cache()
            stats = cache.get_stats()

            return {
                "entries": stats.entries,
                "capacity": stats.capacity,
                "hit_rate": round(stats.hit_rate, 3),
                "hits": stats.hits,
                "misses": stats.misses,
            }
        except Exception as e:
            logger.warning(f"Cache usage check failed: {e}")
            return {"available": False, "error": str(e)[:100]}


# Global singleton access
def get_health() -> HealthChecker:
    """
    Get global health checker instance.

    Returns:
        HealthChecker singleton instance
    """
    return HealthChecker.get_health()
