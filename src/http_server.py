"""
HTTP Server for Health Checks and Metrics Endpoints

This module provides optional HTTP endpoints for production monitoring when
Token Saver 5000 is deployed in Kubernetes or other environments requiring
HTTP health probes and Prometheus metrics scraping.

Features:
- GET /health/liveness - Liveness probe (always returns healthy if server running)
- GET /health/readiness - Readiness probe (checks component health)
- GET /health/diagnostics - Detailed diagnostics (performance metrics, resource usage)
- GET /metrics - Prometheus metrics in text format

Configuration via environment variables:
- HTTP_ENABLED (bool, default: false) - Enable HTTP server
- HTTP_HOST (str, default: "0.0.0.0") - Bind host
- HTTP_PORT (int, default: 8080) - Bind port

Usage:
    # Start HTTP server alongside MCP stdio server
    import asyncio
    from src.http_server import start_http_server

    async def main():
        # Start HTTP server in background
        http_task = None
        if os.getenv("HTTP_ENABLED", "false").lower() == "true":
            http_task = asyncio.create_task(start_http_server())

        # Run MCP stdio server
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, InitializationOptions(...))

        # Cleanup
        if http_task:
            http_task.cancel()

Architecture:
- Async aiohttp server for non-blocking operations
- Integration with existing health.py and metrics.py modules
- Minimal overhead (separate from MCP stdio transport)
- Graceful shutdown handling
- Comprehensive logging for visibility

Example Kubernetes Configuration:
    apiVersion: v1
    kind: Pod
    metadata:
      name: token-saver-5000
    spec:
      containers:
      - name: token-saver
        image: token-saver-5000:latest
        env:
        - name: HTTP_ENABLED
          value: "true"
        - name: HTTP_PORT
          value: "8080"
        ports:
        - containerPort: 8080
          name: http-health
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
"""

import asyncio
import logging
import os
from typing import Any, Dict

from aiohttp import web

from src.health import get_health
from src.metrics import get_metrics

logger = logging.getLogger(__name__)

# Environment variable configuration with defaults
HTTP_ENABLED = os.getenv("HTTP_ENABLED", "false").lower() == "true"
HTTP_HOST = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))


async def handle_liveness(request: web.Request) -> web.Response:
    """
    Liveness probe endpoint.

    Returns 200 OK if server is running. This endpoint should never fail
    unless the server is completely down. Used by Kubernetes/load balancers
    to determine if the process is alive.

    Args:
        request: aiohttp web request

    Returns:
        HTTP 200 with {"status": "healthy", "timestamp": "..."}

    Example Response:
        {
            "status": "healthy",
            "timestamp": "2025-11-27T12:34:56.789Z"
        }
    """
    health = get_health()
    result = health.check_liveness()

    logger.debug(f"Liveness check: {result['status']}")
    return web.json_response(result, status=200)


async def handle_readiness(request: web.Request) -> web.Response:
    """
    Readiness probe endpoint.

    Returns 200 OK if all components are ready, 503 if any component is unhealthy.
    Checks:
    - Embedding manager (model loaded, inference working)
    - Persistence (storage accessible)
    - Cache (LRU eviction working)
    - Disk space (sufficient for operations)

    Used by Kubernetes to determine if the pod should receive traffic.

    Args:
        request: aiohttp web request

    Returns:
        HTTP 200 if healthy/degraded, HTTP 503 if unhealthy

    Example Response (healthy):
        {
            "status": "healthy",
            "timestamp": "2025-11-27T12:34:56.789Z",
            "components": {
                "embedding_manager": {"status": "healthy", "message": "..."},
                "persistence": {"status": "healthy", "message": "..."},
                "cache": {"status": "healthy", "message": "..."},
                "disk_space": {"status": "healthy", "message": "..."}
            }
        }

    Example Response (unhealthy):
        {
            "status": "unhealthy",
            "timestamp": "2025-11-27T12:34:56.789Z",
            "components": {
                "embedding_manager": {"status": "unhealthy", "message": "Model unavailable"}
            }
        }
    """
    health = get_health()
    result = health.check_readiness()

    # Return 503 if unhealthy, 200 otherwise (including degraded)
    # Degraded means service is operational but with reduced capacity
    status = 503 if result["status"] == "unhealthy" else 200

    logger.debug(f"Readiness check: {result['status']} (HTTP {status})")
    return web.json_response(result, status=status)


async def handle_diagnostics(request: web.Request) -> web.Response:
    """
    Diagnostics endpoint with detailed metrics.

    Returns detailed diagnostics including:
    - Component health (from readiness check)
    - Performance metrics (p50/p95/p99 latency percentiles)
    - Resource usage (memory, disk, cache)
    - Error rates by operation

    This endpoint is intended for debugging and monitoring dashboards,
    not for automated health checks (use /health/readiness instead).

    Args:
        request: aiohttp web request

    Returns:
        HTTP 200 with detailed diagnostics

    Example Response:
        {
            "status": "healthy",
            "timestamp": "2025-11-27T12:34:56.789Z",
            "components": {...},
            "performance": {
                "latencies": {
                    "compress": {"p50": 0.35, "p95": 0.72, "p99": 1.15, "count": 1234}
                },
                "error_rates": {
                    "compress": {"errors": 5, "successes": 1229, "total": 1234, "error_rate": 0.004}
                }
            },
            "resources": {
                "memory": {"rss_mb": 450.2, "vms_mb": 1024.5},
                "disk": {"free_gb": 12.5, "used_gb": 87.5, "total_gb": 100.0},
                "cache": {"entries": 850, "capacity": 10000, "hit_rate": 0.75}
            }
        }
    """
    health = get_health()
    result = health.get_diagnostics()

    logger.debug("Diagnostics check completed")
    return web.json_response(result, status=200)


async def handle_metrics(request: web.Request) -> web.Response:
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping:
    - compression_ratio: Histogram of compression effectiveness
    - processing_latency_seconds: Histogram of operation latency
    - documents_processed_total: Counter of processed documents
    - cache_hit_ratio: Gauge of current cache hit rate
    - active_documents: Gauge of currently loaded documents
    - errors_total: Counter of errors by type
    - batch_size: Histogram of batch processing sizes

    Args:
        request: aiohttp web request

    Returns:
        HTTP 200 with Prometheus text format (content-type: text/plain)

    Example Response:
        # HELP compression_ratio Compression ratio distribution
        # TYPE compression_ratio histogram
        compression_ratio_bucket{fidelity_level="BALANCED",le="1.0"} 0
        compression_ratio_bucket{fidelity_level="BALANCED",le="7.0"} 15
        compression_ratio_bucket{fidelity_level="BALANCED",le="10.0"} 42
        compression_ratio_sum{fidelity_level="BALANCED"} 315.5
        compression_ratio_count{fidelity_level="BALANCED"} 42
        ...
    """
    metrics = get_metrics()
    metrics_text = metrics.generate_metrics_text()

    logger.debug("Metrics endpoint scraped")
    return web.Response(text=metrics_text, content_type="text/plain; version=0.0.4")


async def handle_root(request: web.Request) -> web.Response:
    """
    Root endpoint handler.

    Redirects to /health/readiness for convenience. This allows simple
    HTTP GET requests to the root to check overall health status.

    Args:
        request: aiohttp web request

    Returns:
        Same as /health/readiness endpoint
    """
    # Reuse readiness handler
    return await handle_readiness(request)


async def create_app() -> web.Application:
    """
    Create and configure aiohttp application.

    Configures all routes and middleware for the HTTP server:
    - /health/liveness - Liveness probe
    - /health/readiness - Readiness probe
    - /health/diagnostics - Detailed diagnostics
    - /metrics - Prometheus metrics
    - / - Root (redirects to readiness)

    Returns:
        Configured aiohttp.web.Application

    Example:
        >>> app = await create_app()
        >>> runner = web.AppRunner(app)
        >>> await runner.setup()
        >>> site = web.TCPSite(runner, "0.0.0.0", 8080)
        >>> await site.start()
    """
    app = web.Application()

    # Add routes
    app.router.add_get("/health/liveness", handle_liveness)
    app.router.add_get("/health/readiness", handle_readiness)
    app.router.add_get("/health/diagnostics", handle_diagnostics)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/", handle_root)

    logger.info(
        "HTTP server configured with routes: "
        "/health/liveness, /health/readiness, /health/diagnostics, /metrics, /"
    )

    return app


async def start_http_server(host: str = HTTP_HOST, port: int = HTTP_PORT) -> None:
    """
    Start HTTP server for health checks and metrics.

    This function should be run as an asyncio background task alongside
    the main MCP stdio server. It will run indefinitely until cancelled.

    The server lifecycle is managed by aiohttp's AppRunner and TCPSite,
    which handle graceful shutdown when the task is cancelled.

    Args:
        host: Bind host (default: from HTTP_HOST env var, fallback to "0.0.0.0")
        port: Bind port (default: from HTTP_PORT env var, fallback to 8080)

    Raises:
        asyncio.CancelledError: When the task is cancelled for shutdown
        OSError: If port is already in use or binding fails

    Example:
        >>> import asyncio
        >>> from src.http_server import start_http_server
        >>>
        >>> async def main():
        ...     # Start HTTP server in background
        ...     http_task = asyncio.create_task(start_http_server())
        ...
        ...     # Run MCP stdio server
        ...     async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        ...         await server.run(read_stream, write_stream, InitializationOptions(...))
        ...
        ...     # Cleanup
        ...     http_task.cancel()
        ...     try:
        ...         await http_task
        ...     except asyncio.CancelledError:
        ...         pass
        >>>
        >>> asyncio.run(main())
    """
    app = await create_app()

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)

    try:
        await site.start()
        logger.info(f"HTTP server started on http://{host}:{port}")
        logger.info(
            f"Health endpoints: http://{host}:{port}/health/{{liveness,readiness,diagnostics}}"
        )
        logger.info(f"Metrics endpoint: http://{host}:{port}/metrics")

        # Run forever (until task is cancelled)
        while True:
            await asyncio.sleep(3600)  # Sleep forever, waking every hour
    except asyncio.CancelledError:
        logger.info("HTTP server shutting down...")
        await runner.cleanup()
        logger.info("HTTP server stopped")
        raise  # Re-raise to allow proper task cancellation
    except Exception as e:
        logger.error(f"HTTP server error: {e}")
        await runner.cleanup()
        raise


def is_http_enabled() -> bool:
    """
    Check if HTTP server is enabled via environment variable.

    Returns:
        True if HTTP_ENABLED="true", False otherwise

    Example:
        >>> import os
        >>> from src.http_server import is_http_enabled
        >>>
        >>> os.environ["HTTP_ENABLED"] = "true"
        >>> print(is_http_enabled())  # True
        >>>
        >>> os.environ["HTTP_ENABLED"] = "false"
        >>> print(is_http_enabled())  # False
    """
    return HTTP_ENABLED


def get_http_config() -> Dict[str, Any]:
    """
    Get HTTP server configuration from environment variables.

    Returns:
        Dict with enabled, host, and port configuration

    Example:
        >>> from src.http_server import get_http_config
        >>> config = get_http_config()
        >>> print(config)
        {"enabled": True, "host": "0.0.0.0", "port": 8080}
    """
    return {
        "enabled": HTTP_ENABLED,
        "host": HTTP_HOST,
        "port": HTTP_PORT,
    }
