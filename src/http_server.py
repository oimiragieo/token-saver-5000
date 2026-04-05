"""
HTTP Server for Health Checks and Metrics Endpoints

Optional HTTP endpoints for production monitoring (Kubernetes, Docker).
Activated via HTTP_ENABLED=true environment variable.

Routes:
- GET /health/liveness   — always 200 if process is alive
- GET /health/readiness  — 200 if components ready, 503 if unhealthy
- GET /health/diagnostics — performance metrics, resource usage
- GET /metrics           — Prometheus text format

Configuration:
- HTTP_ENABLED (bool, default: false)
- HTTP_HOST   (str,  default: "0.0.0.0")
- HTTP_PORT   (int,  default: 8080)

Wired into the server via bootstrap.async_main() when HTTP_ENABLED=true.
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
    """Liveness probe. Returns 200 if the process is alive."""
    health = get_health()
    result = health.check_liveness()
    return web.json_response(result, status=200)


async def handle_readiness(request: web.Request) -> web.Response:
    """Readiness probe. Returns 200 if components are ready, 503 if unhealthy."""
    health = get_health()
    result = health.check_readiness()
    status = 503 if result["status"] == "unhealthy" else 200
    return web.json_response(result, status=status)


async def handle_diagnostics(request: web.Request) -> web.Response:
    """Detailed diagnostics: component health, latency percentiles, resource usage."""
    health = get_health()
    result = health.get_diagnostics()
    return web.json_response(result, status=200)


async def handle_metrics(request: web.Request) -> web.Response:
    """Prometheus metrics in text exposition format."""
    metrics = get_metrics()
    metrics_text = metrics.generate_metrics_text()
    return web.Response(text=metrics_text, content_type="text/plain; version=0.0.4")


async def handle_root(request: web.Request) -> web.Response:
    """Root handler — delegates to readiness probe."""
    return await handle_readiness(request)


async def create_app() -> web.Application:
    """Create and configure the aiohttp application with all routes."""
    app = web.Application()
    app.router.add_get("/health/liveness", handle_liveness)
    app.router.add_get("/health/readiness", handle_readiness)
    app.router.add_get("/health/diagnostics", handle_diagnostics)
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/", handle_root)
    return app


async def start_http_server(host: str = HTTP_HOST, port: int = HTTP_PORT) -> None:
    """Start HTTP health/metrics server as a background task.

    Runs indefinitely until the task is cancelled. Called by bootstrap.async_main()
    when HTTP_ENABLED=true.
    """
    app = await create_app()

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)

    try:
        await site.start()
        logger.info(f"HTTP server started on http://{host}:{port}")

        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await runner.cleanup()
        raise
    except Exception as e:
        logger.error(f"HTTP server error: {e}")
        await runner.cleanup()
        raise


def is_http_enabled() -> bool:
    """Return True if HTTP_ENABLED=true in the environment."""
    return HTTP_ENABLED


def get_http_config() -> Dict[str, Any]:
    """Return current HTTP server configuration."""
    return {
        "enabled": HTTP_ENABLED,
        "host": HTTP_HOST,
        "port": HTTP_PORT,
    }
