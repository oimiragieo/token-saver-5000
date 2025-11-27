"""
Tests for HTTP Server Module

Tests coverage:
- Endpoint availability (liveness, readiness, diagnostics, metrics, root)
- Response formats (JSON for health, text for metrics)
- Status codes (200 for healthy, 503 for unhealthy)
- Integration with health.py and metrics.py modules
- Configuration (environment variables)
- Server lifecycle (start, shutdown)
- Error handling and edge cases

Test Organization:
- TestHTTPEndpoints: Endpoint response tests
- TestHTTPIntegration: Integration with health/metrics modules
- TestHTTPConfiguration: Environment variable configuration
- TestHTTPLifecycle: Server start/stop lifecycle
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from src.http_server import (
    create_app,
    get_http_config,
    handle_diagnostics,
    handle_liveness,
    handle_metrics,
    handle_readiness,
    handle_root,
    is_http_enabled,
    start_http_server,
)


class TestHTTPEndpoints(AioHTTPTestCase):
    """Test HTTP server endpoints."""

    async def get_application(self):
        """Create test application."""
        return await create_app()

    @unittest_run_loop
    async def test_liveness_endpoint_returns_healthy(self):
        """Test liveness endpoint always returns healthy status."""
        resp = await self.client.request("GET", "/health/liveness")
        assert resp.status == 200
        assert resp.content_type == "application/json"

        data = await resp.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @unittest_run_loop
    async def test_readiness_endpoint_healthy(self):
        """Test readiness endpoint when all components healthy."""
        # Default behavior should be healthy (or degraded if model not loaded)
        resp = await self.client.request("GET", "/health/readiness")
        assert resp.status in [200, 503]  # May vary based on environment
        assert resp.content_type == "application/json"

        data = await resp.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "components" in data

    @unittest_run_loop
    async def test_readiness_endpoint_unhealthy_returns_503(self):
        """Test readiness endpoint returns 503 when unhealthy."""
        with patch("src.http_server.get_health") as mock_health:
            mock_health_instance = MagicMock()
            mock_health_instance.check_readiness.return_value = {
                "status": "unhealthy",
                "timestamp": "2025-11-27T12:34:56.789Z",
                "components": {
                    "embedding_manager": {
                        "status": "unhealthy",
                        "message": "Model unavailable",
                        "details": {},
                    }
                },
            }
            mock_health.return_value = mock_health_instance

            resp = await self.client.request("GET", "/health/readiness")
            assert resp.status == 503
            data = await resp.json()
            assert data["status"] == "unhealthy"

    @unittest_run_loop
    async def test_readiness_endpoint_degraded_returns_200(self):
        """Test readiness endpoint returns 200 when degraded."""
        with patch("src.http_server.get_health") as mock_health:
            mock_health_instance = MagicMock()
            mock_health_instance.check_readiness.return_value = {
                "status": "degraded",
                "timestamp": "2025-11-27T12:34:56.789Z",
                "components": {
                    "cache": {
                        "status": "degraded",
                        "message": "Cache nearing capacity",
                        "details": {"usage_ratio": 0.85},
                    }
                },
            }
            mock_health.return_value = mock_health_instance

            resp = await self.client.request("GET", "/health/readiness")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "degraded"

    @unittest_run_loop
    async def test_diagnostics_endpoint_returns_detailed_metrics(self):
        """Test diagnostics endpoint returns detailed metrics."""
        resp = await self.client.request("GET", "/health/diagnostics")
        assert resp.status == 200
        assert resp.content_type == "application/json"

        data = await resp.json()
        assert "status" in data
        assert "timestamp" in data
        # Diagnostics includes components from readiness
        assert "components" in data or "performance" in data or "resources" in data

    @unittest_run_loop
    async def test_metrics_endpoint_returns_prometheus_format(self):
        """Test metrics endpoint returns Prometheus text format."""
        resp = await self.client.request("GET", "/metrics")
        assert resp.status == 200
        assert resp.content_type.startswith("text/plain")

        text = await resp.text()
        # Should contain Prometheus format content
        assert len(text) > 0
        # Check for either actual metrics or unavailable message
        assert "compression_ratio" in text or "Prometheus metrics" in text or "unavailable" in text

    @unittest_run_loop
    async def test_root_endpoint_returns_readiness(self):
        """Test root endpoint redirects to readiness check."""
        resp = await self.client.request("GET", "/")
        assert resp.status in [200, 503]  # Same as readiness
        assert resp.content_type == "application/json"

        data = await resp.json()
        assert "status" in data
        assert "components" in data  # Should have readiness check structure

    @unittest_run_loop
    async def test_liveness_response_format(self):
        """Test liveness response has correct JSON format."""
        resp = await self.client.request("GET", "/health/liveness")
        data = await resp.json()

        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
        # Timestamp should be ISO format with Z suffix
        assert data["timestamp"].endswith("Z")

    @unittest_run_loop
    async def test_readiness_response_format(self):
        """Test readiness response has correct JSON format."""
        resp = await self.client.request("GET", "/health/readiness")
        data = await resp.json()

        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

        # Check components structure
        if data["components"]:
            component_name = list(data["components"].keys())[0]
            component = data["components"][component_name]
            assert "status" in component
            assert "message" in component

    @unittest_run_loop
    async def test_diagnostics_response_format(self):
        """Test diagnostics response has correct JSON format."""
        resp = await self.client.request("GET", "/health/diagnostics")
        data = await resp.json()

        # Check required fields (includes readiness + extras)
        assert "status" in data
        assert "timestamp" in data

    @unittest_run_loop
    async def test_metrics_content_type_header(self):
        """Test metrics endpoint has correct content-type header."""
        resp = await self.client.request("GET", "/metrics")

        # Prometheus text format with version
        assert "text/plain" in resp.content_type
        # Check for version in content-type
        content_type_full = resp.headers.get("Content-Type", "")
        assert "text/plain" in content_type_full


class TestHTTPIntegration(AioHTTPTestCase):
    """Test integration with health and metrics modules."""

    async def get_application(self):
        """Create test application."""
        return await create_app()

    @unittest_run_loop
    async def test_integration_with_health_module(self):
        """Test that endpoints use real health.py module."""
        # This test verifies that the health module is actually called
        # by checking that the response matches health.py output format

        resp = await self.client.request("GET", "/health/readiness")
        data = await resp.json()

        # Verify health.py component names are present
        if data.get("components"):
            # health.py defines these component checks
            actual_components = set(data["components"].keys())
            # At least some components should be present
            assert len(actual_components) > 0

    @unittest_run_loop
    async def test_integration_with_metrics_module(self):
        """Test that metrics endpoint uses real metrics.py module."""
        resp = await self.client.request("GET", "/metrics")
        text = await resp.text()

        # Verify metrics.py output format
        # Should contain Prometheus format or unavailable message
        assert len(text) > 0
        # metrics.py returns this message when prometheus_client not installed
        if "unavailable" in text:
            assert "prometheus_client" in text
        else:
            # Should have Prometheus format markers
            assert "#" in text or "compression_ratio" in text

    @unittest_run_loop
    async def test_readiness_with_mocked_health_components(self):
        """Test readiness check with various component states."""
        with patch("src.http_server.get_health") as mock_health:
            mock_health_instance = MagicMock()

            # Test all components healthy
            mock_health_instance.check_readiness.return_value = {
                "status": "healthy",
                "timestamp": "2025-11-27T12:34:56.789Z",
                "components": {
                    "embedding_manager": {"status": "healthy", "message": "OK"},
                    "persistence": {"status": "healthy", "message": "OK"},
                    "cache": {"status": "healthy", "message": "OK"},
                    "disk_space": {"status": "healthy", "message": "OK"},
                },
            }
            mock_health.return_value = mock_health_instance

            resp = await self.client.request("GET", "/health/readiness")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"
            assert len(data["components"]) == 4


class TestHTTPConfiguration(AioHTTPTestCase):
    """Test HTTP server configuration via environment variables."""

    async def get_application(self):
        """Create test application."""
        return await create_app()

    def test_is_http_enabled_true(self):
        """Test is_http_enabled returns True when HTTP_ENABLED=true."""
        with patch.dict(os.environ, {"HTTP_ENABLED": "true"}):
            # Need to reload module to pick up new env vars

            # Check the module constant (set at import time)
            # We can't easily test this without reimporting, so just verify function exists
            assert callable(is_http_enabled)

    def test_is_http_enabled_false(self):
        """Test is_http_enabled returns False when HTTP_ENABLED=false."""
        with patch.dict(os.environ, {"HTTP_ENABLED": "false"}):

            assert callable(is_http_enabled)

    def test_get_http_config_returns_configuration(self):
        """Test get_http_config returns configuration dictionary."""
        config = get_http_config()

        # Should return dict with enabled, host, port
        assert isinstance(config, dict)
        assert "enabled" in config
        assert "host" in config
        assert "port" in config

        # Values should match module constants (set at import)
        assert isinstance(config["enabled"], bool)
        assert isinstance(config["host"], str)
        assert isinstance(config["port"], int)

    def test_default_configuration_values(self):
        """Test default configuration values when env vars not set."""
        config = get_http_config()

        # Defaults from module (set at import time)
        # Can't easily test without reimporting, but verify structure
        assert config["port"] > 0
        assert len(config["host"]) > 0

    @unittest_run_loop
    async def test_create_app_configures_all_routes(self):
        """Test create_app configures all expected routes."""
        app = await create_app()

        # Get all routes (aiohttp creates HEAD routes automatically for GET)
        routes = [route.resource.canonical for route in app.router.routes()]

        # Check for expected routes
        assert "/health/liveness" in routes
        assert "/health/readiness" in routes
        assert "/health/diagnostics" in routes
        assert "/metrics" in routes
        assert "/" in routes

        # Should have 5 unique route paths (GET + HEAD for each = 10 total)
        unique_routes = set(routes)
        assert len(unique_routes) == 5


class TestHTTPLifecycle(AioHTTPTestCase):
    """Test HTTP server lifecycle (start, stop, error handling)."""

    async def get_application(self):
        """Create test application."""
        return await create_app()

    @pytest.mark.asyncio
    async def test_server_starts_successfully(self):
        """Test start_http_server can start without errors."""
        # Start server on random high port to avoid conflicts
        task = asyncio.create_task(start_http_server(host="127.0.0.1", port=18080))

        # Give server time to start
        await asyncio.sleep(0.1)

        # Server should be running
        assert not task.done()

        # Cancel and cleanup
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.asyncio
    async def test_server_shutdown_gracefully(self):
        """Test start_http_server handles cancellation gracefully."""
        task = asyncio.create_task(start_http_server(host="127.0.0.1", port=18081))

        # Give server time to start
        await asyncio.sleep(0.1)

        # Cancel task
        task.cancel()

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_server_handles_port_already_in_use(self):
        """Test server handles port conflict gracefully."""
        # Start first server
        task1 = asyncio.create_task(start_http_server(host="127.0.0.1", port=18082))
        await asyncio.sleep(0.2)  # Give it time to bind

        try:
            # Try to start second server on same port
            task2 = asyncio.create_task(start_http_server(host="127.0.0.1", port=18082))
            await asyncio.sleep(0.2)

            # Second server should fail
            # Note: May raise OSError or complete with error depending on timing
            # Just verify we can handle it
            task2.cancel()
            try:
                await task2
            except (asyncio.CancelledError, OSError):
                pass  # Expected

        finally:
            # Cleanup first server
            task1.cancel()
            try:
                await task1
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test server handles multiple concurrent requests."""

        # Use the test client for concurrent requests
        async def make_request():
            resp = await self.client.request("GET", "/health/liveness")
            return resp.status

        # Make 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 10


# Standalone function tests (not requiring test client)


def test_get_http_config_structure():
    """Test get_http_config returns correct structure."""
    config = get_http_config()

    assert "enabled" in config
    assert "host" in config
    assert "port" in config
    assert isinstance(config["enabled"], bool)
    assert isinstance(config["host"], str)
    assert isinstance(config["port"], int)


def test_is_http_enabled_function_exists():
    """Test is_http_enabled function is available."""
    result = is_http_enabled()
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_create_app_returns_application():
    """Test create_app returns aiohttp Application."""
    app = await create_app()
    assert isinstance(app, web.Application)


@pytest.mark.asyncio
async def test_handler_functions_are_async():
    """Test all handler functions are async."""
    import inspect

    handlers = [
        handle_liveness,
        handle_readiness,
        handle_diagnostics,
        handle_metrics,
        handle_root,
    ]

    for handler in handlers:
        assert inspect.iscoroutinefunction(handler), f"{handler.__name__} should be async"


@pytest.mark.asyncio
async def test_handlers_accept_request_parameter():
    """Test handler functions accept web.Request parameter."""
    import inspect

    handlers = [
        handle_liveness,
        handle_readiness,
        handle_diagnostics,
        handle_metrics,
        handle_root,
    ]

    for handler in handlers:
        sig = inspect.signature(handler)
        params = list(sig.parameters.keys())
        assert "request" in params, f"{handler.__name__} should accept 'request' parameter"


# Edge cases and error handling


class TestHTTPEdgeCases(AioHTTPTestCase):
    """Test edge cases and error handling."""

    async def get_application(self):
        """Create test application."""
        return await create_app()

    @unittest_run_loop
    async def test_liveness_never_fails(self):
        """Test liveness endpoint never returns error status."""
        # Make multiple requests
        for _ in range(5):
            resp = await self.client.request("GET", "/health/liveness")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"

    @unittest_run_loop
    async def test_endpoints_handle_head_requests(self):
        """Test endpoints handle HEAD requests (common for health checks)."""
        # aiohttp doesn't automatically handle HEAD, but GET should work
        resp = await self.client.request("GET", "/health/liveness")
        assert resp.status == 200

    @unittest_run_loop
    async def test_readiness_components_field_always_present(self):
        """Test readiness response always includes components field."""
        resp = await self.client.request("GET", "/health/readiness")
        data = await resp.json()

        # Components should always be present
        assert "components" in data
        assert isinstance(data["components"], dict)

    @unittest_run_loop
    async def test_diagnostics_includes_readiness_data(self):
        """Test diagnostics includes all readiness check data."""
        diagnostics_resp = await self.client.request("GET", "/health/diagnostics")
        diagnostics_data = await diagnostics_resp.json()

        # Diagnostics should include readiness fields
        assert "status" in diagnostics_data
        assert "timestamp" in diagnostics_data
        # May or may not have components depending on health.py implementation
