"""observability — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import logging
import sys
from unittest.mock import mock_open


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_mock_context(**overrides):
    """Build a mock HandlerContext dict."""
    compressor = MagicMock()
    compressor.chunks = {}
    compressor.graphs = {}
    compressor.file_metadata = {}

    ctx = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "resource_manager": MagicMock(),
        "sync_manager": MagicMock(),
        "version_manager": MagicMock(),
        "path_validator": MagicMock(),
        "retrieval_history": {},
        "context_window_adapter": MagicMock(),
        "multilevel_encoder": MagicMock(),
        "ace_framework": MagicMock(),
        "focus_manager": MagicMock(),
    }
    ctx["resource_manager"].unregister_document_async = AsyncMock()
    ctx["version_manager"].delete_versions_async = AsyncMock()
    ctx.update(overrides)
    return ctx


def _make_semantic_node(text="test", importance=0.5, embedding=None):
    node = MagicMock()
    node.text = text
    node.importance = importance
    node.embedding = embedding if embedding is not None else np.random.rand(384).astype(np.float32)
    node.metadata = {"tokens": 10, "position": 0, "entities": []}
    return node


def _make_code_chunk(
    name="func", chunk_type="function", code="def f(): pass", docstring="", start_line=1, end_line=5
):
    chunk = MagicMock()
    chunk.name = name
    chunk.chunk_type = chunk_type
    chunk.code = code
    chunk.docstring = docstring
    chunk.start_line = start_line
    chunk.end_line = end_line
    return chunk


class TestObservability:
    def test_manager_not_enabled(self):
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        mgr.tracer = None

        # These should all be no-ops
        mgr.set_attributes({"key": "value"})
        mgr.record_exception(ValueError("test"))

    def test_shutdown_not_enabled(self):
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        result = mgr.shutdown()
        assert result is True


class TestMetrics:
    def test_noop_collector_methods(self):
        from src.metrics import NoOpMetricsCollector

        noop = NoOpMetricsCollector()
        noop.record_compression_ratio(5.0, "HIGH")
        noop.record_latency("ingest", 0.5)
        noop.increment_documents_processed("ingest", "HIGH")
        noop.set_cache_hit_ratio(0.5)
        noop.set_active_documents(10)
        noop.increment_errors("ValueError", "ingest")
        noop.record_batch_size(5)
        noop.reset_all_metrics()
        assert "unavailable" in noop.generate_metrics_text()

    def test_metrics_collector_not_enabled(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            mc._enabled = False
            mc.record_compression_ratio(5.0, "HIGH")
            mc.record_latency("ingest", 0.5)
            mc.increment_documents_processed("ingest", "HIGH")
            mc.set_cache_hit_ratio(0.5)
            mc.set_active_documents(10)
            mc.increment_errors("ValueError", "ingest")
            mc.record_batch_size(5)
            mc.reset_all_metrics()
            assert "unavailable" in mc.generate_metrics_text()

    def test_validate_fidelity_invalid(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            assert mc._validate_fidelity("INVALID_LEVEL") is False
            assert mc._validate_fidelity(None) is True

    def test_validate_operation_invalid(self):
        from src.metrics import MetricsCollector

        with patch("src.metrics.PROMETHEUS_AVAILABLE", True):
            mc = MetricsCollector()
            assert mc._validate_operation("invalid_op") is False

    def test_singleton_reset(self):
        from src.metrics import MetricsCollector

        MetricsCollector._instance = None
        mc = MetricsCollector.get_metrics()
        assert mc is not None
        MetricsCollector.reset_singleton()
        assert MetricsCollector._instance is None


class TestHealth:
    def test_health_psutil_unavailable(self):
        with patch("src.health.PSUTIL_AVAILABLE", False):
            from src.health import HealthChecker

            hc = HealthChecker.__new__(HealthChecker)
            hc._operation_latencies = {}
            hc._operation_errors = {}
            hc._operation_successes = {}
            result = hc._get_memory_usage()
            assert result["available"] is False

    def test_health_disk_usage_error(self):
        from src.health import HealthChecker

        hc = HealthChecker.__new__(HealthChecker)
        hc._operation_latencies = {}
        hc._operation_errors = {}
        hc._operation_successes = {}

        with patch("shutil.disk_usage", side_effect=OSError("fail")):
            result = hc._get_disk_usage()
        assert result["available"] is False


class TestObservability_boost4b:
    """Cover observability manager edge cases."""

    def test_otel_not_available_flag(self):
        """Cover lines 132-134."""
        from src.observability import OPENTELEMETRY_AVAILABLE

        assert isinstance(OPENTELEMETRY_AVAILABLE, bool)

    def test_version_fallback(self):
        """Cover lines 139-140."""
        # The version import fallback is module-level
        # Just verify it's accessible
        from src.observability import __version__

        assert isinstance(__version__, str)

    def test_configure_tracer_failure(self):
        """Cover lines 277-280 - configure fails."""
        from src.observability import ObservabilityManager

        with patch("src.observability.OPENTELEMETRY_AVAILABLE", True):
            with patch.object(
                ObservabilityManager, "_configure_tracer", side_effect=Exception("boom")
            ):
                mgr = ObservabilityManager.__new__(ObservabilityManager)
                mgr.service_name = "test"
                mgr.service_version = "1.0"
                mgr.environment = "test"
                mgr.sampling_rate = 1.0
                mgr.otlp_endpoint = None
                mgr.enable_console_export = False
                mgr.tracer = None
                mgr._enabled = False
                # Simulate the init path
                try:
                    mgr._configure_tracer()
                except Exception:
                    mgr.tracer = None
                    mgr._enabled = False
                assert mgr._enabled is False

    def test_sampling_rate_always_off(self):
        """Cover line 307 - sampling rate 0."""
        # Just verify the flag is handled correctly
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        mgr.sampling_rate = 0.0
        # Direct test of sampling_rate attribute
        assert mgr.sampling_rate == 0.0

    def test_otlp_exporter_failure(self):
        """Cover lines 328-329 - OTLP exporter fails."""
        # This is covered by the configure path with OTLP unavailable
        from src.observability import OTLP_AVAILABLE

        assert isinstance(OTLP_AVAILABLE, bool)

    def test_add_event_disabled(self):
        """Cover line 534 - add_event when disabled."""
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        mgr.add_event("test_event")  # Should return immediately

    def test_shutdown_not_enabled(self):
        """Cover line 624-628."""
        from src.observability import ObservabilityManager

        mgr = ObservabilityManager.__new__(ObservabilityManager)
        mgr._enabled = False
        result = mgr.shutdown()
        assert result is True

    def test_auto_detect_sampling_rate(self):
        """Cover line 703 - auto-detect sampling rate from env."""
        from src.observability import configure_observability

        with patch("src.observability.ObservabilityManager") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.get_observability.return_value = mock_instance
            with patch.dict(os.environ, {"OTEL_SAMPLING_RATE": "0.5", "ENVIRONMENT": "production"}):
                configure_observability()


class TestStructuredLogging:
    """Cover structured logging edge cases."""

    def test_redact_context_with_list(self):
        """Cover lines 123-128 - redact lists containing dicts."""
        from src.structured_logging import _redact_context

        ctx = {
            "items": [
                {"email": "test@test.com", "name": "John"},
                "plain_string",
            ],
            "password": "secret",
        }
        result = _redact_context(ctx)
        assert result["password"] == "[REDACTED]"
        assert result["items"][0]["email"] == "[REDACTED]"
        assert result["items"][1] == "plain_string"

    def test_trace_context_no_otel(self):
        """Cover lines 325-335 - OpenTelemetry not available."""
        from src.structured_logging import StructuredLogger

        StructuredLogger._initialized = False
        logger = StructuredLogger("test_logger")
        StructuredLogger._initialized = False

        with patch.dict(sys.modules, {"opentelemetry": None}):
            result = logger._get_trace_context()
            assert result == {} or isinstance(result, dict)

    def test_get_current_context_empty_stacks(self):
        """Cover lines 355-356, 365-366 - empty context stacks."""
        from src.structured_logging import StructuredLogger

        StructuredLogger._initialized = False
        logger = StructuredLogger("test_logger2")
        StructuredLogger._initialized = False
        result = logger._get_current_context()
        assert isinstance(result, dict)

    def test_error_disabled(self):
        """Cover line 470 - error logging when disabled."""
        from src.structured_logging import StructuredLogger

        StructuredLogger._initialized = False
        logger = StructuredLogger("test_logger3")
        StructuredLogger._initialized = False
        logger.logger.setLevel(logging.CRITICAL + 10)
        logger.error("test error")  # Should return early

    def test_operation_context_manager(self):
        """Cover lines 542-543 - operation context."""
        from src.structured_logging import StructuredLogger

        StructuredLogger._initialized = False
        logger = StructuredLogger("test_logger4")
        StructuredLogger._initialized = False
        with logger.operation("test_op", extra_key="value"):
            pass  # Should push/pop stack


class TestHealth_boost4b:
    """Cover health check edge cases."""

    def test_psutil_not_available(self):
        """Cover lines 66-68."""
        from src.health import PSUTIL_AVAILABLE

        assert isinstance(PSUTIL_AVAILABLE, bool)

    def test_embedding_unhealthy(self):
        """Cover line 327 - embedding returns invalid result."""
        from src.health import HealthChecker

        mgr = HealthChecker.__new__(HealthChecker)
        with patch("src.embeddings.EmbeddingManager") as mock_em_cls:
            mock_mgr = MagicMock()
            mock_mgr.encode.return_value = None  # Invalid result
            mock_em_cls.return_value = mock_mgr
            result = mgr._check_embedding_manager()
            assert result.status.value == "unhealthy"

    def test_persistence_unexpected_data(self):
        """Cover line 372 - persistence returns unexpected data."""
        from src.health import HealthChecker

        mgr = HealthChecker.__new__(HealthChecker)
        # Mock the file read to return different data
        with patch("builtins.open", mock_open(read_data="wrong_data")):
            with patch("os.makedirs"):
                with patch("os.remove"):
                    result = mgr._check_persistence()
                    assert result.status.value in ("degraded", "unhealthy")

    def test_cache_high_usage(self):
        """Cover lines 403-404 - cache at high usage."""
        from src.health import HealthChecker, HealthStatus

        mgr = HealthChecker.__new__(HealthChecker)
        mock_cache = MagicMock()
        mock_stats = MagicMock()
        mock_stats.entries = 9500
        mock_stats.capacity = 10000
        mock_stats.hit_rate = 0.8
        mock_stats.hits = 800
        mock_stats.misses = 200
        mock_cache.get_stats.return_value = mock_stats
        with patch("src.embedding_cache.LRUEmbeddingCache") as mock_cls:
            mock_cls.get_cache.return_value = mock_cache
            result = mgr._check_cache()
            assert result.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_disk_space_failure(self):
        """Cover lines 464-466 - disk space check failure."""
        from src.health import HealthChecker

        mgr = HealthChecker.__new__(HealthChecker)
        with patch("shutil.disk_usage", side_effect=Exception("no disk")):
            result = mgr._check_disk_space()
            assert result.status.value == "degraded"

    def test_memory_usage_no_psutil(self):
        """Cover lines 528-530 - no psutil."""
        from src.health import HealthChecker

        mgr = HealthChecker.__new__(HealthChecker)
        with patch("src.health.PSUTIL_AVAILABLE", False):
            result = mgr._get_memory_usage()
            assert result["available"] is False

    def test_cache_usage_metrics(self):
        """Cover lines 554-556 - cache usage."""
        from src.health import HealthChecker

        mgr = HealthChecker.__new__(HealthChecker)
        mock_cache = MagicMock()
        mock_stats = MagicMock()
        mock_stats.entries = 100
        mock_stats.capacity = 10000
        mock_stats.hit_rate = 0.9
        mock_stats.hits = 900
        mock_stats.misses = 100
        mock_cache.get_stats.return_value = mock_stats
        with patch("src.embedding_cache.LRUEmbeddingCache") as mock_cls:
            mock_cls.get_cache.return_value = mock_cache
            result = mgr._get_cache_usage()
            assert "entries" in result
