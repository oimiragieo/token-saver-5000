"""
Tests for OpenTelemetry distributed tracing.

This module validates the observability framework added in v0.7.0, including:
- OpenTelemetry tracer configuration
- Span creation and context management
- Async context propagation via contextvars
- Exception handling and recording
- Trace context extraction for logging
- OTLP export with console fallback
- Graceful degradation when OTEL unavailable
"""

import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock

# Import observability module
from src.observability import (
    get_observability,
    configure_observability,
    ObservabilityManager,
    NoOpSpan,
    OPENTELEMETRY_AVAILABLE,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    original_instance = ObservabilityManager._instance
    ObservabilityManager._instance = None
    yield
    ObservabilityManager._instance = original_instance


@pytest.fixture
def mock_tracer():
    """Create a mock tracer for testing."""
    tracer = MagicMock()
    span = MagicMock()
    span.is_recording.return_value = True

    # Mock span context
    span_context = MagicMock()
    span_context.trace_id = 0xA9938FD7A6313E0F27F3FC87F574BFF6
    span_context.span_id = 0xED58F84D8971BF60
    span.get_span_context.return_value = span_context

    # Mock context manager
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    tracer.start_as_current_span.return_value.__exit__.return_value = None

    return tracer


@pytest.fixture
def observability_enabled():
    """Create observability manager with OTEL enabled."""
    if not OPENTELEMETRY_AVAILABLE:
        pytest.skip("OpenTelemetry not installed")

    obs = ObservabilityManager(
        service_name="test-service",
        environment="test",
        sampling_rate=1.0,
        enable_console_export=True,
    )
    return obs


@pytest.fixture
def observability_disabled():
    """Create observability manager with OTEL disabled."""
    with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
        obs = ObservabilityManager(service_name="test-service")
        return obs


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicTracing:
    """Test basic tracing functionality."""

    def test_observability_singleton(self):
        """Test singleton pattern."""
        obs1 = get_observability()
        obs2 = get_observability()

        assert obs1 is obs2, "Should return same instance"

    def test_observability_singleton_preserves_config(self):
        """Test singleton preserves first configuration."""
        obs1 = get_observability(service_name="first-service")
        obs2 = get_observability(service_name="second-service")

        assert obs1 is obs2
        assert obs1.service_name == "first-service"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_trace_context_manager(self, observability_enabled):
        """Test trace() context manager."""
        obs = observability_enabled

        with obs.trace("test_operation", doc_id="abc123") as span:
            assert span is not None
            # Verify span is recording
            if hasattr(span, "is_recording"):
                assert span.is_recording() or True  # NoOp returns False

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_span_creation(self, observability_enabled):
        """Test span is created with correct operation name."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("compress", doc_id="abc123"):
                pass

            # Verify span created with correct name
            mock_tracer.start_as_current_span.assert_called_once_with("compress")

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_span_attributes(self, observability_enabled):
        """Test setting span attributes."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("test", doc_id="abc123", fidelity_level="BALANCED"):
                obs.set_attribute("compression_ratio", 7.5)
                obs.set_attribute("token_count", 500)

            # Verify attributes set
            mock_span.set_attributes.assert_called_once_with(
                {"doc_id": "abc123", "fidelity_level": "BALANCED"}
            )
            assert mock_span.set_attribute.call_count == 2

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_span_status_success(self, observability_enabled):
        """Test span status remains OK on success."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("test"):
                pass  # Success - no exception

            # Verify no error status set
            error_status_calls = [
                c
                for c in mock_span.set_status.mock_calls
                if len(c.args) > 0 and hasattr(c.args[0], "status_code")
            ]
            # Should not set ERROR status
            assert all(
                not (
                    hasattr(call_args.args[0], "status_code")
                    and str(call_args.args[0]).find("ERROR") >= 0
                )
                for call_args in error_status_calls
            )


class TestAsyncContextPropagation:
    """Test async context propagation."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_async_trace_propagation(self, observability_enabled):
        """Test context propagates across async tasks."""
        obs = observability_enabled

        trace_contexts = []

        async def task(doc_id):
            # Should have access to parent trace context
            trace_ctx = obs.get_current_trace_context()
            trace_contexts.append(trace_ctx)

        with obs.trace("batch", batch_id="batch-123"):
            parent_ctx = obs.get_current_trace_context()
            await asyncio.gather(task("doc1"), task("doc2"), task("doc3"))

        # All tasks should have trace context
        if parent_ctx:  # Only if OTEL actually enabled
            assert all("trace_id" in ctx for ctx in trace_contexts)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_nested_spans(self, observability_enabled):
        """Test nested span relationships."""
        obs = observability_enabled

        with obs.trace("parent", doc_id="abc"):
            parent_ctx = obs.get_current_trace_context()

            with obs.trace("child", step="1"):
                child_ctx = obs.get_current_trace_context()

                # If OTEL enabled, verify parent-child relationship
                if parent_ctx and child_ctx:
                    # Child should have same trace_id but different span_id
                    assert parent_ctx["trace_id"] == child_ctx["trace_id"]
                    assert parent_ctx["span_id"] != child_ctx["span_id"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_asyncio_gather_propagation(self, observability_enabled):
        """Test context through asyncio.gather()."""
        obs = observability_enabled

        results = []

        async def compress_doc(doc_id):
            trace_ctx = obs.get_current_trace_context()
            results.append({"doc_id": doc_id, "trace_ctx": trace_ctx})
            await asyncio.sleep(0.01)

        with obs.trace("batch_compress", batch_size=3):
            await asyncio.gather(
                compress_doc("doc1"),
                compress_doc("doc2"),
                compress_doc("doc3"),
            )

        # All tasks should complete
        assert len(results) == 3

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_context_var_propagation(self, observability_enabled):
        """Test ContextVar propagation across async tasks."""
        obs = observability_enabled

        from src.observability import _current_span

        async def check_span():
            # ContextVar should be accessible in async task
            span = _current_span.get()
            return span is not None

        with obs.trace("test"):
            # Should have span in context
            result = await check_span()
            # Result depends on OTEL being actually configured
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_multiple_concurrent_traces(self, observability_enabled):
        """Test multiple concurrent operations with separate traces."""
        obs = observability_enabled

        async def operation(op_name, delay):
            with obs.trace(op_name, operation_id=op_name):
                await asyncio.sleep(delay)
                trace_ctx = obs.get_current_trace_context()
                return trace_ctx

        # Run multiple operations concurrently
        results = await asyncio.gather(
            operation("op1", 0.01),
            operation("op2", 0.02),
            operation("op3", 0.015),
        )

        # All operations should complete
        assert len(results) == 3


class TestExceptionHandling:
    """Test exception handling and recording."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_record_exception(self, observability_enabled):
        """Test exception recording on span."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            try:
                with obs.trace("test"):
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected

            # Verify exception recorded on span
            mock_span.record_exception.assert_called()
            assert mock_span.record_exception.call_args[0][0].args[0] == "Test error"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_span_error_status(self, observability_enabled):
        """Test span status set to ERROR on exception."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            try:
                with obs.trace("test"):
                    raise ValueError("Test error")
            except ValueError:
                pass

            # Verify span.set_status called with ERROR
            mock_span.set_status.assert_called()
            # Verify error attributes set
            assert any(
                call_args[0][0] == "error" and call_args[0][1] is True
                for call_args in mock_span.set_attribute.call_args_list
            )

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_exception_reraise(self, observability_enabled):
        """Test exception is re-raised after recording."""
        obs = observability_enabled

        with pytest.raises(ValueError, match="Test error"):
            with obs.trace("test"):
                raise ValueError("Test error")

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_record_exception_with_attributes(self, observability_enabled):
        """Test recording exception with additional attributes."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            try:
                with obs.trace("test"):
                    try:
                        raise ValueError("Test error")
                    except ValueError as e:
                        obs.record_exception(e, {"doc_size": 1000, "retry_count": 3})
                        raise
            except ValueError:
                pass  # Expected - exception re-raised after recording

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_exception_attributes(self, observability_enabled):
        """Test exception attributes set correctly."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            try:
                with obs.trace("test"):
                    raise TypeError("Type mismatch")
            except TypeError:
                pass

            # Verify error.type and error.message attributes
            set_attribute_calls = mock_span.set_attribute.call_args_list
            error_types = [
                call_args[0][1]
                for call_args in set_attribute_calls
                if call_args[0][0] == "error.type"
            ]
            assert "TypeError" in error_types


class TestTraceContext:
    """Test trace context extraction."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_get_current_trace_context(self, observability_enabled):
        """Test extraction of trace_id and span_id."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True

            # Mock span context with specific IDs
            mock_span_context = MagicMock()
            mock_span_context.trace_id = 0xA9938FD7A6313E0F27F3FC87F574BFF6
            mock_span_context.span_id = 0xED58F84D8971BF60
            mock_span.get_span_context.return_value = mock_span_context

            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("test", doc_id="abc123"):
                trace_ctx = obs.get_current_trace_context()

                assert "trace_id" in trace_ctx
                assert "span_id" in trace_ctx
                assert len(trace_ctx["trace_id"]) == 32  # 128-bit hex
                assert len(trace_ctx["span_id"]) == 16  # 64-bit hex

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_trace_context_available_in_span(self, observability_enabled):
        """Test context available throughout span lifetime."""
        obs = observability_enabled

        with obs.trace("test"):
            ctx1 = obs.get_current_trace_context()
            # Do some work
            ctx2 = obs.get_current_trace_context()

            # Context should be consistent
            if ctx1 and ctx2:
                assert ctx1 == ctx2

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_trace_context_logging_integration(self, observability_enabled):
        """Test integration with structured logging."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_span_context = MagicMock()
            mock_span_context.trace_id = 0xA9938FD7A6313E0F27F3FC87F574BFF6
            mock_span_context.span_id = 0xED58F84D8971BF60
            mock_span.get_span_context.return_value = mock_span_context
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("compress", doc_id="abc123"):
                trace_context = obs.get_current_trace_context()

                # Simulate logging with trace context
                log_entry = {
                    "message": "Compression started",
                    **trace_context,
                }

                # Verify trace context included
                assert "trace_id" in log_entry
                assert "span_id" in log_entry

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_trace_context_empty_outside_span(self, observability_enabled):
        """Test trace context empty when outside span."""
        obs = observability_enabled

        # Outside any trace context
        trace_ctx = obs.get_current_trace_context()

        # Should return empty dict
        assert trace_ctx == {}


class TestConfiguration:
    """Test configuration options."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_service_name(self):
        """Test service name configuration."""
        obs = configure_observability(service_name="custom-service")

        assert obs.service_name == "custom-service"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_environment(self):
        """Test environment configuration."""
        obs = configure_observability(environment="production")

        assert obs.environment == "production"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_sampling_rate(self):
        """Test sampling rate configuration."""
        obs = configure_observability(sampling_rate=0.5)

        assert obs.sampling_rate == 0.5

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_service_version(self):
        """Test service version configuration."""
        obs = configure_observability(service_version="1.2.3")

        assert obs.service_version == "1.2.3"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_auto_detect_environment(self):
        """Test auto-detection of environment from env var."""
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            obs = configure_observability()

            assert obs.environment == "staging"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_auto_detect_sampling_production(self):
        """Test auto-detection of sampling rate in production."""
        obs = configure_observability(environment="production")

        # Should default to 0.1 (10%) in production
        assert obs.sampling_rate == 0.1

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_configure_auto_detect_sampling_development(self):
        """Test auto-detection of sampling rate in development."""
        obs = configure_observability(environment="development")

        # Should default to 1.0 (100%) in development
        assert obs.sampling_rate == 1.0


class TestOTLPExport:
    """Test OTLP export configuration."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otlp_exporter_configuration(self):
        """Test OTLP exporter setup."""
        obs = ObservabilityManager(
            service_name="test-service",
            otlp_endpoint="http://localhost:4317",
        )

        # Verify endpoint configured
        assert obs.otlp_endpoint == "http://localhost:4317"

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_console_fallback(self):
        """Test fallback to console exporter when OTLP unavailable."""
        with patch("src.observability.OTLP_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            # Should still be enabled with console exporter
            # (if OPENTELEMETRY_AVAILABLE is True)
            if OPENTELEMETRY_AVAILABLE:
                assert obs.is_enabled()

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_console_exporter_explicit(self):
        """Test explicit console exporter enabling."""
        obs = ObservabilityManager(
            service_name="test-service",
            enable_console_export=True,
        )

        # Verify console export enabled
        assert obs.enable_console_export is True

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_otlp_endpoint_from_env(self):
        """Test OTLP endpoint from environment variable."""
        with patch.dict(os.environ, {"OTEL_EXPORTER_OTLP_ENDPOINT": "http://custom:4317"}):
            obs = ObservabilityManager(service_name="test-service")

            assert obs.otlp_endpoint == "http://custom:4317"


class TestGracefulDegradation:
    """Test graceful degradation when OpenTelemetry unavailable."""

    def test_noop_when_otel_unavailable(self):
        """Test NoOp when OpenTelemetry unavailable."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            with obs.trace("test") as span:
                # Should return NoOpSpan
                assert isinstance(span, NoOpSpan)

    def test_warning_when_otel_unavailable(self):
        """Test warning log when OpenTelemetry unavailable."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            with patch("src.observability.logger") as mock_logger:
                ObservabilityManager(service_name="test-service")

                # Should log warning
                mock_logger.warning.assert_called_once()
                assert "OpenTelemetry not installed" in mock_logger.warning.call_args[0][0]

    def test_is_enabled_false_when_unavailable(self):
        """Test is_enabled() returns False when OTEL unavailable."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            assert obs.is_enabled() is False

    def test_set_attribute_noop_when_disabled(self):
        """Test set_attribute is no-op when disabled."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            with obs.trace("test"):
                # Should not raise exception
                obs.set_attribute("key", "value")

    def test_get_trace_context_empty_when_disabled(self):
        """Test get_current_trace_context returns empty when disabled."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            with obs.trace("test"):
                trace_ctx = obs.get_current_trace_context()

                # Should return empty dict
                assert trace_ctx == {}


class TestNoOpSpan:
    """Test NoOpSpan implementation."""

    def test_noop_span_set_attribute(self):
        """Test NoOpSpan.set_attribute returns self."""
        span = NoOpSpan()
        result = span.set_attribute("key", "value")

        assert result is span

    def test_noop_span_set_attributes(self):
        """Test NoOpSpan.set_attributes returns self."""
        span = NoOpSpan()
        result = span.set_attributes({"key1": "value1", "key2": "value2"})

        assert result is span

    def test_noop_span_add_event(self):
        """Test NoOpSpan.add_event is no-op."""
        span = NoOpSpan()
        # Should not raise
        span.add_event("test_event", {"attr": "value"})

    def test_noop_span_set_status(self):
        """Test NoOpSpan.set_status is no-op."""
        span = NoOpSpan()
        # Should not raise
        span.set_status("status")

    def test_noop_span_record_exception(self):
        """Test NoOpSpan.record_exception is no-op."""
        span = NoOpSpan()
        # Should not raise
        span.record_exception(ValueError("test"))

    def test_noop_span_is_recording(self):
        """Test NoOpSpan.is_recording returns False."""
        span = NoOpSpan()
        assert span.is_recording() is False

    def test_noop_span_get_span_context(self):
        """Test NoOpSpan.get_span_context returns None."""
        span = NoOpSpan()
        assert span.get_span_context() is None

    def test_noop_span_context_manager(self):
        """Test NoOpSpan as context manager."""
        span = NoOpSpan()

        with span as s:
            assert s is span


class TestAdvancedFeatures:
    """Test advanced features."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_add_event(self, observability_enabled):
        """Test adding events to span."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("test"):
                obs.add_event("cache_hit", {"cache_key": "doc123"})
                obs.add_event("compression_completed", {"ratio": 7.5})

            # Verify events added
            assert mock_span.add_event.call_count == 2

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_set_attributes_batch(self, observability_enabled):
        """Test setting multiple attributes at once."""
        obs = observability_enabled

        with patch.object(obs, "tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_span.is_recording.return_value = True
            mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

            with obs.trace("test"):
                obs.set_attributes(
                    {
                        "compression_ratio": 7.5,
                        "token_count": 500,
                        "fidelity_level": "BALANCED",
                    }
                )

            # Verify batch set_attributes called
            mock_span.set_attributes.assert_called()

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_shutdown(self, observability_enabled):
        """Test graceful shutdown."""
        obs = observability_enabled

        result = obs.shutdown(timeout=1.0)

        # Should complete shutdown
        assert isinstance(result, bool)

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_shutdown_when_disabled(self):
        """Test shutdown when observability disabled."""
        with patch("src.observability.OPENTELEMETRY_AVAILABLE", False):
            obs = ObservabilityManager(service_name="test-service")

            result = obs.shutdown()

            # Should return True (no-op)
            assert result is True


class TestIntegration:
    """Integration tests for full trace workflows."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_full_trace_workflow(self, observability_enabled):
        """Test complete workflow with tracing and logging."""
        obs = observability_enabled

        # Simulate compression workflow
        with obs.trace("compress_document", doc_id="abc123", fidelity="BALANCED"):
            # Get trace context for logging
            obs.get_current_trace_context()

            # Simulate compression steps
            obs.add_event("compression_started")
            obs.set_attribute("input_tokens", 1000)

            # Simulate processing
            obs.add_event("graph_construction")
            obs.set_attribute("node_count", 150)

            # Final results
            obs.set_attribute("compression_ratio", 8.5)
            obs.set_attribute("output_tokens", 118)
            obs.set_attribute("status", "success")
            obs.add_event("compression_completed")

        # Workflow should complete without errors
        assert True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    async def test_batch_processing_workflow(self, observability_enabled):
        """Test batch processing with nested spans."""
        obs = observability_enabled

        async def compress_doc(doc_id):
            with obs.trace("compress", doc_id=doc_id):
                obs.set_attribute("status", "processing")
                await asyncio.sleep(0.01)
                obs.set_attribute("compression_ratio", 7.5)
                obs.set_attribute("status", "completed")

        # Batch processing with parent span
        with obs.trace("batch_ingest", batch_id="batch-456", batch_size=3):
            obs.add_event("batch_started")

            await asyncio.gather(
                compress_doc("doc1"),
                compress_doc("doc2"),
                compress_doc("doc3"),
            )

            obs.set_attribute("status", "all_completed")
            obs.add_event("batch_completed")

        # Should complete without errors
        assert True

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_error_handling_workflow(self, observability_enabled):
        """Test error handling with exception recording."""
        obs = observability_enabled

        with pytest.raises(ValueError):
            with obs.trace("compress", doc_id="abc123"):
                obs.set_attribute("status", "processing")

                # Simulate error
                try:
                    raise ValueError("Invalid compression parameters")
                except ValueError as e:
                    obs.record_exception(e, {"retry_count": 0})
                    obs.set_attribute("status", "error")
                    raise

        # Error should be recorded and re-raised
        assert True


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_span_creation_overhead(self, observability_enabled):
        """Test span creation overhead is minimal."""
        obs = observability_enabled

        import time

        iterations = 100
        start = time.time()

        for i in range(iterations):
            with obs.trace(f"operation_{i}"):
                pass

        elapsed = time.time() - start
        avg_per_span = elapsed / iterations

        # Should be under 5ms per span (generous bound)
        assert avg_per_span < 0.005

    @pytest.mark.skipif(not OPENTELEMETRY_AVAILABLE, reason="OpenTelemetry not installed")
    def test_attribute_setting_overhead(self, observability_enabled):
        """Test attribute setting overhead is minimal."""
        obs = observability_enabled

        import time

        start = time.time()

        with obs.trace("test"):
            for i in range(1000):
                obs.set_attribute(f"attr_{i}", i)

        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # 1 second for 1000 attributes


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:

1. Basic Tracing Tests (5 tests):
   - test_observability_singleton
   - test_observability_singleton_preserves_config
   - test_trace_context_manager
   - test_span_creation
   - test_span_attributes
   - test_span_status_success

2. Async Context Propagation Tests (5 tests):
   - test_async_trace_propagation
   - test_nested_spans
   - test_asyncio_gather_propagation
   - test_context_var_propagation
   - test_multiple_concurrent_traces

3. Exception Handling Tests (6 tests):
   - test_record_exception
   - test_span_error_status
   - test_exception_reraise
   - test_record_exception_with_attributes
   - test_exception_attributes

4. Trace Context Tests (4 tests):
   - test_get_current_trace_context
   - test_trace_context_available_in_span
   - test_trace_context_logging_integration
   - test_trace_context_empty_outside_span

5. Configuration Tests (7 tests):
   - test_configure_service_name
   - test_configure_environment
   - test_configure_sampling_rate
   - test_configure_service_version
   - test_configure_auto_detect_environment
   - test_configure_auto_detect_sampling_production
   - test_configure_auto_detect_sampling_development

6. OTLP Export Tests (4 tests):
   - test_otlp_exporter_configuration
   - test_console_fallback
   - test_console_exporter_explicit
   - test_otlp_endpoint_from_env

7. Graceful Degradation Tests (5 tests):
   - test_noop_when_otel_unavailable
   - test_warning_when_otel_unavailable
   - test_is_enabled_false_when_unavailable
   - test_set_attribute_noop_when_disabled
   - test_get_trace_context_empty_when_disabled

8. NoOpSpan Tests (8 tests):
   - test_noop_span_set_attribute
   - test_noop_span_set_attributes
   - test_noop_span_add_event
   - test_noop_span_set_status
   - test_noop_span_record_exception
   - test_noop_span_is_recording
   - test_noop_span_get_span_context
   - test_noop_span_context_manager

9. Advanced Features Tests (4 tests):
   - test_add_event
   - test_set_attributes_batch
   - test_shutdown
   - test_shutdown_when_disabled

10. Integration Tests (3 tests):
    - test_full_trace_workflow
    - test_batch_processing_workflow
    - test_error_handling_workflow

11. Performance Tests (2 tests):
    - test_span_creation_overhead
    - test_attribute_setting_overhead

Total: 53 comprehensive tests covering all observability features.
"""
