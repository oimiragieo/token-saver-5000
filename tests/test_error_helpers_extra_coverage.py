"""error helpers — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

Moved verbatim (byte-for-byte class bodies); nothing reworded.
"""

from unittest.mock import AsyncMock, MagicMock


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


class TestErrorTypes:
    def test_operation_timeout(self):
        from src.error_types import OperationTimeoutError

        e = OperationTimeoutError("embed", timeout=30.0)
        assert "30" in str(e)
        assert e.operation == "embed"

    def test_circuit_breaker_open(self):
        from src.error_types import CircuitBreakerOpenError

        e = CircuitBreakerOpenError("persistence", failure_count=5)
        assert "5" in str(e)

    def test_circuit_breaker_no_count(self):
        from src.error_types import CircuitBreakerOpenError

        e = CircuitBreakerOpenError("persistence")
        assert "OPEN" in str(e)

    def test_retry_exhausted_with_exception(self):
        from src.error_types import RetryExhaustedError

        inner = ValueError("inner")
        e = RetryExhaustedError("op", max_retries=3, last_exception=inner)
        assert "3" in str(e)
        assert "inner" in str(e)

    def test_retry_exhausted_no_exception(self):
        from src.error_types import RetryExhaustedError

        e = RetryExhaustedError("op", max_retries=3)
        assert "3" in str(e)

    def test_rate_limit_exceeded_with_wait(self):
        from src.error_types import RateLimitExceededError

        e = RateLimitExceededError("ingest", rate=10.0, wait_time=5.0)
        assert "5.0" in str(e)

    def test_graceful_degradation(self):
        from src.error_types import GracefulDegradationError

        e = GracefulDegradationError("embed", "tfidf", reason="OOM")
        assert "OOM" in str(e)

    def test_graceful_degradation_no_reason(self):
        from src.error_types import GracefulDegradationError

        e = GracefulDegradationError("embed", "tfidf")
        assert "degraded" in str(e)


class TestErrorHelpers:
    def test_file_id_not_found_with_matches(self):
        from src.error_helpers import SmartError

        err = SmartError.file_id_not_found("quantum_papper", ["quantum_paper", "neural_nets"])
        assert "quantum_paper" in str(err)

    def test_file_id_not_found_many_available(self):
        from src.error_helpers import SmartError

        ids = [f"doc{i}" for i in range(10)]
        err = SmartError.file_id_not_found("unknown", ids)
        assert "10 total" in str(err)

    def test_node_id_not_found(self):
        from src.error_helpers import SmartError

        err = SmartError.node_id_not_found("doc_n99", ["doc_n0", "doc_n1"], "doc")
        assert "doc_n" in str(err)

    def test_invalid_enum_value(self):
        from src.error_helpers import SmartError

        err = SmartError.invalid_enum_value("BALENCED", ["BALANCED", "HIGH", "LOW"], "fidelity")
        assert "BALANCED" in str(err)
