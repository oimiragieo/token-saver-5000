"""benchmark guard — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

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


class TestBenchmarkGuard:
    def test_missing_thresholds(self):
        from src.benchmark_guard import evaluate_report_against_thresholds

        violations = evaluate_report_against_thresholds(mode="unknown", report={}, thresholds={})
        assert len(violations) == 1
        assert "Missing thresholds" in violations[0].message

    def test_load_json(self, tmp_path):
        from src.benchmark_guard import load_json

        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        data = load_json(f)
        assert data["key"] == "value"
