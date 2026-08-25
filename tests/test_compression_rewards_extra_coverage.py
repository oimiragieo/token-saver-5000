"""compression rewards — consolidated from coverage_boost1-4/4b (backlog N10 re-filing).

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


class TestCompressionRewards:
    def test_schema_validation_scores(self):
        from src.compression_rewards import SchemaValidationResult

        r = SchemaValidationResult(input_valid=True, output_valid=False)
        assert r.score == 0.5
        assert not r.all_valid

    def test_fidelity_adherence_ratio_score_zero_target(self):
        from src.compression_rewards import FidelityAdherenceResult

        r = FidelityAdherenceResult(target_ratio=0)
        assert r.ratio_score == 0.0

    def test_composition_integrity_score(self):
        from src.compression_rewards import CompositionIntegrityResult

        r = CompositionIntegrityResult(graph_connected=False, orphan_nodes=5)
        assert r.score < 1.0

    def test_memory_discipline_zero_budget(self):
        from src.compression_rewards import MemoryDisciplineResult

        r = MemoryDisciplineResult(memory_budget_mb=0)
        assert r.memory_score == 0.0
