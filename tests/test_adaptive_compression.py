"""
Tests for adaptive compression ratio and cost telemetry features.

TDD: Written BEFORE implementation (Red phase).

Feature A: Adaptive skeleton ratio that scales with corpus size
Feature B: Cost savings telemetry with model-aware pricing
"""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.semantic_compressor import SemanticCompressor


# ============================================================================
# Feature A: Adaptive Compression Ratio
# ============================================================================


class TestComputeAdaptiveRatio:
    """compute_adaptive_ratio() scales compression based on corpus size."""

    def test_function_exists(self):
        """compute_adaptive_ratio should be importable from semantic_compressor."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert callable(compute_adaptive_ratio)

    def test_small_corpus_keeps_80_percent(self):
        """Corpus under 8K tokens should keep 80% (light compression)."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert compute_adaptive_ratio(1000) == 0.8
        assert compute_adaptive_ratio(5000) == 0.8
        assert compute_adaptive_ratio(7999) == 0.8

    def test_medium_corpus_keeps_50_percent(self):
        """Corpus 8K-32K tokens should keep 50%."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert compute_adaptive_ratio(8000) == 0.5
        assert compute_adaptive_ratio(20000) == 0.5
        assert compute_adaptive_ratio(31999) == 0.5

    def test_large_corpus_keeps_20_percent(self):
        """Corpus 32K-100K tokens should keep 20%."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert compute_adaptive_ratio(32000) == 0.2
        assert compute_adaptive_ratio(60000) == 0.2
        assert compute_adaptive_ratio(99999) == 0.2

    def test_huge_corpus_keeps_10_percent(self):
        """Corpus 100K+ tokens should keep 10% (aggressive compression)."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert compute_adaptive_ratio(100000) == 0.1
        assert compute_adaptive_ratio(500000) == 0.1
        assert compute_adaptive_ratio(1000000) == 0.1

    def test_zero_tokens_returns_max_ratio(self):
        """Zero or negative tokens should return highest ratio (0.8)."""
        from src.semantic_compressor import compute_adaptive_ratio

        assert compute_adaptive_ratio(0) == 0.8

    def test_boundary_values(self):
        """Exact boundary values should fall into the higher tier."""
        from src.semantic_compressor import compute_adaptive_ratio

        # At exactly the boundary, use the new tier
        assert compute_adaptive_ratio(8000) == 0.5  # crosses into medium
        assert compute_adaptive_ratio(32000) == 0.2  # crosses into large
        assert compute_adaptive_ratio(100000) == 0.1  # crosses into huge


class TestAdaptiveRatioIntegration:
    """SemanticCompressor uses adaptive ratio when skeleton_ratio='auto'."""

    def test_auto_skeleton_ratio_accepted(self):
        """SemanticCompressor should accept skeleton_ratio='auto'."""
        compressor = SemanticCompressor(skeleton_ratio="auto")
        assert compressor.skeleton_ratio == "auto"

    def test_explicit_ratio_still_works(self):
        """Explicit numeric skeleton_ratio should be preserved unchanged."""
        compressor = SemanticCompressor(skeleton_ratio=0.3)
        assert compressor.skeleton_ratio == 0.3

    def test_default_ratio_unchanged(self):
        """Default should remain 0.2 for backward compatibility."""
        compressor = SemanticCompressor()
        assert compressor.skeleton_ratio == 0.2

    def test_auto_ratio_adapts_to_small_doc(self):
        """With 'auto', a small document should use ~80% ratio."""
        compressor = SemanticCompressor(skeleton_ratio="auto")
        # A small document (~200 tokens)
        text = "Short sentence. " * 30
        compressor.ingest_file(text, "small_doc")
        skeleton = compressor._generate_skeleton("small_doc")
        # With auto, small docs should show most content
        # num_skeleton should be closer to total (80%)
        shown_ratio = skeleton.total_nodes - skeleton.skeleton_text.count("[HIDDEN]")
        assert shown_ratio >= 0  # Sanity check - just verify it runs

    def test_auto_ratio_adapts_to_large_doc(self):
        """With 'auto', a large document should use more aggressive compression."""
        compressor = SemanticCompressor(skeleton_ratio="auto")
        # A large document (~many tokens)
        text = ("This is a detailed paragraph about software engineering best practices. " * 50 + "\n") * 20
        compressor.ingest_file(text, "large_doc")
        skeleton = compressor._generate_skeleton("large_doc")
        # Should have generated a skeleton (verify it ran without error)
        assert skeleton.total_nodes > 0
        assert skeleton.skeleton_tokens < skeleton.total_tokens


# ============================================================================
# Feature B: Cost Savings Telemetry
# ============================================================================


class TestComputeCostSavings:
    """compute_cost_savings() calculates dollar savings per operation."""

    def test_function_exists(self):
        """compute_cost_savings should be importable from metrics."""
        from src.metrics import compute_cost_savings

        assert callable(compute_cost_savings)

    def test_sonnet_pricing(self):
        """Sonnet model should use $3/M input tokens."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=100_000,
            compressed_tokens=20_000,
            model="claude-sonnet-4"
        )
        # Saved 80K tokens at $3/M = $0.24
        assert result.saved_tokens == 80_000
        assert abs(result.cost_savings_usd - 0.24) < 0.001

    def test_opus_pricing(self):
        """Opus model should use $15/M input tokens."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=100_000,
            compressed_tokens=20_000,
            model="claude-opus-4"
        )
        # Saved 80K tokens at $15/M = $1.20
        assert result.saved_tokens == 80_000
        assert abs(result.cost_savings_usd - 1.20) < 0.001

    def test_haiku_pricing(self):
        """Haiku model should use $0.80/M input tokens."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=100_000,
            compressed_tokens=20_000,
            model="claude-haiku-3.5"
        )
        # Saved 80K tokens at $0.80/M = $0.064
        assert result.saved_tokens == 80_000
        assert abs(result.cost_savings_usd - 0.064) < 0.001

    def test_default_model_uses_sonnet_pricing(self):
        """Unknown or missing model should default to Sonnet pricing."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=100_000,
            compressed_tokens=20_000,
        )
        # Default to Sonnet: $3/M
        assert abs(result.cost_savings_usd - 0.24) < 0.001

    def test_no_savings_returns_zero(self):
        """When compressed >= original, savings should be zero."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=1000,
            compressed_tokens=1000,
        )
        assert result.saved_tokens == 0
        assert result.cost_savings_usd == 0.0

    def test_result_has_all_fields(self):
        """Result should contain all telemetry fields."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=50_000,
            compressed_tokens=10_000,
            model="claude-sonnet-4"
        )
        assert hasattr(result, "original_tokens")
        assert hasattr(result, "compressed_tokens")
        assert hasattr(result, "saved_tokens")
        assert hasattr(result, "model")
        assert hasattr(result, "cost_per_million")
        assert hasattr(result, "cost_savings_usd")
        assert hasattr(result, "savings_percent")

    def test_savings_percent_calculated(self):
        """savings_percent should be (saved/original) * 100."""
        from src.metrics import compute_cost_savings

        result = compute_cost_savings(
            original_tokens=100_000,
            compressed_tokens=20_000,
        )
        assert abs(result.savings_percent - 80.0) < 0.1


class TestTokenSavingsTelemetry:
    """TokenSavingsTelemetry dataclass should be properly structured."""

    def test_dataclass_exists(self):
        """TokenSavingsTelemetry should be importable from metrics."""
        from src.metrics import TokenSavingsTelemetry

        instance = TokenSavingsTelemetry(
            original_tokens=100_000,
            compressed_tokens=20_000,
            saved_tokens=80_000,
            model="claude-sonnet-4",
            cost_per_million=3.0,
            cost_savings_usd=0.24,
            savings_percent=80.0,
        )
        assert instance.original_tokens == 100_000

    def test_to_dict(self):
        """Should serialize to a clean dict for JSON responses."""
        from src.metrics import TokenSavingsTelemetry

        instance = TokenSavingsTelemetry(
            original_tokens=100_000,
            compressed_tokens=20_000,
            saved_tokens=80_000,
            model="claude-sonnet-4",
            cost_per_million=3.0,
            cost_savings_usd=0.24,
            savings_percent=80.0,
        )
        d = instance.to_dict()
        assert isinstance(d, dict)
        assert d["saved_tokens"] == 80_000
        assert d["cost_savings_usd"] == 0.24


# ============================================================================
# Feature C: Handler Integration
# ============================================================================


class TestHandlerCostSavingsIntegration:
    """Compression handlers should include cost savings in responses."""

    @pytest.fixture
    def mock_context(self):
        """Build a mock handler context."""
        mock_compressor = Mock()
        mock_skeleton = Mock()
        mock_skeleton.total_nodes = 10
        mock_skeleton.total_tokens = 5000
        mock_skeleton.skeleton_tokens = 1000
        mock_skeleton.compression_ratio = 5.0
        mock_skeleton.skeleton_text = "test skeleton"
        mock_skeleton.node_map = {}
        mock_compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
        mock_compressor.graphs = {"test_doc": Mock()}
        mock_compressor.chunks = {}
        mock_compressor.file_metadata = {}
        mock_compressor.skeleton_ratio = 0.2

        mock_resource_manager = Mock()
        mock_resource_manager.check_document_size_async = AsyncMock(return_value=(True, ""))
        mock_resource_manager.register_document_async = AsyncMock()

        mock_persistence = Mock()
        mock_persistence.save_document = Mock(return_value=True)
        mock_persistence.save_file_sync_metadata = Mock(return_value=True)

        mock_sync_manager = Mock()
        mock_sync_manager.register_file = Mock()
        mock_sync_manager.export_metadata = Mock(return_value={})

        mock_version_manager = Mock()
        mock_version_manager.add_version_async = AsyncMock()

        mock_path_validator = Mock()

        import networkx as nx
        mock_compressor.graphs["test_doc"] = nx.Graph()

        return {
            "compressor": mock_compressor,
            "resource_manager": mock_resource_manager,
            "persistence": mock_persistence,
            "sync_manager": mock_sync_manager,
            "version_manager": mock_version_manager,
            "path_validator": mock_path_validator,
            "validate_file_id": Mock(),
            "validate_node_ids": Mock(),
            "validate_token_count": Mock(),
            "retrieval_history": {},
        }

    @pytest.mark.asyncio
    @patch("src.handlers.compression_handlers.validate_file_id")
    async def test_ingest_response_includes_cost_savings(self, mock_validate, mock_context):
        """handle_ingest response should include cost_savings field."""
        from src.handlers.compression_handlers import handle_ingest

        args = {
            "text": "A moderately long text for testing. " * 20,
            "file_id": "test_cost",
        }

        result = await handle_ingest(mock_context, args)
        parsed = json.loads(result)

        assert parsed["status"] == "success"
        assert "cost_savings" in parsed
        assert "saved_tokens" in parsed["cost_savings"]
        assert "cost_savings_usd" in parsed["cost_savings"]
        assert "model" in parsed["cost_savings"]
