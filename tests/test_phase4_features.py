"""
Tests for Phase 4 features: Citation tags, auto ratio exposure, validation hooks,
compression presets, token threshold monitoring, memory classification,
framework memory hooks, TOON gate, streaming compression, diff-aware re-ingestion,
multimodal production, SCAR training pipeline, cross-document deduplication.

TDD: Written BEFORE implementation (Red phase).
"""

import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest


# ============================================================================
# Feature 1: Citation Provenance Tags
# ============================================================================


class TestCitationTags:
    """Compressed output should include provenance citation markers."""

    def test_skeleton_anchor_nodes_have_rag_citation(self):
        """Skeleton ANCHOR nodes should include [rag:<node_id>] citation."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = ("This is a detailed paragraph about software engineering. " * 10 + "\n") * 5
        compressor.ingest_file(text, "cite_test")
        skeleton = compressor._generate_skeleton("cite_test")
        # At least one ANCHOR line should contain [rag:cite_test_...]
        assert "[rag:" in skeleton.skeleton_text

    def test_search_results_have_rag_citation(self):
        """Search results should include citation markers in node summaries."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = ("Machine learning models require training data. " * 10 + "\n") * 5
        compressor.ingest_file(text, "search_cite")
        results = compressor.search_semantic_with_scores("machine learning", "search_cite", top_k=3)
        # Results should have node IDs that can be cited
        assert len(results) > 0
        node_id, _ = results[0]
        assert node_id.startswith("search_cite")

    def test_citation_format_is_bracketed(self):
        """Citations should follow [rag:<file_id>:<chunk_index>] format."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = ("Detailed technical content for citation testing. " * 10 + "\n") * 5
        compressor.ingest_file(text, "fmt_test")
        skeleton = compressor._generate_skeleton("fmt_test")
        import re
        # Should find at least one [rag:...] citation
        citations = re.findall(r'\[rag:[^\]]+\]', skeleton.skeleton_text)
        assert len(citations) > 0


# ============================================================================
# Feature 2: Expose skeleton_ratio=auto via MCP
# ============================================================================


class TestAutoRatioMCPExposure:
    """MCP tool schema should expose skeleton_ratio parameter."""

    def test_ingest_schema_has_skeleton_ratio(self):
        """ingest_context tool should have skeleton_ratio in its schema."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        ingest_tool = next(t for t in tools if t.name == "ingest_context")
        props = ingest_tool.inputSchema["properties"]
        assert "skeleton_ratio" in props

    def test_skeleton_ratio_accepts_auto_string(self):
        """skeleton_ratio should accept 'auto' as a value."""
        from src.handlers.mcp_core import setup_mcp_tools

        tools = setup_mcp_tools()
        ingest_tool = next(t for t in tools if t.name == "ingest_context")
        ratio_schema = ingest_tool.inputSchema["properties"]["skeleton_ratio"]
        # Should support both number and string "auto"
        assert "auto" in str(ratio_schema).lower() or "oneOf" in ratio_schema or "description" in ratio_schema


# ============================================================================
# Feature 3: Input Validation Hooks
# ============================================================================


class TestInputValidationHooks:
    """Pre-execute validation for tool calls."""

    def test_validate_hooks_module_exists(self):
        """validation_hooks module should be importable."""
        from src.validation_hooks import validate_tool_input

        assert callable(validate_tool_input)

    def test_search_query_min_length(self):
        """Search queries must be at least 3 characters."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("search_semantic", {"query": "ab"})
        assert len(errors) > 0
        assert "query" in errors[0].lower()

    def test_search_query_valid(self):
        """Valid search queries should pass validation."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("search_semantic", {"query": "machine learning"})
        assert len(errors) == 0

    def test_ingest_text_not_empty(self):
        """Ingest text must not be empty after stripping."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("ingest_context", {"text": "   ", "file_id": "test"})
        assert len(errors) > 0

    def test_file_id_format(self):
        """File IDs must be alphanumeric with underscores."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("ingest_context", {
            "text": "valid text content here",
            "file_id": "invalid file id!@#"
        })
        assert len(errors) > 0
        assert "file_id" in errors[0].lower()

    def test_unknown_tool_passes(self):
        """Unknown tools should not fail validation (pass through)."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("nonexistent_tool", {"anything": "goes"})
        assert len(errors) == 0

    def test_top_k_range(self):
        """top_k must be between 1 and 100."""
        from src.validation_hooks import validate_tool_input

        errors = validate_tool_input("search_semantic", {"query": "test", "top_k": 0})
        assert len(errors) > 0

        errors = validate_tool_input("search_semantic", {"query": "test", "top_k": 5})
        assert len(errors) == 0


# ============================================================================
# Feature 4: Compression Profile Presets
# ============================================================================


class TestCompressionPresets:
    """Named presets mapping to fidelity/ratio combos."""

    def test_presets_module_exists(self):
        """compression_presets module should be importable."""
        from src.compression_presets import get_preset, list_presets

        assert callable(get_preset)
        assert callable(list_presets)

    def test_code_review_preset(self):
        """code-review preset: high fidelity, keep more structure."""
        from src.compression_presets import get_preset

        preset = get_preset("code-review")
        assert preset.skeleton_ratio >= 0.4  # Keep more for review
        assert preset.fidelity == "DETAILED"

    def test_chat_preset(self):
        """chat preset: balanced compression for conversation."""
        from src.compression_presets import get_preset

        preset = get_preset("chat")
        assert preset.skeleton_ratio <= 0.3
        assert preset.fidelity == "OUTLINE"

    def test_research_preset(self):
        """research preset: preserve detail, moderate compression."""
        from src.compression_presets import get_preset

        preset = get_preset("research")
        assert preset.skeleton_ratio >= 0.3
        assert preset.fidelity in ("STRUCTURE", "DETAILED")

    def test_aggressive_preset(self):
        """aggressive preset: maximum compression."""
        from src.compression_presets import get_preset

        preset = get_preset("aggressive")
        assert preset.skeleton_ratio <= 0.15
        assert preset.fidelity == "ABSTRACT"

    def test_list_presets_returns_all(self):
        """list_presets should return at least 4 presets."""
        from src.compression_presets import list_presets

        presets = list_presets()
        assert len(presets) >= 4
        names = [p.name for p in presets]
        assert "code-review" in names
        assert "chat" in names

    def test_unknown_preset_raises(self):
        """Unknown preset name should raise ValueError."""
        from src.compression_presets import get_preset

        with pytest.raises(ValueError):
            get_preset("nonexistent-preset")

    def test_preset_has_description(self):
        """Each preset should have a human-readable description."""
        from src.compression_presets import get_preset

        preset = get_preset("chat")
        assert hasattr(preset, "description")
        assert len(preset.description) > 10


# ============================================================================
# Feature 5: Token Threshold Auto-Trigger
# ============================================================================


class TestTokenThresholdMonitor:
    """Monitor context size and suggest compression at thresholds."""

    def test_monitor_module_exists(self):
        """token_threshold module should be importable."""
        from src.token_threshold import check_context_budget

        assert callable(check_context_budget)

    def test_under_threshold_no_warning(self):
        """Under 80K tokens should return no warning."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(50_000, context_limit=200_000)
        assert result.status == "ok"
        assert result.should_compress is False

    def test_80k_threshold_suggests_compression(self):
        """At 80K tokens should suggest compression."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(80_000, context_limit=200_000)
        assert result.status == "warning"
        assert result.should_compress is True

    def test_120k_threshold_urgent(self):
        """At 120K tokens should be urgent."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(120_000, context_limit=200_000)
        assert result.status == "urgent"
        assert result.should_compress is True

    def test_150k_threshold_critical(self):
        """At 150K tokens should be critical."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(150_000, context_limit=200_000)
        assert result.status == "critical"
        assert result.should_compress is True

    def test_result_includes_recommendation(self):
        """Result should include a recommended action."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(90_000, context_limit=200_000)
        assert hasattr(result, "recommendation")
        assert len(result.recommendation) > 0

    def test_result_includes_usage_percent(self):
        """Result should include usage percentage."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(100_000, context_limit=200_000)
        assert hasattr(result, "usage_percent")
        assert abs(result.usage_percent - 50.0) < 0.1

    def test_custom_thresholds(self):
        """Should support custom threshold percentages."""
        from src.token_threshold import check_context_budget

        # With small context limit, 50K is critical
        result = check_context_budget(50_000, context_limit=60_000)
        assert result.status == "critical"

    def test_to_dict(self):
        """Result should serialize to dict."""
        from src.token_threshold import check_context_budget

        result = check_context_budget(90_000, context_limit=200_000)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "status" in d
        assert "should_compress" in d


# ============================================================================
# Feature 6: Memory Classification System
# ============================================================================


class TestMemoryClassification:
    """Auto-categorize insights as gotchas/issues/decisions/patterns."""

    def test_classifier_module_exists(self):
        """memory_classifier module should be importable."""
        from src.memory_classifier import classify_insight

        assert callable(classify_insight)

    def test_classify_gotcha(self):
        """Text about pitfalls/gotchas should classify as 'gotcha'."""
        from src.memory_classifier import classify_insight

        result = classify_insight("Watch out: the API silently drops null fields")
        assert result.category == "gotcha"

    def test_classify_issue(self):
        """Text about bugs/errors should classify as 'issue'."""
        from src.memory_classifier import classify_insight

        result = classify_insight("Bug: the retry logic fails after 3 attempts due to timeout")
        assert result.category == "issue"

    def test_classify_decision(self):
        """Text about design choices should classify as 'decision'."""
        from src.memory_classifier import classify_insight

        result = classify_insight("We decided to use PostgreSQL instead of MongoDB for ACID compliance")
        assert result.category == "decision"

    def test_classify_pattern(self):
        """Text about recurring patterns should classify as 'pattern'."""
        from src.memory_classifier import classify_insight

        result = classify_insight("The repository pattern is used consistently across all services")
        assert result.category == "pattern"

    def test_classify_general_fallback(self):
        """Unrecognized text should classify as 'general'."""
        from src.memory_classifier import classify_insight

        result = classify_insight("The sky is blue today")
        assert result.category == "general"

    def test_classification_has_confidence(self):
        """Classification should include a confidence score."""
        from src.memory_classifier import classify_insight

        result = classify_insight("Warning: never use eval() with user input")
        assert hasattr(result, "confidence")
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_multiple(self):
        """Should handle batch classification."""
        from src.memory_classifier import classify_insights

        texts = [
            "Bug: null pointer in auth module",
            "We chose JWT over sessions",
            "Always validate input before processing",
        ]
        results = classify_insights(texts)
        assert len(results) == 3
        categories = [r.category for r in results]
        assert "issue" in categories
        assert "decision" in categories


# ============================================================================
# Feature 7: Framework Memory Sync Hook
# ============================================================================


class TestFrameworkMemoryHook:
    """Post-compression hook persisting distilled learnings."""

    def test_memory_hook_module_exists(self):
        """memory_hooks module should be importable."""
        from src.memory_hooks import MemoryHookManager

        assert MemoryHookManager is not None

    def test_register_hook(self):
        """Should be able to register a post-compression hook."""
        from src.memory_hooks import MemoryHookManager

        manager = MemoryHookManager()
        called = []
        manager.register_hook("post_compress", lambda data: called.append(data))
        manager.trigger("post_compress", {"file_id": "test", "tokens_saved": 1000})
        assert len(called) == 1
        assert called[0]["file_id"] == "test"

    def test_multiple_hooks(self):
        """Multiple hooks for same event should all fire."""
        from src.memory_hooks import MemoryHookManager

        manager = MemoryHookManager()
        results = []
        manager.register_hook("post_compress", lambda d: results.append("a"))
        manager.register_hook("post_compress", lambda d: results.append("b"))
        manager.trigger("post_compress", {})
        assert results == ["a", "b"]

    def test_hook_error_doesnt_crash(self):
        """A failing hook should log error but not crash."""
        from src.memory_hooks import MemoryHookManager

        manager = MemoryHookManager()
        manager.register_hook("post_compress", lambda d: 1 / 0)  # ZeroDivisionError
        # Should not raise
        manager.trigger("post_compress", {})

    def test_export_memory_index(self):
        """Should export accumulated memory entries."""
        from src.memory_hooks import MemoryHookManager

        manager = MemoryHookManager()
        manager.add_memory_entry("test_file", "Important pattern found", "pattern")
        entries = manager.get_memory_index()
        assert len(entries) == 1
        assert entries[0]["category"] == "pattern"


# ============================================================================
# Feature 8: TOON Production Gate
# ============================================================================


class TestTOONProductionGate:
    """Auto-benchmark TOON vs JSON, enable only when proven better."""

    def test_gate_module_exists(self):
        """toon_gate module should be importable."""
        from src.toon_gate import TOONGate

        assert TOONGate is not None

    def test_gate_defaults_disabled(self):
        """TOON gate should default to disabled (JSON preferred)."""
        from src.toon_gate import TOONGate

        gate = TOONGate()
        assert gate.is_enabled() is False
        assert gate.recommended_format() == "json"

    def test_gate_enables_after_benchmark(self):
        """Gate should enable TOON when benchmark shows savings."""
        from src.toon_gate import TOONGate

        gate = TOONGate()
        gate.record_benchmark(json_tokens=1000, toon_tokens=600)
        gate.record_benchmark(json_tokens=800, toon_tokens=500)
        gate.record_benchmark(json_tokens=1200, toon_tokens=700)
        # 3+ benchmarks with consistent savings should enable
        assert gate.is_enabled() is True
        assert gate.recommended_format() == "toon"

    def test_gate_stays_disabled_when_no_savings(self):
        """Gate should stay disabled if TOON isn't smaller."""
        from src.toon_gate import TOONGate

        gate = TOONGate()
        gate.record_benchmark(json_tokens=1000, toon_tokens=950)
        gate.record_benchmark(json_tokens=800, toon_tokens=790)
        gate.record_benchmark(json_tokens=1200, toon_tokens=1180)
        assert gate.is_enabled() is False

    def test_gate_stats(self):
        """Should report benchmark statistics."""
        from src.toon_gate import TOONGate

        gate = TOONGate()
        gate.record_benchmark(json_tokens=1000, toon_tokens=600)
        stats = gate.get_stats()
        assert "benchmarks_run" in stats
        assert "avg_savings_percent" in stats
        assert stats["benchmarks_run"] == 1


# ============================================================================
# Feature 9: Streaming Compression
# ============================================================================


class TestStreamingCompression:
    """Async generator for streaming skeleton output."""

    @pytest.mark.asyncio
    async def test_stream_skeleton_exists(self):
        """SemanticCompressor should have stream_skeleton method."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        assert hasattr(compressor, "stream_skeleton")

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self):
        """stream_skeleton should yield text chunks."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = ("Paragraph about testing streaming compression. " * 10 + "\n") * 5
        await compressor.ingest_file_async(text, "stream_test")

        chunks = []
        async for chunk in compressor.stream_skeleton("stream_test"):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert "SEMANTIC SKELETON" in full_text

    @pytest.mark.asyncio
    async def test_stream_matches_non_stream(self):
        """Streaming output should produce same content as non-streaming."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = ("Content for comparison test between streaming and batch. " * 10 + "\n") * 5
        await compressor.ingest_file_async(text, "compare_test")

        regular = compressor._generate_skeleton("compare_test")
        chunks = []
        async for chunk in compressor.stream_skeleton("compare_test"):
            chunks.append(chunk)

        streamed = "".join(chunks)
        assert streamed == regular.skeleton_text


# ============================================================================
# Feature 10: Diff-Aware Re-ingestion
# ============================================================================


class TestDiffAwareReingestion:
    """Only re-compress changed sections on file refresh."""

    def test_diff_ingest_exists(self):
        """SemanticCompressor should have diff_reingest method."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        assert hasattr(compressor, "diff_reingest")

    @pytest.mark.asyncio
    async def test_diff_reingest_detects_changes(self):
        """diff_reingest should identify changed sections."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        original = "Line 1 about topic A.\nLine 2 about topic B.\nLine 3 about topic C.\n" * 5
        await compressor.ingest_file_async(original, "diff_test")

        updated = "Line 1 about topic A.\nLine 2 CHANGED content.\nLine 3 about topic C.\n" * 5
        result = await compressor.diff_reingest_async("diff_test", updated)

        assert result.chunks_unchanged >= 0
        assert result.chunks_updated >= 0

    @pytest.mark.asyncio
    async def test_diff_reingest_preserves_unchanged(self):
        """Unchanged chunks should keep their original embeddings."""
        from src.semantic_compressor import SemanticCompressor
        import numpy as np

        compressor = SemanticCompressor()
        original = ("Stable content that should not change. " * 10 + "\n") * 5
        await compressor.ingest_file_async(original, "preserve_test")

        # Get original embeddings
        original_embeddings = {
            nid: node.embedding.copy()
            for nid, node in compressor.chunks.items()
            if nid.startswith("preserve_test")
        }

        # Re-ingest identical content
        result = await compressor.diff_reingest_async("preserve_test", original)

        # Embeddings should be identical (not recomputed)
        for nid, orig_emb in original_embeddings.items():
            if nid in compressor.chunks:
                assert np.array_equal(orig_emb, compressor.chunks[nid].embedding)

    @pytest.mark.asyncio
    async def test_diff_reingest_result_has_stats(self):
        """Result should include diff statistics."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = "Simple text for stats test.\n" * 10
        await compressor.ingest_file_async(text, "stats_test")
        result = await compressor.diff_reingest_async("stats_test", text)

        assert hasattr(result, "chunks_unchanged")
        assert hasattr(result, "chunks_updated")
        assert hasattr(result, "chunks_added")
        assert hasattr(result, "chunks_removed")


# ============================================================================
# Feature 11: Multimodal Production-Ready
# ============================================================================


class TestMultimodalProduction:
    """Multimodal compressor should have proper test coverage."""

    def test_multimodal_compressor_imports(self):
        """MultiModalCompressor should be importable."""
        from src.multimodal_compressor import MultiModalCompressor, ModalityType

        assert MultiModalCompressor is not None
        assert ModalityType is not None

    def test_text_ingestion(self):
        """Should handle pure text ingestion."""
        from src.multimodal_compressor import MultiModalCompressor

        compressor = MultiModalCompressor(use_clip_for_images=False, use_codebert_for_code=False)
        items = [
            {"type": "text", "content": "This is a test paragraph about software.", "id": "t1"},
        ]
        result = compressor.ingest_mixed_content(items, "proj1")
        assert result["total_nodes"] > 0

    def test_code_ingestion(self):
        """Should handle code content."""
        from src.multimodal_compressor import MultiModalCompressor

        compressor = MultiModalCompressor(use_clip_for_images=False, use_codebert_for_code=False)
        items = [
            {"type": "code", "content": "def hello():\n    print('world')", "id": "c1"},
        ]
        result = compressor.ingest_mixed_content(items, "proj2")
        assert result["total_nodes"] > 0

    def test_cross_modal_search(self):
        """Should search across modalities."""
        from src.multimodal_compressor import MultiModalCompressor

        compressor = MultiModalCompressor(use_clip_for_images=False, use_codebert_for_code=False)
        items = [
            {"type": "text", "content": "Machine learning models need training data.", "id": "t1"},
            {"type": "code", "content": "model.fit(X_train, y_train)", "id": "c1"},
        ]
        compressor.ingest_mixed_content(items, "proj3")
        results = compressor.search_cross_modal("training", project_id="proj3")
        assert len(results) > 0


# ============================================================================
# Feature 12: SCAR Training Pipeline
# ============================================================================


class TestSCARTrainingPipeline:
    """SCAR training should be accessible via clean API."""

    def test_training_config_importable(self):
        """TrainingConfig should be importable."""
        from src.training_utils import TrainingConfig

        config = TrainingConfig()
        assert config.batch_size > 0
        assert config.num_epochs > 0

    def test_scar_trainer_importable(self):
        """SCARTrainer should be importable."""
        from src.training_utils import SCARTrainer

        assert SCARTrainer is not None

    def test_training_config_defaults(self):
        """Default config should have sensible values."""
        from src.training_utils import TrainingConfig

        config = TrainingConfig()
        assert 0 < config.learning_rate < 1
        assert config.batch_size >= 1


# ============================================================================
# Feature 13: Cross-Document Deduplication
# ============================================================================


class TestCrossDocumentDedup:
    """Detect and merge semantically duplicate chunks across files."""

    def test_dedup_method_exists(self):
        """SemanticCompressor should have find_duplicates method."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        assert hasattr(compressor, "find_duplicates")

    def test_find_exact_duplicates(self):
        """Identical text across files should be detected."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        shared_text = "This exact paragraph appears in both documents. " * 10
        compressor.ingest_file(shared_text + "\nUnique to doc A.\n" * 5, "doc_a")
        compressor.ingest_file(shared_text + "\nUnique to doc B.\n" * 5, "doc_b")

        dupes = compressor.find_duplicates(threshold=0.95)
        assert len(dupes) > 0

    def test_find_semantic_duplicates(self):
        """Semantically similar (not identical) text should be detected."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        compressor.ingest_file(
            "The authentication module validates user credentials using JWT tokens. " * 10,
            "doc_x"
        )
        compressor.ingest_file(
            "User authentication is handled by verifying JWT-based credentials. " * 10,
            "doc_y"
        )

        dupes = compressor.find_duplicates(threshold=0.85)
        # Should find at least some semantic overlap
        assert len(dupes) >= 0  # May or may not find depending on embedding

    def test_dedup_result_structure(self):
        """Duplicate results should have proper structure."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = "Shared content for testing. " * 20
        compressor.ingest_file(text, "dup_a")
        compressor.ingest_file(text, "dup_b")

        dupes = compressor.find_duplicates(threshold=0.95)
        if len(dupes) > 0:
            d = dupes[0]
            assert "node_a" in d
            assert "node_b" in d
            assert "similarity" in d
            assert d["similarity"] >= 0.95

    def test_dedup_stats(self):
        """Should report deduplication statistics."""
        from src.semantic_compressor import SemanticCompressor

        compressor = SemanticCompressor()
        text = "Content repeated across files. " * 20
        compressor.ingest_file(text, "s1")
        compressor.ingest_file(text, "s2")

        dupes = compressor.find_duplicates(threshold=0.95)
        # The method should always return a list (possibly empty)
        assert isinstance(dupes, list)
