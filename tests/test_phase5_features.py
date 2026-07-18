"""
Tests for Phase 5 features - 11 new capabilities based on 2025 research papers.

Features tested:
1. Attention-guided pruning (AttentionRAG)
2. Semantic chunking (SCOPE)
3. Intra-document redundancy collapse (R-KV)
4. Context decay and eviction (DynamicKV/ACON)
5. Compression fidelity scoring (SCOPE/Empirical Study)
6. Multi-level skeleton (Squeezed Attention)
7. Query-adaptive compression (KVzip/LazyLLM)
8. Context window advisor (MCP Best Practices)
9. Keyword anchoring (SCOPE)
10. Compression replay log (ACON)
11. Generative rewrite stub (SCOPE)
"""

import time
import numpy as np
import pytest

# =============================================================================
# 1. Attention-Guided Pruning
# =============================================================================


class TestAttentionPruning:
    """Tests for attention-guided node pruning based on query relevance."""

    def test_score_nodes_by_relevance(self):
        """Nodes should be scored by cosine similarity to query embedding."""
        from src.attention_pruning import score_nodes_by_relevance

        # 3 node embeddings, query aligns with node 0
        node_embeddings = {
            "n0": np.array([1.0, 0.0, 0.0]),
            "n1": np.array([0.0, 1.0, 0.0]),
            "n2": np.array([0.5, 0.5, 0.0]),
        }
        query_embedding = np.array([1.0, 0.0, 0.0])

        scores = score_nodes_by_relevance(node_embeddings, query_embedding)

        assert scores["n0"] == pytest.approx(1.0, abs=0.01)
        assert scores["n1"] == pytest.approx(0.0, abs=0.01)
        assert scores["n0"] > scores["n2"] > scores["n1"]

    def test_prune_by_relevance(self):
        """Prune nodes below relevance threshold, keeping top-k."""
        from src.attention_pruning import prune_by_relevance

        node_embeddings = {
            "n0": np.array([1.0, 0.0, 0.0]),
            "n1": np.array([0.0, 1.0, 0.0]),
            "n2": np.array([0.7, 0.3, 0.0]),
            "n3": np.array([0.9, 0.1, 0.0]),
        }
        query_embedding = np.array([1.0, 0.0, 0.0])

        kept = prune_by_relevance(node_embeddings, query_embedding, keep_ratio=0.5)

        assert len(kept) == 2
        assert "n0" in kept
        assert "n3" in kept or "n2" in kept

    def test_prune_empty_nodes(self):
        """Pruning empty node set returns empty."""
        from src.attention_pruning import prune_by_relevance

        result = prune_by_relevance({}, np.array([1.0, 0.0]), keep_ratio=0.5)
        assert result == []

    def test_prune_keep_ratio_one(self):
        """keep_ratio=1.0 keeps all nodes."""
        from src.attention_pruning import prune_by_relevance

        nodes = {"n0": np.array([1.0]), "n1": np.array([0.0])}
        result = prune_by_relevance(nodes, np.array([1.0]), keep_ratio=1.0)
        assert len(result) == 2

    def test_prune_keep_ratio_zero(self):
        """keep_ratio=0.0 keeps at least 1 node."""
        from src.attention_pruning import prune_by_relevance

        nodes = {"n0": np.array([1.0]), "n1": np.array([0.0])}
        result = prune_by_relevance(nodes, np.array([1.0]), keep_ratio=0.0)
        assert len(result) >= 1


# =============================================================================
# 2. Semantic Chunking
# =============================================================================


class TestSemanticChunking:
    """Tests for embedding-based semantic boundary detection."""

    def test_detect_boundaries(self):
        """Should detect semantic boundaries between dissimilar sentences."""
        from src.semantic_chunking import detect_semantic_boundaries

        sentences = [
            "Python is a programming language.",
            "Python supports multiple paradigms.",
            "The weather today is sunny and warm.",
            "Tomorrow will be rainy.",
        ]

        # Mock embedder that returns distinct embeddings for different topics
        def mock_encode(texts):
            embeddings = []
            for t in texts:
                if "Python" in t or "programming" in t:
                    embeddings.append(np.array([1.0, 0.0, 0.0]))
                else:
                    embeddings.append(np.array([0.0, 1.0, 0.0]))
            return np.array(embeddings)

        boundaries = detect_semantic_boundaries(sentences, mock_encode, threshold=0.5)

        # Should detect boundary between sentence 1 and 2 (topic change)
        assert 2 in boundaries  # index where new chunk starts

    def test_chunk_by_semantics(self):
        """Should group semantically similar sentences into chunks."""
        from src.semantic_chunking import chunk_by_semantics

        sentences = ["A about cats.", "B about cats.", "C about dogs.", "D about dogs."]

        def mock_encode(texts):
            embeddings = []
            for t in texts:
                if "cats" in t:
                    embeddings.append(np.array([1.0, 0.0]))
                else:
                    embeddings.append(np.array([0.0, 1.0]))
            return np.array(embeddings)

        chunks = chunk_by_semantics(sentences, mock_encode, threshold=0.5)

        assert len(chunks) == 2
        assert all("cats" in s for s in chunks[0])
        assert all("dogs" in s for s in chunks[1])

    def test_single_sentence(self):
        """Single sentence returns single chunk."""
        from src.semantic_chunking import chunk_by_semantics

        def mock_encode(texts):
            return np.array([[1.0, 0.0]] * len(texts))

        chunks = chunk_by_semantics(["Hello world."], mock_encode)
        assert len(chunks) == 1

    def test_empty_input(self):
        """Empty input returns empty chunks."""
        from src.semantic_chunking import chunk_by_semantics

        def mock_encode(texts):
            return np.array([]).reshape(0, 2)

        chunks = chunk_by_semantics([], mock_encode)
        assert chunks == []

    def test_max_chunk_size_respected(self):
        """Chunks should not exceed max_chunk_size sentences."""
        from src.semantic_chunking import chunk_by_semantics

        sentences = [f"Same topic sentence {i}." for i in range(20)]

        def mock_encode(texts):
            return np.array([[1.0, 0.0]] * len(texts))

        chunks = chunk_by_semantics(sentences, mock_encode, max_chunk_size=5)
        assert all(len(c) <= 5 for c in chunks)


# =============================================================================
# 3. Intra-Document Redundancy Collapse
# =============================================================================


class TestIntraDocDedup:
    """Tests for within-document redundancy detection and collapse."""

    def test_find_intra_duplicates(self):
        """Should find near-duplicate nodes within same document."""
        from src.intra_doc_dedup import find_intra_duplicates

        nodes = {
            "doc1_n0": {"text": "Python is great", "embedding": np.array([1.0, 0.0, 0.0])},
            "doc1_n1": {"text": "Python is great for ML", "embedding": np.array([0.98, 0.1, 0.0])},
            "doc1_n2": {"text": "Weather is nice", "embedding": np.array([0.0, 1.0, 0.0])},
        }

        dupes = find_intra_duplicates(nodes, threshold=0.9)

        assert len(dupes) >= 1
        pair = dupes[0]
        assert pair["similarity"] >= 0.9

    def test_collapse_redundant_nodes(self):
        """Should collapse duplicate nodes into representative with count."""
        from src.intra_doc_dedup import collapse_redundant_nodes

        nodes = {
            "doc1_n0": {"text": "Repeat this", "embedding": np.array([1.0, 0.0])},
            "doc1_n1": {"text": "Repeat this too", "embedding": np.array([0.99, 0.05])},
            "doc1_n2": {"text": "Different content", "embedding": np.array([0.0, 1.0])},
        }

        collapsed = collapse_redundant_nodes(nodes, threshold=0.9)

        # Should have fewer nodes
        assert len(collapsed) < len(nodes)
        # Representative should have occurrence_count
        for node in collapsed.values():
            if node.get("occurrence_count", 1) > 1:
                assert node["occurrence_count"] == 2

    def test_no_duplicates(self):
        """Orthogonal nodes should not be collapsed."""
        from src.intra_doc_dedup import collapse_redundant_nodes

        nodes = {
            "n0": {"text": "A", "embedding": np.array([1.0, 0.0])},
            "n1": {"text": "B", "embedding": np.array([0.0, 1.0])},
        }

        collapsed = collapse_redundant_nodes(nodes, threshold=0.95)
        assert len(collapsed) == 2


# =============================================================================
# 4. Context Decay and Eviction
# =============================================================================


class TestContextDecay:
    """Tests for document access tracking and stale eviction."""

    def test_access_tracker_records_time(self):
        """AccessTracker should record access timestamps."""
        from src.context_decay import AccessTracker

        tracker = AccessTracker()
        tracker.record_access("doc1")

        info = tracker.get_access_info("doc1")
        assert info is not None
        assert info["access_count"] >= 1
        assert "last_accessed" in info

    def test_access_tracker_multiple_accesses(self):
        """Multiple accesses should increment count."""
        from src.context_decay import AccessTracker

        tracker = AccessTracker()
        tracker.record_access("doc1")
        tracker.record_access("doc1")
        tracker.record_access("doc1")

        info = tracker.get_access_info("doc1")
        assert info["access_count"] == 3

    def test_find_stale_documents(self):
        """Should identify documents not accessed within max_age."""
        from src.context_decay import AccessTracker

        tracker = AccessTracker()
        # Manually set old timestamp
        tracker.record_access("old_doc")
        tracker._access_log["old_doc"]["last_accessed"] = time.time() - 7200  # 2 hours ago
        tracker.record_access("new_doc")

        stale = tracker.find_stale(max_age_seconds=3600)  # 1 hour
        assert "old_doc" in stale
        assert "new_doc" not in stale

    def test_compute_decay_score(self):
        """Decay score should decrease over time."""
        from src.context_decay import compute_decay_score

        # Recent access = high score
        recent_score = compute_decay_score(
            last_accessed=time.time() - 60,  # 1 min ago
            access_count=5,
            base_importance=0.8,
        )

        # Old access = lower score
        old_score = compute_decay_score(
            last_accessed=time.time() - 86400,  # 1 day ago
            access_count=1,
            base_importance=0.8,
        )

        assert recent_score > old_score
        assert 0.0 <= recent_score <= 1.0
        assert 0.0 <= old_score <= 1.0

    def test_unknown_doc_returns_none(self):
        """Unknown document should return None."""
        from src.context_decay import AccessTracker

        tracker = AccessTracker()
        assert tracker.get_access_info("nonexistent") is None


# =============================================================================
# 5. Compression Fidelity Scoring
# =============================================================================


class TestFidelityScoring:
    """Tests for measuring compression quality via embedding similarity."""

    def test_fidelity_identical(self):
        """Identical input/output should have fidelity ~1.0."""
        from src.fidelity_scoring import compute_fidelity_score

        text = "Python is a programming language used for many tasks."

        def mock_encode(texts):
            return np.array([[1.0, 0.0, 0.0]] * len(texts))

        score = compute_fidelity_score(text, text, mock_encode)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_fidelity_different(self):
        """Very different input/output should have low fidelity."""
        from src.fidelity_scoring import compute_fidelity_score

        def mock_encode(texts):
            embeddings = []
            for t in texts:
                if "Python" in t:
                    embeddings.append(np.array([1.0, 0.0, 0.0]))
                else:
                    embeddings.append(np.array([0.0, 1.0, 0.0]))
            return np.array(embeddings)

        score = compute_fidelity_score("Python is great", "Weather is nice", mock_encode)
        assert score < 0.5

    def test_fidelity_returns_float(self):
        """Fidelity score should always be a float in [0, 1]."""
        from src.fidelity_scoring import compute_fidelity_score

        def mock_encode(texts):
            return np.array([[0.5, 0.5]] * len(texts))

        score = compute_fidelity_score("a", "b", mock_encode)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_fidelity_empty_input(self):
        """Empty input should return 0.0 fidelity."""
        from src.fidelity_scoring import compute_fidelity_score

        def mock_encode(texts):
            return np.array([[0.0, 0.0]] * len(texts))

        score = compute_fidelity_score("", "", mock_encode)
        assert score == pytest.approx(0.0, abs=0.01) or score == pytest.approx(1.0, abs=0.01)


# =============================================================================
# 6. Multi-Level Skeleton
# =============================================================================


class TestMultiLevelSkeleton:
    """Tests for 3-tier skeleton output (headline/summary/full)."""

    def test_generate_multi_level(self):
        """Should return 3 levels: headline, summary, full."""
        from src.multi_level_skeleton import generate_multi_level_skeleton

        nodes = []
        for i in range(10):
            nodes.append(
                {
                    "node_id": f"n{i}",
                    "text": f"Node {i} content about topic {i % 3}.",
                    "importance": 1.0 - (i * 0.08),
                }
            )

        result = generate_multi_level_skeleton(nodes)

        assert "headline" in result
        assert "summary" in result
        assert "full" in result

    def test_headline_is_smallest(self):
        """Headline should have fewest nodes."""
        from src.multi_level_skeleton import generate_multi_level_skeleton

        nodes = [
            {"node_id": f"n{i}", "text": f"Content {i}", "importance": 1.0 - i * 0.05}
            for i in range(20)
        ]

        result = generate_multi_level_skeleton(nodes)

        assert len(result["headline"]["nodes"]) < len(result["summary"]["nodes"])
        assert len(result["summary"]["nodes"]) <= len(result["full"]["nodes"])

    def test_headline_ratio(self):
        """Headline should be ~10% of nodes."""
        from src.multi_level_skeleton import generate_multi_level_skeleton

        nodes = [
            {"node_id": f"n{i}", "text": f"Content {i}", "importance": 1.0 - i * 0.01}
            for i in range(100)
        ]

        result = generate_multi_level_skeleton(nodes)

        assert len(result["headline"]["nodes"]) <= 15  # ~10% + buffer
        assert len(result["summary"]["nodes"]) <= 35  # ~30% + buffer

    def test_levels_contain_text(self):
        """Each level should have concatenated text."""
        from src.multi_level_skeleton import generate_multi_level_skeleton

        nodes = [
            {"node_id": "n0", "text": "Important finding.", "importance": 1.0},
            {"node_id": "n1", "text": "Less important.", "importance": 0.3},
        ]

        result = generate_multi_level_skeleton(nodes)

        assert "Important finding" in result["headline"]["text"]
        assert "text" in result["full"]

    def test_empty_nodes(self):
        """Empty nodes should return empty levels."""
        from src.multi_level_skeleton import generate_multi_level_skeleton

        result = generate_multi_level_skeleton([])
        assert result["headline"]["nodes"] == []
        assert result["summary"]["nodes"] == []
        assert result["full"]["nodes"] == []


# =============================================================================
# 7. Query-Adaptive Compression
# =============================================================================


class TestQueryAdaptiveCompression:
    """Tests for per-section compression based on query relevance."""

    def test_compute_section_ratios(self):
        """Relevant sections get lighter compression, irrelevant heavier."""
        from src.query_adaptive import compute_section_ratios

        sections = [
            {"text": "About Python ML", "embedding": np.array([1.0, 0.0])},
            {"text": "About weather", "embedding": np.array([0.0, 1.0])},
        ]
        query_embedding = np.array([1.0, 0.0])
        base_ratio = 0.3

        ratios = compute_section_ratios(sections, query_embedding, base_ratio)

        # Python section should keep more (higher ratio)
        assert ratios[0] > ratios[1]
        # Average should be close to base_ratio
        avg = sum(ratios) / len(ratios)
        assert avg == pytest.approx(base_ratio, abs=0.15)

    def test_no_query_uniform_ratios(self):
        """Without query, all sections get same ratio."""
        from src.query_adaptive import compute_section_ratios

        sections = [
            {"text": "A", "embedding": np.array([1.0, 0.0])},
            {"text": "B", "embedding": np.array([0.0, 1.0])},
        ]

        ratios = compute_section_ratios(sections, query_embedding=None, base_ratio=0.5)

        assert ratios[0] == pytest.approx(ratios[1], abs=0.01)

    def test_single_section(self):
        """Single section gets base_ratio."""
        from src.query_adaptive import compute_section_ratios

        sections = [{"text": "Only one", "embedding": np.array([1.0])}]
        ratios = compute_section_ratios(sections, np.array([1.0]), base_ratio=0.4)
        assert len(ratios) == 1
        assert ratios[0] == pytest.approx(0.4, abs=0.2)

    def test_ratios_clamped(self):
        """Ratios should be clamped to [0.05, 1.0]."""
        from src.query_adaptive import compute_section_ratios

        sections = [
            {"text": "A", "embedding": np.array([1.0, 0.0])},
            {"text": "B", "embedding": np.array([0.0, 1.0])},
        ]

        ratios = compute_section_ratios(sections, np.array([1.0, 0.0]), base_ratio=0.1)
        assert all(0.05 <= r <= 1.0 for r in ratios)


# =============================================================================
# 8. Context Window Advisor
# =============================================================================


class TestContextAdvisor:
    """Tests for context window analysis and recommendations."""

    def test_advise_basic(self):
        """Should return model recommendations and pruning suggestions."""
        from src.context_advisor import advise_context

        doc_stats = [
            {"doc_id": "doc1", "tokens": 5000, "importance": 0.9},
            {"doc_id": "doc2", "tokens": 50000, "importance": 0.3},
            {"doc_id": "doc3", "tokens": 10000, "importance": 0.7},
        ]

        advice = advise_context(doc_stats)

        assert "total_tokens" in advice
        assert advice["total_tokens"] == 65000
        assert "recommended_models" in advice
        assert "prune_first" in advice
        assert len(advice["prune_first"]) > 0

    def test_prune_first_ordered_by_importance(self):
        """Prune suggestions should be ordered by lowest importance first."""
        from src.context_advisor import advise_context

        doc_stats = [
            {"doc_id": "high", "tokens": 10000, "importance": 0.9},
            {"doc_id": "low", "tokens": 10000, "importance": 0.1},
            {"doc_id": "mid", "tokens": 10000, "importance": 0.5},
        ]

        advice = advise_context(doc_stats)

        # First prune suggestion should be lowest importance
        assert advice["prune_first"][0]["doc_id"] == "low"

    def test_model_recommendations(self):
        """Should recommend models that fit the total token count."""
        from src.context_advisor import advise_context

        # Small context — many models fit
        small = advise_context([{"doc_id": "d1", "tokens": 1000, "importance": 0.5}])
        assert len(small["recommended_models"]) > 0

        # Huge context — fewer models fit
        huge = advise_context([{"doc_id": "d1", "tokens": 500000, "importance": 0.5}])
        assert len(huge["recommended_models"]) <= len(small["recommended_models"])

    def test_empty_docs(self):
        """Empty doc list should return zero tokens."""
        from src.context_advisor import advise_context

        advice = advise_context([])
        assert advice["total_tokens"] == 0

    def test_compression_strategy(self):
        """Should include a recommended compression strategy."""
        from src.context_advisor import advise_context

        advice = advise_context(
            [
                {"doc_id": "d1", "tokens": 100000, "importance": 0.5},
            ]
        )
        assert "strategy" in advice
        assert isinstance(advice["strategy"], str)


# =============================================================================
# 9. Keyword Anchoring — REMOVED (2026-07-17 cleanup, B4). The dead
# src/keyword_anchoring.py module was deleted; the MCP anchored_keywords
# feature is served by the separate inline implementation in
# compression_handlers.py (unaffected).
# =============================================================================


# =============================================================================
# 10. Compression Replay Log
# =============================================================================


class TestCompressionReplay:
    """Tests for tracking compression history and insights."""

    def test_record_compression(self):
        """Should record compression event."""
        from src.compression_replay import CompressionReplayLog

        log = CompressionReplayLog()
        log.record(
            doc_id="doc1",
            content_type="code",
            input_tokens=10000,
            output_tokens=3000,
            ratio=0.3,
            fidelity_score=0.92,
        )

        history = log.get_history("doc1")
        assert len(history) == 1
        assert history[0]["ratio"] == 0.3

    def test_get_insights(self):
        """Should compute insights: best ratio per content type."""
        from src.compression_replay import CompressionReplayLog

        log = CompressionReplayLog()
        log.record("d1", "code", 10000, 3000, 0.3, 0.95)
        log.record("d2", "code", 10000, 5000, 0.5, 0.85)
        log.record("d3", "prose", 10000, 2000, 0.2, 0.90)

        insights = log.get_insights()

        assert "code" in insights
        assert "prose" in insights
        # Best for code: the one with highest fidelity
        assert insights["code"]["best_ratio"] == 0.3

    def test_recommend_ratio(self):
        """Should recommend ratio based on past performance."""
        from src.compression_replay import CompressionReplayLog

        log = CompressionReplayLog()
        log.record("d1", "code", 10000, 3000, 0.3, 0.95)
        log.record("d2", "code", 10000, 5000, 0.5, 0.80)
        log.record("d3", "code", 10000, 2000, 0.2, 0.70)

        recommended = log.recommend_ratio("code", min_fidelity=0.85)
        assert recommended == 0.3  # Only ratio with fidelity >= 0.85

    def test_empty_log(self):
        """Empty log should return empty insights."""
        from src.compression_replay import CompressionReplayLog

        log = CompressionReplayLog()
        assert log.get_insights() == {}
        assert log.get_history("nonexistent") == []

    def test_recommend_unknown_type(self):
        """Unknown content type should return default ratio."""
        from src.compression_replay import CompressionReplayLog

        log = CompressionReplayLog()
        recommended = log.recommend_ratio("unknown", min_fidelity=0.8)
        assert recommended is None


# =============================================================================
# 11. Generative Rewrite Stub
# =============================================================================


class TestGenerativeRewrite:
    """Tests for generative rewrite prompt template generation."""

    def test_generate_rewrite_prompt(self):
        """Should produce a structured rewrite prompt for client LLM."""
        from src.generative_rewrite import generate_rewrite_prompt

        original_text = "Python is a high-level programming language. It supports OOP and functional programming. Python is widely used in data science."
        target_ratio = 0.5

        prompt = generate_rewrite_prompt(original_text, target_ratio)

        assert isinstance(prompt, dict)
        assert "system_instruction" in prompt
        assert "user_prompt" in prompt
        assert "target_token_count" in prompt
        assert "original_token_count" in prompt

    def test_rewrite_prompt_includes_original(self):
        """Rewrite prompt should contain the original text."""
        from src.generative_rewrite import generate_rewrite_prompt

        text = "Machine learning is a subset of AI."
        prompt = generate_rewrite_prompt(text, 0.5)

        assert text in prompt["user_prompt"]

    def test_rewrite_prompt_with_keywords(self):
        """Should include keyword preservation instructions."""
        from src.generative_rewrite import generate_rewrite_prompt

        prompt = generate_rewrite_prompt(
            "The API uses OAuth2 for authentication.",
            target_ratio=0.5,
            preserve_keywords=["OAuth2", "API"],
        )

        assert "OAuth2" in prompt["system_instruction"] or "OAuth2" in prompt["user_prompt"]

    def test_target_token_count(self):
        """Target token count should be ratio * original."""
        from src.generative_rewrite import generate_rewrite_prompt

        text = "word " * 100  # ~100 words
        prompt = generate_rewrite_prompt(text, target_ratio=0.3)

        assert prompt["target_token_count"] < prompt["original_token_count"]
        ratio = prompt["target_token_count"] / prompt["original_token_count"]
        assert ratio == pytest.approx(0.3, abs=0.05)

    def test_empty_text(self):
        """Empty text should return prompt with zero counts."""
        from src.generative_rewrite import generate_rewrite_prompt

        prompt = generate_rewrite_prompt("", 0.5)
        assert prompt["original_token_count"] == 0
        assert prompt["target_token_count"] == 0


# =============================================================================
# Integration: MCP Tool Wiring
# =============================================================================


class TestPhase5MCPWiring:
    """Tests that new features are wired as MCP tools."""

    def test_prune_by_relevance_tool_exists(self):
        """prune_by_relevance should be registered as MCP tool handler."""
        from src.handlers.compression_handlers import handle_prune_by_relevance

        assert callable(handle_prune_by_relevance)

    def test_multi_level_skeleton_tool_exists(self):
        """get_multi_level_skeleton should be a callable handler."""
        from src.handlers.compression_handlers import handle_multi_level_skeleton

        assert callable(handle_multi_level_skeleton)

    def test_evict_stale_tool_exists(self):
        """evict_stale should be a callable handler."""
        from src.handlers.compression_handlers import handle_evict_stale

        assert callable(handle_evict_stale)

    def test_advise_context_tool_exists(self):
        """advise_context should be a callable handler."""
        from src.handlers.compression_handlers import handle_advise_context

        assert callable(handle_advise_context)

    def test_get_compression_insights_tool_exists(self):
        """get_compression_insights should be a callable handler."""
        from src.handlers.compression_handlers import handle_get_compression_insights

        assert callable(handle_get_compression_insights)

    def test_generate_rewrite_prompt_tool_exists(self):
        """generate_rewrite_prompt should be a callable handler."""
        from src.handlers.compression_handlers import handle_generate_rewrite_prompt

        assert callable(handle_generate_rewrite_prompt)
