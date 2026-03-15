"""
Tests for prompt caching optimization improvements.

These tests verify that tool responses and context outputs are structured
to maximize LLM prompt cache hit rates by:
1. Excluding volatile fields (UUIDs, timestamps) from display output
2. Placing dynamic content (queries) at the end of output
3. Separating stable vs volatile content layers

TDD: Written BEFORE implementation (Red phase).
"""

import json
import time

import numpy as np
import pytest

from src.ace_framework import ACEBullet, ACEContext, BulletType


# ============================================================================
# Issue 1: ACE Bullet/Context volatile field exclusion
# ============================================================================


class TestACEBulletDisplayDict:
    """ACEBullet.to_display_dict() should exclude cache-busting volatile fields."""

    @pytest.fixture
    def bullet(self):
        return ACEBullet(
            text="Be concise in explanations",
            bullet_type=BulletType.PRINCIPLE,
            embedding=np.zeros(384),
            confidence=0.75,
            success_count=3,
            failure_count=1,
        )

    def test_to_display_dict_exists(self, bullet):
        """to_display_dict method should exist on ACEBullet."""
        assert hasattr(bullet, "to_display_dict")
        result = bullet.to_display_dict()
        assert isinstance(result, dict)

    def test_to_display_dict_excludes_volatile_fields(self, bullet):
        """Display dict must NOT contain timestamps or UUIDs."""
        result = bullet.to_display_dict()
        assert "bullet_id" not in result
        assert "created_at" not in result
        assert "updated_at" not in result

    def test_to_display_dict_preserves_semantic_fields(self, bullet):
        """Display dict must preserve all semantically meaningful fields."""
        result = bullet.to_display_dict()
        assert result["text"] == "Be concise in explanations"
        assert result["bullet_type"] == "principle"
        assert result["confidence"] == 0.75
        assert result["success_count"] == 3
        assert result["failure_count"] == 1
        assert result["success_rate"] == 0.75
        assert result["total_usage"] == 4
        assert result["source"] == "manual"

    def test_to_display_dict_excludes_embedding(self, bullet):
        """Display dict should not include large embedding arrays."""
        result = bullet.to_display_dict()
        assert "embedding" not in result

    def test_to_display_dict_is_stable_across_time(self, bullet):
        """Two calls should produce identical output (no time-dependent fields)."""
        result1 = bullet.to_display_dict()
        time.sleep(0.01)
        result2 = bullet.to_display_dict()
        assert result1 == result2

    def test_to_dict_still_includes_volatile_fields(self, bullet):
        """Original to_dict must remain unchanged for persistence/internal use."""
        result = bullet.to_dict()
        assert "bullet_id" in result
        assert "created_at" in result
        assert "updated_at" in result


class TestACEContextDisplayDict:
    """ACEContext display serialization should exclude volatile fields."""

    @pytest.fixture
    def context_with_bullets(self):
        ctx = ACEContext()
        bullet = ACEBullet(
            text="Test bullet",
            bullet_type=BulletType.STRATEGY,
            embedding=np.zeros(384),
            confidence=0.8,
        )
        ctx.add_bullet(bullet, "Added test bullet")
        return ctx

    def test_to_display_dict_exists(self, context_with_bullets):
        """to_display_dict method should exist on ACEContext."""
        assert hasattr(context_with_bullets, "to_display_dict")
        result = context_with_bullets.to_display_dict()
        assert isinstance(result, dict)

    def test_to_display_dict_excludes_context_timestamps(self, context_with_bullets):
        """Context display dict must NOT contain context-level timestamps."""
        result = context_with_bullets.to_display_dict()
        assert "created_at" not in result
        assert "updated_at" not in result

    def test_to_display_dict_excludes_context_id(self, context_with_bullets):
        """Context display dict must NOT contain the UUID context_id."""
        result = context_with_bullets.to_display_dict()
        assert "context_id" not in result

    def test_to_display_dict_bullets_exclude_volatile(self, context_with_bullets):
        """Bullets inside context display dict should also exclude volatile fields."""
        result = context_with_bullets.to_display_dict()
        for bullet_dict in result["bullets"].values():
            assert "bullet_id" not in bullet_dict
            assert "created_at" not in bullet_dict
            assert "updated_at" not in bullet_dict

    def test_delta_history_excludes_timestamps(self, context_with_bullets):
        """Delta history entries in display dict should not include timestamps."""
        result = context_with_bullets.to_display_dict()
        for delta in result.get("delta_history", []):
            assert "timestamp" not in delta

    def test_delta_history_excludes_bullet_ids(self, context_with_bullets):
        """Delta history entries in display dict should not include bullet_ids."""
        result = context_with_bullets.to_display_dict()
        for delta in result.get("delta_history", []):
            assert "bullet_id" not in delta

    def test_to_display_dict_preserves_version(self, context_with_bullets):
        """Display dict should still contain version for semantic meaning."""
        result = context_with_bullets.to_display_dict()
        assert "version" in result

    def test_to_display_dict_preserves_stats(self, context_with_bullets):
        """Display dict should still contain performance stats."""
        result = context_with_bullets.to_display_dict()
        assert "stats" in result

    def test_to_dict_still_includes_volatile_fields(self, context_with_bullets):
        """Original to_dict must remain unchanged for persistence."""
        result = context_with_bullets.to_dict()
        assert "context_id" in result
        assert "created_at" in result
        assert "updated_at" in result


# ============================================================================
# Issue 2: ACE handlers use display dict
# ============================================================================


class TestACEHandlersCacheOptimized:
    """ACE handlers should use display-safe serialization in responses."""

    @pytest.fixture
    def ace_context_with_data(self):
        ctx = ACEContext()
        for i in range(3):
            bullet = ACEBullet(
                text=f"Bullet {i}",
                bullet_type=BulletType.PRINCIPLE,
                embedding=np.zeros(384),
                confidence=0.5 + i * 0.1,
            )
            ctx.add_bullet(bullet, f"Added bullet {i}")
        return ctx

    def test_filter_serialize_excludes_volatile(self, ace_context_with_data):
        """_filter_and_serialize_bullets should use display dict by default."""
        from src.handlers.ace_handlers import _filter_and_serialize_bullets

        bullets = _filter_and_serialize_bullets(
            ace_context_with_data,
            include_embeddings=False,
            min_confidence=None,
            bullet_type_filter=None,
        )
        for bullet_dict in bullets:
            assert "bullet_id" not in bullet_dict
            assert "created_at" not in bullet_dict
            assert "updated_at" not in bullet_dict


# ============================================================================
# Issue 3: Skeleton query placement (volatile content at end)
# ============================================================================


class TestSkeletonQueryPlacement:
    """Query metadata should be at the END of skeleton output, not the top."""

    @pytest.fixture
    def compressor(self):
        from src.semantic_compressor import SemanticCompressor

        return SemanticCompressor()

    def test_query_text_at_end_of_skeleton(self, compressor):
        """When query is provided, query text should appear after node content."""
        # Ingest a document first
        compressor.ingest_file("This is a test document with some content. " * 20, "test_doc")

        skeleton = compressor._generate_skeleton("test_doc", query="find the test")
        lines = skeleton.skeleton_text.strip().split("\n")

        # Find where query-related lines are
        query_line_indices = [
            i for i, line in enumerate(lines) if "Query:" in line or "QUERY_GUIDED" in line
        ]
        # Find where node content lines are ([node_id] lines)
        node_line_indices = [i for i, line in enumerate(lines) if line.strip().startswith("[")]

        if query_line_indices and node_line_indices:
            last_node_line = max(node_line_indices)
            first_query_line = min(query_line_indices)
            assert first_query_line > last_node_line, (
                f"Query metadata (line {first_query_line}) should come AFTER "
                f"node content (last node at line {last_node_line}) for cache-friendly ordering"
            )

    def test_skeleton_without_query_unchanged(self, compressor):
        """Skeleton without query should not have query metadata at all."""
        compressor.ingest_file("Another test document content. " * 20, "test_doc2")
        skeleton = compressor._generate_skeleton("test_doc2")
        assert "Query:" not in skeleton.skeleton_text
        assert "QUERY_GUIDED" not in skeleton.skeleton_text
