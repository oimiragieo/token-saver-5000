"""
Tests for Audited Semantic Graph

Tests for:
- NodeProvenance tracking
- QualityHistory management
- AuditedSemanticNode creation and properties
- CompositionEdge constraints
- AuditedSemanticGraph operations
- Evidence bundle integration
"""

import pytest
import time
import numpy as np

from src.audited_graph import (
    AuditedSemanticGraph,
    AuditedSemanticNode,
    NodeProvenance,
    QualityHistory,
    CompositionEdge,
)
from src.evidence_bundle import EvidenceBundle


class TestNodeProvenance:
    """Tests for NodeProvenance dataclass"""

    def test_basic_creation(self):
        """Test creating provenance"""
        prov = NodeProvenance(
            created_at=time.time(),
            created_by="ingest",
        )

        assert prov.created_by == "ingest"
        assert prov.creation_bundle_id is None

    def test_with_source_info(self):
        """Test provenance with source file info"""
        prov = NodeProvenance(
            created_at=time.time(),
            created_by="ingest",
            source_file="test.py",
            source_line_start=10,
            source_line_end=25,
        )

        assert prov.source_file == "test.py"
        assert prov.source_line_start == 10
        assert prov.source_line_end == 25

    def test_parent_child_relationships(self):
        """Test parent/child node tracking"""
        prov = NodeProvenance(
            created_at=time.time(),
            created_by="split",
            parent_node_ids=["parent1", "parent2"],
            child_node_ids=["child1"],
        )

        assert len(prov.parent_node_ids) == 2
        assert len(prov.child_node_ids) == 1

    def test_to_dict(self):
        """Test serialization"""
        prov = NodeProvenance(
            created_at=1234567890.0,
            created_by="test_op",
            creation_bundle_id="bundle_123",
            source_file="test.py",
        )

        d = prov.to_dict()
        assert d["created_at"] == 1234567890.0
        assert d["created_by"] == "test_op"
        assert d["creation_bundle_id"] == "bundle_123"

    def test_from_dict(self):
        """Test deserialization"""
        data = {
            "created_at": 1234567890.0,
            "created_by": "test",
            "source_file": "test.py",
            "parent_node_ids": ["p1"],
        }

        prov = NodeProvenance.from_dict(data)
        assert prov.created_at == 1234567890.0
        assert prov.parent_node_ids == ["p1"]


class TestQualityHistory:
    """Tests for QualityHistory dataclass"""

    def test_empty_history(self):
        """Test empty quality history"""
        history = QualityHistory()

        assert history.latest_score is None
        assert history.average_score == 0.0
        assert history.trend == "stable"

    def test_add_score(self):
        """Test adding scores"""
        history = QualityHistory()
        history.add_score(0.8, "compress")
        history.add_score(0.9, "modulate")

        assert len(history.scores) == 2
        assert history.latest_score == 0.9

    def test_average_score(self):
        """Test average calculation"""
        history = QualityHistory()
        history.add_score(0.6, "op1")
        history.add_score(0.8, "op2")
        history.add_score(1.0, "op3")

        assert abs(history.average_score - 0.8) < 0.001

    def test_trend_stable(self):
        """Test stable trend"""
        history = QualityHistory()
        history.add_score(0.8, "op1")
        history.add_score(0.81, "op2")
        history.add_score(0.79, "op3")

        assert history.trend == "stable"

    def test_trend_improving(self):
        """Test improving trend"""
        history = QualityHistory()
        for i in range(5):
            history.add_score(0.5 + i * 0.1, f"op{i}")

        assert history.trend == "improving"

    def test_trend_declining(self):
        """Test declining trend"""
        history = QualityHistory()
        for i in range(5):
            history.add_score(0.9 - i * 0.1, f"op{i}")

        assert history.trend == "declining"

    def test_to_dict(self):
        """Test serialization"""
        history = QualityHistory()
        history.add_score(0.8, "test")

        d = history.to_dict()
        assert "scores" in d
        assert "average_score" in d
        assert "trend" in d


class TestAuditedSemanticNode:
    """Tests for AuditedSemanticNode"""

    def test_basic_creation(self):
        """Test creating a node"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test_1",
            text="Test content",
            embedding=embedding,
            importance=0.75,
        )

        assert node.node_id == "test_1"
        assert node.importance == 0.75

    def test_record_access(self):
        """Test recording access"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
        )

        assert node.access_count == 0
        assert node.last_accessed is None

        node.record_access()

        assert node.access_count == 1
        assert node.last_accessed is not None

    def test_add_quality_score(self):
        """Test adding quality score"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
        )

        node.add_quality_score(0.9, "compress")

        assert node.quality_history.latest_score == 0.9

    def test_is_active(self):
        """Test active status"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
        )

        # Not accessed yet
        assert not node.is_active

        # After access
        node.record_access()
        assert node.is_active

    def test_promotion_eligible_false(self):
        """Test promotion eligibility - not eligible"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
        )

        assert not node.promotion_eligible

    def test_promotion_eligible_true(self):
        """Test promotion eligibility - eligible"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
        )

        # Add multiple high quality scores
        for i in range(5):
            node.add_quality_score(0.85, f"op{i}")

        # Access multiple times
        for _ in range(5):
            node.record_access()

        assert node.promotion_eligible

    def test_to_dict(self):
        """Test serialization"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
            importance=0.5,
            metadata={"key": "value"},
        )

        d = node.to_dict()
        assert d["node_id"] == "test"
        assert d["importance"] == 0.5
        assert "provenance" in d
        assert "quality_history" in d


class TestCompositionEdge:
    """Tests for CompositionEdge"""

    def test_basic_creation(self):
        """Test creating an edge"""
        edge = CompositionEdge(
            source_id="node1",
            target_id="node2",
        )

        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.edge_type == "semantic"
        assert edge.weight == 1.0

    def test_with_constraints(self):
        """Test edge with constraints"""
        edge = CompositionEdge(
            source_id="n1",
            target_id="n2",
            edge_type="structural",
            constraints=["requires:n3", "excludes:n4"],
        )

        assert len(edge.constraints) == 2
        assert edge.edge_type == "structural"

    def test_to_dict(self):
        """Test serialization"""
        edge = CompositionEdge(
            source_id="a",
            target_id="b",
            weight=0.5,
        )

        d = edge.to_dict()
        assert d["source_id"] == "a"
        assert d["target_id"] == "b"
        assert d["weight"] == 0.5


class TestAuditedSemanticGraph:
    """Tests for AuditedSemanticGraph"""

    def test_empty_graph(self):
        """Test empty graph properties"""
        graph = AuditedSemanticGraph()

        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.evidence_count == 0

    def test_add_node(self):
        """Test adding a node"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        bundle = graph.add_node(
            node_id="test_1",
            text="Test content",
            embedding=embedding,
            importance=0.8,
        )

        assert graph.node_count == 1
        assert isinstance(bundle, EvidenceBundle)
        assert bundle.operation == "add_node"

    def test_add_node_with_source(self):
        """Test adding node with source file info"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        bundle = graph.add_node(
            node_id="test",
            text="content",
            embedding=embedding,
            source_file="test.py",
            operation="ingest",
        )

        node = graph.get_node("test")
        assert node.provenance.source_file == "test.py"

    def test_add_node_with_quality(self):
        """Test adding node with initial quality score"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node(
            node_id="test",
            text="content",
            embedding=embedding,
            quality_score=0.9,
        )

        node = graph.get_node("test")
        assert node.quality_history.latest_score == 0.9

    def test_add_edge(self):
        """Test adding an edge"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        # Add nodes first
        graph.add_node("n1", "content1", embedding)
        graph.add_node("n2", "content2", embedding)

        # Add edge
        bundle = graph.add_edge("n1", "n2", edge_type="semantic")

        assert graph.edge_count == 1
        assert isinstance(bundle, EvidenceBundle)

    def test_add_edge_missing_node(self):
        """Test adding edge with missing node"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "content", embedding)

        # Try to add edge to non-existent node
        bundle = graph.add_edge("n1", "n2")

        assert bundle is None
        assert graph.edge_count == 0

    def test_add_edge_updates_provenance(self):
        """Test that adding edge updates provenance"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "content1", embedding)
        graph.add_node("n2", "content2", embedding)
        graph.add_edge("n1", "n2")

        n1 = graph.get_node("n1")
        n2 = graph.get_node("n2")

        assert "n2" in n1.provenance.child_node_ids
        assert "n1" in n2.provenance.parent_node_ids

    def test_get_node(self):
        """Test getting a node"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("test", "content", embedding)
        node = graph.get_node("test")

        assert node is not None
        assert node.node_id == "test"
        assert node.access_count == 1  # Access was recorded

    def test_get_node_not_found(self):
        """Test getting non-existent node"""
        graph = AuditedSemanticGraph()

        node = graph.get_node("nonexistent")
        assert node is None

    def test_get_node_with_history(self):
        """Test getting node with evidence history"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("test", "content", embedding)
        node, bundles = graph.get_node_with_history("test")

        assert node is not None
        assert len(bundles) >= 1

    def test_get_edges_for_node(self):
        """Test getting edges for a node"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding)
        graph.add_node("n2", "c2", embedding)
        graph.add_node("n3", "c3", embedding)

        graph.add_edge("n1", "n2")
        graph.add_edge("n1", "n3")

        edges = graph.get_edges_for_node("n1")
        assert len(edges) == 2

    def test_update_quality(self):
        """Test updating quality score"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("test", "content", embedding)

        result = graph.update_quality("test", 0.85, "modulate")
        assert result is True

        node = graph.get_node("test")
        assert node.quality_history.latest_score == 0.85

    def test_update_quality_not_found(self):
        """Test updating quality for non-existent node"""
        graph = AuditedSemanticGraph()

        result = graph.update_quality("nonexistent", 0.5)
        assert result is False

    def test_get_promotion_candidates(self):
        """Test getting promotion candidates"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        # Add a node that will be promotion eligible
        graph.add_node("good", "content", embedding)
        node = graph.get_node("good")

        # Add quality scores
        for i in range(5):
            node.add_quality_score(0.9, f"op{i}")

        # Access multiple times
        for _ in range(5):
            node.record_access()

        candidates = graph.get_promotion_candidates()
        assert len(candidates) >= 1

    def test_get_demotion_candidates(self):
        """Test getting demotion candidates"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        # Add a node with declining quality
        graph.add_node("bad", "content", embedding)
        node = graph.get_node("bad")

        # Add declining quality scores
        for i in range(5):
            node.add_quality_score(0.9 - i * 0.15, f"op{i}")

        candidates = graph.get_demotion_candidates()
        assert len(candidates) >= 1

    def test_verify_composition_constraints_valid(self):
        """Test valid composition verification"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding)
        graph.add_node("n2", "c2", embedding)

        valid, violations = graph.verify_composition_constraints(["n1", "n2"])

        assert valid
        assert len(violations) == 0

    def test_verify_composition_constraints_missing_node(self):
        """Test composition verification with missing node"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding)

        valid, violations = graph.verify_composition_constraints(["n1", "n2"])

        assert not valid
        assert any("not found" in v for v in violations)

    def test_check_constraint_requires(self):
        """Test requires constraint"""
        graph = AuditedSemanticGraph()

        # Test requires constraint
        assert graph._check_constraint("requires:n1", ["n1", "n2"])
        assert not graph._check_constraint("requires:n3", ["n1", "n2"])

    def test_check_constraint_excludes(self):
        """Test excludes constraint"""
        graph = AuditedSemanticGraph()

        # Test excludes constraint
        assert graph._check_constraint("excludes:n3", ["n1", "n2"])
        assert not graph._check_constraint("excludes:n1", ["n1", "n2"])

    def test_get_statistics(self):
        """Test getting statistics"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding, quality_score=0.9)
        graph.add_node("n2", "c2", embedding, quality_score=0.8)
        graph.add_edge("n1", "n2")

        stats = graph.get_statistics()

        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["evidence_count"] >= 2
        assert "avg_quality_score" in stats
        assert "chain_valid" in stats

    def test_export_for_visualization(self):
        """Test export for visualization"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "content1", embedding, importance=0.9)
        graph.add_node("n2", "content2", embedding, importance=0.7)
        graph.add_edge("n1", "n2")

        export = graph.export_for_visualization()

        assert "nodes" in export
        assert "edges" in export
        assert "statistics" in export
        assert len(export["nodes"]) == 2
        assert len(export["edges"]) == 1

    def test_clear(self):
        """Test clearing the graph"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding)
        graph.add_node("n2", "c2", embedding)
        graph.add_edge("n1", "n2")

        assert graph.node_count == 2

        graph.clear()

        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.evidence_count == 0

    def test_evidence_chain_integrity(self):
        """Test evidence chain integrity"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        # Add multiple nodes
        for i in range(5):
            graph.add_node(f"n{i}", f"content{i}", embedding)

        stats = graph.get_statistics()
        assert stats["chain_valid"] is True


class TestGraphWithConstraints:
    """Tests for composition constraints in graph"""

    def test_edge_with_constraints(self):
        """Test edge with composition constraints"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "c1", embedding)
        graph.add_node("n2", "c2", embedding)
        graph.add_node("n3", "c3", embedding)

        graph.add_edge(
            "n1", "n2",
            constraints=["requires:n3"]
        )

        # Composition with n3 should pass
        valid, _ = graph.verify_composition_constraints(["n1", "n2", "n3"])
        assert valid

    def test_compatible_fidelities(self):
        """Test compatible fidelities on node"""
        embedding = np.array([1.0, 0.0, 0.0])
        node = AuditedSemanticNode(
            node_id="test",
            text="content",
            embedding=embedding,
            compatible_fidelities=["ABSTRACT", "OUTLINE"],
        )

        assert "ABSTRACT" in node.compatible_fidelities
        assert "DETAILED" not in node.compatible_fidelities


class TestEdgeCases:
    """Edge case tests"""

    def test_empty_text_node(self):
        """Test node with empty text"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        bundle = graph.add_node("empty", "", embedding)

        assert graph.node_count == 1
        assert bundle is not None

    def test_large_embedding(self):
        """Test node with large embedding"""
        graph = AuditedSemanticGraph()
        embedding = np.random.randn(768)  # BERT-sized embedding

        bundle = graph.add_node("large", "content", embedding)

        node = graph.get_node("large")
        assert node.embedding.shape == (768,)

    def test_many_edges(self):
        """Test node with many edges"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        # Add central node and many connected nodes
        graph.add_node("center", "center content", embedding)
        for i in range(20):
            graph.add_node(f"n{i}", f"content{i}", embedding)
            graph.add_edge("center", f"n{i}")

        edges = graph.get_edges_for_node("center")
        assert len(edges) == 20

    def test_self_loop(self):
        """Test adding self-loop edge"""
        graph = AuditedSemanticGraph()
        embedding = np.array([1.0, 0.0, 0.0])

        graph.add_node("n1", "content", embedding)

        # Self-loop should work
        bundle = graph.add_edge("n1", "n1")

        assert bundle is not None
        assert graph.edge_count == 1
