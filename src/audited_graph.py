"""
Audited Semantic Graph with Provenance Tracking

Extends the semantic graph with audit capabilities inspired by ASG-SI:
- Each node tracks its creation provenance
- Operations produce evidence bundles
- Quality history enables skill promotion/demotion
- Composition constraints are explicitly tracked

Usage:
    from src.audited_graph import AuditedSemanticGraph, AuditedSemanticNode

    graph = AuditedSemanticGraph()

    # Add node with full audit trail
    bundle = graph.add_node(
        node_id="chunk_1",
        text="Original text content...",
        embedding=embedding_vector,
        importance=0.85,
        operation="ingest"
    )

    # Query with provenance
    node, history = graph.get_node_with_history("chunk_1")
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import networkx as nx
import numpy as np

from .evidence_bundle import (
    EvidenceBundle,
    EvidenceStore,
    QualityMetrics,
    ContractResult,
)
from .compression_verifier import CompressionContract

if TYPE_CHECKING:
    from .semantic_compressor import FidelityLevel

logger = logging.getLogger(__name__)


@dataclass
class NodeProvenance:
    """Provenance information for a semantic node"""
    created_at: float
    created_by: str  # Operation that created this node
    creation_bundle_id: Optional[str] = None
    source_file: Optional[str] = None
    source_line_start: Optional[int] = None
    source_line_end: Optional[int] = None
    parent_node_ids: List[str] = field(default_factory=list)
    child_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "created_at": self.created_at,
            "created_by": self.created_by,
            "creation_bundle_id": self.creation_bundle_id,
            "source_file": self.source_file,
            "source_line_start": self.source_line_start,
            "source_line_end": self.source_line_end,
            "parent_node_ids": self.parent_node_ids,
            "child_node_ids": self.child_node_ids,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeProvenance":
        return cls(
            created_at=data.get("created_at", time.time()),
            created_by=data.get("created_by", "unknown"),
            creation_bundle_id=data.get("creation_bundle_id"),
            source_file=data.get("source_file"),
            source_line_start=data.get("source_line_start"),
            source_line_end=data.get("source_line_end"),
            parent_node_ids=data.get("parent_node_ids", []),
            child_node_ids=data.get("child_node_ids", []),
        )


@dataclass
class QualityHistory:
    """Quality metrics history for a node"""
    scores: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    operations: List[str] = field(default_factory=list)

    def add_score(self, score: float, operation: str) -> None:
        """Add a quality score"""
        self.scores.append(score)
        self.timestamps.append(time.time())
        self.operations.append(operation)

    @property
    def latest_score(self) -> Optional[float]:
        return self.scores[-1] if self.scores else None

    @property
    def average_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def trend(self) -> str:
        """Calculate score trend (improving, declining, stable)"""
        if len(self.scores) < 2:
            return "stable"
        recent = self.scores[-5:]
        if len(recent) < 2:
            return "stable"
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        return "stable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scores": self.scores,
            "timestamps": self.timestamps,
            "operations": self.operations,
            "average_score": self.average_score,
            "trend": self.trend,
        }


@dataclass
class AuditedSemanticNode:
    """
    Semantic node with full audit trail.

    Extends the basic SemanticNode with:
    - Creation provenance
    - Quality history
    - Access tracking
    - Composition constraints
    """
    node_id: str
    text: str
    embedding: np.ndarray
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Audit fields
    provenance: NodeProvenance = field(default_factory=lambda: NodeProvenance(
        created_at=time.time(),
        created_by="unknown"
    ))
    quality_history: QualityHistory = field(default_factory=QualityHistory)
    access_count: int = 0
    last_accessed: Optional[float] = None

    # Composition constraints
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    compatible_fidelities: List[str] = field(default_factory=lambda: [
        "ABSTRACT", "OUTLINE", "STRUCTURE", "DETAILED", "RAW"
    ])

    def record_access(self) -> None:
        """Record node access for usage tracking"""
        self.access_count += 1
        self.last_accessed = time.time()

    def add_quality_score(self, score: float, operation: str) -> None:
        """Add quality score to history"""
        self.quality_history.add_score(score, operation)

    @property
    def is_active(self) -> bool:
        """Check if node is actively used"""
        if self.last_accessed is None:
            return False
        # Consider active if accessed in last hour
        return time.time() - self.last_accessed < 3600

    @property
    def promotion_eligible(self) -> bool:
        """Check if node is eligible for promotion (frequently used, high quality)"""
        return (
            self.access_count >= 3 and
            self.quality_history.average_score >= 0.7 and
            self.quality_history.trend != "declining"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "importance": self.importance,
            "metadata": self.metadata,
            "provenance": self.provenance.to_dict(),
            "quality_history": self.quality_history.to_dict(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "compatible_fidelities": self.compatible_fidelities,
        }


@dataclass
class CompositionEdge:
    """Edge in the audited graph with composition constraints"""
    source_id: str
    target_id: str
    edge_type: str = "semantic"  # semantic, structural, temporal
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)
    created_by: str = "unknown"
    constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "weight": self.weight,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "constraints": self.constraints,
        }


class AuditedSemanticGraph:
    """
    Directed multigraph with full audit trail.

    Implements ASG-SI's audited skill graph concept:
    - Nodes represent compression "skills" (chunking strategies)
    - Edges encode composition constraints
    - All operations produce evidence bundles
    - Quality tracking enables promotion/demotion
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize audited graph.

        Args:
            storage_path: Optional path for evidence store
        """
        self.graph = nx.DiGraph()
        self._nodes: Dict[str, AuditedSemanticNode] = {}
        self._edges: List[CompositionEdge] = []
        self._evidence_store = EvidenceStore(
            storage_path=storage_path if storage_path else None
        )
        self._operation_count = 0

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def evidence_count(self) -> int:
        return len(self._evidence_store)

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: np.ndarray,
        importance: float = 0.0,
        operation: str = "add_node",
        source_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
    ) -> EvidenceBundle:
        """
        Add node with full audit trail.

        Args:
            node_id: Unique node identifier
            text: Node text content
            embedding: Embedding vector
            importance: PageRank importance score
            operation: Operation creating this node
            source_file: Source file path
            metadata: Additional metadata
            quality_score: Optional initial quality score

        Returns:
            EvidenceBundle for this operation
        """
        # Create provenance
        provenance = NodeProvenance(
            created_at=time.time(),
            created_by=operation,
            source_file=source_file,
        )

        # Create node
        node = AuditedSemanticNode(
            node_id=node_id,
            text=text,
            embedding=embedding,
            importance=importance,
            metadata=metadata or {},
            provenance=provenance,
        )

        # Add quality score if provided
        if quality_score is not None:
            node.add_quality_score(quality_score, operation)

        # Create evidence bundle
        bundle = self._create_bundle(
            operation=operation,
            input_data=text,
            output_data=node_id,
            input_tokens=len(text.split()),
            output_tokens=1,
            parameters={"importance": importance, "source_file": source_file},
        )

        # Link bundle to provenance
        node.provenance.creation_bundle_id = bundle.bundle_id

        # Store node
        self._nodes[node_id] = node
        self.graph.add_node(node_id, data=node)

        # Store evidence
        self._evidence_store.append(bundle)
        self._operation_count += 1

        return bundle

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "semantic",
        weight: float = 1.0,
        operation: str = "add_edge",
        constraints: Optional[List[str]] = None,
    ) -> Optional[EvidenceBundle]:
        """
        Add edge with composition constraints.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of relationship
            weight: Edge weight
            operation: Operation creating this edge
            constraints: Composition constraints

        Returns:
            EvidenceBundle or None if nodes don't exist
        """
        # Validate nodes exist
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning(f"Cannot add edge: nodes {source_id} or {target_id} not found")
            return None

        # Create edge
        edge = CompositionEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            created_by=operation,
            constraints=constraints or [],
        )

        self._edges.append(edge)
        self.graph.add_edge(source_id, target_id, weight=weight, edge_type=edge_type)

        # Update provenance
        self._nodes[source_id].provenance.child_node_ids.append(target_id)
        self._nodes[target_id].provenance.parent_node_ids.append(source_id)

        # Create evidence bundle
        bundle = self._create_bundle(
            operation=operation,
            input_data=f"{source_id}->{target_id}",
            output_data=edge_type,
            input_tokens=2,
            output_tokens=1,
            parameters={"weight": weight, "edge_type": edge_type},
        )

        self._evidence_store.append(bundle)
        self._operation_count += 1

        return bundle

    def get_node(self, node_id: str) -> Optional[AuditedSemanticNode]:
        """Get node by ID and record access"""
        node = self._nodes.get(node_id)
        if node:
            node.record_access()
        return node

    def get_node_with_history(
        self,
        node_id: str
    ) -> Tuple[Optional[AuditedSemanticNode], List[EvidenceBundle]]:
        """
        Get node with its evidence history.

        Args:
            node_id: Node ID

        Returns:
            Tuple of (node, list of related evidence bundles)
        """
        node = self.get_node(node_id)
        if not node:
            return None, []

        # Find all bundles mentioning this node
        related_bundles = []
        for bundle in self._evidence_store:
            if node_id in str(bundle.parameters) or node_id in bundle.output_hash:
                related_bundles.append(bundle)

        return node, related_bundles

    def get_edges_for_node(self, node_id: str) -> List[CompositionEdge]:
        """Get all edges connected to a node"""
        return [
            e for e in self._edges
            if e.source_id == node_id or e.target_id == node_id
        ]

    def update_quality(
        self,
        node_id: str,
        quality_score: float,
        operation: str = "quality_update"
    ) -> bool:
        """
        Update quality score for a node.

        Args:
            node_id: Node ID
            quality_score: New quality score (0.0-1.0)
            operation: Operation updating quality

        Returns:
            True if update succeeded
        """
        node = self._nodes.get(node_id)
        if not node:
            return False

        node.add_quality_score(quality_score, operation)
        return True

    def get_promotion_candidates(self) -> List[AuditedSemanticNode]:
        """Get nodes eligible for promotion (high quality, frequently used)"""
        return [
            node for node in self._nodes.values()
            if node.promotion_eligible
        ]

    def get_demotion_candidates(self) -> List[AuditedSemanticNode]:
        """Get nodes that should be demoted (declining quality, rarely used)"""
        return [
            node for node in self._nodes.values()
            if (
                node.quality_history.trend == "declining" or
                (node.access_count < 2 and node.quality_history.average_score < 0.5)
            )
        ]

    def verify_composition_constraints(
        self,
        node_ids: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Verify composition constraints for a set of nodes.

        Args:
            node_ids: List of node IDs to compose

        Returns:
            Tuple of (all_valid, list of violations)
        """
        violations = []

        # Check all nodes exist
        for nid in node_ids:
            if nid not in self._nodes:
                violations.append(f"Node {nid} not found")

        if violations:
            return False, violations

        # Check edge constraints
        for edge in self._edges:
            if edge.source_id in node_ids and edge.target_id in node_ids:
                for constraint in edge.constraints:
                    if not self._check_constraint(constraint, node_ids):
                        violations.append(f"Constraint violated: {constraint}")

        return len(violations) == 0, violations

    def _check_constraint(self, constraint: str, node_ids: List[str]) -> bool:
        """Check a single composition constraint"""
        # Simple constraint checking (can be extended)
        if constraint.startswith("requires:"):
            required = constraint.split(":")[1]
            return required in node_ids
        if constraint.startswith("excludes:"):
            excluded = constraint.split(":")[1]
            return excluded not in node_ids
        return True  # Unknown constraints pass by default

    def _create_bundle(
        self,
        operation: str,
        input_data: str,
        output_data: str,
        input_tokens: int,
        output_tokens: int,
        parameters: Dict[str, Any],
    ) -> EvidenceBundle:
        """Create an evidence bundle for an operation"""
        # Get previous bundle hash for chaining
        prev_hash = None
        if len(self._evidence_store) > 0:
            prev_hash = self._evidence_store[-1].bundle_hash

        # Check basic contracts
        preconditions = ContractResult()
        preconditions.add_check("input_provided", bool(input_data))
        preconditions.add_check("operation_valid", bool(operation))

        postconditions = ContractResult()
        postconditions.add_check("output_generated", bool(output_data))

        return EvidenceBundle.create(
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            parameters=parameters,
            quality_metrics=QualityMetrics(),
            preconditions=preconditions,
            postconditions=postconditions,
            previous_bundle_hash=prev_hash,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        quality_scores = [
            node.quality_history.average_score
            for node in self._nodes.values()
            if node.quality_history.scores
        ]

        access_counts = [node.access_count for node in self._nodes.values()]

        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "evidence_count": self.evidence_count,
            "operation_count": self._operation_count,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "avg_access_count": sum(access_counts) / len(access_counts) if access_counts else 0,
            "promotion_candidates": len(self.get_promotion_candidates()),
            "demotion_candidates": len(self.get_demotion_candidates()),
            "chain_valid": self._evidence_store.verify_chain()[0],
        }

    def export_for_visualization(self) -> Dict[str, Any]:
        """Export graph data for visualization"""
        nodes = []
        for node in self._nodes.values():
            nodes.append({
                "id": node.node_id,
                "importance": node.importance,
                "access_count": node.access_count,
                "quality_score": node.quality_history.average_score,
                "text_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
            })

        edges = []
        for edge in self._edges:
            edges.append({
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.edge_type,
                "weight": edge.weight,
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": self.get_statistics(),
        }

    def clear(self) -> None:
        """Clear all data (for testing)"""
        self._nodes.clear()
        self._edges.clear()
        self.graph.clear()
        self._evidence_store.clear()
        self._operation_count = 0
