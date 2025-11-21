"""
Blind Spot Detector - Self-Correcting Context Loop

Implements the "Holodeck Context" concept:
When an AI generates a response, this detector:
1. Embeds the AI's answer
2. Compares it to hidden nodes in the local graph
3. Alerts if relevant context was missed
4. Auto-injects critical missing information

This prevents hallucination by ensuring fidelity preservation.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .semantic_compressor import SemanticCompressor, SemanticNode


@dataclass
class BlindSpot:
    """Represents a detected blind spot in AI reasoning"""
    node_id: str
    similarity_to_response: float
    was_retrieved: bool
    urgency: str  # "low", "medium", "high", "critical"
    reason: str


@dataclass
class BlindSpotReport:
    """Report of detected blind spots"""
    response_analyzed: str
    total_blind_spots: int
    critical_blind_spots: int
    blind_spots: List[BlindSpot]
    recommendations: List[str]
    auto_inject: List[str]  # Node IDs to auto-inject


class BlindSpotDetector:
    """
    Detects when an AI response might be missing critical context.

    Algorithm:
    1. Embed the AI's response
    2. Compare to all nodes in the document graph
    3. Find nodes with high similarity that weren't retrieved
    4. Rank by urgency (importance × similarity)
    5. Generate alerts and recommendations
    """

    def __init__(
        self,
        compressor: SemanticCompressor,
        similarity_threshold: float = 0.70,
        urgency_threshold: float = 0.50,
    ):
        """
        Initialize blind spot detector.

        Args:
            compressor: The semantic compressor with document graphs
            similarity_threshold: Minimum similarity to consider relevant
            urgency_threshold: Combined score threshold for critical alerts
        """
        self.compressor = compressor
        self.similarity_threshold = similarity_threshold
        self.urgency_threshold = urgency_threshold

    def _calculate_urgency(self, similarity: float, importance: float) -> Tuple[str, float]:
        """
        Calculate urgency score and level.

        Urgency = similarity × importance
        - similarity: How relevant is this node to the response?
        - importance: How central is this node to the document?

        Returns:
            (urgency_level, urgency_score)
        """
        urgency_score = similarity * importance

        if urgency_score >= 0.6:
            return "critical", urgency_score
        elif urgency_score >= 0.4:
            return "high", urgency_score
        elif urgency_score >= 0.25:
            return "medium", urgency_score
        else:
            return "low", urgency_score

    def analyze_response(
        self,
        ai_response: str,
        file_id: str,
        retrieved_node_ids: List[str],
        auto_inject_threshold: float = 0.6,
    ) -> BlindSpotReport:
        """
        Analyze an AI response for blind spots.

        Args:
            ai_response: The AI's generated response
            file_id: Which document was being discussed
            retrieved_node_ids: Which nodes the AI actually saw
            auto_inject_threshold: Urgency threshold for auto-injection

        Returns:
            BlindSpotReport with detected issues and recommendations
        """
        # 1. Embed the AI response
        response_embedding = self.compressor.model.encode([ai_response])[0]

        # 2. Get all nodes for this file
        graph = self.compressor.graphs.get(file_id)
        if not graph:
            raise ValueError(f"File {file_id} not found")

        file_nodes = [
            nid for nid in graph.nodes()
            if nid.startswith(file_id)
        ]

        # 3. Compare response to each node
        blind_spots = []
        retrieved_set = set(retrieved_node_ids)

        for node_id in file_nodes:
            node = self.compressor.chunks[node_id]

            # Calculate similarity
            similarity = cosine_similarity(
                [response_embedding],
                [node.embedding]
            )[0][0]

            # Is this node relevant but missed?
            if similarity >= self.similarity_threshold:
                was_retrieved = node_id in retrieved_set
                urgency_level, urgency_score = self._calculate_urgency(
                    similarity, node.importance
                )

                # Determine reason
                if not was_retrieved and urgency_level in ["critical", "high"]:
                    reason = "High-relevance content not retrieved"
                elif not was_retrieved:
                    reason = "Potentially relevant context missed"
                elif urgency_level == "critical":
                    reason = "Critical content was retrieved (good)"
                else:
                    reason = "Relevant content was retrieved"

                blind_spot = BlindSpot(
                    node_id=node_id,
                    similarity_to_response=similarity,
                    was_retrieved=was_retrieved,
                    urgency=urgency_level,
                    reason=reason,
                )

                # Only report if it's a real blind spot (not retrieved + relevant)
                if not was_retrieved and urgency_level != "low":
                    blind_spots.append(blind_spot)

        # 4. Sort by urgency
        blind_spots.sort(key=lambda x: x.similarity_to_response, reverse=True)

        # 5. Generate recommendations
        recommendations = []
        auto_inject = []

        critical_spots = [bs for bs in blind_spots if bs.urgency == "critical"]
        high_spots = [bs for bs in blind_spots if bs.urgency == "high"]

        if critical_spots:
            recommendations.append(
                f"⚠️  CRITICAL: Found {len(critical_spots)} highly relevant nodes that were NOT retrieved!"
            )
            for spot in critical_spots:
                node = self.compressor.chunks[spot.node_id]
                summary = self.compressor._generate_summary(node.text, max_length=60)
                recommendations.append(
                    f"  • [{spot.node_id}] similarity={spot.similarity_to_response:.2f}, "
                    f"importance={node.importance:.2f}: {summary}"
                )
                auto_inject.append(spot.node_id)

        if high_spots:
            recommendations.append(
                f"⚡ HIGH: Found {len(high_spots)} relevant nodes that might improve the answer"
            )
            for spot in high_spots[:3]:  # Show top 3
                node = self.compressor.chunks[spot.node_id]
                summary = self.compressor._generate_summary(node.text, max_length=60)
                recommendations.append(
                    f"  • [{spot.node_id}] similarity={spot.similarity_to_response:.2f}: {summary}"
                )

        if not blind_spots:
            recommendations.append("✅ No significant blind spots detected. Response appears well-grounded.")

        return BlindSpotReport(
            response_analyzed=ai_response[:200] + "..." if len(ai_response) > 200 else ai_response,
            total_blind_spots=len(blind_spots),
            critical_blind_spots=len(critical_spots),
            blind_spots=blind_spots,
            recommendations=recommendations,
            auto_inject=auto_inject,
        )

    def format_report(self, report: BlindSpotReport) -> str:
        """Format a blind spot report for display"""
        lines = []
        lines.append("=" * 60)
        lines.append("🔍 BLIND SPOT ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append(f"\nAnalyzed Response: {report.response_analyzed}\n")

        lines.append(f"📊 Summary:")
        lines.append(f"  • Total blind spots detected: {report.total_blind_spots}")
        lines.append(f"  • Critical blind spots: {report.critical_blind_spots}")

        if report.auto_inject:
            lines.append(f"  • Auto-injecting {len(report.auto_inject)} critical nodes\n")

        lines.append(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            lines.append(rec)

        if report.auto_inject:
            lines.append(f"\n🔧 Auto-Injection:")
            lines.append(f"The following nodes should be retrieved immediately:")
            for node_id in report.auto_inject:
                lines.append(f"  • {node_id}")
            lines.append(f"\nUse: modulate_region({report.auto_inject}, fidelity_level='RAW')")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)

    def validate_response_fidelity(
        self,
        ai_response: str,
        file_id: str,
        retrieved_node_ids: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """
        Quick validation: Is the response safe or does it need correction?

        Returns:
            (is_valid, correction_message)
        """
        report = self.analyze_response(ai_response, file_id, retrieved_node_ids)

        if report.critical_blind_spots > 0:
            correction = f"⚠️  WARNING: Response may be incomplete or inaccurate!\n"
            correction += f"Found {report.critical_blind_spots} critical blind spots.\n"
            correction += f"Recommend retrieving: {report.auto_inject}\n"
            return False, correction

        return True, None


class HaloEffectDetector:
    """
    Detects when AI might be hallucinating based on:
    1. Claiming content that doesn't exist in any node
    2. Contradicting high-importance nodes
    3. Overconfident statements without evidence
    """

    def __init__(self, compressor: SemanticCompressor):
        self.compressor = compressor

    def detect_hallucination(
        self,
        ai_response: str,
        file_id: str,
        confidence_threshold: float = 0.3,
    ) -> Tuple[bool, List[str]]:
        """
        Detect potential hallucination.

        Algorithm:
        1. Extract key claims from AI response (simple: look for definitive statements)
        2. Check if each claim has supporting evidence in the graph
        3. Flag claims with low similarity to any node

        Returns:
            (is_hallucinating, list_of_suspicious_claims)
        """
        # This is a simplified version - real implementation would use
        # more sophisticated claim extraction

        # Embed the response
        response_embedding = self.compressor.model.encode([ai_response])[0]

        # Get all nodes for the file
        graph = self.compressor.graphs.get(file_id)
        if not graph:
            return False, []

        file_nodes = [nid for nid in graph.nodes() if nid.startswith(file_id)]

        # Find maximum similarity to any node
        max_similarities = []
        for node_id in file_nodes:
            node = self.compressor.chunks[node_id]
            similarity = cosine_similarity(
                [response_embedding],
                [node.embedding]
            )[0][0]
            max_similarities.append(similarity)

        max_sim = max(max_similarities) if max_similarities else 0

        # If the response has very low similarity to ALL nodes, it might be hallucinating
        if max_sim < confidence_threshold:
            return True, [
                f"Response has low similarity to all document nodes (max: {max_sim:.2f})",
                "AI may be generating content not present in the source document",
            ]

        return False, []
