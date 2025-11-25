"""
Detection Handler Functions

This module contains handler functions for blind spot and hallucination detection:
- check_blind_spots: Analyze if AI response missed critical context
- detect_hallucination: Check if response is grounded in source material

These handlers implement self-correcting context loops and response validation
to ensure fidelity and prevent fabrication.
"""

import logging
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


def handle_check_blind_spots(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_blind_spots tool call.

    Analyzes if AI response missed critical context by comparing response
    embedding to ALL nodes in the document. If relevant content was not
    retrieved, alerts and suggests auto-injection.

    This implements the 'Self-Correcting Context Loop' pattern.

    Args:
        context: Server context containing blind_spot_detector
        args: Tool arguments with 'ai_response', 'file_id', and 'retrieved_nodes'

    Returns:
        Formatted blind spot report with auto-correction suggestions if needed
    """
    ai_response = args["ai_response"]
    file_id = args["file_id"]
    retrieved_nodes = args["retrieved_nodes"]
    blind_spot_detector = context["blind_spot_detector"]

    logger.info(f"Checking blind spots for response about {file_id}")

    report = blind_spot_detector.analyze_response(ai_response, file_id, retrieved_nodes)

    result = blind_spot_detector.format_report(report)

    # If critical blind spots found, auto-suggest retrieval
    if report.auto_inject:
        result += "\n\n🔧 AUTO-CORRECTION SUGGESTED:\n"
        result += f"Retrieve these nodes: {report.auto_inject}\n"
        result += f"Command: modulate_region({report.auto_inject}, 'RAW')"

    return result


def handle_detect_hallucination(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle detect_hallucination tool call.

    Checks if a response is grounded in source material by comparing response
    embedding to document graph. Flags responses with low similarity to all
    nodes (possible fabrication).

    Args:
        context: Server context containing halo_detector
        args: Tool arguments with 'ai_response' and 'file_id'

    Returns:
        Hallucination detection report with warnings or confirmation
    """
    ai_response = args["ai_response"]
    file_id = args["file_id"]
    halo_detector = context["halo_detector"]

    logger.info(f"Checking for hallucination in response about {file_id}")

    is_hallucinating, warnings = halo_detector.detect_hallucination(ai_response, file_id)

    if is_hallucinating:
        result = "🚨 HALLUCINATION ALERT 🚨\n\n"
        result += "The response may contain fabricated information:\n"
        for warning in warnings:
            result += f"  • {warning}\n"
        result += "\nRecommendation: Re-examine source material and regenerate response."
    else:
        result = "✅ Response appears grounded in source material.\n"
        result += "No hallucination detected."

    return result
