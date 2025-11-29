"""
Detection Handler Functions

This module contains handler functions for blind spot and hallucination detection:
- check_blind_spots: Analyze if AI response missed critical context
- detect_hallucination: Check if response is grounded in source material

These handlers implement self-correcting context loops and response validation
to ensure fidelity and prevent fabrication.

Version: 0.7.0 - Added file_id validation and rate limiting
"""

import logging
import re
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context
from ..rate_limiter import RATE_LIMITERS
from ..error_types import RateLimitExceededError

logger = logging.getLogger("semantic-modulator")


def _validate_file_id(file_id: str, compressor: Any) -> None:
    """
    Validate file_id parameter for security and existence.

    Args:
        file_id: Document identifier to validate
        compressor: SemanticCompressor instance with documents dict

    Raises:
        ValueError: If file_id is invalid or document not found

    Security checks (v0.7.0):
        - Empty/whitespace-only file_id rejected
        - Path traversal sequences (../, ..\\) rejected
        - Null bytes rejected
        - Document existence verified in compressor
    """
    # Check for empty or whitespace-only
    if not file_id or not file_id.strip():
        raise ValueError(
            "file_id cannot be empty.\n"
            "Tip: Use ingest_context() to add a document first, then use the returned file_id."
        )

    # Check for path traversal attempts (CWE-22 prevention)
    path_traversal_pattern = re.compile(r"(\.\.[/\\]|[/\\]\.\.|\.\.)")
    if path_traversal_pattern.search(file_id):
        raise ValueError(
            f"Invalid file_id: '{file_id}' contains path traversal sequences.\n"
            "Tip: Use the file_id returned by ingest_context() directly without modification."
        )

    # Check for null bytes (injection prevention)
    if "\x00" in file_id:
        raise ValueError(
            "Invalid file_id: contains null bytes.\n"
            "Tip: Use the file_id returned by ingest_context() directly without modification."
        )

    # Check document existence
    if file_id not in compressor.graphs:
        available_docs = list(compressor.graphs.keys())[:5]  # Show first 5
        available_str = ", ".join(f"'{d}'" for d in available_docs) if available_docs else "(none)"
        raise ValueError(
            f"Document '{file_id}' not found.\n"
            f"[TIP] Available documents: {available_str}\n"
            "Tip: Use ingest_context() to add the document first."
        )


async def handle_check_blind_spots(context: HandlerContext, args: Dict[str, Any]) -> str:
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

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for detection operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 detection operations/second to prevent resource exhaustion."
        )

    ai_response = args["ai_response"]
    file_id = args["file_id"]
    retrieved_nodes = args["retrieved_nodes"]
    blind_spot_detector = context["blind_spot_detector"]
    compressor = context["compressor"]

    # Validate file_id (v0.7.0 security hardening)
    _validate_file_id(file_id, compressor)

    logger.info(f"Checking blind spots for response about {file_id}")

    report = blind_spot_detector.analyze_response(ai_response, file_id, retrieved_nodes)

    result = blind_spot_detector.format_report(report)

    # If critical blind spots found, auto-suggest retrieval
    if report.auto_inject:
        result += "\n\n[FIX] AUTO-CORRECTION SUGGESTED:\n"
        result += f"Retrieve these nodes: {report.auto_inject}\n"
        result += f"Command: modulate_region({report.auto_inject}, 'RAW')"

    return result


async def handle_detect_hallucination(context: HandlerContext, args: Dict[str, Any]) -> str:
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

    Raises:
        ValueError: If file_id is invalid or document not found
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for detection operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 detection operations/second to prevent resource exhaustion."
        )

    ai_response = args["ai_response"]
    file_id = args["file_id"]
    halo_detector = context["halo_detector"]
    compressor = context["compressor"]

    # Validate file_id (v0.7.0 security hardening)
    _validate_file_id(file_id, compressor)

    logger.info(f"Checking for hallucination in response about {file_id}")

    is_hallucinating, warnings = halo_detector.detect_hallucination(ai_response, file_id)

    if is_hallucinating:
        result = "[ALERT] HALLUCINATION ALERT [ALERT]\n\n"
        result += "The response may contain fabricated information:\n"
        for warning in warnings:
            result += f"  -{warning}\n"
        result += "\nRecommendation: Re-examine source material and regenerate response."
    else:
        result = "[OK] Response appears grounded in source material.\n"
        result += "No hallucination detected."

    return result
