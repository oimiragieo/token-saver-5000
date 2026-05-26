"""
ACE (Agentic Context Engineering) MCP Tool Handlers

Exposes the ACE framework through MCP tools for evolving contexts as playbooks.

Available Tools:
1. ace_generate - Generate reasoning trajectory for a task
2. ace_reflect - Extract insights from a trajectory
3. ace_curate - Integrate insights into playbook
4. ace_grow_context - Manually add bullets to playbook
5. ace_refine_context - Update bullet performance based on feedback
6. ace_get_playbook - Retrieve current playbook state

Integration with Token Saver 5000:
- ACE playbooks guide semantic node selection
- Delta updates enable incremental context evolution
- Grow-and-refine prevents context collapse during compression

References:
- Paper: Agentic Context Engineering (arXiv:2510.04618v1)
- Core Implementation: src/ace_framework.py
"""

import json
import logging
from typing import Dict, Any, List

from ..types import HandlerContext
from ..rate_limiter import RATE_LIMITERS
from ..error_types import RateLimitExceededError
from ..ace_framework import (
    ACEFramework,
    ACEContext,
    BulletType,
    ACEBullet,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _get_or_create_context(
    ace_contexts: Dict[str, ACEContext],
    ace_framework: ACEFramework,
    context_id: str,
) -> ACEContext:
    """
    Get existing ACE context or create new one if not exists.

    Args:
        ace_contexts: Dictionary of context_id -> ACEContext
        ace_framework: ACE framework instance for creating new contexts
        context_id: Context identifier

    Returns:
        ACEContext instance
    """
    if context_id not in ace_contexts:
        logger.info(f"Creating new ACE context: {context_id}")
        ace_contexts[context_id] = ace_framework.create_initial_context()
    return ace_contexts[context_id]


def _calculate_avg_confidence(items: List[Dict[str, Any]]) -> float:
    """
    Calculate average confidence from list of items with 'confidence' field.

    Args:
        items: List of dictionaries containing 'confidence' key

    Returns:
        Average confidence value, or 0.0 if empty list
    """
    if not items:
        return 0.0
    return sum(item["confidence"] for item in items) / len(items)


def _build_success_response(data: Dict[str, Any]) -> str:
    """
    Build JSON success response with consistent formatting.

    Args:
        data: Response data dictionary

    Returns:
        JSON string with status="success" and formatted data
    """
    return json.dumps({"status": "success", **data}, indent=2)


def _build_error_response(error_msg: str, operation: str = "ACE operation") -> str:
    """
    Build JSON error response with logging.

    Args:
        error_msg: Error message to include in response
        operation: Operation description for logging

    Returns:
        JSON string with status="error" and error message
    """
    logger.error(f"Error in {operation}: {error_msg}", exc_info=True)
    return json.dumps({"status": "error", "message": error_msg})


def _add_bullet_to_context(
    ace_context: ACEContext, bullet_data: Dict[str, Any], text_model: Any
) -> None:
    """
    Create and add a bullet to an ACE context.

    Args:
        ace_context: ACE context to add bullet to
        bullet_data: Dictionary with 'text', 'bullet_type', and optional 'confidence'
        text_model: Sentence transformer model for creating embeddings
    """
    text = bullet_data["text"]
    raw_bullet_type = bullet_data.get("bullet_type")
    if not raw_bullet_type:
        raise ValueError("bullet_type is required for each bullet (got missing or empty key)")
    bullet_type = BulletType(raw_bullet_type)
    confidence = bullet_data.get("confidence", 0.5)

    # Create embedding
    embedding = text_model.encode(text, convert_to_numpy=True)

    # Create and add bullet
    bullet = ACEBullet(
        text=text,
        bullet_type=bullet_type,
        embedding=embedding,
        confidence=confidence,
        source="manual",
    )
    ace_context.add_bullet(bullet, delta_description=f"Manually added: {text[:50]}...")


def _update_bullets_performance(
    ace_context: ACEContext,
    bullet_ids: List[str],
    success: bool,
    confidence_boost: float,
    context_id: str,
) -> tuple[int, List[float]]:
    """
    Update performance metrics for multiple bullets.

    Args:
        ace_context: ACE context containing bullets
        bullet_ids: List of bullet IDs to update
        success: Whether operation using these bullets succeeded
        confidence_boost: Amount to adjust confidence
        context_id: Context ID for logging

    Returns:
        Tuple of (updated_count, list of updated confidences)
    """
    updated_count = 0
    confidences = []

    for bullet_id in bullet_ids:
        if bullet_id in ace_context.bullets:
            bullet = ace_context.bullets[bullet_id]
            bullet.update_performance(success=success, confidence_boost=confidence_boost)
            updated_count += 1
            confidences.append(bullet.confidence)
        else:
            logger.warning(f"Bullet {bullet_id} not found in context {context_id}")

    return updated_count, confidences


def _filter_and_serialize_bullets(
    ace_context: ACEContext,
    include_embeddings: bool,
    min_confidence: float | None,
    bullet_type_filter: str | None,
) -> List[Dict[str, Any]]:
    """
    Filter and serialize bullets based on criteria.

    Args:
        ace_context: ACE context containing bullets
        include_embeddings: Whether to include embedding data
        min_confidence: Minimum confidence threshold (None = no filter)
        bullet_type_filter: Bullet type to filter by (None = no filter)

    Returns:
        List of serialized bullet dictionaries
    """
    bullets = []
    for bullet_id, bullet in ace_context.bullets.items():
        # Apply filters
        if min_confidence and bullet.confidence < min_confidence:
            continue
        if bullet_type_filter and bullet.bullet_type.value != bullet_type_filter:
            continue

        bullet_dict = bullet.to_display_dict()
        # Include bullet_id so callers can use it with ace_refine_context.
        # to_display_dict() intentionally strips it for cache-friendliness, but the
        # MCP-facing playbook MUST expose IDs to make refinement usable end-to-end.
        bullet_dict["bullet_id"] = bullet_id

        # Add embedding only when explicitly requested
        if include_embeddings and hasattr(bullet, "embedding"):
            bullet_dict["embedding"] = bullet.embedding.tolist()

        bullets.append(bullet_dict)

    return bullets


async def handle_ace_generate(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Generate reasoning trajectory for a task using ACE playbook.

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - task (str): The task or query to reason about
            - context_id (str, optional): ID of existing ACE context (default: "default")
            - max_steps (int, optional): Maximum trajectory steps (default: 5)
            - top_k_bullets (int, optional): Bullets to consider per step (default: 5)

    Returns:
        JSON string with trajectory steps

    Example:
        Input: {"task": "Explain quantum error correction", "context_id": "quantum_domain"}
        Output: {
            "status": "success",
            "trajectory": [...],
            "context_id": "quantum_domain",
            "stats": {"total_steps": 3, "avg_confidence": 0.85}
        }
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        task = args["task"]
        context_id = args.get("context_id", "default")
        max_steps = args.get("max_steps", 5)
        top_k_bullets = args.get("top_k_bullets", 5)

        # Get or create context
        ace_framework = context["ace_framework"]
        ace_contexts = context["ace_contexts"]
        ace_context = _get_or_create_context(ace_contexts, ace_framework, context_id)

        # Generate trajectory
        trajectory = ace_framework.generator.generate_trajectory(
            task=task,
            context=ace_context,
            max_steps=max_steps,
            top_k_bullets=top_k_bullets,
        )

        return _build_success_response(
            {
                "trajectory": trajectory,
                "context_id": context_id,
                "stats": {
                    "total_steps": len(trajectory),
                    "avg_confidence": _calculate_avg_confidence(trajectory),
                },
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE trajectory generation")


async def handle_ace_reflect(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Reflect on a trajectory to extract insights.

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - trajectory (list): Generated reasoning trajectory
            - outcome (str): What actually happened
            - success (bool): Whether the trajectory led to success
            - context_id (str, optional): Context ID (default: "default")

    Returns:
        JSON string with extracted insights

    Example:
        Input: {"trajectory": [...], "outcome": "Solved", "success": true}
        Output: {
            "status": "success",
            "insights": [...],
            "stats": {"total_insights": 2, "avg_confidence": 0.7}
        }
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        trajectory = args["trajectory"]
        outcome = args["outcome"]
        success = args["success"]
        context_id = args.get("context_id", "default")

        # Reflect on trajectory
        ace_framework = context["ace_framework"]
        insights = ace_framework.reflector.reflect_on_trajectory(
            trajectory=trajectory, outcome=outcome, success=success
        )

        return _build_success_response(
            {
                "insights": insights,
                "context_id": context_id,
                "stats": {
                    "total_insights": len(insights),
                    "avg_confidence": _calculate_avg_confidence(insights),
                },
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE trajectory reflection")


async def handle_ace_curate(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Curate insights into playbook via delta updates.

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - insights (list): Insights from reflection
            - context_id (str, optional): Context ID (default: "default")
            - max_bullets (int, optional): Maximum bullets (triggers pruning)

    Returns:
        JSON string with updated context stats

    Example:
        Input: {"insights": [...], "context_id": "quantum_domain"}
        Output: {
            "status": "success",
            "version": 5,
            "total_bullets": 12,
            "deltas_applied": 2
        }
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        insights = args["insights"]
        context_id = args.get("context_id", "default")
        max_bullets = args.get("max_bullets")

        # Get or create context
        ace_framework = context["ace_framework"]
        ace_contexts = context["ace_contexts"]
        ace_context = _get_or_create_context(ace_contexts, ace_framework, context_id)
        initial_version = ace_context.version

        # Curate insights
        updated_context = ace_framework.curator.curate_insights(
            context=ace_context, insights=insights, max_bullets=max_bullets
        )
        ace_contexts[context_id] = updated_context

        return _build_success_response(
            {
                "context_id": context_id,
                "version": updated_context.version,
                "total_bullets": len(updated_context.bullets),
                "deltas_applied": updated_context.version - initial_version,
                "stats": updated_context.get_performance_stats(),
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE insight curation")


async def handle_ace_grow_context(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Manually add bullets to playbook (grow operation).

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - bullets (list): List of bullet dictionaries with:
                - text (str): Bullet content
                - bullet_type (str): Type (principle/strategy/tactic/constraint/preference/learned)
                - confidence (float, optional): Initial confidence (default: 0.5)
            - context_id (str, optional): Context ID (default: "default")

    Returns:
        JSON string with updated context stats

    Example:
        Input: {"bullets": [{"text": "...", "bullet_type": "strategy"}], "context_id": "quantum"}
        Output: {"status": "success", "bullets_added": 1, "total_bullets": 13}
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        bullets_data = args["bullets"]
        context_id = args.get("context_id", "default")

        # Get or create context
        ace_framework = context["ace_framework"]
        ace_contexts = context["ace_contexts"]
        ace_context = _get_or_create_context(ace_contexts, ace_framework, context_id)
        text_model = ace_framework.text_model

        # Add bullets to context
        for bullet_data in bullets_data:
            _add_bullet_to_context(ace_context, bullet_data, text_model)

        return _build_success_response(
            {
                "context_id": context_id,
                "bullets_added": len(bullets_data),
                "total_bullets": len(ace_context.bullets),
                "version": ace_context.version,
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE context growth")


async def handle_ace_refine_context(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Update bullet performance based on feedback (refine operation).

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - bullet_ids (list): List of bullet IDs to update
            - success (bool): Whether these bullets led to success
            - confidence_boost (float, optional): Adjustment amount (default: 0.05)
            - context_id (str, optional): Context ID (default: "default")

    Returns:
        JSON string with update results

    Example:
        Input: {"bullet_ids": ["abc-123"], "success": true, "confidence_boost": 0.1}
        Output: {"status": "success", "bullets_updated": 1, "avg_confidence_after": 0.85}
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        bullet_ids = args["bullet_ids"]
        success = args["success"]
        confidence_boost = args.get("confidence_boost", 0.05)
        context_id = args.get("context_id", "default")

        # Get context
        ace_contexts = context["ace_contexts"]
        if context_id not in ace_contexts:
            return _build_error_response(
                f"Context '{context_id}' not found", "ACE context refinement"
            )

        ace_context = ace_contexts[context_id]
        updated_count, confidences = _update_bullets_performance(
            ace_context, bullet_ids, success, confidence_boost, context_id
        )
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return _build_success_response(
            {
                "context_id": context_id,
                "bullets_updated": updated_count,
                "avg_confidence_after": avg_confidence,
                "success_rate": (
                    ace_context.get_performance_stats()["avg_success_rate"]
                    if ace_context.bullets
                    else 0.0
                ),
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE context refinement")


async def handle_ace_get_playbook(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Retrieve current ACE playbook state.

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - context_id (str, optional): Context ID (default: "default")
            - include_embeddings (bool, optional): Include bullet embeddings (default: false)
            - min_confidence (float, optional): Filter bullets below this confidence
            - bullet_type (str, optional): Filter by bullet type

    Returns:
        JSON string with playbook state

    Example:
        Input: {"context_id": "quantum_domain", "min_confidence": 0.7}
        Output: {"status": "success", "version": 5, "bullets": [...], "stats": {...}}
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        context_id = args.get("context_id", "default")
        include_embeddings = args.get("include_embeddings", False)
        min_confidence = args.get("min_confidence")
        bullet_type_filter = args.get("bullet_type")

        # Get or create context
        ace_framework = context["ace_framework"]
        ace_contexts = context["ace_contexts"]
        ace_context = _get_or_create_context(ace_contexts, ace_framework, context_id)

        # Filter and serialize bullets
        bullets = _filter_and_serialize_bullets(
            ace_context, include_embeddings, min_confidence, bullet_type_filter
        )

        # Strip volatile fields (timestamps, UUIDs) from delta history for cache-friendly output
        clean_deltas = []
        for delta in ace_context.delta_history[-5:]:
            if isinstance(delta, dict):
                clean = {k: v for k, v in delta.items() if k not in ("timestamp", "bullet_id")}
                clean_deltas.append(clean)
            else:
                clean_deltas.append(delta)

        return _build_success_response(
            {
                "context_id": context_id,
                "version": ace_context.version,
                "total_bullets": len(ace_context.bullets),
                "filtered_bullets": len(bullets),
                "bullets": bullets,
                "stats": ace_context.get_performance_stats(),
                "delta_history": clean_deltas,
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE playbook retrieval")


# ============================================================================
# Full ACE Cycle Handler
# ============================================================================


async def handle_ace_execute_cycle(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Execute complete ACE cycle: Generate → Reflect → Curate.

    This is a convenience handler that combines the three-step ACE process
    into a single tool call.

    Args:
        context: Handler context containing ace_framework and ace_contexts
        args: Tool arguments containing:
            - task (str): The task or query
            - outcome (str): What actually happened
            - success (bool): Whether the task succeeded
            - context_id (str, optional): Context ID (default: "default")
            - max_trajectory_steps (int, optional): Max trajectory steps (default: 5)

    Returns:
        JSON string with full cycle results

    Example:
        Input: {"task": "Explain entanglement", "outcome": "Explained", "success": true}
        Output: {
            "status": "success",
            "trajectory": [...],
            "version_before": 3,
            "version_after": 5,
            "deltas_applied": 2
        }
    """
    # Rate limiting (v0.7.0 security hardening)
    try:
        await RATE_LIMITERS["compression"].acquire(blocking=True)
    except RateLimitExceededError:
        raise ValueError(
            "Rate limit exceeded for ACE operations. Please retry in a moment.\n"
            "Tip: The server allows ~5 ACE operations/second to prevent resource exhaustion."
        )

    try:
        task = args["task"]
        outcome = args["outcome"]
        success = args["success"]
        context_id = args.get("context_id", "default")
        max_trajectory_steps = args.get("max_trajectory_steps", 5)

        # Get or create context
        ace_framework = context["ace_framework"]
        ace_contexts = context["ace_contexts"]
        ace_context = _get_or_create_context(ace_contexts, ace_framework, context_id)
        version_before = ace_context.version

        # Execute full ACE cycle
        updated_context, trajectory = ace_framework.execute_ace_cycle(
            task=task,
            context=ace_context,
            outcome=outcome,
            success=success,
            max_trajectory_steps=max_trajectory_steps,
        )
        ace_contexts[context_id] = updated_context

        return _build_success_response(
            {
                "context_id": context_id,
                "trajectory": trajectory,
                "version_before": version_before,
                "version_after": updated_context.version,
                "deltas_applied": updated_context.version - version_before,
                "stats": updated_context.get_performance_stats(),
            }
        )

    except Exception as e:
        return _build_error_response(str(e), "ACE cycle execution")
