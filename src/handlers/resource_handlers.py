"""
Resource Management Handler Functions

This module contains handler functions for resource monitoring and health checks:
- check_resource_health: Check resource usage and system health
- create_progress_bar: Helper function for text-based progress bars

These handlers provide proactive monitoring of storage, memory, and document
count metrics with warnings and recommendations.
"""

import logging
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


def handle_check_resource_health(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_resource_health tool call.

    Checks resource usage and system health, returning storage, memory, and
    document count metrics with proactive warnings and recommendations.

    Args:
        context: Server context containing resource_manager
        args: Tool arguments (unused)

    Returns:
        Formatted resource health report with metrics, warnings, and recommendations
    """
    resource_manager = context["resource_manager"]
    health = resource_manager.check_health()

    metrics = health["metrics"]
    warnings = health["warnings"]
    recommendations = health["recommendations"]
    healthy = health["healthy"]

    lines = ["💾 Resource Health Check", "=" * 70, ""]

    # Storage metrics
    storage_pct = metrics["storage_usage_pct"]
    storage_bar = create_progress_bar(storage_pct)
    lines.append("📊 Storage Usage:")
    lines.append(
        f"   {metrics['storage_mb']:.1f} MB / {metrics['storage_limit_mb']:.1f} MB ({storage_pct:.1f}%)"
    )
    lines.append(f"   {storage_bar}")
    lines.append("")

    # Document count
    doc_pct = metrics["document_usage_pct"]
    doc_bar = create_progress_bar(doc_pct)
    lines.append("📄 Document Count:")
    lines.append(
        f"   {metrics['document_count']} / {metrics['document_limit']} documents ({doc_pct:.1f}%)"
    )
    lines.append(f"   {doc_bar}")
    lines.append("")

    # Memory (if available)
    if metrics["memory_mb"] is not None:
        memory_pct = (metrics["memory_mb"] / metrics["memory_limit_mb"]) * 100
        memory_bar = create_progress_bar(memory_pct)
        lines.append("🧠 Memory Usage:")
        lines.append(
            f"   {metrics['memory_mb']:.1f} MB / {metrics['memory_limit_mb']:.1f} MB ({memory_pct:.1f}%)"
        )
        lines.append(f"   {memory_bar}")
        lines.append("")

    # Overall status
    lines.append("=" * 70)
    if healthy:
        lines.append("Status: ✅ Healthy - All systems operating normally")
    else:
        lines.append("Status: ⚠️  Warnings Detected - Review below")

    # Warnings
    if warnings:
        lines.append("")
        lines.append("⚠️  Warnings:")
        for warning in warnings:
            lines.append(f"  • {warning}")

    # Recommendations
    if recommendations:
        lines.append("")
        lines.append("💡 Recommendations:")
        for rec in recommendations:
            lines.append(f"  • {rec}")

    return "\n".join(lines)


def create_progress_bar(percentage: float, width: int = 40) -> str:
    """
    Create a text-based progress bar.

    Helper function that generates a visual progress bar with appropriate
    status indicators based on usage percentage.

    Args:
        percentage: Usage percentage (0-100)
        width: Character width of the bar (default: 40)

    Returns:
        Formatted progress bar string with status indicator

    Examples:
        >>> create_progress_bar(25.0)
        '[██████████░░░░░░░░░░░░░░░░░░░░] ✅ 25%'

        >>> create_progress_bar(85.0)
        '[██████████████████████████████████░░░░░░] ⚠️  85%'

        >>> create_progress_bar(100.0)
        '[████████████████████████████████████████] 🔴 FULL'
    """
    filled = int((percentage / 100) * width)
    empty = width - filled

    if percentage >= 100:
        bar = "█" * width
        return f"[{bar}] 🔴 FULL"
    elif percentage >= 80:
        bar = "█" * filled + "░" * empty
        return f"[{bar}] ⚠️  {percentage:.0f}%"
    else:
        bar = "█" * filled + "░" * empty
        return f"[{bar}] ✅ {percentage:.0f}%"
