"""
Resource Management Handler Functions

This module contains handler functions for resource monitoring and health checks:
- check_resource_health: Check resource usage and system health
- check_environment: Check environment health (models, cache, stale docs)
- create_progress_bar: Helper function for text-based progress bars

These handlers provide proactive monitoring of storage, memory, and document
count metrics with warnings and recommendations.

v0.9.0: Added check_environment handler for comprehensive environment status.
"""

import json
import logging
import os
from typing import Any, Dict
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")


async def handle_check_resource_health(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_resource_health tool call.

    Checks resource usage and system health, returning storage, memory, and
    document count metrics with proactive warnings and recommendations.

    v0.8.0 audit fix: Converted to async and uses check_health_async() to avoid
    blocking event loop when ResourceManager uses threading.Lock internally.

    Args:
        context: Server context containing resource_manager
        args: Tool arguments (unused)

    Returns:
        Formatted resource health report with metrics, warnings, and recommendations
    """
    resource_manager = context["resource_manager"]
    # v0.8.0 audit fix: use async wrapper to avoid blocking event loop
    health = await resource_manager.check_health_async()

    metrics = health["metrics"]
    warnings = health["warnings"]
    recommendations = health["recommendations"]
    healthy = health["healthy"]

    lines = ["[STORAGE] Resource Health Check", "=" * 70, ""]

    # Storage metrics
    storage_pct = metrics["storage_usage_pct"]
    storage_bar = create_progress_bar(storage_pct)
    lines.append("[STATS] Storage Usage:")
    lines.append(
        f"   {metrics['storage_mb']:.1f} MB / {metrics['storage_limit_mb']:.1f} MB ({storage_pct:.1f}%)"
    )
    lines.append(f"   {storage_bar}")
    lines.append("")

    # Document count
    doc_pct = metrics["document_usage_pct"]
    doc_bar = create_progress_bar(doc_pct)
    lines.append("[DOCS] Document Count:")
    lines.append(
        f"   {metrics['document_count']} / {metrics['document_limit']} documents ({doc_pct:.1f}%)"
    )
    lines.append(f"   {doc_bar}")
    lines.append("")

    # Memory (if available)
    if metrics["memory_mb"] is not None:
        memory_pct = (metrics["memory_mb"] / metrics["memory_limit_mb"]) * 100
        memory_bar = create_progress_bar(memory_pct)
        lines.append("[MEM] Memory Usage:")
        lines.append(
            f"   {metrics['memory_mb']:.1f} MB / {metrics['memory_limit_mb']:.1f} MB ({memory_pct:.1f}%)"
        )
        lines.append(f"   {memory_bar}")
        lines.append("")

    # Overall status
    lines.append("=" * 70)
    if healthy:
        lines.append("Status: [OK] Healthy - All systems operating normally")
    else:
        lines.append("Status: [WARN]  Warnings Detected - Review below")

    # Warnings
    if warnings:
        lines.append("")
        lines.append("[WARN]  Warnings:")
        for warning in warnings:
            lines.append(f"  -{warning}")

    # Recommendations
    if recommendations:
        lines.append("")
        lines.append("[TIP] Recommendations:")
        for rec in recommendations:
            lines.append(f"  -{rec}")

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
        '[██████████░░░░░░░░░░░░░░░░░░░░] [OK] 25%'

        >>> create_progress_bar(85.0)
        '[██████████████████████████████████░░░░░░] [WARN]  85%'

        >>> create_progress_bar(100.0)
        '[████████████████████████████████████████] [CRIT] FULL'
    """
    filled = int((percentage / 100) * width)
    empty = width - filled

    if percentage >= 100:
        bar = "█" * width
        return f"[{bar}] [CRIT] FULL"
    elif percentage >= 80:
        bar = "█" * filled + "░" * empty
        return f"[{bar}] [WARN]  {percentage:.0f}%"
    else:
        bar = "█" * filled + "░" * empty
        return f"[{bar}] [OK] {percentage:.0f}%"


async def handle_check_environment(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle check_environment MCP tool (v0.9.0).

    Provides comprehensive environment health status including:
    - Embedding models loaded and their status
    - Memory usage estimates
    - Cache statistics and hit ratios
    - Disk space availability
    - Stale document detection
    - Recommendations for optimization

    Args:
        context: Server context containing compressor, sync_manager, etc.
        args: Tool arguments (unused)

    Returns:
        JSON string with comprehensive environment status
    """
    compressor = context["compressor"]
    sync_manager = context["sync_manager"]

    # Gather environment information
    status = "healthy"
    warnings = []
    recommendations = []

    # 1. Embedding model status
    models_loaded = []
    estimated_memory_mb = 0

    # Check text model
    if hasattr(compressor, "_text_compressor"):
        # Using CodeCompressionAdapter
        text_compressor = compressor._text_compressor
        if hasattr(text_compressor, "model") and text_compressor.model is not None:
            models_loaded.append(
                {
                    "name": "all-MiniLM-L6-v2",
                    "type": "text",
                    "status": "loaded",
                }
            )
            estimated_memory_mb += 80  # ~80MB for MiniLM

        # Check code model
        if hasattr(compressor, "_code_compressor") and compressor._code_compressor is not None:
            models_loaded.append(
                {
                    "name": "microsoft/codebert-base",
                    "type": "code",
                    "status": "loaded",
                }
            )
            estimated_memory_mb += 400  # ~400MB for CodeBERT
        else:
            models_loaded.append(
                {
                    "name": "microsoft/codebert-base",
                    "type": "code",
                    "status": "not_loaded (lazy)",
                }
            )
    else:
        # Using SemanticCompressor directly
        if hasattr(compressor, "model") and compressor.model is not None:
            models_loaded.append(
                {
                    "name": "all-MiniLM-L6-v2",
                    "type": "text",
                    "status": "loaded",
                }
            )
            estimated_memory_mb += 80

    # 2. Determine embedding tier
    embedding_tier = "standard"
    if hasattr(compressor, "_text_compressor"):
        tc = compressor._text_compressor
        if hasattr(tc, "_current_tier"):
            embedding_tier = (
                tc._current_tier.value
                if hasattr(tc._current_tier, "value")
                else str(tc._current_tier)
            )

    # 3. Cache statistics
    cache_stats = {}
    cache_hit_ratio = 0.0
    if hasattr(compressor, "_text_compressor"):
        tc = compressor._text_compressor
        if hasattr(tc, "embedding_cache") and tc.embedding_cache:
            cache = tc.embedding_cache
            if hasattr(cache, "get_stats"):
                cache_stats = cache.get_stats()
                hits = cache_stats.get("hits", 0)
                misses = cache_stats.get("misses", 0)
                if hits + misses > 0:
                    cache_hit_ratio = hits / (hits + misses)

    # 4. Check for stale documents
    stale_documents = []
    file_metadata = sync_manager.export_metadata()
    for doc_id, metadata in file_metadata.items():
        if metadata.get("file_path"):
            try:
                file_path = metadata["file_path"]
                if os.path.exists(file_path):
                    current_mtime = os.path.getmtime(file_path)
                    cached_mtime = metadata.get("mtime", 0)
                    if current_mtime > cached_mtime:
                        stale_documents.append(doc_id)
            except Exception:
                pass  # Ignore errors checking individual files

    if stale_documents:
        status = "degraded"
        warnings.append(f"{len(stale_documents)} documents are stale and need refresh")

    # 5. Disk space check
    disk_space_mb = None
    try:
        import shutil

        total, used, free = shutil.disk_usage(".")
        disk_space_mb = free // (1024 * 1024)
        if disk_space_mb < 100:
            status = "degraded"
            warnings.append(f"Low disk space: {disk_space_mb}MB free")
            recommendations.append("Free up disk space or delete unused documents")
    except Exception:
        pass

    # 6. Document statistics
    total_documents = len(compressor.graphs) if hasattr(compressor, "graphs") else 0
    total_chunks = len(compressor.chunks) if hasattr(compressor, "chunks") else 0

    # 7. Generate recommendations
    if cache_hit_ratio < 0.5 and total_chunks > 100:
        recommendations.append("Consider increasing cache size for better performance")

    if embedding_tier == "standard" and estimated_memory_mb > 200:
        recommendations.append("Consider ONNX tier for lower memory usage")

    if not models_loaded:
        status = "unhealthy"
        warnings.append("No embedding models loaded")

    # Build response
    response = {
        "status": status,
        "embedding_tier": embedding_tier,
        "models_loaded": models_loaded,
        "estimated_memory_mb": estimated_memory_mb,
        "cache_hit_ratio": round(cache_hit_ratio, 3),
        "cache_stats": cache_stats,
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "stale_documents": stale_documents[:10],  # Limit to first 10
        "stale_count": len(stale_documents),
    }

    if disk_space_mb is not None:
        response["disk_space_mb"] = disk_space_mb

    if warnings:
        response["warnings"] = warnings

    if recommendations:
        response["recommendations"] = recommendations

    if not warnings and not recommendations:
        response["message"] = "All systems operating normally"

    return json.dumps(response, indent=2)
