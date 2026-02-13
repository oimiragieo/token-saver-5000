"""
Resource Management Handler Functions

This module contains handler functions for resource monitoring and health checks:
- check_resource_health: Check resource usage and system health
- check_environment: Check environment health (models, cache, stale docs)
- should_compress: Estimate token count and recommend compression strategy
- create_progress_bar: Helper function for text-based progress bars

These handlers provide proactive monitoring of storage, memory, and document
count metrics with warnings and recommendations.

v0.9.0: Added check_environment handler for comprehensive environment status.
v0.9.1: Added should_compress handler for pre-read token estimation.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple
from ..types import HandlerContext  # TypedDict for handler context

logger = logging.getLogger("semantic-modulator")

# Token estimation constants
# Average characters per token varies by content type
CHARS_PER_TOKEN_PROSE = 4.0  # English prose averages ~4 chars/token
CHARS_PER_TOKEN_CODE = 3.5  # Code tends to be slightly more token-dense
CHARS_PER_TOKEN_DEFAULT = 3.8  # Conservative default

# Thresholds for compression recommendations (v0.9.3: removed unused COMPRESS_THRESHOLD_TOKENS)
SKIP_THRESHOLD_TOKENS = 100  # Below this, skip compression entirely
DIRECT_READ_THRESHOLD_TOKENS = 500  # Below this, read directly without compression
# Note: COMPRESS recommendation = DIRECT_READ_THRESHOLD_TOKENS or higher (500+)
STRONG_COMPRESS_THRESHOLD = 2000  # Strongly recommend compression
MUST_COMPRESS_THRESHOLD = 10000  # Must compress to fit context

# Binary file detection constants (v0.9.2)
BINARY_SNIFF_SIZE = 8192  # Read first 8KB for content sniffing
NULL_BYTE_THRESHOLD = 0.01  # >1% null bytes = likely binary

# Binary file extensions - require conversion before compression
BINARY_EXTENSIONS = {
    # Documents (need MarkItDown or similar)
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".odt",
    ".ods",
    ".odp",
    ".rtf",
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".webp",
    ".svg",
    ".ico",
    # Media
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".mkv",
    ".flac",
    ".ogg",
    ".webm",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".bz2",
    ".xz",
    # Executables
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    # Other binary
    ".dat",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".pickle",
    ".pkl",
}

# Text file extensions - can be read directly
TEXT_EXTENSIONS = {
    # Prose/documentation
    ".md",
    ".txt",
    ".rst",
    ".adoc",
    ".tex",
    ".org",
    # Code
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".r",
    ".m",
    ".mm",
    ".pl",
    ".pm",
    ".lua",
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hs",
    ".clj",
    ".cljs",
    ".jl",
    ".nim",
    ".v",
    ".zig",
    # Shell
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".bat",
    ".cmd",
    ".fish",
    # Config/Data
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".csv",
    ".tsv",
    ".ini",
    ".cfg",
    ".conf",
    ".env",
    # Web
    ".vue",
    ".svelte",
    ".astro",
    ".mdx",
    # SQL
    ".sql",
    ".prisma",
    # Makefiles and build
    ".mk",
    ".cmake",
    ".gradle",
    ".sbt",
}

# Code-like extensions - use lower chars/token ratio for token estimation (v0.9.3)
# Subset of TEXT_EXTENSIONS that typically have denser token usage
CODE_LIKE_EXTENSIONS = {
    # Programming languages
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    # Shell scripts
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    # Data/Config (typically dense syntax)
    ".sql",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    # Web frameworks
    ".vue",
    ".svelte",
}


def is_binary_content(
    file_path: str, sniff_size: int = BINARY_SNIFF_SIZE
) -> Tuple[bool, Optional[str]]:
    """
    Check if file contains binary content by sampling first N bytes.

    Uses null byte ratio heuristic: >1% null bytes indicates binary content.
    This is similar to the binaryornot library approach.

    Args:
        file_path: Path to the file to check
        sniff_size: Number of bytes to sample (default: 8192)

    Returns:
        Tuple of (is_binary, error_message):
        - (True, None) if file contains binary content
        - (False, None) if file contains text content
        - (False, "error message") if file could not be read

    v0.9.2 Hardening: Now returns error message instead of silently
    treating read errors as text (Issue 2 fix).
    """
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(sniff_size)
        if not chunk:
            return False, None  # Empty file is not binary
        null_count = chunk.count(b"\x00")
        is_binary = (null_count / len(chunk)) > NULL_BYTE_THRESHOLD
        return is_binary, None
    except (IOError, OSError) as e:
        return False, f"Cannot read file for binary detection: {e}"


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

    v0.9.3: Uses ASCII-only characters (#, -) for terminal compatibility.

    Args:
        percentage: Usage percentage (0-100)
        width: Character width of the bar (default: 40)

    Returns:
        Formatted progress bar string with status indicator

    Examples:
        >>> create_progress_bar(25.0)
        '[##########------------------------------] [OK] 25%'

        >>> create_progress_bar(85.0)
        '[##################################------] [WARN]  85%'

        >>> create_progress_bar(100.0)
        '[########################################] [CRIT] FULL'
    """
    filled = int((percentage / 100) * width)
    empty = width - filled

    # v0.9.3: Use ASCII-only characters for terminal compatibility
    if percentage >= 100:
        bar = "#" * width
        return f"[{bar}] [CRIT] FULL"
    elif percentage >= 80:
        bar = "#" * filled + "-" * empty
        return f"[{bar}] [WARN]  {percentage:.0f}%"
    else:
        bar = "#" * filled + "-" * empty
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

    enabled_tool_names = list(context.get("enabled_tool_names", []))
    response["tool_profile"] = {
        "profile": str(context.get("tool_profile", "full")),
        "enabled_tool_count": len(enabled_tool_names),
        "enabled_tools": enabled_tool_names,
    }

    if warnings:
        response["warnings"] = warnings

    if recommendations:
        response["recommendations"] = recommendations

    if not warnings and not recommendations:
        response["message"] = "All systems operating normally"

    return json.dumps(response, indent=2)


async def handle_should_compress(context: HandlerContext, args: Dict[str, Any]) -> str:
    """
    Handle should_compress MCP tool (v0.9.2).

    Estimates token count for a file WITHOUT reading it into context.
    Uses file size heuristics AND binary content detection to recommend
    the optimal ingestion workflow.

    This tool is critical for token efficiency - it allows AI to decide
    whether to use compression BEFORE wasting tokens reading the full file.

    v0.9.2: Added binary file detection with content sniffing.
    - Detects binary files by extension AND content (null byte heuristic)
    - Returns CONVERT_THEN_COMPRESS for binary files with MarkItDown suggestion
    - New fields: needs_conversion, is_text_readable, conversion_tool, reason

    Args:
        context: Server context (unused for this tool)
        args: Tool arguments:
            - file_path (str): Path to file to estimate
            - content_type (str, optional): "prose", "code", or "auto" (default)

    Returns:
        JSON with token estimate, binary detection, and compression recommendation

    Recommendations:
        - SKIP: File <100 tokens, read directly
        - DIRECT_READ: File 100-500 tokens, read directly
        - COMPRESS: File >500 tokens, use ingest_context
        - CONVERT_THEN_COMPRESS: Binary file, use MarkItDown first
    """
    file_path = args.get("file_path")
    content_type = args.get("content_type", "auto")

    if not file_path:
        return json.dumps(
            {
                "error": "file_path is required",
                "suggestion": "Provide a valid file path to assess",
                "recommendation": "UNKNOWN",
            }
        )

    # SECURITY: Validate path before any file I/O (v0.9.3 - PathValidator REQUIRED)
    # Prevents path traversal attacks (CWE-22) like ../../etc/passwd
    path_validator = context.get("path_validator")
    if not path_validator:
        # v0.9.3: Fail-fast for security - PathValidator is required
        return json.dumps(
            {
                "error": "Path validation unavailable",
                "suggestion": "Configure PathValidator in server context",
                "recommendation": "UNKNOWN",
            }
        )

    try:
        file_path = path_validator.validate(file_path)  # Returns absolute path
    except ValueError as e:
        return json.dumps(
            {
                "error": f"Path validation failed: {args.get('file_path')}",
                "suggestion": "File path must be within allowed directories",
                "details": str(e),
                "recommendation": "UNKNOWN",
            }
        )

    # Check if file exists and get size
    if not os.path.exists(file_path):
        return json.dumps(
            {
                "error": f"File not found: {file_path}",
                "suggestion": "Verify the file path exists and is accessible",
                "recommendation": "UNKNOWN",
            }
        )

    try:
        file_size_bytes = os.path.getsize(file_path)
    except OSError as e:
        return json.dumps(
            {
                "error": f"Cannot read file size: {e}",
                "suggestion": "Check file permissions and accessibility",
                "recommendation": "UNKNOWN",
            }
        )

    # Get file extension for initial classification
    ext = os.path.splitext(file_path)[1].lower()

    # v0.9.3: Short-circuit binary detection by extension BEFORE content sniffing
    # This avoids unnecessary file reads for known binary/text extensions
    is_binary = False
    detected_by = None

    if ext in BINARY_EXTENSIONS:
        # Known binary extension - skip content sniffing entirely
        is_binary = True
        detected_by = "extension"
    elif ext in TEXT_EXTENSIONS:
        # Known text extension - skip content sniffing entirely
        is_binary = False
        detected_by = "extension"
    elif file_size_bytes > 0:
        # Unknown extension - content sniffing required
        # v0.9.2 Hardening: is_binary_content returns tuple (is_binary, error_msg)
        is_binary, read_error = is_binary_content(file_path)
        if read_error:
            return json.dumps(
                {
                    "error": read_error,
                    "suggestion": "Check file permissions and accessibility",
                    "recommendation": "UNKNOWN",
                }
            )
        detected_by = "content"
    else:
        # Empty file with unknown extension - treat as text
        is_binary = False
        detected_by = "empty_file"

    # If binary file detected, return CONVERT_THEN_COMPRESS recommendation
    if is_binary:
        # Determine file type for helpful message
        if ext in {
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".odt",
            ".ods",
            ".odp",
            ".rtf",
        }:
            file_type = "document"
        elif ext in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp", ".svg", ".ico"}:
            file_type = "image"
        elif ext in {".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac", ".ogg", ".webm"}:
            file_type = "media"
        elif ext in {".zip", ".tar", ".gz", ".rar", ".7z", ".bz2", ".xz"}:
            file_type = "archive"
        elif ext in {".exe", ".dll", ".so", ".dylib", ".bin"}:
            file_type = "executable"
        else:
            file_type = "binary"

        reason = (
            f"Binary file ({file_type}) detected. "
            "Use MarkItDown (or pandoc, pdftotext) to convert to text, "
            "then call ingest_context with the converted output."
        )

        return json.dumps(
            {
                # Existing fields (backward compat)
                "file_path": file_path,
                "file_size_bytes": file_size_bytes,
                "estimated_tokens": 0,  # Cannot estimate for binary
                "content_type_detected": file_type,
                "recommendation": "CONVERT_THEN_COMPRESS",
                "reason": reason,
                "potential_token_savings": 0,
                "estimated_compression_ratio": "N/A",
                # New fields (v0.9.2)
                "needs_conversion": True,
                "is_text_readable": False,
                "conversion_tool": "MarkItDown",
                "detected_by": detected_by,
            },
            indent=2,
        )

    # Text file processing
    # Estimate character count (roughly 1 byte per char for UTF-8 text)
    estimated_chars = file_size_bytes

    # Determine chars-per-token ratio based on content type
    if content_type == "code":
        chars_per_token = CHARS_PER_TOKEN_CODE
        detected_type = "code"
    elif content_type == "prose":
        chars_per_token = CHARS_PER_TOKEN_PROSE
        detected_type = "prose"
    elif content_type == "auto":
        # Auto-detect based on file extension (v0.9.3: use module-level constant)
        if ext in CODE_LIKE_EXTENSIONS:
            chars_per_token = CHARS_PER_TOKEN_CODE
            detected_type = "code"
        else:
            chars_per_token = CHARS_PER_TOKEN_DEFAULT
            detected_type = "prose/mixed"
    else:
        chars_per_token = CHARS_PER_TOKEN_DEFAULT
        detected_type = "unknown"

    # Estimate token count
    estimated_tokens = int(estimated_chars / chars_per_token)

    # Determine recommendation using v0.9.2 thresholds
    if file_size_bytes == 0:
        recommendation = "SKIP"
        reason = "File is empty. No content to process."
    elif estimated_tokens < SKIP_THRESHOLD_TOKENS:
        recommendation = "SKIP"
        reason = f"File is tiny (~{estimated_tokens} tokens). Too small for compression overhead."
    elif estimated_tokens < DIRECT_READ_THRESHOLD_TOKENS:
        recommendation = "DIRECT_READ"
        reason = f"File is small (~{estimated_tokens} tokens). Read directly without compression."
    elif estimated_tokens < STRONG_COMPRESS_THRESHOLD:
        recommendation = "COMPRESS"
        reason = f"File is medium (~{estimated_tokens} tokens). Compression recommended for token savings."
    elif estimated_tokens < MUST_COMPRESS_THRESHOLD:
        recommendation = "COMPRESS"
        reason = f"File is large (~{estimated_tokens} tokens). Compression strongly recommended."
    else:
        recommendation = "COMPRESS"
        reason = (
            f"File is very large (~{estimated_tokens} tokens). Must compress to fit in context."
        )

    # Calculate potential savings
    # Typical compression ratios: small docs 2-4x, medium 5-10x, large 15-20x
    if estimated_tokens < 500:
        compression_ratio = 2.0
    elif estimated_tokens < 2000:
        compression_ratio = 5.0
    elif estimated_tokens < 10000:
        compression_ratio = 10.0
    else:
        compression_ratio = 15.0

    potential_savings = int(estimated_tokens * (1 - 1 / compression_ratio))

    return json.dumps(
        {
            # Existing fields (backward compat)
            "file_path": file_path,
            "file_size_bytes": file_size_bytes,
            "estimated_tokens": estimated_tokens,
            "content_type_detected": detected_type,
            "recommendation": recommendation,
            "reason": reason,
            "potential_token_savings": potential_savings,
            "estimated_compression_ratio": f"{compression_ratio}x",
            # New fields (v0.9.2)
            "needs_conversion": False,
            "is_text_readable": True,
            "conversion_tool": None,
        },
        indent=2,
    )
