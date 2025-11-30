"""
Experimental handlers for MCP tools.

These handlers expose experimental features that are NOT production-ready.
All responses include "experimental": true flag.

Features exposed:
- SCAR (Semantic Compression with Alignment and Retrieval) - requires PyTorch
- TOON (Token-Oriented Object Notation) - pure Python
- Multimodal compression - requires Pillow for images

Dependencies are lazily imported to avoid startup failures.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from src.types import HandlerContext

logger = logging.getLogger(__name__)


# =============================================================================
# Lazy Import Helpers
# =============================================================================


def _get_toon_serializer():
    """Lazy import for TOONSerializer (pure Python, always available)."""
    from src.toon_serializer import TOONSerializer

    return TOONSerializer()


def _get_multimodal_compressor(use_clip: bool = False, use_codebert: bool = False):
    """
    Lazy import for MultiModalCompressor.

    Args:
        use_clip: Enable CLIP for image embeddings (requires additional deps)
        use_codebert: Enable CodeBERT for code embeddings (requires transformers)

    Raises:
        ImportError: If Pillow is not installed for image support
    """
    from src.multimodal_compressor import MultiModalCompressor

    return MultiModalCompressor(use_clip_for_images=use_clip, use_codebert_for_code=use_codebert)


def _get_scar_compressor(context: "HandlerContext"):
    """
    Lazy import for SCAREnhancedCompressor.

    SCAR requires a base SemanticCompressor instance and PyTorch.
    Uses untrained random weights by default - NOT production-ready.

    Args:
        context: Handler context containing compressor

    Raises:
        ImportError: If PyTorch is not installed
    """
    from src.scar_compressor import SCAREnhancedCompressor

    return SCAREnhancedCompressor(base_compressor=context["compressor"])


# =============================================================================
# TOON Handlers (Pure Python - Always Available)
# =============================================================================


async def handle_toon_encode(context: "HandlerContext", args: Dict[str, Any]) -> str:
    """
    Encode data to TOON format (~40% smaller than JSON).

    TOON = Token-Oriented Object Notation
    Pure Python implementation, always available.

    Args:
        context: Handler context (unused for TOON)
        args: {"data": dict|list} - Data to encode

    Returns:
        JSON string with toon_output, original_chars, toon_chars,
        savings_percent, experimental flag
    """
    data = args.get("data")
    if data is None:
        return json.dumps({"error": "Missing required argument: data", "experimental": True})

    try:
        serializer = _get_toon_serializer()

        # TOON's serialize_search_results expects List[Dict[str, Any]]
        # Convert data to the expected format
        if isinstance(data, dict) and "results" in data:
            # Extract the results list from dict with "results" key
            results_list = data["results"]
        elif isinstance(data, list):
            # Already a list, use directly
            results_list = data
        else:
            # Single dict item, wrap in list
            results_list = [data]

        toon_output = serializer.serialize_search_results(results_list)

        original_json = json.dumps(data, indent=2)
        original_chars = len(original_json)
        toon_chars = len(toon_output)
        savings = (
            ((original_chars - toon_chars) / original_chars * 100) if original_chars > 0 else 0
        )

        return json.dumps(
            {
                "toon_output": toon_output,
                "original_chars": original_chars,
                "toon_chars": toon_chars,
                "savings_percent": round(savings, 1),
                "experimental": True,
                "note": "TOON format is experimental - NOT production-ready",
            }
        )

    except Exception as e:
        logger.error(f"TOON encode failed: {e}")
        return json.dumps({"error": f"TOON encoding failed: {str(e)}", "experimental": True})


async def handle_toon_decode(context: "HandlerContext", args: Dict[str, Any]) -> str:
    """
    Decode TOON format back to structured data.

    Note: TOON is a lossy format optimized for LLM consumption.
    Decoding reconstructs approximate structure, not exact original.

    Args:
        context: Handler context (unused for TOON)
        args: {"toon_input": str} - TOON-formatted string

    Returns:
        JSON string with data and experimental flag
    """
    toon_input = args.get("toon_input")
    if not toon_input:
        return json.dumps({"error": "Missing required argument: toon_input", "experimental": True})

    try:
        # TOON is designed for LLM consumption, not round-trip serialization
        # Basic parsing of the TOON format
        lines = toon_input.strip().split("\n")
        results = []
        current_item = {}

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                if current_item:
                    results.append(current_item)
                current_item = {"raw": line[2:]}
            elif ":" in line and current_item:
                key, _, value = line.partition(":")
                current_item[key.strip()] = value.strip()

        if current_item:
            results.append(current_item)

        return json.dumps(
            {
                "data": results if len(results) > 1 else (results[0] if results else {}),
                "experimental": True,
                "note": "TOON decode is approximate - format optimized for LLM consumption",
            }
        )

    except Exception as e:
        logger.error(f"TOON decode failed: {e}")
        return json.dumps({"error": f"TOON decoding failed: {str(e)}", "experimental": True})


# =============================================================================
# SCAR Handlers (Requires PyTorch)
# =============================================================================


async def handle_scar_compress(context: "HandlerContext", args: Dict[str, Any]) -> str:
    """
    Compress embeddings using SCAR (learnable compression).

    WARNING: Uses UNTRAINED random weights by default.
    Requires PyTorch. NOT production-ready without model training.

    Args:
        context: Handler context containing compressor
        args: {
            "doc_id": str,  # Document to compress
            "target_dim": int (optional, default 128)
        }

    Returns:
        JSON string with compressed_dim, original_dim, reduction_ratio,
        model_trained, experimental flag
    """
    doc_id = args.get("doc_id")
    if not doc_id:
        return json.dumps({"error": "Missing required argument: doc_id", "experimental": True})

    try:
        # Check PyTorch availability first
        try:
            import torch  # noqa: F401
        except ImportError:
            return json.dumps(
                {
                    "error": "SCAR requires PyTorch. Install with: pip install torch",
                    "experimental": True,
                }
            )

        # Access compressor via dict-style context access
        compressor = context["compressor"]

        # Check if document exists in compressor's graphs
        if doc_id not in compressor.graphs:
            return json.dumps(
                {
                    "error": f"Document '{doc_id}' not found. Ingest it first.",
                    "experimental": True,
                }
            )

        # Extract embeddings from chunks
        # Match both semantic (_n) and code (::) chunk ID patterns
        embeddings = []
        for chunk_id, chunk_data in compressor.chunks.items():
            # Semantic chunks: {doc_id}_n0, {doc_id}_n1, etc.
            # Code chunks: {doc_id}::function_name, {doc_id}::imports, etc.
            if not (chunk_id.startswith(f"{doc_id}_") or chunk_id.startswith(f"{doc_id}::")):
                continue

            # Handle both object-based (SemanticNode/CodeChunk) and dict-based chunks
            embedding = None
            if hasattr(chunk_data, "embedding"):
                # Object with .embedding attribute (SemanticNode, CodeChunk)
                embedding = chunk_data.embedding
            elif isinstance(chunk_data, dict) and "embedding" in chunk_data:
                # Dict with "embedding" key (test mocks, etc.)
                embedding = chunk_data["embedding"]

            if embedding is not None:
                embeddings.append(embedding)

        if not embeddings:
            return json.dumps(
                {
                    "error": f"No embeddings found for document '{doc_id}'",
                    "experimental": True,
                }
            )

        import torch

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

        scar = _get_scar_compressor(context)
        target_dim = args.get("target_dim", 128)
        compressed = scar.compress_embeddings(embeddings_tensor, target_dim=target_dim)

        original_dim = embeddings_tensor.shape[-1]
        compressed_dim = compressed.shape[-1]

        return json.dumps(
            {
                "doc_id": doc_id,
                "original_dim": int(original_dim),
                "compressed_dim": int(compressed_dim),
                "reduction_ratio": round(original_dim / compressed_dim, 2),
                "num_vectors": len(embeddings),
                "model_trained": False,  # Always false without explicit training
                "experimental": True,
                "warning": "Using UNTRAINED random weights - results are NOT meaningful without training",
            }
        )

    except ImportError as e:
        return json.dumps({"error": f"SCAR dependency missing: {str(e)}", "experimental": True})
    except Exception as e:
        logger.error(f"SCAR compress failed: {e}")
        return json.dumps({"error": f"SCAR compression failed: {str(e)}", "experimental": True})


async def handle_scar_get_stats(context: "HandlerContext", args: Dict[str, Any]) -> str:
    """
    Get SCAR compressor statistics and model state.

    Args:
        context: Handler context containing compressor
        args: {} (no arguments needed)

    Returns:
        JSON string with pytorch_available, model_trained,
        components, experimental flag
    """
    try:
        # Check PyTorch availability
        pytorch_available = False
        pytorch_version = None
        try:
            import torch

            pytorch_available = True
            pytorch_version = torch.__version__
        except ImportError:
            pass

        stats = {
            "pytorch_available": pytorch_available,
            "pytorch_version": pytorch_version,
            "model_trained": False,  # No persistent training state yet
            "experimental": True,
        }

        if pytorch_available:
            try:
                _get_scar_compressor(context)  # Verify SCAR can be initialized
                stats["components"] = ["LearnableSemanticCompressor", "SemanticAlignmentModule"]
                stats["default_compression_dim"] = 128
                stats["warning"] = "Model uses random weights - train before production use"
            except Exception as e:
                stats["initialization_error"] = str(e)
        else:
            stats["install_hint"] = "pip install torch"

        return json.dumps(stats)

    except Exception as e:
        logger.error(f"SCAR stats failed: {e}")
        return json.dumps({"error": f"Failed to get SCAR stats: {str(e)}", "experimental": True})


# =============================================================================
# Multimodal Handlers (Requires Pillow for Images)
# =============================================================================


async def handle_multimodal_ingest(context: "HandlerContext", args: Dict[str, Any]) -> str:
    """
    Ingest mixed content (text, code, images) into compression graph.

    Requires Pillow for image support. Image paths are validated
    against allowed directories (CWE-22 protection).

    Args:
        context: Handler context containing path_validator
        args: {
            "doc_id": str,
            "text_content": str (optional),
            "code_content": str (optional),
            "code_language": str (optional, default "python"),
            "image_paths": list[str] (optional) - paths validated via PathValidator
        }

    Returns:
        JSON string with doc_id, content_types, node_count, experimental flag
    """
    doc_id = args.get("doc_id")
    if not doc_id:
        return json.dumps({"error": "Missing required argument: doc_id", "experimental": True})

    text_content = args.get("text_content")
    code_content = args.get("code_content")
    image_paths = args.get("image_paths", [])

    if not any([text_content, code_content, image_paths]):
        return json.dumps(
            {
                "error": "At least one content type required: text_content, code_content, or image_paths",
                "experimental": True,
            }
        )

    try:
        # Validate image paths if provided (CWE-22 protection)
        validated_image_paths = []
        if image_paths:
            path_validator = context.get("path_validator")
            if path_validator is None:
                return json.dumps(
                    {
                        "error": "PathValidator not configured - cannot safely handle file paths",
                        "experimental": True,
                    }
                )

            for path in image_paths:
                try:
                    # Use validate() method, not validate_path()
                    validated_path = path_validator.validate(path)
                    validated_image_paths.append(validated_path)
                except ValueError as e:
                    return json.dumps(
                        {
                            "error": f"Invalid image path '{path}': {str(e)}",
                            "experimental": True,
                        }
                    )

        # Check Pillow availability if images requested
        if validated_image_paths:
            try:
                from PIL import Image  # noqa: F401
            except ImportError:
                return json.dumps(
                    {
                        "error": "Image support requires Pillow. Install with: pip install Pillow",
                        "experimental": True,
                    }
                )

        # Initialize multimodal compressor
        compressor = _get_multimodal_compressor(use_clip=False, use_codebert=False)

        # Build content list for ingestion
        content_items = []
        content_types = []

        if text_content:
            content_items.append({"type": "text", "content": text_content})
            content_types.append("text")

        if code_content:
            content_items.append(
                {
                    "type": "code",
                    "content": code_content,
                    "language": args.get("code_language", "python"),
                }
            )
            content_types.append("code")

        for img_path in validated_image_paths:
            content_items.append({"type": "image", "path": img_path})
            content_types.append("image")

        # Ingest mixed content
        result = compressor.ingest_mixed_content(doc_id, content_items)

        return json.dumps(
            {
                "doc_id": doc_id,
                "content_types": content_types,
                "node_count": result.get("node_count", len(content_items)),
                "experimental": True,
                "note": "Multimodal compression is experimental - NOT production-ready",
            }
        )

    except ImportError as e:
        return json.dumps(
            {"error": f"Multimodal dependency missing: {str(e)}", "experimental": True}
        )
    except Exception as e:
        logger.error(f"Multimodal ingest failed: {e}")
        return json.dumps({"error": f"Multimodal ingestion failed: {str(e)}", "experimental": True})


# =============================================================================
# Handler Registry
# =============================================================================

EXPERIMENTAL_HANDLERS = {
    "toon_encode": handle_toon_encode,
    "toon_decode": handle_toon_decode,
    "scar_compress": handle_scar_compress,
    "scar_get_stats": handle_scar_get_stats,
    "multimodal_ingest": handle_multimodal_ingest,
}
