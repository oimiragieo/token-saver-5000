"""
Centralized Embedding Management for Semantic Modulator

This module provides a singleton EmbeddingManager that eliminates duplicate
SentenceTransformer model initializations across the codebase.

**Problem Solved:**
- Previously: 4-5 separate SentenceTransformer initializations (~80MB each)
- Now: Single shared instance per model with lazy loading and caching

**Benefits:**
- Reduced memory usage (~320MB → ~80MB for 4 modules using same model)
- Faster initialization (model only loaded once)
- Thread-safe model caching
- Centralized model configuration

Version: 0.4.0
"""

import logging
import threading
from typing import Dict, Optional

from sentence_transformers import SentenceTransformer

from .constants import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_CODE_MODEL,
    DEFAULT_IMAGE_MODEL,
)

# Enable HuggingFace Hub progress bars for model downloads (v0.4.1+)
try:
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
except ImportError:
    # If huggingface_hub is not available, progress bars won't show
    # but model downloads will still work
    pass

logger = logging.getLogger("embeddings")


class EmbeddingManager:
    """
    Singleton manager for SentenceTransformer models with caching.

    This class ensures only one instance of each embedding model is loaded,
    even when multiple components request the same model.

    Thread-safe implementation using double-checked locking pattern.

    Example usage:
        >>> manager = EmbeddingManager()
        >>> text_model = manager.get_text_embedder()  # Loads model
        >>> text_model2 = manager.get_text_embedder()  # Returns cached model
        >>> assert text_model is text_model2  # Same instance
    """

    _instance: Optional["EmbeddingManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls):
        """
        Singleton pattern with thread-safe lazy initialization.

        Returns:
            The single EmbeddingManager instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    logger.info("Initializing EmbeddingManager (singleton)")
                    instance = super().__new__(cls)
                    instance._model_cache: Dict[str, SentenceTransformer] = {}
                    instance._cache_lock = threading.Lock()
                    cls._instance = instance
        return cls._instance

    def get_text_embedder(self, model_name: str = DEFAULT_TEXT_MODEL) -> SentenceTransformer:
        """
        Get or create text embedding model.

        Args:
            model_name: Name of the SentenceTransformer model
                       (default: "all-MiniLM-L6-v2")

        Returns:
            Cached or newly loaded SentenceTransformer model

        Example:
            >>> manager = EmbeddingManager()
            >>> model = manager.get_text_embedder()
            >>> embeddings = model.encode(["Hello world"])
        """
        return self._get_or_create_model(model_name, "text")

    def get_code_embedder(self, model_name: str = DEFAULT_CODE_MODEL) -> SentenceTransformer:
        """
        Get or create code embedding model.

        Args:
            model_name: Name of the SentenceTransformer model
                       (default: "microsoft/codebert-base")

        Returns:
            Cached or newly loaded SentenceTransformer model

        Fallback behavior:
            If codebert model unavailable, falls back to DEFAULT_TEXT_MODEL
        """
        try:
            return self._get_or_create_model(model_name, "code")
        except Exception as e:
            logger.warning(
                f"Failed to load code model {model_name}: {e}\n"
                f"Falling back to text model: {DEFAULT_TEXT_MODEL}"
            )
            return self.get_text_embedder(DEFAULT_TEXT_MODEL)

    def get_image_embedder(self, model_name: str = DEFAULT_IMAGE_MODEL) -> SentenceTransformer:
        """
        Get or create image-text embedding model (CLIP).

        Args:
            model_name: Name of the CLIP model
                       (default: "clip-ViT-B-32")

        Returns:
            Cached or newly loaded CLIP model

        Note:
            CLIP models support both text and image inputs,
            useful for multimodal compression.
        """
        return self._get_or_create_model(model_name, "image")

    def _get_or_create_model(self, model_name: str, model_type: str) -> SentenceTransformer:
        """
        Thread-safe model retrieval or creation.

        Args:
            model_name: Name of the SentenceTransformer model
            model_type: Type hint for logging ("text", "code", "image")

        Returns:
            Cached or newly loaded SentenceTransformer model
        """
        # Fast path: check cache without lock
        if model_name in self._model_cache:
            logger.debug(f"Using cached {model_type} embedder: {model_name}")
            return self._model_cache[model_name]

        # Slow path: acquire lock and load model
        with self._cache_lock:
            # Double-check pattern: another thread may have loaded it
            if model_name in self._model_cache:
                logger.debug(f"Using cached {model_type} embedder: {model_name}")
                return self._model_cache[model_name]

            # Load model (this is the expensive operation)
            logger.info(
                f"Loading {model_type} embedding model: {model_name} "
                f"(~80MB download if not cached)"
            )
            model = SentenceTransformer(model_name)

            # Cache for future use
            self._model_cache[model_name] = model
            logger.info(
                f"Cached {model_type} embedder: {model_name} "
                f"(total cached: {len(self._model_cache)})"
            )

            return model

    def clear_cache(self) -> None:
        """
        Clear all cached models from memory.

        Use this to free memory if you're done with certain models.
        The next request will reload the model.

        Warning:
            This will clear ALL cached models. Be careful in concurrent
            environments where other threads may be using cached models.
        """
        with self._cache_lock:
            num_models = len(self._model_cache)
            self._model_cache.clear()
            logger.info(f"Cleared embedding cache ({num_models} models released)")

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the embedding cache.

        Returns:
            Dictionary with cache statistics:
            - num_models: Number of cached models
            - model_names: List of cached model names
            - estimated_memory_mb: Rough estimate of memory usage
        """
        with self._cache_lock:
            model_names = list(self._model_cache.keys())
            num_models = len(model_names)

            # Rough estimate: ~80MB per model (text/code), ~150MB for CLIP
            estimated_memory_mb = 0
            for name in model_names:
                if "clip" in name.lower():
                    estimated_memory_mb += 150
                else:
                    estimated_memory_mb += 80

            return {
                "num_models": num_models,
                "model_names": model_names,
                "estimated_memory_mb": estimated_memory_mb,
            }

    def __repr__(self) -> str:
        """String representation of EmbeddingManager."""
        stats = self.get_cache_stats()
        return (
            f"EmbeddingManager(cached_models={stats['num_models']}, "
            f"estimated_memory={stats['estimated_memory_mb']}MB)"
        )


# Convenience function for backward compatibility
def get_embedding_manager() -> EmbeddingManager:
    """
    Get the singleton EmbeddingManager instance.

    This is a convenience function for modules that prefer functional style.

    Returns:
        The singleton EmbeddingManager instance

    Example:
        >>> from src.embeddings import get_embedding_manager
        >>> manager = get_embedding_manager()
        >>> model = manager.get_text_embedder()
    """
    return EmbeddingManager()
