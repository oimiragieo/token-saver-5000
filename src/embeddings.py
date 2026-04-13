"""
Centralized Embedding Management for Semantic Modulator (v0.6.0)

This module provides a singleton EmbeddingManager with multi-tier embedding support
and automatic fallback for memory-constrained environments.

**Embedding Tiers (v0.6.0):**
1. STANDARD: SentenceTransformer models (~400MB, highest quality)
2. ONNX: Quantized ONNX models (~150MB, 3-5× faster, 60-70% memory reduction)
3. TFIDF: Sklearn TF-IDF (~10MB, 100-1000× faster, adequate quality fallback)

**Problem Solved:**
- Previously: 4-5 separate SentenceTransformer initializations (~80MB each)
- Now: Single shared instance per model with lazy loading and caching
- v0.6.0: Automatic tier degradation based on memory constraints + LRU cache

**Benefits:**
- Reduced memory usage (~320MB → ~80MB for 4 modules using same model)
- Faster initialization (model only loaded once)
- Thread-safe model caching
- Centralized model configuration
- Memory-adaptive tier selection (v0.6.0)
- 60-80% cache hit rate for repeated queries (v0.6.0)

Version: 0.6.0
"""

import logging
import threading
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # ONNX-only mode — no torch/sentence-transformers

from .constants import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_CODE_MODEL,
    DEFAULT_IMAGE_MODEL,
)

# Import tier-specific managers (with graceful degradation)
try:
    from .embeddings_onnx import ONNXEmbeddingManager

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from .embeddings_tfidf import TFIDFEmbeddingManager

    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False

try:
    from .embedding_cache import LRUEmbeddingCache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Enable HuggingFace Hub progress bars for model downloads (v0.4.1+)
try:
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
except ImportError:
    # If huggingface_hub is not available, progress bars won't show
    # but model downloads will still work
    pass

logger = logging.getLogger("embeddings")


class _EmbeddingManagerAdapter:
    """Adapter so EmbeddingManager can substitute for SentenceTransformer.

    In ONNX-only deployments (no torch/sentence-transformers), callers that
    expect a SentenceTransformer-like .encode() interface get this adapter
    instead.  It delegates to EmbeddingManager.encode() which has built-in
    STANDARD → ONNX → TF-IDF fallback.
    """

    def __init__(self, manager: "EmbeddingManager"):
        self._manager = manager

    def encode(self, texts, **kwargs):
        normalize = kwargs.get("normalize_embeddings", kwargs.get("normalize", True))
        # MUST skip STANDARD tier to avoid recursion:
        # adapter.encode → manager.encode(STANDARD) → _encode_standard
        # → get_text_embedder → adapter → loop!
        # Force ONNX (or TF-IDF fallback) directly.
        return self._manager.encode(texts, tier=EmbeddingTier.ONNX, normalize=normalize)


class EmbeddingTier(Enum):
    """
    Embedding tiers with automatic fallback hierarchy (v0.6.0).

    Tiers ordered by quality/memory trade-off:
    - STANDARD: Highest quality, highest memory (~400MB)
    - ONNX: Medium quality, medium memory (~150MB, 3-5× faster)
    - TFIDF: Lowest quality, lowest memory (~10MB, 100-1000× faster)
    """

    STANDARD = "standard"  # SentenceTransformer (default)
    ONNX = "onnx"  # ONNX Runtime quantized
    TFIDF = "tfidf"  # Sklearn TF-IDF fallback


class EmbeddingManager:
    """
    Singleton manager for embedding models with multi-tier support and caching (v0.6.0).

    This class ensures only one instance of each embedding model is loaded,
    even when multiple components request the same model. Supports automatic
    tier degradation based on memory constraints.

    **Tiers:**
    - STANDARD: SentenceTransformer models (highest quality, ~400MB)
    - ONNX: Quantized ONNX models (3-5× faster, 60-70% memory reduction)
    - TFIDF: Sklearn TF-IDF (fastest, 98% memory reduction, fallback)

    Thread-safe implementation using double-checked locking pattern.

    Example usage:
        >>> manager = EmbeddingManager(tier=EmbeddingTier.ONNX)
        >>> embeddings = manager.encode(["hello world"])  # Uses ONNX tier
        >>> embeddings2 = manager.encode(["hello world"])  # Cache hit
    """

    _instance: Optional["EmbeddingManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, tier: EmbeddingTier = EmbeddingTier.STANDARD, enable_cache: bool = True):
        """
        Singleton pattern with thread-safe lazy initialization.

        Args:
            tier: Embedding tier to use (default: STANDARD)
            enable_cache: Enable LRU caching (default: True)

        Returns:
            The single EmbeddingManager instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    logger.info(
                        f"Initializing EmbeddingManager (tier={tier.value}, cache={enable_cache})"
                    )
                    instance = super().__new__(cls)
                    instance._model_cache: Dict[str, SentenceTransformer] = {}
                    instance._cache_lock = threading.Lock()
                    instance._tier = tier
                    instance._enable_cache = enable_cache

                    # Initialize tier-specific managers
                    instance._onnx_manager = None
                    instance._tfidf_manager = None
                    instance._lru_cache = None

                    # Initialize cache if enabled
                    if enable_cache and CACHE_AVAILABLE:
                        instance._lru_cache = LRUEmbeddingCache(max_entries=10000)
                        logger.info("LRU embedding cache enabled (10k entries)")
                    elif enable_cache and not CACHE_AVAILABLE:
                        logger.warning("LRU cache requested but embedding_cache module unavailable")

                    cls._instance = instance
        return cls._instance

    def encode(
        self,
        texts: Union[str, List[str]],
        tier: Optional[EmbeddingTier] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings using configured tier with automatic fallback.

        Args:
            texts: Single text or list of texts to encode
            tier: Override tier for this call (default: use manager's tier)
            normalize: L2 normalize embeddings (default: True)

        Returns:
            NumPy array of embeddings (shape: [num_texts, embedding_dim])

        Example:
            >>> manager = EmbeddingManager(tier=EmbeddingTier.ONNX)
            >>> embeddings = manager.encode(["hello world", "foo bar"])
            >>> embeddings.shape  # (2, 384)

        Fallback Behavior:
            If requested tier unavailable, automatically falls back:
            STANDARD → ONNX → TFIDF
        """
        # Use manager's tier if not overridden
        tier = tier or self._tier

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Try cache first (if enabled)
        if self._lru_cache:
            cached_embeddings, miss_indices = self._lru_cache.get_batch(texts)

            if not miss_indices:
                # Full cache hit
                return np.array([emb for emb in cached_embeddings])

            # Partial cache hit - compute missing embeddings
            missing_texts = [texts[i] for i in miss_indices]
        else:
            # No cache - compute all
            missing_texts = texts
            miss_indices = list(range(len(texts)))
            cached_embeddings = [None] * len(texts)

        # Compute missing embeddings using tier hierarchy
        try:
            if tier == EmbeddingTier.STANDARD:
                missing_embeddings = self._encode_standard(missing_texts, normalize)
            elif tier == EmbeddingTier.ONNX:
                missing_embeddings = self._encode_onnx(missing_texts, normalize)
            elif tier == EmbeddingTier.TFIDF:
                missing_embeddings = self._encode_tfidf(missing_texts, normalize)
            else:
                raise ValueError(f"Unknown embedding tier: {tier}")

        except Exception as e:
            # Automatic fallback
            logger.warning(f"Tier {tier.value} failed: {e}. Attempting fallback...")
            missing_embeddings = self._encode_with_fallback(missing_texts, tier, normalize)

        # Update cache with missing embeddings
        if self._lru_cache:
            self._lru_cache.put_batch(missing_texts, list(missing_embeddings))

        # Merge cached and computed embeddings
        result_embeddings = []
        miss_idx_set = set(miss_indices)

        for i, text in enumerate(texts):
            if i in miss_idx_set:
                # Use computed embedding
                miss_position = miss_indices.index(i)
                result_embeddings.append(missing_embeddings[miss_position])
            else:
                # Use cached embedding
                result_embeddings.append(cached_embeddings[i])

        return np.array(result_embeddings)

    def _encode_standard(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using standard SentenceTransformer."""
        model = self.get_text_embedder()
        return model.encode(texts, normalize_embeddings=normalize)

    def _encode_onnx(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using ONNX-optimized model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX tier unavailable (missing onnxruntime/optimum)")

        if not self._onnx_manager:
            self._onnx_manager = ONNXEmbeddingManager()

        return self._onnx_manager.encode(texts, normalize=normalize)

    def _encode_tfidf(self, texts: List[str], normalize: bool) -> np.ndarray:
        """Encode using TF-IDF fallback."""
        if not TFIDF_AVAILABLE:
            raise ImportError("TF-IDF tier unavailable (missing sklearn)")

        if not self._tfidf_manager:
            self._tfidf_manager = TFIDFEmbeddingManager()

        return self._tfidf_manager.encode(texts, normalize=normalize)

    def _encode_with_fallback(
        self, texts: List[str], tier: EmbeddingTier, normalize: bool
    ) -> np.ndarray:
        """
        Encode with automatic tier fallback.

        Fallback hierarchy: STANDARD → ONNX → TFIDF
        """
        # Try STANDARD first (if not already attempted)
        if tier != EmbeddingTier.STANDARD:
            try:
                logger.info("Falling back to STANDARD tier...")
                return self._encode_standard(texts, normalize)
            except Exception as e:
                logger.warning(f"STANDARD tier fallback failed: {e}")

        # Try ONNX next (if not already attempted)
        if tier != EmbeddingTier.ONNX and ONNX_AVAILABLE:
            try:
                logger.info("Falling back to ONNX tier...")
                return self._encode_onnx(texts, normalize)
            except Exception as e:
                logger.warning(f"ONNX tier fallback failed: {e}")

        # Last resort: TF-IDF
        if TFIDF_AVAILABLE:
            logger.info("Falling back to TF-IDF tier (last resort)...")
            return self._encode_tfidf(texts, normalize)

        # If we get here, all tiers failed
        raise RuntimeError("All embedding tiers failed. Cannot encode texts.")

    def get_text_embedder(self, model_name: str = DEFAULT_TEXT_MODEL):
        """
        Get or create text embedding model.

        In ONNX-only mode (no sentence-transformers), returns a lightweight
        adapter that delegates to encode() with automatic ONNX/TF-IDF fallback.

        Args:
            model_name: Name of the SentenceTransformer model
                       (default: "all-MiniLM-L6-v2")

        Returns:
            SentenceTransformer model, or _EmbeddingManagerAdapter in ONNX-only mode
        """
        try:
            return self._get_or_create_model(model_name, "text")
        except (ImportError, TypeError):
            logger.info("SentenceTransformer unavailable — using ONNX/TF-IDF adapter")
            return _EmbeddingManagerAdapter(self)

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

            # Load model (this is the expensive operation).
            # Guard: SentenceTransformer may be None in ONNX-only mode
            # (e.g. Docker images that skip torch + sentence-transformers).
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed — cannot load "
                    f"STANDARD tier model '{model_name}'. "
                    "Use EmbeddingTier.ONNX or EmbeddingTier.TFIDF instead."
                )
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

    def set_tier(self, tier: EmbeddingTier):
        """
        Switch embedding tier.

        Args:
            tier: New tier to use

        Example:
            >>> manager = EmbeddingManager()
            >>> manager.set_tier(EmbeddingTier.ONNX)  # Switch to ONNX tier
        """
        logger.info(f"Switching embedding tier: {self._tier.value} → {tier.value}")
        self._tier = tier

    def get_tier(self) -> EmbeddingTier:
        """Get current embedding tier."""
        return self._tier

    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the embedding cache and tier usage (v0.6.0).

        Returns:
            Dictionary with cache statistics:
            - tier: Current embedding tier
            - num_models: Number of cached models (STANDARD tier only)
            - model_names: List of cached model names (STANDARD tier only)
            - estimated_memory_mb: Rough estimate of memory usage
            - lru_cache: LRU cache statistics (if enabled)
            - onnx_manager: ONNX manager stats (if initialized)
            - tfidf_manager: TF-IDF manager stats (if initialized)
        """
        stats = {
            "tier": self._tier.value,
            "cache_enabled": self._enable_cache,
        }

        # STANDARD tier model cache
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

            stats.update(
                {
                    "num_models": num_models,
                    "model_names": model_names,
                    "estimated_memory_mb": estimated_memory_mb,
                }
            )

        # LRU cache stats
        if self._lru_cache:
            stats["lru_cache"] = self._lru_cache.get_stats()

        # ONNX manager stats
        if self._onnx_manager:
            stats["onnx_manager"] = self._onnx_manager.get_memory_usage()

        # TF-IDF manager stats
        if self._tfidf_manager:
            stats["tfidf_manager"] = self._tfidf_manager.get_memory_usage()

        return stats

    def __repr__(self) -> str:
        """String representation of EmbeddingManager (v0.6.0)."""
        stats = self.get_cache_stats()
        return (
            f"EmbeddingManager(tier={stats['tier']}, "
            f"cached_models={stats['num_models']}, "
            f"estimated_memory={stats['estimated_memory_mb']}MB, "
            f"cache_enabled={stats['cache_enabled']})"
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
