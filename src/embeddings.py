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

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    # Annotations only. `from __future__ import annotations` above makes every
    # annotation a string, so this import never runs at runtime — which is the
    # whole point: the real import is deferred into _sentence_transformer_cls().
    # Without this block the `-> SentenceTransformer` annotations are undefined
    # names to a type checker, even though they are harmless to the interpreter.
    from sentence_transformers import SentenceTransformer


def _sentence_transformer_cls() -> Any:
    """Return the SentenceTransformer class, or None in ONNX-only mode.

    DEFERRED ON PURPOSE. This import pulls sentence_transformers ->
    transformers -> torch: roughly 7.5 seconds and ~1,070 modules. EVERY module
    that touches compression imports this one, so binding it at module scope put
    that cost in every boot, including boots that never build a torch model.
    Measured: app import 14.1s -> 6.3s, with all 113 routes still registering.

    THE RESULT IS CACHED IN THE MODULE GLOBAL, not a private variable, and that
    is load-bearing rather than stylistic. Four test files simulate ONNX-only
    mode with `monkeypatch.setattr(emb, "SentenceTransformer", None)` — a
    deliberate seam for exercising the no-torch path. Caching privately would
    leave that patch inert: the tests would set an attribute nothing reads, and
    pass while testing nothing. Reading the global means a patched None steers
    this accessor exactly as the old module-level binding did.

    Pairs with `__getattr__` below, which is what keeps the attribute itself
    accessible without an eager import.
    """
    g = globals()
    if "SentenceTransformer" not in g:
        try:
            from sentence_transformers import SentenceTransformer as _ST

            g["SentenceTransformer"] = _ST
        except ImportError:
            # ONNX-only mode — no torch/sentence-transformers installed.
            g["SentenceTransformer"] = None
    return g["SentenceTransformer"]


def __getattr__(name: str) -> Any:
    """PEP 562 module-level attribute hook.

    Keeps `src.embeddings.SentenceTransformer` readable — for the tests that
    patch it, and for any caller that still reads it — WITHOUT importing
    sentence_transformers at module load. Once resolved, the real global
    shadows this hook, so it costs one call.
    """
    if name == "SentenceTransformer":
        return _sentence_transformer_cls()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    from huggingface_hub.utils.tqdm import enable_progress_bars

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
        self._cached_dim = None

    def get_sentence_embedding_dimension(self) -> int:
        """SentenceTransformer-compatible embedding-dimension accessor.

        ``MultiModalCompressor`` and ``SCAREnhancedCompressor`` call this on
        whatever text encoder they hold. In ONNX-only deployments that encoder
        is THIS adapter, which previously lacked the method -> AttributeError
        (a latent 500 on ``multimodal_ingest`` / ``scar_compress``). Probe the
        backend once with a tiny input and cache the dimension.
        """
        if self._cached_dim is None:
            import numpy as np

            vec = np.asarray(self.encode(["x"], normalize_embeddings=False))
            self._cached_dim = int(vec.shape[-1])
        return self._cached_dim

    def encode(self, texts, **kwargs):
        normalize = kwargs.get("normalize_embeddings", kwargs.get("normalize", True))
        # MUST skip STANDARD tier to avoid recursion:
        # adapter.encode → manager.encode(STANDARD) → _encode_standard
        # → get_text_embedder → adapter → loop!
        # Force ONNX directly.
        #
        # RECURSION GUARD (defense-in-depth): this adapter only exists because
        # sentence-transformers is unavailable, so STANDARD can never load here.
        # If ONNX is ALSO unavailable, re-entering ``manager.encode`` would let
        # the fallback machinery bounce STANDARD↔ONNX indefinitely. There is no
        # usable neural embedder offline, so raise ONE clean error immediately
        # at bounded depth rather than recursing.
        if not ONNX_AVAILABLE:
            raise RuntimeError(
                "No usable embedding backend: sentence-transformers is "
                "unavailable (ONNX-only adapter active) and the ONNX tier is "
                "also unavailable (missing onnxruntime/optimum). Install one of "
                "sentence-transformers or onnxruntime+optimum, or request "
                "EmbeddingTier.TFIDF explicitly."
            )
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

        Note on tier locking (Phase 7c-4 war story):
            The tier is fixed at FIRST construction and shared process-wide. A
            second construction that requests a CONFLICTING tier does NOT switch
            the locked instance — it logs CRITICAL and returns the existing one.
            To avoid silent poisoning, set ``EMBEDDING_TIER`` at process start
            (before the first ``EmbeddingManager(...)`` call). Use
            :meth:`reset_for_testing` to clear the singleton between tests.
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

        # Singleton already exists: warn loudly if the caller asked for a
        # DIFFERENT tier than the locked one. Silently returning a STANDARD
        # instance to an ONNX requester is the Phase 7c-4 silent-poisoning bug.
        existing = cls._instance
        locked_tier = getattr(existing, "_tier", None)
        if locked_tier is not None and locked_tier != tier:
            logger.critical(
                "EmbeddingManager tier conflict: singleton is locked to "
                "tier=%s but a new construction requested tier=%s. The locked "
                "tier WINS — the requested tier is ignored. Set EMBEDDING_TIER "
                "at process start to avoid silent tier poisoning.",
                locked_tier.value,
                tier.value,
            )
        return existing

    @classmethod
    def reset_for_testing(cls) -> None:
        """Clear the process-wide singleton so the next construction is fresh.

        Test-only helper. Lets a test exercise a specific tier without being
        poisoned by a tier some earlier test (or import-time call) locked in.
        Never call this in production code paths — it would orphan the model
        cache held by the previous instance.
        """
        with cls._lock:
            cls._instance = None

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

        CORRECTNESS (audit P1-6): TF-IDF vectors are vocabulary-order-dependent
        and semantically meaningless when used as drop-in replacements for SBERT
        / ONNX neural embeddings (cosine distances become garbage). We therefore
        REFUSE to silently fall through to TF-IDF when the caller requested a
        neural tier (STANDARD or ONNX) and both neural tiers failed. Silently
        returning TF-IDF garbage corrupts every downstream cosine comparison.
        Only an explicit TFIDF request is allowed to use TF-IDF here.
        """
        last_exc: Optional[Exception] = None

        # Try STANDARD first (if not already attempted).
        #
        # INFINITE-RECURSION GUARD (offline/degraded path): when
        # sentence-transformers is unavailable (``SentenceTransformer is None``),
        # ``_encode_standard`` cannot load a real model — ``get_text_embedder``
        # returns an ``_EmbeddingManagerAdapter`` whose ``.encode`` re-enters
        # ``manager.encode(tier=ONNX)``. If ONNX also can't load, that bounces
        # straight back into ``_encode_with_fallback(tier=ONNX)``, which would
        # re-enter the STANDARD branch below, building an O(n²) nested error
        # string ~330 frames deep (~14s CPU) before finally raising. The
        # STANDARD fallback is structurally useless in this state, so skip it:
        #   _encode_with_fallback(ONNX) → _encode_standard → adapter
        #     → encode(ONNX) → _encode_onnx FAILS → _encode_with_fallback(ONNX) → …
        # Only attempt STANDARD when a real SentenceTransformer can actually load.
        if tier != EmbeddingTier.STANDARD and _sentence_transformer_cls() is not None:
            try:
                logger.info("Falling back to STANDARD tier...")
                return self._encode_standard(texts, normalize)
            except Exception as e:
                last_exc = e
                logger.warning(f"STANDARD tier fallback failed: {e}")

        # Try ONNX next (if not already attempted)
        if tier != EmbeddingTier.ONNX and ONNX_AVAILABLE:
            try:
                logger.info("Falling back to ONNX tier...")
                return self._encode_onnx(texts, normalize)
            except Exception as e:
                last_exc = e
                logger.warning(f"ONNX tier fallback failed: {e}")

        # TF-IDF is only a legitimate result when it was explicitly requested.
        # For a neural-tier request whose neural fallbacks failed, returning
        # TF-IDF vectors would be silently-wrong, so we raise instead.
        if tier == EmbeddingTier.TFIDF and TFIDF_AVAILABLE:
            logger.info("Using TF-IDF tier (explicitly requested)...")
            return self._encode_tfidf(texts, normalize)

        # Neural tier requested but SBERT + ONNX both failed: refuse to return
        # garbage TF-IDF vectors. Surface the original failure at error level so
        # it reaches Sentry rather than being swallowed as a warning.
        if last_exc is not None:
            logger.error(
                "Embedding tier %s requested but SBERT/ONNX neural tiers failed; "
                "refusing to fall back to semantically-meaningless TF-IDF vectors. "
                "Original error: %s",
                tier.value,
                last_exc,
                exc_info=last_exc,
            )
            raise RuntimeError(
                f"Embedding tier {tier.value} failed and TF-IDF fallback is not "
                f"a valid substitute for neural embeddings. Original error: {last_exc}"
            ) from last_exc

        # No neural fallback attempted and no explicit TF-IDF: nothing worked.
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
            _st_cls = _sentence_transformer_cls()
            if _st_cls is None:
                raise ImportError(
                    "sentence-transformers not installed — cannot load "
                    f"STANDARD tier model '{model_name}'. "
                    "Use EmbeddingTier.ONNX or EmbeddingTier.TFIDF instead."
                )
            logger.info(
                f"Loading {model_type} embedding model: {model_name} "
                f"(~80MB download if not cached)"
            )
            model = _st_cls(model_name)

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
