"""
ONNX Embedding Manager for Token Saver 5000 (v0.6.0)

Provides optimized embedding inference using ONNX Runtime with quantization support.
This tier offers 3-5× faster inference and 60-70% lower memory usage compared to PyTorch.

Key Features:
- Quantized INT8 models for reduced memory footprint
- CPU-optimized execution providers
- Batch inference with configurable parallelism
- Automatic model download and caching
- Fallback to standard embeddings on failure
- Per-model pooling selector (CLS for ModernBERT/granite, mean for bge/MiniLM)

Performance Characteristics:
- Memory: ~150MB (vs ~400MB for PyTorch sentence-transformers)
- Inference: 3-5× faster on CPU (batch processing)
- Quality: Identical to PyTorch (same model weights)

Supported Models:
- all-MiniLM-L6-v2 (default, 384 dimensions)
- BAAI/bge-small-en-v1.5 (384 dimensions, mean pooling)
- onnx-community/granite-embedding-small-english-r2-ONNX (384 dimensions, CLS pooling)
- Optimized ONNX models from Hugging Face Optimum
"""

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .constants import DEFAULT_TEXT_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model pooling policy
# ---------------------------------------------------------------------------
# granite-r2 (ibm-granite/ModernBERT architecture) uses CLS pooling —
# the model card and sentence-transformers Pooling config both confirm
# pooling_mode_cls_token=True, pooling_mode_mean_tokens=False.
# All other models we ship (bge-small, MiniLM) use mean pooling.
# Key = any substring that uniquely identifies the model family.
_CLS_POOLING_SUBSTRINGS = frozenset(
    [
        "granite-embedding",
        "granite-embedding-small-english-r2",
    ]
)


def _uses_cls_pooling(model_name: str) -> bool:
    """Return True when *model_name* is a granite/ModernBERT CLS-pooling model."""
    name_lower = model_name.lower()
    return any(sub in name_lower for sub in _CLS_POOLING_SUBSTRINGS)


class ONNXEmbeddingManager:
    """
    Manages ONNX-optimized embedding inference for reduced memory usage.

    Uses ONNX Runtime with quantized INT8 models for 3-5× speedup and
    60-70% memory reduction compared to standard PyTorch models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        cache_dir: Optional[str] = None,
        quantized: bool = True,
    ):
        """
        Initialize ONNX embedding manager.

        Args:
            model_name: Hugging Face model identifier
            cache_dir: Optional directory for model caching (default: ~/.cache/token-saver-5000)
            quantized: Use quantized INT8 model for reduced memory (default: True)
        """
        self.model_name = model_name
        self.quantized = quantized
        self.cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/token-saver-5000"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy initialization
        self._session = None
        self._tokenizer = None
        self._initialized = False
        # Which backend _initialize() selected. Set here, not only on the
        # branch that uses it: an attribute that exists on one path is the
        # shape that raises AttributeError somewhere else entirely.
        self._raw_ort = False
        self._session_input_names: set = set()
        self._init_lock = threading.Lock()

    def _initialize(self):
        """Lazy initialization of ONNX session and tokenizer (thread-safe)."""
        if self._initialized:
            return

        with self._init_lock:
            # Double-checked locking pattern
            if self._initialized:
                return

            try:
                import onnxruntime as ort

                logger.info(f"Initializing ONNX embedding manager: {self.model_name}")

                model_path = self.cache_dir / self.model_name.replace("/", "_")

                # FAST PATH — torch-free. Everything this branch touches
                # (onnxruntime + the Rust `tokenizers` lib) is measured clean of
                # torch, whereas `optimum.onnxruntime` and
                # `transformers.AutoTokenizer` both drag torch in at first
                # ENCODE. This is the path production takes on every boot after
                # the model is cached, so serving no longer needs torch at all.
                # Equivalence is not assumed: the two paths agree to 2.22e-16
                # (machine epsilon) -- see
                # docs/spikes/2026-08-19-torch-free-onnx-equivalence.py in the
                # platform repo, which prints its own verdict.
                onnx_file = self._find_onnx_file(model_path)
                tok_file = self._find_tokenizer_file(model_path)
                if onnx_file is not None and tok_file is not None:
                    from tokenizers import Tokenizer

                    tok = Tokenizer.from_file(str(tok_file))
                    tok.enable_truncation(max_length=512)
                    tok.enable_padding()
                    self._tokenizer = tok
                    self._session = ort.InferenceSession(
                        str(onnx_file), providers=["CPUExecutionProvider"]
                    )
                    self._session_input_names = {i.name for i in self._session.get_inputs()}
                    self._raw_ort = True
                    self._initialized = True
                    logger.info("ONNX embedding manager initialized (torch-free path)")
                    return

                # EXPORT PATH — needs optimum, and therefore torch. Only reached
                # the first time a model is used, which in the deployed image is
                # BUILD time (the Dockerfile pre-exports both models). If this
                # runs in a torch-free runtime it will fail loudly at import,
                # which is the correct outcome: silently degrading here would
                # mean serving embeddings from a different code path than the
                # one that was measured.
                from transformers import AutoTokenizer
                from optimum.onnxruntime import ORTModelForFeatureExtraction

                self._raw_ort = False
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=str(self.cache_dir)
                )
                self._tokenizer = tokenizer

                if not model_path.exists():
                    logger.info("Downloading and optimizing ONNX model (first-time setup)...")

                    # onnx-community pre-exported repos ship an onnx/ subfolder;
                    # use subfolder='onnx' when the model name signals it.
                    if "onnx-community" in self.model_name.lower():
                        ort_model = ORTModelForFeatureExtraction.from_pretrained(
                            self.model_name,
                            subfolder="onnx",
                            file_name="model.onnx",
                            cache_dir=str(self.cache_dir),
                        )
                    else:
                        # Export model to ONNX format on the fly
                        ort_model = ORTModelForFeatureExtraction.from_pretrained(
                            self.model_name, export=True, cache_dir=str(self.cache_dir)
                        )

                    # Save to cache. The TOKENIZER is saved beside the model on
                    # purpose: without it the torch-free fast path above has no
                    # tokenizer.json to read and would silently fall back here
                    # forever, re-importing optimum (and torch) on every boot.
                    ort_model.save_pretrained(str(model_path))
                    try:
                        tokenizer.save_pretrained(str(model_path))
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning(f"Could not save tokenizer beside ONNX model: {exc}")
                    logger.info(f"ONNX model cached to {model_path}")
                else:
                    # Load from cache (already in local subfolder form)
                    if "onnx-community" in self.model_name.lower():
                        ort_model = ORTModelForFeatureExtraction.from_pretrained(
                            str(model_path),
                            subfolder="onnx",
                            file_name="model.onnx",
                        )
                    else:
                        ort_model = ORTModelForFeatureExtraction.from_pretrained(str(model_path))

                self._session = ort_model

                self._initialized = True
                logger.info("ONNX embedding manager initialized successfully")

            except ImportError as e:
                logger.error(
                    f"ONNX dependencies not available: {e}. "
                    "Install with: pip install onnxruntime optimum"
                )
                raise
            except Exception as e:
                logger.error(f"Failed to initialize ONNX embedding manager: {e}")
                raise

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text(s) into embeddings using ONNX Runtime.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for inference (default: 32)
            normalize: L2 normalize embeddings (default: True)

        Returns:
            NumPy array of embeddings (shape: [num_texts, embedding_dim])

        Example:
            ```python
            manager = ONNXEmbeddingManager()
            embeddings = manager.encode(["hello world", "foo bar"])
            # Shape: (2, 384)
            ```
        """
        # Ensure initialization
        self._initialize()

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        try:
            if self._raw_ort:
                # Torch-free path: numpy in, numpy out, no framework tensors.
                embeddings = self._encode_raw(texts)
            else:
                encoded = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                outputs = self._session(**encoded)
                # Per-model pooling: CLS for granite/ModernBERT, mean for bge/MiniLM.
                # L2-normalize EXACTLY ONCE below — do not call normalize inside
                # pooling helpers.
                if _uses_cls_pooling(self.model_name):
                    embeddings = self._cls_pooling(outputs.last_hidden_state)
                else:
                    embeddings = self._mean_pooling(
                        outputs.last_hidden_state, encoded["attention_mask"]
                    )
                embeddings = embeddings.detach().cpu().numpy()

            # Normalize. Guard against a zero-norm row (all-zero model output
            # from pad-only input or a silent failure): dividing by 0 yields NaN
            # which then propagates silently into cosine ranking and corrupts
            # node selection. Replace a zero norm with 1.0 so the row stays
            # all-zero instead of NaN. (audit 2026-06-24)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                embeddings = embeddings / norms

            return embeddings

        except Exception as e:
            logger.error(f"ONNX inference failed: {e}")
            raise

    @staticmethod
    def _find_onnx_file(model_path):
        """Locate the exported model.onnx, or None if this model is not cached.

        Two layouts exist because two export routes exist: `save_pretrained`
        writes `model.onnx` at the top level, while onnx-community repos ship an
        `onnx/` subfolder. Returning None (rather than guessing) is what routes
        a cold model to the optimum export path instead of failing.
        """
        if not model_path.exists():
            return None
        for candidate in (model_path / "model.onnx", model_path / "onnx" / "model.onnx"):
            if candidate.is_file():
                return candidate
        return None

    def _find_tokenizer_file(self, model_path):
        """Locate tokenizer.json for the torch-free path, or None.

        Prefers the copy saved beside the model. Falls back to the HuggingFace
        hub cache so installs that predate the save-tokenizer-beside-the-model
        change still take the fast path instead of silently importing optimum
        forever -- a fallback that costs a glob once per process and removes an
        upgrade cliff.
        """
        if model_path.exists():
            direct = model_path / "tokenizer.json"
            if direct.is_file():
                return direct

        hub_dir = self.cache_dir / f"models--{self.model_name.replace('/', '--')}"
        if hub_dir.is_dir():
            matches = sorted(hub_dir.rglob("tokenizer.json"))
            if matches:
                return matches[-1]
        return None

    def _encode_raw(self, texts):
        """Torch-free encode: Rust tokenizer -> InferenceSession -> numpy pooling.

        Mirrors the optimum path exactly, including the per-model pooling
        selector -- a single global pooling choice is correct for whichever
        family you happened to test and silently wrong for the other. (A first
        port of this used CLS for everything and drifted 4.68e-02 cosine on
        bge-small, which is mean-pooled.)
        """
        encodings = self._tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._session_input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        hidden = self._session.run(None, feed)[0]

        if _uses_cls_pooling(self.model_name):
            return hidden[:, 0, :]

        # Mask-weighted mean. Clip the denominator rather than letting an
        # all-padding row divide by zero -- the caller's normalize step guards
        # zero NORMS, not zero token COUNTS, so a NaN introduced here would slip
        # past it and corrupt cosine ranking silently.
        m3 = mask[:, :, None].astype(np.float32)
        summed = (hidden * m3).sum(axis=1)
        counts = np.clip(m3.sum(axis=1), 1e-9, None)
        return summed / counts

    def _cls_pooling(self, token_embeddings):
        """
        CLS pooling — extracts the [CLS] token representation (position 0).

        Required for ModernBERT-architecture models such as granite-embedding-r2.
        The model card and sentence-transformers Pooling config both confirm
        pooling_mode_cls_token=True for these models.  Using mean pooling on
        granite yields cosine ~0.91-0.96 vs the oracle; CLS yields cos=1.000.

        Args:
            token_embeddings: Tensor of shape (batch, seq_len, hidden_size)

        Returns:
            Tensor of shape (batch, hidden_size) — the [CLS] token.
        """
        # [CLS] is at position 0 in every BERT-family model with add_special_tokens=True
        return token_embeddings[:, 0, :]

    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Mean pooling with attention mask weighting.

        Args:
            token_embeddings: Token-level embeddings
            attention_mask: Attention mask for padding tokens

        Returns:
            Pooled sentence embeddings
        """
        import torch

        # Expand mask to match embedding dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # Sum embeddings with mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / sum_mask

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension for the loaded model.

        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        self._initialize()

        # Get dimension from tokenizer config
        if hasattr(self._tokenizer, "model_max_length"):
            # Default for all-MiniLM-L6-v2
            return 384

        # Fallback: encode dummy text
        dummy_embedding = self.encode("test")
        return dummy_embedding.shape[1]

    def get_memory_usage(self) -> dict:
        """
        Get current memory usage statistics.

        Returns:
            Dict with memory usage info (in MB)
        """
        import psutil
        import sys

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "model_size_mb": (sys.getsizeof(self._session) / 1024 / 1024 if self._session else 0),
        }


# Singleton instance for global access
_onnx_manager_instance: Optional[ONNXEmbeddingManager] = None
_onnx_singleton_lock = threading.Lock()


def get_onnx_embedding_manager(
    model_name: str = DEFAULT_TEXT_MODEL,
    cache_dir: Optional[str] = None,
    quantized: bool = True,
) -> ONNXEmbeddingManager:
    """
    Get or create singleton ONNX embedding manager (thread-safe).

    Args:
        model_name: Hugging Face model identifier
        cache_dir: Optional directory for model caching
        quantized: Use quantized INT8 model (default: True)

    Returns:
        ONNXEmbeddingManager instance
    """
    global _onnx_manager_instance

    if _onnx_manager_instance is None:
        with _onnx_singleton_lock:
            if _onnx_manager_instance is None:
                _onnx_manager_instance = ONNXEmbeddingManager(
                    model_name=model_name,
                    cache_dir=cache_dir,
                    quantized=quantized,
                )

    return _onnx_manager_instance
