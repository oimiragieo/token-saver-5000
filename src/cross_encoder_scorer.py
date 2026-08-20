"""ONNX cross-encoder relevance scorer for the recall-gated rerank stage (#187).

Implements the ``reranker_gate.RerankScorer`` protocol: ``(query, [docs]) ->
[scores]``, exactly one float per document (higher = more relevant). Wraps an
INT8-quantized ONNX cross-encoder (``ms-marco-MiniLM-L-6-v2`` by default -- the
quantized ``onnx/model_quint8_avx2.onnx`` is ~23MB and runs CPU-only, so it fits
the CPU-only Fly image from #139 without the ~90MB fp32 weights).

The model + tokenizer load LAZILY on the first non-empty ``__call__`` (not at
construction), so importing this module and wiring it behind a default-OFF flag
costs nothing until a rerank actually runs.

Feasibility receipt (2026-07-11): on the dev box the INT8 model loads in ~7.4s
and scores a 3-pair batch in ~13ms, ranking a relevant doc (+4.2) above an
irrelevant one (-11.4). The pure recall-gate logic that CONSUMES this scorer
lives in ``reranker_gate.rerank_candidates`` and is unit-tested model-free with a
mock scorer; this module is the real model plug-in, still dormant/unwired.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

# A small INT8-quantized cross-encoder that runs CPU-only (no CUDA). The AVX2
# quantized variant is broadly supported on x86 servers; swap in an AVX-512 or
# fp32 variant via the constructor if the target CPU warrants it.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_ONNX_SUBFOLDER = "onnx"
DEFAULT_ONNX_FILE = "model_quint8_avx2.onnx"


class CrossEncoderScorer:
    """Lazy ONNX cross-encoder implementing ``reranker_gate.RerankScorer``.

    Construction is cheap (no model load). The model + tokenizer load on the
    FIRST non-empty ``__call__``; subsequent calls reuse them. ``__call__``
    returns one float per input document, higher = more relevant, matching the
    scorer contract ``rerank_candidates`` expects.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANK_MODEL,
        *,
        onnx_subfolder: str = DEFAULT_ONNX_SUBFOLDER,
        onnx_file_name: str = DEFAULT_ONNX_FILE,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._onnx_subfolder = onnx_subfolder
        self._onnx_file_name = onnx_file_name
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        # Which backend _ensure_loaded() selected. Set on every path (not only
        # the branch that uses it) -- an attribute that exists on one path is
        # the shape that raises AttributeError somewhere else entirely.
        self._raw_ort = False
        self._session_input_names: set = set()
        # Guards _ensure_loaded() so two concurrent callers can't both build
        # (and one publish a half-constructed pair). Construction is a rare,
        # one-time event (first non-empty __call__), so a plain Lock costs
        # nothing on the hot path once loaded (the readiness check below is
        # lock-free and happens first).
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        """True once BOTH the model and tokenizer have been published.

        Deliberately AND, not OR: _ensure_loaded() builds both in locals and
        publishes them together at the very end, so any observer -- including
        a concurrent caller racing the first load -- must never see one
        published without the other. A scorer that failed mid-construction
        (e.g. InferenceSession() raised) must read as NOT loaded, not as
        half-loaded.
        """
        return self._model is not None and self._tokenizer is not None

    def _find_cached_raw_files(self) -> tuple[Path, Path] | None:
        """Locate a cached ONNX artifact + tokenizer.json for the raw path.

        Mirrors embeddings_onnx.py::ONNXEmbeddingManager._find_onnx_file /
        _find_tokenizer_file: return None (never raise, never guess) when the
        files aren't cached locally, which routes to the optimum export
        fallback below instead of failing outright. Resolution goes through
        huggingface_hub's own cache lookup -- never a hand-rolled glob -- so
        it agrees with whatever optimum/transformers would also resolve.
        """
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return None

        onnx_rel = (
            f"{self._onnx_subfolder}/{self._onnx_file_name}"
            if self._onnx_subfolder
            else self._onnx_file_name
        )
        onnx_path = try_to_load_from_cache(self._model_id, onnx_rel)
        tok_path = try_to_load_from_cache(self._model_id, "tokenizer.json")
        if not onnx_path or not tok_path:
            return None
        return Path(onnx_path), Path(tok_path)

    def _ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        with self._load_lock:
            # Double-checked: another thread may have finished loading (or a
            # failed attempt may have left nothing published) while this
            # thread waited for the lock.
            if self.is_loaded:
                return

            # Build everything in LOCALS first. Nothing on `self` is touched
            # until the very end, and every attribute is published in one
            # block together -- a concurrent reader (or a reader on the next
            # call, if construction raises partway through) can never observe
            # a tokenizer without a model or vice versa. A half-published
            # scorer previously could pass a stale `self._tokenizer is not
            # None` guard while `self._model` was never set (or was set to a
            # session that failed construction), which would send a real
            # request into a scorer that could never actually score.
            raw_ort: bool
            tokenizer_local = None
            model_local = None
            session_input_names: set = set()

            # FAST PATH -- torch-free. onnxruntime + the Rust `tokenizers`
            # lib never import torch (measured: docs/spikes/2026-08-19-
            # torch-free-onnx-equivalence.py); optimum.onnxruntime and
            # transformers.AutoTokenizer both do, at import time. Equivalence
            # with the optimum path is not assumed -- docs/spikes/2026-08-20-
            # cross-encoder-raw-ort-equivalence.py measured 0.000e+00 worst
            # |logit delta| between the two on all 4 shipped ONNX artifacts,
            # across single/mixed-length/near-512/heavily-padded/truncation-
            # asymmetry batches, with identical ranking on every batch.
            cached = self._find_cached_raw_files()
            if cached is not None:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                onnx_path, tok_path = cached
                tok = Tokenizer.from_file(str(tok_path))
                # PARITY, not a new choice: `padding=True, truncation=True`
                # on a transformers tokenizer pair defaults to
                # strategy="longest_first" -- reproduced exactly, not
                # switched to only_second (that is a separate, optional
                # follow-up needing its own retrieval-quality evidence per
                # the plan; bundling it here was a v1 defect).
                tok.enable_truncation(max_length=self._max_length, strategy="longest_first")
                tok.enable_padding()
                # If InferenceSession() raises here, nothing on `self` has
                # been touched -- the next call retries the load cleanly
                # instead of reading a stale half-loaded state.
                session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                tokenizer_local = tok
                model_local = session
                session_input_names = {i.name for i in session.get_inputs()}
                raw_ort = True
            else:
                # EXPORT PATH -- needs optimum, and therefore torch. Reached
                # only when the raw files aren't cached locally (first-time
                # setup / build time). Imports are local so the module has no
                # hard optimum/torch dependency at plain import time (it
                # stays importable in the default, rerank-OFF path).
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer

                tokenizer_local = AutoTokenizer.from_pretrained(self._model_id)
                model_local = ORTModelForSequenceClassification.from_pretrained(
                    self._model_id,
                    subfolder=self._onnx_subfolder,
                    file_name=self._onnx_file_name,
                )
                raw_ort = False

            # Publish atomically -- every attribute a caller reads (is_loaded,
            # __call__'s self._raw_ort branch, _score_raw's session input
            # names) becomes visible together.
            self._tokenizer = tokenizer_local
            self._model = model_local
            self._session_input_names = session_input_names
            self._raw_ort = raw_ort

    def __call__(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        self._ensure_loaded()

        if self._raw_ort:
            return self._score_raw(query, docs)

        import torch

        encoded = self._tokenizer(
            [[query, doc] for doc in docs],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**dict(encoded))
        logits = output.logits
        # ms-marco cross-encoders emit a single relevance logit per pair -> [N, 1].
        if logits.ndim == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return [float(x) for x in logits.tolist()]

    def _score_raw(self, query: str, docs: list[str]) -> list[float]:
        """Torch-free score: Rust tokenizer batch-encode -> InferenceSession -> numpy.

        Every document for this query is encoded and run in ONE batch, on
        purpose: the spike measured the quantized graphs to be batch-padding
        sensitive (batched vs one-doc-at-a-time can differ by >0.3 on the same
        graph) -- per-document calls would silently diverge from what
        equivalence was actually measured against.
        """
        import numpy as np

        encodings = self._tokenizer.encode_batch([[query, doc] for doc in docs])
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._session_input_names:
            # ORT silently zero-fills an absent declared input; on a
            # cross-encoder that means every doc segment reads as query
            # tokens, corrupting relevance without erroring. Always supply
            # real pair type_ids when the graph declares the input.
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        out = self._model.run(None, feed)[0]
        if out.ndim != 2 or out.shape[-1] != 1:
            raise ValueError(
                f"expected cross-encoder logits shape [N,1], got {out.shape} for "
                f"{self._model_id}/{self._onnx_file_name} -- this scorer only "
                "supports single-logit relevance heads."
            )
        return [float(x) for x in out[:, 0].tolist()]
