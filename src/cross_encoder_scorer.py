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

    @property
    def is_loaded(self) -> bool:
        """True once the ONNX model has been lazily loaded."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # Imports are local so the module has no hard optimum/torch dependency at
        # import time (it stays importable in the default, rerank-OFF path).
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = ORTModelForSequenceClassification.from_pretrained(
            self._model_id,
            subfolder=self._onnx_subfolder,
            file_name=self._onnx_file_name,
        )

    def __call__(self, query: str, documents: Sequence[str]) -> list[float]:
        docs = list(documents)
        if not docs:
            return []
        self._ensure_loaded()
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
