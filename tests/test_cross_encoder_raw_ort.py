"""The cross-encoder rerank scorer must run torch-free on a cached model.

WHY. The embeddings path already proved (docs/spikes/2026-08-19-torch-free-
onnx-equivalence.py, machine-epsilon agreement) that onnxruntime + tokenizers
alone can replace optimum + transformers + torch. The cross-encoder scorer
(``src/cross_encoder_scorer.py``) still hard-imports ``optimum.onnxruntime``
and ``transformers.AutoTokenizer`` in ``_ensure_loaded``, and ``torch`` in
``__call__``, so it is the ONE reranker still requiring the full torch stack
even in a runtime image (docs/spikes/2026-08-19...) that has already dropped
torch everywhere else. Task B of
docs/superpowers/plans/2026-08-20-backlog-closeout-plan.md ports it, gated by
docs/spikes/2026-08-20-cross-encoder-raw-ort-equivalence.py, which measured
bit-identical (0.000e+00 worst |logit delta|) agreement between raw
onnxruntime + tokenizers and the production optimum path, on all 4 shipped
ONNX artifacts including the production default (model_quint8_avx2.onnx).

WHY A SUBPROCESS for the torch-free tests. Same reason as
test_mcp_chain_imports_without_torch.py: in this pytest process torch is
already imported by something else, so ``sys.modules`` is poisoned before the
first assertion. Blocking the import at the meta-path in a fresh interpreter
is the only way to prove the CONSTRUCT and SCORE path never touches torch.

The contract tests below (token_type_ids population, dtype, batching parity,
logits-shape rejection) test CORRECTNESS of the raw path's own numerics, not
torch-freedom, so they run in-process directly against ``CrossEncoderScorer``
-- no subprocess needed, and torch already being imported by pytest's own
collection is irrelevant to what they check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.cross_encoder_scorer import DEFAULT_RERANK_MODEL, CrossEncoderScorer

REPO_ROOT = Path(__file__).resolve().parents[1]

# Same block as test_mcp_chain_imports_without_torch.py -- kept verbatim so the
# two guards stay comparable and any future fix to one is obviously a fix to
# apply to the other too.
_BLOCK_TORCH = """
import sys

class _BlockTorch:
    def find_module(self, name, path=None):
        return self if name == "torch" or name.startswith("torch.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(f"No module named {name!r} (blocked by the test)")
        return None

sys.meta_path.insert(0, _BlockTorch())
assert "torch" not in sys.modules, "torch was already imported - the block is too late"
"""

# The exact substring the blocker's own ImportError carries. A block failing
# for any OTHER reason (a typo in the subprocess body, an unrelated crash, a
# missing dependency) must NOT be mistaken for proof the block works -- a
# soft "IMPORTED not in stdout" check passes on every one of those too.
_BLOCK_SIGNATURE = "(blocked by the test)"


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )


def _default_artifacts_cached() -> bool:
    """True only if the ACTUAL files the raw path will read are cached.

    A prior version checked only config.json's presence -- config.json is
    fetched by the optimum EXPORT path too, so it can exist while the raw
    path's onnx artifact or tokenizer.json is still missing, which would
    silently degrade the headline test to always exercising (and always
    passing through) the torch-requiring fallback instead of the path this
    test exists to prove.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    from src.cross_encoder_scorer import DEFAULT_ONNX_FILE, DEFAULT_ONNX_SUBFOLDER

    onnx_rel = f"{DEFAULT_ONNX_SUBFOLDER}/{DEFAULT_ONNX_FILE}"
    return (
        try_to_load_from_cache(DEFAULT_RERANK_MODEL, onnx_rel) is not None
        and try_to_load_from_cache(DEFAULT_RERANK_MODEL, "tokenizer.json") is not None
    )


def test_the_block_itself_works() -> None:
    """NON-VACUITY: if the block does not block for the RIGHT reason, every
    test below is theatre.

    Two assertions, not one: torch must not import (the outcome), AND the
    process must have died with the block's OWN error signature (the
    mechanism). A body that fails for any unrelated reason -- a stray typo,
    an unrelated missing module, a crash before the import line even runs --
    would also leave "IMPORTED" out of stdout, and a check that only looks
    for that string's absence cannot tell the two apart.
    """
    proc = _run("\nimport torch\nprint('IMPORTED')\n")
    assert "IMPORTED" not in proc.stdout, (
        "torch imported despite the block - the remaining tests in this file "
        "prove nothing about a torch-free cross-encoder path"
    )
    assert _BLOCK_SIGNATURE in proc.stderr, (
        "torch import failed, but NOT with the blocker's own signature - this "
        "proves nothing about the block actually firing (could be any other "
        f"crash).\nstderr tail: {proc.stderr.strip()[-500:]}"
    )
    assert (
        "ImportError" in proc.stderr
    ), f"expected an ImportError, got:\n{proc.stderr.strip()[-500:]}"


def test_cross_encoder_scores_one_pair_torch_free() -> None:
    """THE HEADLINE -- construct + score without torch ever entering sys.modules.

    Requires the RAW files the fast path reads (the default ONNX artifact +
    tokenizer.json, not merely config.json) to already be in the local HF
    cache -- this mirrors test_cross_encoder_scorer.py's own ``_model_cached``
    skip so a clean-cache CI box doesn't force a download, but checks the
    specific files this test exercises rather than a file every backend
    fetches regardless of which path will run.
    """
    if not _default_artifacts_cached():
        pytest.skip("raw ONNX artifact + tokenizer.json not cached locally")

    proc = _run(
        "\n"
        "from src.cross_encoder_scorer import CrossEncoderScorer\n"
        "scorer = CrossEncoderScorer()\n"
        "scores = scorer('a query', ['a document'])\n"
        "import math\n"
        "assert isinstance(scores, list), f'expected list, got {type(scores)}'\n"
        "assert len(scores) == 1, f'expected 1 score, got {len(scores)}'\n"
        "assert isinstance(scores[0], float), f'expected float, got {type(scores[0])}'\n"
        "assert math.isfinite(scores[0]), f'non-finite score: {scores[0]}'\n"
        "assert scorer._raw_ort is True, 'expected the raw path, got the optimum fallback'\n"
        "assert 'torch' not in sys.modules, 'torch entered sys.modules during scoring'\n"
        "print('OK', scores[0])\n"
    )
    assert "OK" in proc.stdout, (
        "CrossEncoderScorer could not construct+score without torch.\n"
        f"stdout: {proc.stdout.strip()[-800:]}\n"
        f"stderr tail: {proc.stderr.strip()[-800:]}"
    )


# ---------------------------------------------------------------------------
# Contract tests -- correctness of the raw path's own numerics. In-process,
# no torch-block needed: these check WHAT gets fed to the graph and WHAT
# comes back, not whether torch was imported to get there.
# ---------------------------------------------------------------------------


def _real_scorer_or_skip(**kwargs) -> CrossEncoderScorer:
    if not _default_artifacts_cached():
        pytest.skip("raw ONNX artifact + tokenizer.json not cached locally")
    scorer = CrossEncoderScorer(**kwargs)
    scorer._ensure_loaded()
    if not scorer._raw_ort:
        pytest.skip("cached files present but the raw path was not selected")
    return scorer


def test_token_type_ids_populated_for_document_segment() -> None:
    """The doc segment must carry NONZERO type_ids -- not ORT's silent
    zero-fill, which would make every doc segment read as query tokens."""
    scorer = _real_scorer_or_skip()
    assert "token_type_ids" in scorer._session_input_names, (
        "this test assumes the cached graph declares token_type_ids; if it "
        "doesn't, the zero-fill risk this test guards against doesn't apply"
    )
    enc = scorer._tokenizer.encode("a query", "a document")
    assert any(
        t == 1 for t in enc.type_ids
    ), f"expected at least one segment-1 (document) type id, got all: {enc.type_ids}"
    # negative control: the query segment must be all segment 0
    query_only = scorer._tokenizer.encode("a query")
    assert all(
        t == 0 for t in query_only.type_ids
    ), f"single-segment encode should be all type_id 0, got: {query_only.type_ids}"


def test_feed_arrays_are_int64() -> None:
    """input_ids / attention_mask / token_type_ids must all be int64 -- ORT
    is picky about integer input dtype and a silent int32/float cast would
    either error opaquely or, worse, run and produce wrong numbers."""
    scorer = _real_scorer_or_skip()
    captured: dict = {}
    original_run = scorer._model.run

    def _capturing_run(output_names, feed, *a, **kw):
        captured.update(feed)
        return original_run(output_names, feed, *a, **kw)

    scorer._model.run = _capturing_run
    try:
        scorer("a query", ["doc one", "doc two"])
    finally:
        scorer._model.run = original_run

    assert captured, "the session's run() was never called"
    for name, arr in captured.items():
        assert isinstance(arr, np.ndarray), f"{name} is not a numpy array: {type(arr)}"
        assert arr.dtype == np.int64, f"{name} has dtype {arr.dtype}, expected int64"


def test_batching_matches_singleton_on_fp32_artifact() -> None:
    """Correct padding + masking means BATCHED and one-doc-at-a-time scoring
    must agree, on an artifact where numerical noise from padding can't hide
    a real bug: the FP32 graph.

    Deliberately NOT run against the default quantized artifact. The
    2026-08-20 spike measured the shipped INT8 graphs to be genuinely
    batch-padding sensitive (batched vs singleton differed by up to ~0.3 on
    the SAME correct feed) -- an inherent property of that quantized graph,
    not a defect in this scorer's tokenization or masking. Asserting exact
    batched==singleton equality against a quantized artifact would either
    mask a real feed bug behind quantization noise, or flag quantization
    noise as a feed bug; the FP32 artifact has neither problem (measured
    2026-08-20: batched-vs-singleton delta 0.000e+00), so it is the
    discriminating artifact for THIS property.
    """
    scorer = _real_scorer_or_skip(onnx_file_name="model.onnx")
    query = "a somewhat longer query about semantic compression pipelines"
    docs = ["short doc", "x" * 800, "d"]

    batched = scorer(query, docs)
    singleton = [scorer(query, [d])[0] for d in docs]

    assert len(batched) == len(singleton) == len(docs)
    for i, (b, s) in enumerate(zip(batched, singleton)):
        assert abs(b - s) < 1e-4, (
            f"doc {i}: batched={b} singleton={s} delta={abs(b - s):.3e} -- "
            "correct attention-mask padding should make these agree on the "
            "FP32 artifact; a divergence here indicates a real feed/padding "
            "bug, not quantization noise"
        )


def test_logits_shape_n2_is_rejected() -> None:
    """[N,2] logits must raise a clear error, never be silently squeezed or
    silently indexed into the wrong column.

    No cached model needed: a session double standing in for the real
    InferenceSession is enough to exercise the shape-check branch
    deterministically and fast, matching the spike's explicit REJECT-[N,2]
    requirement without a network dependency.
    """

    class _FakeSessionN2:
        def get_inputs(self):
            return []

        def run(self, output_names, feed):
            n = feed["input_ids"].shape[0]
            return [np.zeros((n, 2), dtype=np.float32)]

    scorer = CrossEncoderScorer()
    scorer._tokenizer = _RealTokenizerStub()
    scorer._model = _FakeSessionN2()
    scorer._session_input_names = set()
    scorer._raw_ort = True

    with pytest.raises(ValueError, match=r"\[N,1\]"):
        scorer._score_raw("q", ["d1", "d2"])


def test_logits_shape_n1_squeezes_to_flat_list() -> None:
    """[N,1] logits must squeeze to a flat length-N list of floats."""

    class _FakeSessionN1:
        def get_inputs(self):
            return []

        def run(self, output_names, feed):
            n = feed["input_ids"].shape[0]
            return [np.arange(n, dtype=np.float32).reshape(n, 1)]

    scorer = CrossEncoderScorer()
    scorer._tokenizer = _RealTokenizerStub()
    scorer._model = _FakeSessionN1()
    scorer._session_input_names = set()
    scorer._raw_ort = True

    out = scorer._score_raw("q", ["d1", "d2", "d3"])
    assert out == [0.0, 1.0, 2.0]
    assert all(isinstance(x, float) for x in out)


class _RealTokenizerStub:
    """Minimal encode_batch stand-in -- shape tests don't care about real
    token ids, only that ids/mask/type_ids arrays have the right cardinality."""

    class _Enc:
        def __init__(self, n: int) -> None:
            self.ids = [0] * n
            self.attention_mask = [1] * n
            self.type_ids = [0] * n

    def encode_batch(self, pairs):
        return [self._Enc(4) for _ in pairs]
