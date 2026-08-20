"""The ONNX embedder must serve without torch, and must not change its vectors.

WHY THIS EXISTS. `optimum.onnxruntime` and `transformers.AutoTokenizer` both
pull torch in when the first embedding is COMPUTED -- not at import, which is
what made this expensive to see: `embeddings_onnx` imports torch inside
functions and so measured clean at boot while dragging ~1.25GB of torch into
every runtime image anyway.

    optimum.onnxruntime                    pulls torch -> True
    transformers.AutoTokenizer             pulls torch -> True
    onnxruntime alone                      pulls torch -> False
    tokenizers (the Rust lib) alone        pulls torch -> False

So the cached-model path now builds its session with
`onnxruntime.InferenceSession` and tokenizes with the Rust `tokenizers` lib.
Measured old-vs-new on the real class: worst per-element difference 1.49e-08
(float32 precision) with `torch_loaded` False vs True.

THE TRAP THIS LOCKS. A first port used CLS pooling for every model and drifted
4.68e-02 cosine on bge-small, which is MEAN-pooled -- one step from being filed
as "raw onnxruntime is not equivalent" when the probe was what was wrong.
`_uses_cls_pooling` must therefore be honoured on the raw path too: a single
global pooling choice is right for whichever family you tested and silently
wrong for the other, and nothing about the output SHAPE reveals it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.embeddings_onnx import ONNXEmbeddingManager, _uses_cls_pooling

REPO_ROOT = Path(__file__).resolve().parents[1]


def _model_is_cached() -> bool:
    """Is the exported model on disk? A PURE FILESYSTEM fact, deliberately.

    The first version of this called `_find_tokenizer_file`, which is part of
    the mechanism under test -- so a red-check that disabled that function
    turned the headline test from FAIL into SKIP. A skip reads as green, so a
    regression in the torch-free path would have silently switched its own
    guard off. A population gate must never share a mechanism with the thing it
    gates.

    Tokenizer discovery is now inside the assertions where it belongs: if the
    model is cached and the tokenizer cannot be found, that is a FAILURE to
    report, not a reason to stay quiet.
    """
    m = ONNXEmbeddingManager()
    d = m.cache_dir / m.model_name.replace("/", "_")
    return (d / "model.onnx").is_file() or (d / "onnx" / "model.onnx").is_file()


class _StubSession:
    """Minimal InferenceSession stand-in returning a known hidden state."""

    def __init__(self, hidden: np.ndarray, input_names=("input_ids", "attention_mask")):
        self._hidden = hidden
        self._names = input_names
        self.last_feed: dict | None = None

    def get_inputs(self):
        return [type("I", (), {"name": n})() for n in self._names]

    def run(self, _outputs, feed):
        self.last_feed = feed
        return [self._hidden]


class _StubTokenizer:
    def __init__(self, n_tokens: int = 3):
        self._n = n_tokens

    def encode_batch(self, texts):
        out = []
        for _ in texts:
            out.append(
                type(
                    "E",
                    (),
                    {
                        "ids": list(range(self._n)),
                        "attention_mask": [1] * self._n,
                        "type_ids": [0] * self._n,
                    },
                )()
            )
        return out


def _manager_with_stubs(
    model_name: str, hidden: np.ndarray, input_names=("input_ids", "attention_mask")
):
    m = ONNXEmbeddingManager(model_name=model_name)
    m._tokenizer = _StubTokenizer(n_tokens=hidden.shape[1])
    m._session = _StubSession(hidden, input_names)
    m._session_input_names = set(input_names)
    m._raw_ort = True
    m._initialized = True
    return m


def test_raw_path_honours_the_per_model_pooling_selector() -> None:
    """CLS for granite, mask-weighted MEAN for bge -- on the RAW path too.

    Built so the two poolings CANNOT coincide: token 0 is all ones and the rest
    are all zeros, so CLS gives 1.0 and mean over 3 tokens gives 1/3.
    """
    hidden = np.zeros((1, 3, 4), dtype=np.float32)
    hidden[0, 0, :] = 1.0

    granite = _manager_with_stubs("onnx-community/granite-embedding-small-english-r2-ONNX", hidden)
    bge = _manager_with_stubs("BAAI/bge-small-en-v1.5", hidden)

    assert _uses_cls_pooling(granite.model_name) is True
    assert _uses_cls_pooling(bge.model_name) is False

    cls_vec = granite._encode_raw(["x"])
    mean_vec = bge._encode_raw(["x"])

    assert np.allclose(cls_vec, 1.0), f"granite must use CLS pooling, got {cls_vec}"
    assert np.allclose(mean_vec, 1.0 / 3.0), f"bge must use MEAN pooling, got {mean_vec}"
    assert not np.allclose(cls_vec, mean_vec), "the fixture cannot discriminate the two poolings"


def test_token_type_ids_are_sent_only_when_the_session_declares_them() -> None:
    """Feeding an undeclared input raises in onnxruntime; omitting a declared one also fails."""
    hidden = np.ones((1, 2, 4), dtype=np.float32)

    without = _manager_with_stubs("BAAI/bge-small-en-v1.5", hidden)
    without._encode_raw(["x"])
    assert "token_type_ids" not in without._session.last_feed

    with_tt = _manager_with_stubs(
        "BAAI/bge-small-en-v1.5",
        hidden,
        input_names=("input_ids", "attention_mask", "token_type_ids"),
    )
    with_tt._encode_raw(["x"])
    assert "token_type_ids" in with_tt._session.last_feed


def test_an_all_padding_row_does_not_become_nan() -> None:
    """The caller guards zero NORMS, not zero token COUNTS.

    A NaN introduced by dividing by a zero token count would slip past that
    guard and corrupt cosine ranking silently, which is the worst available
    outcome: no error, wrong order.
    """
    hidden = np.ones((1, 2, 4), dtype=np.float32)
    m = _manager_with_stubs("BAAI/bge-small-en-v1.5", hidden)

    class _AllPad(_StubTokenizer):
        def encode_batch(self, texts):
            encs = super().encode_batch(texts)
            for e in encs:
                e.attention_mask = [0] * len(e.ids)
            return encs

    m._tokenizer = _AllPad(n_tokens=2)
    out = m._encode_raw(["x"])
    assert not np.isnan(out).any(), f"all-padding row produced NaN: {out}"


def test_a_cold_model_routes_to_the_export_path(tmp_path: Path) -> None:
    """`_find_onnx_file` must return None rather than guess.

    Returning a path that does not exist would fail inside InferenceSession
    with a confusing error instead of falling back to the optimum export.
    """
    m = ONNXEmbeddingManager(cache_dir=str(tmp_path))
    assert m._find_onnx_file(tmp_path / "not-there") is None

    d = tmp_path / "model"
    d.mkdir()
    assert m._find_onnx_file(d) is None, "an empty dir is not an exported model"

    (d / "model.onnx").write_bytes(b"stub")
    assert m._find_onnx_file(d) == d / "model.onnx"

    nested = tmp_path / "community"
    (nested / "onnx").mkdir(parents=True)
    (nested / "onnx" / "model.onnx").write_bytes(b"stub")
    assert m._find_onnx_file(nested) == nested / "onnx" / "model.onnx"


@pytest.mark.skipif(not _model_is_cached(), reason="ONNX model not in the local cache")
def test_a_real_encode_does_not_load_torch() -> None:
    """THE HEADLINE, in a SUBPROCESS.

    In-process this can only ever pass, because pytest has already imported
    torch through some other test. The claim is about a fresh interpreter --
    which is what the runtime image actually is.
    """
    code = (
        "import sys; "
        "from src.embeddings_onnx import ONNXEmbeddingManager as M; "
        "m=M(); v=m.encode(['hello world']); "
        "print('RESULT', m._raw_ort, v.shape[1], 'torch' in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    line = next((x for x in proc.stdout.splitlines() if x.startswith("RESULT")), None)
    assert line, f"probe produced no result; stderr tail: {proc.stderr.strip()[-400:]}"

    _, raw_ort, dim, torch_loaded = line.split()
    assert raw_ort == "True", "cached model did not take the torch-free path"
    assert dim == "384", f"unexpected embedding dim {dim}"
    assert torch_loaded == "False", (
        "a real encode still loaded torch -- the runtime image cannot drop it. "
        "Most likely the tokenizer.json beside the model is missing, so "
        "_initialize fell back to the optimum export path."
    )
