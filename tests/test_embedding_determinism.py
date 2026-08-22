"""Prove the compression path's embedding bytes are DETERMINISTIC, run to run.

WHY THIS EXISTS. arXiv 2607.15516 (2026-07-17) measured that compression which
is not byte-stable DEFEATS provider prefix caching (Anthropic `cache_control`,
OpenAI automatic caching): the cached prefix changes every call so it never
hits, and below ~6x compression that can cost the customer MORE than not
compressing. Our engine separates a query-agnostic `cache_stable_prefix` from
the query-conditioned `skeleton_text` -- architecturally the good case -- but
nothing PROVED the bytes are stable.

Measured concern: `src/embeddings_onnx.py` constructs `ort.InferenceSession`
with no `SessionOptions`, so `intra_op_num_threads` defaults to 0 ("runtime
picks"). Unpinned multi-threaded float reduction CAN reorder sums and shift
float32 bits. Whether it DOES for our graph is exactly what this test settles.

WHAT IS TESTED (both arms compare exact bytes, never np.allclose):
  1. SAME-PROCESS -- embed the same input twice in one interpreter.
  2. CROSS-PROCESS -- THE CLAIM THAT MATTERS. The deployed runtime is a fresh
     interpreter every time; in-process-only can pass while cross-process
     differs (thread-pool sizing, import order, env). Mirrors the subprocess
     style in tests/test_onnx_torch_free_path.py.

Chose `ONNXEmbeddingManager.encode()` directly rather than a full
`SemanticCompressor` ingest: it is the smallest real path that exercises the
ONNX session + tokenizer, which is exactly the mechanism arXiv 2607.15516's
concern lives in. A full compressor ingest would additionally exercise
ranking/chunking, which is not what the byte-determinism claim is about and
would slow the test down for no added discrimination.

NON-VACUITY (mandatory, not optional):
  - assert the embedding is non-empty and has the expected dimensionality
    BEFORE any determinism comparison -- two empty outputs compare equal and
    prove nothing.
  - assert two DIFFERENT inputs produce DIFFERENT bytes (the discrimination
    control) -- without this, a stubbed/broken encoder returning a constant
    vector would pass every determinism arm trivially.

GATING: reuses `_model_is_cached()` from test_onnx_torch_free_path.py so a
developer without the exported model skips cleanly. Engine CI now WARMS the
model (see .github/workflows/ci.yml "Warm the ..." steps), so this test DOES
run there -- a skip here on CI would itself be the bug this file exists to
catch, so ran_or_skipped must be reported honestly by whoever runs this.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.embeddings_onnx import ONNXEmbeddingManager

from tests.test_onnx_torch_free_path import _model_is_cached

REPO_ROOT = Path(__file__).resolve().parents[1]

_TEXT_A = (
    "Semantic compression separates a query-agnostic cache_stable_prefix from "
    "a query-conditioned skeleton_text so that provider prefix caching can hit "
    "on repeated calls against the same document."
)
_TEXT_B = (
    "A completely different sentence about token-saver-5000's ranking and "
    "chunking pipeline, chosen to be lexically and semantically distinct from "
    "the other fixture string used in this determinism test."
)


def _assert_non_vacuous_embedding(vec: np.ndarray, *, expected_dim: int = 384) -> None:
    """Guard against comparing two empty/constant outputs and calling it proof."""
    assert vec is not None, "embedding must not be None"
    arr = np.asarray(vec)
    assert arr.size > 0, "embedding must be non-empty"
    assert arr.shape[-1] == expected_dim, f"unexpected embedding dim: {arr.shape}"
    assert not np.all(arr == 0), "embedding is all-zero -- looks like a broken/stub encoder"


@pytest.mark.skipif(not _model_is_cached(), reason="ONNX model not in the local cache")
def test_same_process_embedding_is_byte_identical() -> None:
    """Embed the same input twice in ONE interpreter; bytes must match exactly."""
    m = ONNXEmbeddingManager()

    vec1 = np.asarray(m.encode([_TEXT_A])[0])
    vec2 = np.asarray(m.encode([_TEXT_A])[0])
    _assert_non_vacuous_embedding(vec1)
    _assert_non_vacuous_embedding(vec2)

    b1, b2 = vec1.tobytes(), vec2.tobytes()
    if b1 != b2:
        first_diff = next(i for i in range(min(len(b1), len(b2))) if b1[i] != b2[i])
        delta = float(np.max(np.abs(vec1 - vec2)))
        pytest.fail(
            "BLOCKER: embeddings are non-deterministic across same-process calls: "
            f"first differing byte offset={first_diff}, max abs float delta={delta:.3e}"
        )

    # Discrimination control: a different input must NOT produce the same bytes,
    # or this whole test would pass against a broken constant-output encoder.
    vec_other = np.asarray(m.encode([_TEXT_B])[0])
    _assert_non_vacuous_embedding(vec_other)
    assert vec_other.tobytes() != b1, (
        "different inputs produced identical embedding bytes -- the encoder "
        "looks constant/stubbed, which would make the determinism claim above vacuous"
    )


@pytest.mark.skipif(not _model_is_cached(), reason="ONNX model not in the local cache")
def test_cross_process_embedding_is_byte_identical() -> None:
    """THE CLAIM THAT MATTERS: a fresh interpreter must reproduce the same bytes.

    The deployed runtime is a fresh interpreter on every boot/request path, so
    in-process-only determinism is not sufficient evidence -- thread-pool
    sizing, import order, and env can all differ across processes even when
    they never differ within one.
    """
    code = (
        "import sys, numpy as np; "
        "from src.embeddings_onnx import ONNXEmbeddingManager as M; "
        "m = M(); "
        "v = np.asarray(m.encode([" + repr(_TEXT_A) + "])[0]); "
        "sys.stdout.buffer.write(b'RESULT_BYTES_BEGIN'); "
        "sys.stdout.buffer.write(v.tobytes()); "
        "sys.stdout.buffer.write(b'RESULT_BYTES_END'); "
        "sys.stdout.buffer.write(('RESULT_SHAPE ' + str(v.shape[-1])).encode()); "
    )

    def _run_once() -> tuple[bytes, int]:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            cwd=str(REPO_ROOT),
            timeout=600,
        )
        out = proc.stdout
        begin = out.find(b"RESULT_BYTES_BEGIN")
        end = out.find(b"RESULT_BYTES_END")
        assert begin != -1 and end != -1, (
            f"subprocess produced no parsable result; stderr tail: "
            f"{proc.stderr[-800:].decode(errors='replace')}"
        )
        payload = out[begin + len(b"RESULT_BYTES_BEGIN") : end]
        shape_marker = out[end + len(b"RESULT_BYTES_END") :].decode(errors="replace")
        dim = int(shape_marker.strip().split()[-1])
        return payload, dim

    b1, dim1 = _run_once()
    b2, dim2 = _run_once()

    assert len(b1) > 0 and len(b2) > 0, "cross-process embedding bytes must be non-empty"
    assert dim1 == 384 and dim2 == 384, f"unexpected embedding dim: {dim1}, {dim2}"
    assert not all(b == 0 for b in b1), "cross-process embedding is all-zero bytes"

    if b1 != b2:
        n = min(len(b1), len(b2))
        first_diff = next((i for i in range(n) if b1[i] != b2[i]), n)
        v1 = np.frombuffer(b1, dtype=np.float32)
        v2 = np.frombuffer(b2, dtype=np.float32)
        delta = float(np.max(np.abs(v1 - v2))) if len(v1) == len(v2) else float("nan")
        pytest.fail(
            "BLOCKER: embeddings are non-deterministic across processes: "
            f"first differing byte offset={first_diff}, max abs float delta={delta:.3e}"
        )
