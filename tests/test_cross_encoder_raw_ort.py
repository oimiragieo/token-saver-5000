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

WHY A SUBPROCESS. Same reason as test_mcp_chain_imports_without_torch.py: in
this pytest process torch is already imported by something else, so
``sys.modules`` is poisoned before the first assertion. Blocking the import at
the meta-path in a fresh interpreter is the only way to prove the CONSTRUCT
and SCORE path never touches torch.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )


def test_the_block_itself_works() -> None:
    """NON-VACUITY: if the block does not block, every test below is theatre."""
    proc = _run("\nimport torch\nprint('IMPORTED')\n")
    assert "IMPORTED" not in proc.stdout, (
        "torch imported despite the block - the remaining tests in this file "
        "prove nothing about a torch-free cross-encoder path"
    )


def test_cross_encoder_scores_one_pair_torch_free() -> None:
    """THE HEADLINE -- construct + score without torch ever entering sys.modules.

    Requires the default cross-encoder model to already be in the local HF
    cache (populated once via the existing optimum-backed test path or the
    2026-08-20 spike run) -- this mirrors test_cross_encoder_scorer.py's own
    ``_model_cached`` skip so a clean-cache CI box doesn't force a download.
    """
    proc = _run(
        "\n"
        "import importlib.util\n"
        "if importlib.util.find_spec('huggingface_hub') is None:\n"
        "    print('SKIP no huggingface_hub')\n"
        "    raise SystemExit(0)\n"
        "from huggingface_hub import try_to_load_from_cache\n"
        "from src.cross_encoder_scorer import DEFAULT_RERANK_MODEL\n"
        "if try_to_load_from_cache(DEFAULT_RERANK_MODEL, 'config.json') is None:\n"
        "    print('SKIP model not cached')\n"
        "    raise SystemExit(0)\n"
        "from src.cross_encoder_scorer import CrossEncoderScorer\n"
        "scorer = CrossEncoderScorer()\n"
        "scores = scorer('a query', ['a document'])\n"
        "import math\n"
        "assert isinstance(scores, list), f'expected list, got {type(scores)}'\n"
        "assert len(scores) == 1, f'expected 1 score, got {len(scores)}'\n"
        "assert isinstance(scores[0], float), f'expected float, got {type(scores[0])}'\n"
        "assert math.isfinite(scores[0]), f'non-finite score: {scores[0]}'\n"
        "assert 'torch' not in sys.modules, 'torch entered sys.modules during scoring'\n"
        "print('OK', scores[0])\n"
    )
    if "SKIP" in proc.stdout:
        import pytest

        pytest.skip(proc.stdout.strip())
    assert "OK" in proc.stdout, (
        "CrossEncoderScorer could not construct+score without torch.\n"
        f"stdout: {proc.stdout.strip()[-800:]}\n"
        f"stderr tail: {proc.stderr.strip()[-800:]}"
    )
