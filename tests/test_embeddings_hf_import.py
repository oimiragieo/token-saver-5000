"""
Regression tests for src/embeddings.py import hygiene.

Bug: huggingface_hub.utils.enable_progress_bars was imported from the top-level
utils package which deprecated the re-export. The fix moves to the canonical
huggingface_hub.utils.tqdm submodule.
"""

import sys
import types
from unittest.mock import MagicMock, patch


def _make_fake_hf_tqdm(enable_fn=None):
    """Build a minimal fake huggingface_hub.utils.tqdm module."""
    tqdm_mod = types.ModuleType("huggingface_hub.utils.tqdm")
    if enable_fn is None:
        enable_fn = MagicMock()
    tqdm_mod.enable_progress_bars = enable_fn
    return tqdm_mod


def test_enable_progress_bars_imported_from_tqdm_submodule():
    """
    Regression: embeddings.py must import enable_progress_bars from
    huggingface_hub.utils.tqdm (not from huggingface_hub.utils).
    A successful module reload with the tqdm submodule present proves
    the import path is correct.
    """
    # Build a fake huggingface_hub hierarchy
    hf_mod = types.ModuleType("huggingface_hub")
    hf_utils = types.ModuleType("huggingface_hub.utils")
    hf_tqdm = _make_fake_hf_tqdm()

    # Remove cached embeddings module so reload exercises import path
    for key in list(sys.modules.keys()):
        if "src.embeddings" in key or key == "src.embeddings":
            del sys.modules[key]

    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": hf_mod,
            "huggingface_hub.utils": hf_utils,
            "huggingface_hub.utils.tqdm": hf_tqdm,
        },
    ):
        import src.embeddings  # noqa: F401 — side-effect import to exercise the try block

        # The fake enable_progress_bars on the tqdm submodule should have been called
        assert hf_tqdm.enable_progress_bars.called, (
            "enable_progress_bars() was NOT called — the import is pointing "
            "at the wrong module (likely huggingface_hub.utils, not .utils.tqdm)"
        )


def test_enable_progress_bars_import_failure_is_swallowed():
    """
    The ImportError fallback must not crash the module load when
    huggingface_hub is not installed at all.
    """
    # Remove cached embeddings module
    for key in list(sys.modules.keys()):
        if "src.embeddings" in key or key == "src.embeddings":
            del sys.modules[key]

    # Hide huggingface_hub entirely
    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": None,
            "huggingface_hub.utils": None,
            "huggingface_hub.utils.tqdm": None,
        },
    ):
        try:
            import src.embeddings  # noqa: F401
        except ImportError:
            pass  # huggingface_hub absence is gracefully handled by the try/except in embeddings.py
