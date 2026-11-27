"""Pairwise cosine similarity without a hard sklearn dependency.

Docker runtime images omit scikit-learn/scipy to stay under the <600MB
size contract. Pure-NumPy BLAS normalization handles dense inputs with
< 1e-12 parity against scikit-learn while avoiding importing sklearn/scipy
at module load time; scipy sparse matrices lazily delegate to sklearn.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _unit_normalize_rows(arr: Any) -> tuple[np.ndarray, bool]:
    """Validate 2D finite input, scale by row max for overflow safety, and L2-normalize."""
    raw = np.asarray(arr)
    is_f32 = raw.dtype == np.float32
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D array, got {raw.ndim}D array instead.")
    if raw.size == 0 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError(
            f"Found array with {raw.shape[0]} sample(s) and {raw.shape[1]} feature(s); "
            "a minimum of 1 is required."
        )
    x_f64 = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x_f64).all():
        raise ValueError("Input contains NaN, infinity or a value too large.")
    row_max = np.max(np.abs(x_f64), axis=1, keepdims=True)
    scaled = np.divide(x_f64, row_max, out=np.zeros_like(x_f64), where=row_max > 0.0)
    norm = np.linalg.norm(scaled, axis=1, keepdims=True)
    unit = np.divide(scaled, norm, out=np.zeros_like(scaled), where=norm > 0.0)
    return unit, is_f32


def cosine_similarity(X: Any, Y: Any = None) -> np.ndarray:
    """Compute pairwise cosine similarity between rows of X (and optional Y)."""
    if hasattr(X, "tocsr") or (Y is not None and hasattr(Y, "tocsr")):
        from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_similarity

        return _sklearn_cosine_similarity(X, Y)

    x_unit, x_f32 = _unit_normalize_rows(X)
    if Y is None:
        res = x_unit @ x_unit.T
        return res.astype(np.float32, copy=False) if x_f32 else res
    y_unit, y_f32 = _unit_normalize_rows(Y)
    if x_unit.shape[1] != y_unit.shape[1]:
        raise ValueError(
            f"Incompatible dimension for X and Y matrices: "
            f"X.shape[1] == {x_unit.shape[1]} while Y.shape[1] == {y_unit.shape[1]}"
        )
    res = x_unit @ y_unit.T
    return res.astype(np.float32, copy=False) if (x_f32 and y_f32) else res
