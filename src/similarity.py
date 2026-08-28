"""Pairwise cosine similarity without a hard sklearn dependency.

Docker runtime images omit scikit-learn/scipy to stay under the <600MB
size contract. Production code historically called
``sklearn.metrics.pairwise.cosine_similarity``; this module keeps that
call shape while preferring sklearn when installed (local/dev) and
falling back to NumPy otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.metrics.pairwise import cosine_similarity as _sklearn_cosine_similarity

    def cosine_similarity(X: Any, Y: Any = None) -> np.ndarray:
        return _sklearn_cosine_similarity(X, Y)

except ImportError:  # pragma: no cover - exercised in ONNX-only Docker image

    def cosine_similarity(X: Any, Y: Any = None) -> np.ndarray:
        x = np.asarray(X, dtype=np.float64)
        y = x if Y is None else np.asarray(Y, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        x_unit = x / np.clip(x_norm, 1e-12, None)
        y_unit = y / np.clip(y_norm, 1e-12, None)
        return x_unit @ y_unit.T
