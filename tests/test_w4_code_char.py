"""CL 0 characterization tests for baseline CodeSemanticCompressor and TOONSerializer."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import numpy as np

from src.code_compressor import CodeLanguage, CodeSemanticCompressor
from src.toon_serializer import (
    OutputFormat,
    TOONSerializer,
    estimate_token_savings,
    format_response,
)

SAMPLE_PY = '''
import math

def compute_area(radius: float) -> float:
    """Compute circle area."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return math.pi * radius * radius

class GeometryHelper:
    """Helper for geometry calculations."""

    def perimeter(self, radius: float) -> float:
        return 2.0 * math.pi * radius
'''


def _mock_code_compressor() -> CodeSemanticCompressor:
    with patch("src.code_compressor.EmbeddingManager") as mock_mgr_cls:
        mock_mgr = Mock()
        mock_embedder = Mock()

        def _fake_encode(texts, **_kwargs):
            return np.ones((len(texts), 8), dtype=np.float32)

        mock_embedder.encode.side_effect = _fake_encode
        mock_mgr.get_code_embedder.return_value = mock_embedder
        mock_mgr_cls.return_value = mock_mgr
        return CodeSemanticCompressor()


def test_code_compressor_baseline_language_and_chunking():
    comp = _mock_code_compressor()
    assert comp.detect_language("mod.py") == CodeLanguage.PYTHON
    assert comp.detect_language("app.ts") == CodeLanguage.TYPESCRIPT
    assert comp.detect_language("main.rs") == CodeLanguage.RUST
    assert comp.detect_language("notes.txt") == CodeLanguage.UNKNOWN

    chunks = comp.chunk_python_code(SAMPLE_PY, "sample")
    chunk_types = {c.chunk_type for c in chunks}
    names = {c.name for c in chunks}
    assert "import" in chunk_types
    assert "function" in chunk_types
    assert "class" in chunk_types
    assert "compute_area" in names
    assert "GeometryHelper" in names


def test_code_compressor_baseline_ingest_and_skeleton():
    comp = _mock_code_compressor()
    stats = comp.ingest_code_file(SAMPLE_PY, file_id="geom", filepath="geom.py")
    assert stats["file_id"] == "geom"
    assert stats["language"] == "python"
    assert stats["total_chunks"] >= 3

    skeleton = comp.generate_code_skeleton("geom", show_top_n=2)
    assert "=== CODE SKELETON: geom ===" in skeleton
    assert "[IMPORTS]" in skeleton


def test_toon_serializer_baseline_roundtrip_and_savings():
    serializer = TOONSerializer()
    rows = [
        {"node_id": "n1", "importance": 0.9, "summary": "alpha, beta"},
        {"node_id": "n2", "importance": 0.4, "summary": "gamma"},
    ]
    toon = serializer.serialize_search_results(rows)
    assert toon.startswith("results[2]{node_id,importance,summary}:")
    assert "alpha\\, beta" in toon

    compact_json = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    savings = estimate_token_savings(compact_json, toon)
    assert "json_tokens" in savings
    assert "toon_tokens" in savings
    assert savings["json_tokens"] >= 0

    as_json = format_response(rows, OutputFormat.JSON, serializer)
    assert json.loads(as_json) == rows
