"""#190 unit 2: SemanticCompressor._prepare_raw_chunks routes a structured JSON
array to record-level chunking (records survive as nodes) and falls through to
the normal text chunker for everything else.

Model-free: ``object.__new__(SemanticCompressor)`` skips __init__/model load; we
stub the two collaborators the method touches (_count_tokens, _chunk_text).
"""

from __future__ import annotations

import json

from src.semantic_compressor import SemanticCompressor


def _bare_compressor():
    c = object.__new__(SemanticCompressor)
    # Instance attrs shadow the class methods for this stubbed instance.
    c._count_tokens = lambda t: len(t.split())
    c._chunk_text = lambda text, strategy="auto": ["<TEXT_CHUNKER>"]
    return c


def test_json_array_uses_record_chunking_not_text_chunker():
    c = _bare_compressor()
    text = json.dumps([{"id": i, "v": i} for i in range(10)])
    chunks = c._prepare_raw_chunks(text, "json_array")
    assert chunks != ["<TEXT_CHUNKER>"]  # the record path was taken
    joined = "\n".join(chunks)
    for i in range(10):
        # records keep their ORIGINAL (json.dumps-spaced) form — every one survives
        assert f'"id": {i}' in joined  # no data loss


def test_prose_falls_through_to_text_chunker():
    c = _bare_compressor()
    assert c._prepare_raw_chunks("just some prose here", None) == ["<TEXT_CHUNKER>"]


def test_json_array_kind_but_unsplittable_falls_back():
    # kind claims json_array but the payload is an object -> split returns None
    # -> fall back to the text chunker rather than raise.
    c = _bare_compressor()
    assert c._prepare_raw_chunks('{"a": 1}', "json_array") == ["<TEXT_CHUNKER>"]
