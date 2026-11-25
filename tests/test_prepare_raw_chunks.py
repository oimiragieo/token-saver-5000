"""#190 unit 2: SemanticCompressor._prepare_raw_chunks routes a structured JSON
array to record-level chunking (records survive as nodes) and falls through to
the normal text chunker for everything else.

Model-free: ``object.__new__(SemanticCompressor)`` skips __init__/model load; we
stub the two collaborators the method touches (_count_tokens, _chunk_text).
"""

from __future__ import annotations

import json
from threading import Lock

import pytest

from src.semantic_compressor import SemanticCompressor


def _bare_compressor():
    c = object.__new__(SemanticCompressor)
    # Instance attrs shadow the class methods for this stubbed instance.
    c._count_tokens = lambda t: len(t.split())
    c._chunk_text = lambda text, **kwargs: ["<TEXT_CHUNKER>"]
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


def test_jsonl_uses_record_chunking_not_text_chunker():
    c = _bare_compressor()
    text = "\n".join(f'{{"id": {i}, "v": {i}}}' for i in range(10))
    chunks = c._prepare_raw_chunks(text, "jsonl")
    assert chunks != ["<TEXT_CHUNKER>"]  # the record path was taken
    joined = "\n".join(chunks)
    for i in range(10):
        assert f'"id": {i}' in joined  # no data loss


def test_jsonl_kind_but_unsplittable_falls_back():
    # kind claims jsonl but it is a single JSON line -> split returns None ->
    # fall back to the text chunker rather than raise.
    c = _bare_compressor()
    assert c._prepare_raw_chunks('{"a": 1}', "jsonl") == ["<TEXT_CHUNKER>"]


def test_csv_uses_record_chunking_header_as_own_node():
    c = _bare_compressor()
    rows = [f"{i},name{i},{i * 10}" for i in range(10)]
    text = "id,name,score\n" + "\n".join(rows)
    chunks = c._prepare_raw_chunks(text, "csv")
    assert chunks != ["<TEXT_CHUNKER>"]  # the record path was taken
    assert chunks[0] == "id,name,score"  # header is its own first node
    assert chunks.count("id,name,score") == 1  # NOT duplicated per group (MED-4)
    joined = "\n".join(chunks)
    for row in rows:
        assert row in joined  # no data loss


def test_csv_multiline_quoted_falls_through():
    # A multiline-quoted CSV bails from record-chunking (guard) -> text chunker.
    c = _bare_compressor()
    text = 'a,b\n1,"x\ny"\n3,4'
    assert c._prepare_raw_chunks(text, "csv") == ["<TEXT_CHUNKER>"]


def test_ingest_applies_chunk_target_before_embedding(monkeypatch):
    class ReachedEmbedding(Exception):
        pass

    c = object.__new__(SemanticCompressor)
    c._sync_lock = Lock()
    c._count_tokens = lambda text: len(text.split())
    c._clear_cache_for_doc = lambda file_id: None
    paragraphs = [" ".join(f"word{i}_{j}" for j in range(30)) for i in range(4)]
    text = "\n\n".join(paragraphs)
    batches = []

    async def capture_chunks(chunks):
        batches.append(chunks)
        raise ReachedEmbedding

    monkeypatch.setattr(c, "_encode_async", capture_chunks)

    with pytest.raises(ReachedEmbedding):
        c.ingest_file(text, "default")
    with pytest.raises(ReachedEmbedding):
        c.ingest_file(text, "profile", max_chunk_size=50)

    assert batches == [[text], paragraphs]


def test_custom_jsonl_group_budget_keeps_oversized_records_atomic():
    c = _bare_compressor()
    records = ['{"id": 1}', '{"id": 2, "notes": "many words in a long record"}', '{"id": 3}']
    text = "\n".join(records)

    assert c._prepare_raw_chunks(text, "jsonl") == [text]
    assert c._prepare_raw_chunks(text, "jsonl", max_chunk_size=4) == records
