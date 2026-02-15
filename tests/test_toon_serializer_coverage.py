from __future__ import annotations

import json

from src.toon_serializer import (
    OutputFormat,
    TOONSerializer,
    estimate_token_savings,
    format_response,
)


def test_serialize_search_results_handles_types_and_empty():
    serializer = TOONSerializer(indent_size=2)
    assert serializer._indent(2) == "    "
    assert serializer.serialize_search_results([]) == "results[0]{}"

    rows = [
        {"node_id": "a", "importance": 0.12345, "summary": "hello,world\nline2", "x": None},
        {"node_id": "b", "importance": 0.9, "summary": "ok", "x": 7},
    ]
    out = serializer.serialize_search_results(
        rows, fields=["node_id", "importance", "summary", "x"]
    )
    assert "results[2]{node_id,importance,summary,x}:" in out
    assert "\\," in out
    assert "\\n" in out
    assert "0.123" in out


def test_serialize_domain_helpers_and_stats():
    serializer = TOONSerializer()
    docs = [
        {
            "file_id": "f1",
            "total_nodes": 3,
            "total_tokens": 100,
            "skeleton_tokens": 20,
            "compression_ratio": 5.0,
        }
    ]
    inv = serializer.serialize_document_inventory(docs)
    assert "documents" not in inv  # inventory reuses search format
    assert "results[1]" in inv

    afm = serializer.serialize_afm_context(
        [("user", "x" * 150), ("assistant", "short")], {"count": 2, "nested": {"ignored": 1}}
    )
    assert "afm_context:" in afm
    assert "stats:" in afm
    assert "results[2]{role,content}:" in afm

    skel = serializer.serialize_skeleton(
        [{"node_id": "n1", "importance": 0.2, "type": "fact", "summary": "s"}],
        {"file_id": "f1", "total_nodes": 1, "compression_ratio": 3.2},
    )
    assert "skeleton:" in skel
    assert "file_id: f1" in skel

    stats = serializer.serialize_stats(
        {
            "simple": 1,
            "nested": {"a": 2},
            "arr_simple": [1, 2, 3],
            "arr_obj": [{"k": "v"}, {"k": "w"}],
        }
    )
    assert "stats:" in stats
    assert "arr_simple: [1,2,3]" in stats
    assert "results[2]{k}:" in stats


def test_format_response_variants_and_savings():
    serializer = TOONSerializer()
    payload = [{"a": 1}, {"a": 2}]
    as_json = format_response(payload, OutputFormat.JSON, serializer)
    assert json.loads(as_json) == payload

    as_toon_list = format_response(payload, OutputFormat.TOON, serializer)
    assert "results[2]{a}:" in as_toon_list

    as_toon_dict = format_response(
        {"messages": [("user", "q")], "stats": {"k": 1}}, OutputFormat.TOON, serializer
    )
    assert "afm_context:" in as_toon_dict

    as_text = format_response({"k": "v"}, OutputFormat.TEXT, serializer)
    assert as_text == "{'k': 'v'}"

    fallback = format_response(123, OutputFormat.TOON, serializer)
    assert json.loads(fallback) == 123

    toon_json = serializer.to_json("dummy")
    assert "official TOON parser" in toon_json

    savings = estimate_token_savings("one two three", "one")
    assert savings["json_tokens"] > savings["toon_tokens"]
    assert savings["tokens_saved"] > 0
