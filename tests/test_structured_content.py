"""#190 structured-content detection + record splitting (model-free).

The foundation of the JSON/table compression fix: a raw JSON array currently
collapses to 1 mega-node and the skeletonizer hides ~99% of it (dogfood
2026-07-11). These pure functions let the chunker split on record boundaries so
records survive as individually rankable nodes.
"""

from __future__ import annotations

import json

from src.structured_content import detect_structured_content, split_json_records


class TestDetectStructuredContent:
    def test_json_array_of_objects(self):
        text = json.dumps([{"id": 1, "name": "a"}, {"id": 2, "name": "b"}])
        assert detect_structured_content(text) == "json_array"

    def test_json_single_object(self):
        assert detect_structured_content('{"a": 1, "b": 2}') == "json_object"

    def test_jsonl(self):
        text = '{"id": 1}\n{"id": 2}\n{"id": 3}'
        assert detect_structured_content(text) == "jsonl"

    def test_csv(self):
        text = "id,name,score\n1,alice,90\n2,bob,85"
        assert detect_structured_content(text) == "csv"

    def test_prose_is_none(self):
        assert (
            detect_structured_content("The quick brown fox jumps over the lazy dog. It ran fast.")
            is None
        )

    def test_markdown_is_none(self):
        assert detect_structured_content("# Heading\n\nSome **bold** text and a [link](x).") is None

    def test_pretty_printed_array_resolves_as_array_not_jsonl(self):
        text = '[\n  {"id": 1},\n  {"id": 2}\n]'
        assert detect_structured_content(text) == "json_array"

    def test_empty_is_none(self):
        assert detect_structured_content("") is None
        assert detect_structured_content("   ") is None

    def test_malformed_json_is_none(self):
        assert detect_structured_content('[{"id": 1}, ') is None

    def test_single_element_array_not_structured(self):
        assert detect_structured_content('[{"id": 1}]') is None

    def test_prose_with_commas_not_csv(self):
        # Uneven comma counts per line -> not a table.
        assert (
            detect_structured_content("Hello, world, this is prose.\nA single line here.") is None
        )


class TestSplitJsonRecords:
    def test_splits_array_into_records(self):
        records = [{"id": i, "v": i * 2} for i in range(5)]
        out = split_json_records(json.dumps(records))
        assert out is not None
        assert len(out) == 5
        assert [json.loads(r) for r in out] == records

    def test_non_array_returns_none(self):
        assert split_json_records('{"a": 1}') is None

    def test_prose_returns_none(self):
        assert split_json_records("not json at all") is None

    def test_single_element_returns_none(self):
        assert split_json_records('[{"a": 1}]') is None

    def test_scalar_array_splits(self):
        assert split_json_records("[1, 2, 3]") == ["1", "2", "3"]

    def test_records_are_compact(self):
        out = split_json_records('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]')
        assert out == ['{"a":1,"b":2}', '{"a":3,"b":4}']
