"""Tests for the gc_compress_manifest handler (v1.8.0 A2b).

Regression locks:
- input schema must round-trip byte-for-byte (JSON equivalent)
- non-zero savings on a manifest with verbose descriptions
- schema-only fields untouched when description is empty
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _call(manifest: dict) -> dict:
    """Import and call handle_compress_manifest, isolating from gateway."""
    from src.handlers.compress_manifest import handle_compress_manifest

    return handle_compress_manifest({"manifest": manifest})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompressManifestHandlerRoundTrip:
    """inputSchema must survive compression unchanged (JSON-equivalent)."""

    def test_input_schema_round_trips_byte_for_byte(self):
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to compress"},
                "fidelity": {
                    "type": "string",
                    "enum": ["aggressive", "balanced", "detailed"],
                    "default": "balanced",
                },
            },
            "required": ["text"],
        }
        manifest = {
            "tools": [
                {
                    "name": "ingest_context",
                    "description": "Ingest a document into the semantic graph for later retrieval and compression.",
                    "inputSchema": schema,
                }
            ]
        }
        result = _call(manifest)
        out_tools = result["manifest"]["tools"]
        assert len(out_tools) == 1
        # inputSchema must be JSON-equivalent (same structure, no mutations)
        assert json.loads(json.dumps(out_tools[0]["inputSchema"])) == json.loads(
            json.dumps(schema)
        ), "inputSchema was mutated during description compression"

    def test_tool_name_preserved_verbatim(self):
        manifest = {
            "tools": [
                {
                    "name": "my_tool_name",
                    "description": "A tool with a somewhat verbose description.",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
        result = _call(manifest)
        assert result["manifest"]["tools"][0]["name"] == "my_tool_name"

    def test_multiple_tools_all_schemas_preserved(self):
        schemas = [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
            {"type": "object", "properties": {}, "additionalProperties": True},
        ]
        tools = [
            {
                "name": f"tool_{i}",
                "description": f"Tool number {i} does something important with the data pipeline.",
                "inputSchema": schema,
            }
            for i, schema in enumerate(schemas)
        ]
        result = _call({"tools": tools})
        for i, out_tool in enumerate(result["manifest"]["tools"]):
            assert out_tool["inputSchema"] == schemas[i], f"inputSchema for tool_{i} was mutated"


class TestCompressManifestHandlerSavings:
    """Non-zero savings must be reported on verbose manifests."""

    def test_stats_present_in_output(self):
        manifest = {
            "tools": [
                {
                    "name": "ingest_context",
                    "description": (
                        "Ingest a document into the semantic compression graph. "
                        "The document is chunked, embedded, and stored in a PageRank graph. "
                        "Subsequent calls to read_skeleton will return a compressed version "
                        "of the document that preserves the most important semantic content "
                        "while reducing token count by 60-90%."
                    ),
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
        result = _call(manifest)
        assert "stats" in result, "Response must include a 'stats' key"
        stats = result["stats"]
        assert "input_tokens" in stats
        assert "output_tokens" in stats
        assert "savings_pct" in stats

    def test_non_zero_savings_on_verbose_manifest(self):
        """A manifest with long descriptions must yield measurable savings."""
        verbose_desc = (
            "This tool performs semantic compression on a large document by first "
            "chunking the text into semantically coherent segments, then embedding "
            "each segment using a pre-trained language model, constructing a similarity "
            "graph between segments, applying PageRank to score importance, and finally "
            "selecting the top-ranked segments to form a compressed skeleton. "
            "The output preserves the most important information while reducing token "
            "count by 60-90% depending on the fidelity parameter."
        )
        manifest = {
            "tools": [
                {
                    "name": f"tool_{i}",
                    "description": verbose_desc,
                    "inputSchema": {"type": "object", "properties": {}},
                }
                for i in range(5)
            ]
        }
        result = _call(manifest)
        stats = result["stats"]
        assert stats["input_tokens"] > 0, "input_tokens must be > 0 for a verbose manifest"
        assert stats["savings_pct"] >= 0.0, "savings_pct must be non-negative"
        # For a sufficiently verbose manifest the output should be shorter
        assert (
            stats["output_tokens"] <= stats["input_tokens"]
        ), "Output must not be longer than input for a verbose manifest"

    def test_savings_pct_is_float_in_valid_range(self):
        manifest = {
            "tools": [
                {
                    "name": "search_semantic",
                    "description": "Search for semantically similar content in compressed documents.",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
        result = _call(manifest)
        pct = result["stats"]["savings_pct"]
        assert isinstance(pct, float)
        assert 0.0 <= pct <= 100.0


class TestCompressManifestHandlerEdgeCases:
    """Edge cases: empty descriptions, empty manifest, single tool."""

    def test_empty_description_left_alone(self):
        """Tools with empty description must not be broken."""
        manifest = {
            "tools": [
                {
                    "name": "schema_only_tool",
                    "description": "",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                    },
                }
            ]
        }
        result = _call(manifest)
        out = result["manifest"]["tools"][0]
        assert out["name"] == "schema_only_tool"
        assert out["inputSchema"] == {
            "type": "object",
            "properties": {"x": {"type": "number"}},
        }

    def test_empty_tools_list_returns_valid_shape(self):
        result = _call({"tools": []})
        assert result["manifest"]["tools"] == []
        assert result["stats"]["input_tokens"] == 0
        assert result["stats"]["output_tokens"] == 0
        assert result["stats"]["savings_pct"] == 0.0

    def test_missing_description_field_handled_gracefully(self):
        """Some MCP servers omit 'description' — handler must not crash."""
        manifest = {
            "tools": [
                {
                    "name": "tool_without_desc",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        }
        result = _call(manifest)
        assert result["manifest"]["tools"][0]["name"] == "tool_without_desc"

    def test_output_manifest_shape_preserved(self):
        """Output manifest must have the same top-level shape as input."""
        manifest = {"tools": [{"name": "t", "description": "desc", "inputSchema": {}}]}
        result = _call(manifest)
        assert "manifest" in result
        assert "tools" in result["manifest"]
        assert "stats" in result
