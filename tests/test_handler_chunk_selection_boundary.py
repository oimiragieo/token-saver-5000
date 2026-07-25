"""Handler-side chunk selection must be boundary-safe (#420).

Found by dogfooding live prod: `read_skeleton(file_id="400-coldstart-recheck")`
with NO query came back compressed against the H1 of a DIFFERENT document,
`400-coldstart-recheck-large`. Measured harm from the response's own
pipeline.stages: 80 nodes -> 40, and the compression ratio got WORSE
(2.194 -> 1.706), against a query the caller never issued.

Cause: `compression_handlers.py` reconstructed the auto-mode raw_text with
`nid.startswith(scoped_file_id)`. `"400-coldstart-recheck-large_n0"` starts
with `"400-coldstart-recheck"`, so the shorter document vacuumed in the
longer one's chunks and `_extract_h1_query` found its heading.

`semantic_compressor._node_belongs_to_file` already exists for exactly this
-- its docstring names the same collision ("foobar_n0".startswith("foo")) and
lists the surfaces it broke: "skeletons, search, stats and diff re-ingestion".
It was applied in the compressor and never adopted by the handlers. These
tests bind the handler side behaviourally so a source-level regression cannot
pass them.
"""

from __future__ import annotations

from src.handlers.compression_handlers import chunks_for_file


class _Node:
    def __init__(self, text: str):
        self.text = text


def _corpus() -> dict[str, _Node]:
    """Two documents where one id is a strict prefix of the other."""
    return {
        "doc_n0": _Node("SHORT doc heading"),
        "doc_n1": _Node("short body"),
        "doc-large_n0": _Node("LONG doc heading"),
        "doc-large_n1": _Node("long body"),
        "doc-large_n2": _Node("more long body"),
    }


class TestPrefixSiblingIsolation:
    def test_the_shorter_id_does_not_capture_its_prefix_sibling(self):
        selected = chunks_for_file(_corpus(), "doc")
        assert [nid for nid, _ in selected] == ["doc_n0", "doc_n1"]

    def test_the_longer_id_still_gets_its_own_chunks(self):
        selected = chunks_for_file(_corpus(), "doc-large")
        assert [nid for nid, _ in selected] == [
            "doc-large_n0",
            "doc-large_n1",
            "doc-large_n2",
        ]

    def test_reconstructed_text_carries_no_sibling_content(self):
        """The concrete failure: the sibling's heading leaked into raw_text and
        then became the synthesised query."""
        text = "\n\n".join(node.text for _, node in chunks_for_file(_corpus(), "doc"))
        assert "LONG doc heading" not in text
        assert "SHORT doc heading" in text


class TestCodeNodeIdsAlsoIsolate:
    """Code nodes use `{file_id}::{symbol}`, not `{file_id}_n{i}`."""

    def test_code_node_format_respects_the_boundary(self):
        corpus = {
            "main.py::run": _Node("def run"),
            "main.py.bak::run": _Node("def run backup"),
        }
        selected = chunks_for_file(corpus, "main.py")
        assert [nid for nid, _ in selected] == ["main.py::run"]


class TestOrdering:
    def test_selection_is_sorted_for_deterministic_reconstruction(self):
        """Order carries document structure -- the heading heuristic depends
        on the first chunk actually being the first chunk."""
        shuffled = {
            "doc_n1": _Node("second"),
            "doc_n0": _Node("first"),
        }
        assert [nid for nid, _ in chunks_for_file(shuffled, "doc")] == ["doc_n0", "doc_n1"]


class TestEmptyAndMissing:
    def test_unknown_file_id_returns_empty_not_everything(self):
        assert chunks_for_file(_corpus(), "nonexistent") == []

    def test_empty_corpus_is_safe(self):
        assert chunks_for_file({}, "doc") == []
