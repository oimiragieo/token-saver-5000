"""Source-order render locks for the code skeleton + multi-level skeleton surfaces.

World-class compression audit #2 (2026-07-07) fixed the TEXT skeleton to render in
document order (semantic_compressor.py). Fable's follow-up audit found two OTHER
render surfaces still emitting importance order:
  - code_compressor.generate_code_skeleton (code files read via read_skeleton)
  - multi_level_skeleton.generate_multi_level_skeleton (the multilevel MCP tool)

These locks force importance order to DIVERGE from document order and assert each
surface renders in document order regardless. Model-free (no embedding model): the
code test hand-builds chunks + graph; the multi-level test calls a pure function.
"""

import networkx as nx

from src.code_compressor import CodeChunk, CodeSemanticCompressor
from src.multi_level_skeleton import generate_multi_level_skeleton


def _chunk(cid: str, name: str, start_line: int, importance: float) -> CodeChunk:
    return CodeChunk(
        chunk_id=cid,
        chunk_type="function",
        code=f"def {name}():\n    pass",
        name=name,
        docstring=None,
        start_line=start_line,
        end_line=start_line + 1,
        dependencies=[],
        importance=importance,
    )


class TestCodeSkeletonRendersInSourceOrder:
    """generate_code_skeleton must render selected functions in source-line order,
    not importance order (audit #2, code surface)."""

    def test_functions_render_in_start_line_order_not_importance(self) -> None:
        file_id = "srcorder_test.py"
        # Source order: fn_a(10) -> fn_b(20) -> fn_c(30).
        # Importance order (desc): fn_b(0.9) -> fn_c(0.5) -> fn_a(0.1) -- REVERSED-ish
        # vs source, so the pre-fix render loop would emit b, c, a.
        chunks = [
            _chunk(f"{file_id}::fn_a", "fn_a", start_line=10, importance=0.1),
            _chunk(f"{file_id}::fn_b", "fn_b", start_line=20, importance=0.9),
            _chunk(f"{file_id}::fn_c", "fn_c", start_line=30, importance=0.5),
        ]

        compressor = object.__new__(CodeSemanticCompressor)
        compressor.chunks = {c.chunk_id: c for c in chunks}
        graph = nx.DiGraph()
        for c in chunks:
            graph.add_node(c.chunk_id)
        compressor.graphs = {file_id: graph}

        skeleton = compressor.generate_code_skeleton(file_id, show_top_n=3)

        pos_a = skeleton.index("fn_a")
        pos_b = skeleton.index("fn_b")
        pos_c = skeleton.index("fn_c")
        assert pos_a < pos_b < pos_c, (
            "code skeleton must render functions in source-line order (a<b<c), "
            f"got positions a={pos_a} b={pos_b} c={pos_c}\n--- skeleton ---\n{skeleton}"
        )


class TestMultiLevelSkeletonRendersInInputOrder:
    """generate_multi_level_skeleton must render each tier in the caller's input
    (document) order, selecting membership by importance (audit #2, multilevel
    surface)."""

    def _nodes(self):
        # Input/document order n0..n4; importance deliberately NOT monotonic with
        # position so importance-order != document-order.
        return [
            {"node_id": "d_n0", "text": "ALPHA", "importance": 0.20},
            {"node_id": "d_n1", "text": "BETA", "importance": 0.95},
            {"node_id": "d_n2", "text": "GAMMA", "importance": 0.40},
            {"node_id": "d_n3", "text": "DELTA", "importance": 0.80},
            {"node_id": "d_n4", "text": "EPSILON", "importance": 0.10},
        ]

    def test_full_tier_is_document_order(self) -> None:
        result = generate_multi_level_skeleton(self._nodes())
        assert result["full"]["nodes"] == ["d_n0", "d_n1", "d_n2", "d_n3", "d_n4"], (
            "full tier must be document order, not importance order; got "
            f"{result['full']['nodes']}"
        )
        # Text must concatenate in the same document order.
        assert result["full"]["text"] == "ALPHA BETA GAMMA DELTA EPSILON"

    def test_summary_tier_selects_by_importance_renders_in_input_order(self) -> None:
        # summary_ratio 0.6 of 5 -> top-3 by importance = {d_n1(.95), d_n3(.80), d_n2(.40)}.
        result = generate_multi_level_skeleton(self._nodes(), headline_ratio=0.2, summary_ratio=0.6)
        summary_ids = result["summary"]["nodes"]
        assert set(summary_ids) == {
            "d_n1",
            "d_n2",
            "d_n3",
        }, f"summary tier must select the top-3 by importance; got {summary_ids}"
        # ...but rendered in document (input) order: n1 before n2 before n3.
        assert summary_ids == [
            "d_n1",
            "d_n2",
            "d_n3",
        ], f"summary tier must render selected nodes in document order; got {summary_ids}"

    def test_headline_tier_top_importance_in_input_order(self) -> None:
        result = generate_multi_level_skeleton(self._nodes(), headline_ratio=0.4, summary_ratio=0.6)
        # top-2 by importance = {d_n1(.95), d_n3(.80)}, rendered n1 then n3 (doc order).
        assert result["headline"]["nodes"] == ["d_n1", "d_n3"]
