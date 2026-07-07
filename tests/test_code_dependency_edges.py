"""Task #236 rank11 — dependency-edge building must be indexed, not quadratic.

``ingest_code_file`` resolved every ``chunk.dependencies`` entry by rescanning
the ENTIRE chunks list (``for dep in deps: for other_chunk in chunks``) —
O(C x D x C). A function body's ast.Call walk emits one dependency name per
call site, so a large file made this quadratic in input size. The fix ports the
#30 indexing discipline: build a name -> [chunk_id] index once, then O(1)
lookups (``CodeSemanticCompressor._build_dependency_edges``).

Model-free via object.__new__ — the helper is a @staticmethod using only its
argument (same pattern as test_audit_194_async_chunking.py).
"""

from __future__ import annotations

from typing import List

from src.code_compressor import CodeChunk, CodeSemanticCompressor


def _chunk(chunk_id: str, name: str, deps: List[str]) -> CodeChunk:
    return CodeChunk(
        chunk_id=chunk_id,
        chunk_type="function",
        code=f"def {name}(): pass",
        name=name,
        docstring=None,
        start_line=1,
        end_line=2,
        dependencies=deps,
    )


def _naive_reference_edges(chunks: List[CodeChunk]) -> set:
    """The pre-fix nested-loop semantics, as a reference oracle."""
    edges = set()
    for chunk in chunks:
        for dep in chunk.dependencies:
            for other_chunk in chunks:
                if other_chunk.name == dep and other_chunk.chunk_id != chunk.chunk_id:
                    edges.add((chunk.chunk_id, other_chunk.chunk_id))
    return edges


class _IterCountingList(list):
    """List that counts how many times it is iterated end-to-end."""

    iter_count = 0

    def __iter__(self):
        self.iter_count += 1
        return super().__iter__()


class TestDependencyEdgeParity:
    def test_matches_naive_reference_on_mixed_graph(self):
        chunks = [
            _chunk("f::a", "a", ["b", "c", "missing"]),
            _chunk("f::b", "b", ["a"]),
            _chunk("f::c", "c", []),
            # Same NAME as f::a under a different chunk_id (qualified method
            # case) — both must receive an inbound edge for dep "a".
            _chunk("f::Klass.a", "a", ["b"]),
        ]
        got = set(CodeSemanticCompressor._build_dependency_edges(chunks))
        assert got == _naive_reference_edges(chunks)

    def test_self_edge_excluded(self):
        chunks = [_chunk("f::rec", "rec", ["rec"])]
        assert CodeSemanticCompressor._build_dependency_edges(chunks) == []

    def test_duplicate_dependency_mentions_collapse(self):
        # 50 call sites of the same helper -> exactly one edge (nx.add_edge
        # was idempotent pre-fix, so the rendered graph is unchanged).
        chunks = [
            _chunk("f::caller", "caller", ["helper"] * 50),
            _chunk("f::helper", "helper", []),
        ]
        edges = CodeSemanticCompressor._build_dependency_edges(chunks)
        assert edges == [("f::caller", "f::helper")]

    def test_unresolvable_dependency_produces_no_edge(self):
        chunks = [_chunk("f::a", "a", ["not_defined_here"])]
        assert CodeSemanticCompressor._build_dependency_edges(chunks) == []


class TestDependencyEdgeComplexity:
    def test_chunks_list_is_not_rescanned_per_dependency(self):
        """The complexity-sensitive lock: the old code iterated ``chunks`` once
        PER DEPENDENCY (here 40 chunks x 25 deps = 1000 full scans); the
        indexed version walks the list a constant number of times (index build
        + edge pass = 2)."""
        chunks = _IterCountingList(
            _chunk(f"f::fn{i}", f"fn{i}", [f"fn{(i + j) % 40}" for j in range(25)])
            for i in range(40)
        )
        edges = CodeSemanticCompressor._build_dependency_edges(chunks)
        assert edges, "sanity: the dense dep graph must produce edges"
        assert chunks.iter_count <= 2, (
            f"chunks iterated {chunks.iter_count} times — a per-dependency "
            "rescan has been reintroduced (the #236 rank11 quadratic)"
        )


class TestIngestIntegrationModelFree:
    def test_ingest_code_file_builds_dependency_edges_via_index(self):
        """End-to-end through ingest_code_file with a stubbed embedder —
        dependency edges land in the graph exactly as the naive semantics
        dictate."""
        import numpy as np

        class _StubModel:
            def encode(self, texts, show_progress_bar=False):
                return np.zeros((len(texts), 8), dtype=np.float32)

        comp = object.__new__(CodeSemanticCompressor)
        comp.model = _StubModel()
        comp.similarity_threshold = 0.70
        comp.chunks = {}
        comp.graphs = {}
        comp.file_metadata = {}

        code = (
            "def helper():\n"
            "    return 1\n\n"
            "def caller():\n"
            "    return helper() + helper()\n"
        )
        stats = comp.ingest_code_file(code, file_id="mod", filepath="mod.py")
        graph = comp.graphs["mod"]

        dep_edges = {(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "dependency"}
        assert ("mod::caller", "mod::helper") in dep_edges
        # Zero-vector embeddings -> no similarity edges; the dependency edge
        # count is what number_of_edges reports beyond imports/etc.
        assert stats["total_chunks"] >= 2
