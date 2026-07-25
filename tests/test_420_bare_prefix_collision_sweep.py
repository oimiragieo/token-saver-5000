"""#420 sweep of remaining bare-prefix node-id collision sites.

Node ids are ``{file_id}_n{index}`` (text) or ``{file_id}::{symbol}`` (code).
``node_id.startswith(file_id)`` is not a membership test: ``"doc-large_n0"
.startswith("doc")`` is True, so when file_ids ``doc`` and ``doc-large``
coexist, operations scoped to ``doc`` silently reach into ``doc-large``.

``chunks_for_file`` (compression_handlers.py) and ``_node_belongs_to_file``
(semantic_compressor.py) already fix this for the sites they cover. This
file locks the remaining REACHABLE sites found by the #420 sweep:

  - src/handlers/compression_handlers.py -- persist-chunks filter (handle_ingest)
  - src/handlers/compression_handlers.py -- anchored-keywords filter (handle_read_skeleton)
  - src/handlers/compression_handlers.py -- delete-confirmation node count (handle_delete_document)
  - src/handlers/compression_handlers.py -- delete path, plain-SemanticCompressor branch
  - src/handlers/compression_handlers.py -- diff-reingest persist filter
  - src/handlers/connector_handlers.py -- sync-feed persist filter
  - src/handlers/file_sync_handlers.py -- refresh-document persist filter
  - src/code_compressor.py -- search_code file_id filter
  - src/scar_compressor.py -- search_with_alignment file_id filter

Sites established as NOT REACHABLE (the enclosing graph/dict lookup is
already an exact-match on file_id, so a sibling's nodes can never appear)
are documented in the #420 report, not tested here:

  - src/adaptive_rate_allocator.py:290
  - src/blind_spot_detector.py:124, 358
  - src/code_compressor.py:736 (generate_code_skeleton)
  - src/handlers/compression_handlers.py -- temporal-facts filter (handle_ingest)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import networkx as nx
import numpy as np
import pytest


class _Node:
    """Minimal stand-in for a chunk/node value. Only `.text` is read by the
    sites under test; extra attrs are set where a specific handler needs them.
    """

    def __init__(self, text: str, tokens: int = 5):
        self.text = text
        self.metadata = {"tokens": tokens}
        self.embedding = np.zeros(3)
        self.importance = 0.5


# =========================================================================
# src/handlers/compression_handlers.py -- handle_ingest persist-chunks filter
# =========================================================================


@pytest.mark.asyncio
async def test_handle_ingest_persists_only_own_chunks_not_prefix_sibling():
    from src.handlers.compression_handlers import handle_ingest

    mock_skeleton = MagicMock()
    mock_skeleton.total_tokens = 100
    mock_skeleton.skeleton_tokens = 20
    mock_skeleton.total_nodes = 2
    mock_skeleton.compression_ratio = 5.0

    mock_estimate = MagicMock()
    mock_estimate.compression_ratio = 5.0

    compressor = AsyncMock()
    compressor.ingest_file_async = AsyncMock(return_value=mock_skeleton)
    compressor.generate_skeleton = MagicMock(return_value=mock_skeleton)
    compressor.chunks = {
        "doc_n0": _Node("own chunk 0"),
        "doc_n1": _Node("own chunk 1"),
        "doc-large_n0": _Node("sibling chunk"),
    }
    compressor.graphs = {"doc": nx.Graph()}
    compressor.file_metadata = {}

    persistence = MagicMock()
    persistence.save_document = MagicMock(return_value=True)

    context = {
        "compressor": compressor,
        "persistence": persistence,
        "sync_manager": MagicMock(),
        "version_manager": AsyncMock(),
        "resource_manager": AsyncMock(),
        "retrieval_history": {},
    }
    context["resource_manager"].check_document_size_async = AsyncMock(return_value=(True, None))
    context["resource_manager"].register_document_async = AsyncMock()

    args = {
        "text": "This is a sufficiently long test document for semantic analysis purposes.",
        "file_id": "doc",
    }
    await handle_ingest(context, args)

    assert persistence.save_document.called, "persist step must have been reached"
    saved_chunks = persistence.save_document.call_args.kwargs["chunks"]
    assert "doc-large_n0" not in saved_chunks
    assert set(saved_chunks) == {"doc_n0", "doc_n1"}


# =========================================================================
# src/handlers/compression_handlers.py -- handle_diff_reingest persist filter
# =========================================================================


@pytest.mark.asyncio
async def test_handle_diff_reingest_persists_only_own_chunks_not_prefix_sibling():
    from src.handlers.compression_handlers import handle_diff_reingest

    diff_result = SimpleNamespace(
        chunks_unchanged=1, chunks_updated=0, chunks_added=0, chunks_removed=0
    )

    compressor = AsyncMock()
    compressor.diff_reingest_async = AsyncMock(return_value=diff_result)
    compressor.chunks = {
        "doc_n0": _Node("own chunk"),
        "doc-large_n0": _Node("sibling chunk"),
    }
    compressor.graphs = {"doc": nx.Graph()}
    compressor.file_metadata = {}

    persistence = MagicMock()
    persistence.save_document = MagicMock(return_value=True)

    context = {
        "compressor": compressor,
        "persistence": persistence,
        "version_manager": AsyncMock(),
    }

    await handle_diff_reingest(context, {"file_id": "doc", "text": "updated content here"})

    assert persistence.save_document.called
    saved_chunks = persistence.save_document.call_args.kwargs["chunks"]
    assert "doc-large_n0" not in saved_chunks
    assert set(saved_chunks) == {"doc_n0"}


# =========================================================================
# src/handlers/compression_handlers.py -- handle_read_skeleton anchored_keywords
# =========================================================================


def test_resolve_anchored_node_ids_excludes_prefix_sibling():
    """Direct unit test of the extracted anchored-keyword resolver (reuses
    chunks_for_file rather than a bare startswith)."""
    from src.handlers.compression_handlers import resolve_anchored_node_ids

    chunks = {
        "doc_n0": _Node("no match here"),
        "doc-large_n0": _Node("the anchor keyword appears in the sibling"),
    }

    result = resolve_anchored_node_ids(chunks, "doc", ["anchor"])
    assert result == set()

    chunks["doc_n0"] = _Node("this node has the anchor keyword too")
    result = resolve_anchored_node_ids(chunks, "doc", ["anchor"])
    assert result == {"doc_n0"}


# =========================================================================
# src/handlers/compression_handlers.py -- handle_delete_document
# =========================================================================


def _plain_semantic_compressor(chunks: dict, graphs: dict, file_metadata: dict | None = None):
    """Model-free SemanticCompressor stand-in (#420): a plain instance
    deliberately lacks `delete_document_from_memory` (only
    CodeCompressionAdapter has it), so `handle_delete_document` takes the
    bare-dict-mutation branch this test targets."""
    from src.semantic_compressor import SemanticCompressor

    c = object.__new__(SemanticCompressor)
    c.chunks = chunks
    c.graphs = graphs
    c.file_metadata = file_metadata or {}
    c._baseline_skeleton_stats = {fid: {"skeleton_tokens": 10, "ratio": 2.0} for fid in graphs}
    c._baseline_skeleton_cache = {}
    return c


def _doc_graph(*node_ids: str) -> nx.Graph:
    g = nx.Graph()
    for nid in node_ids:
        g.add_node(nid)
    return g


@pytest.mark.asyncio
async def test_delete_confirmation_count_excludes_prefix_sibling():
    from src.handlers.compression_handlers import handle_delete_document

    compressor = _plain_semantic_compressor(
        chunks={
            "doc_n0": _Node("a"),
            "doc_n1": _Node("b"),
            "doc-large_n0": _Node("sibling"),
        },
        graphs={"doc": _doc_graph("doc_n0", "doc_n1")},
    )

    context = {
        "compressor": compressor,
        "persistence": MagicMock(),
        "retrieval_history": {},
    }

    result = await handle_delete_document(context, {"file_id": "doc", "confirm": False})

    # Own document has exactly 2 nodes; the sibling's 1 node must not be counted.
    assert "Remove all 2 semantic nodes" in result


@pytest.mark.asyncio
async def test_delete_document_removes_only_own_chunks_from_plain_compressor():
    from src.handlers.compression_handlers import handle_delete_document

    assert not hasattr(
        _plain_semantic_compressor({}, {}), "delete_document_from_memory"
    ), "test assumes the plain-SemanticCompressor branch is the one taken"

    compressor = _plain_semantic_compressor(
        chunks={
            "doc_n0": _Node("a"),
            "doc_n1": _Node("b"),
            "doc-large_n0": _Node("sibling"),
        },
        graphs={
            "doc": _doc_graph("doc_n0", "doc_n1"),
            "doc-large": _doc_graph("doc-large_n0"),
        },
    )

    context = {
        "compressor": compressor,
        "persistence": MagicMock(delete_document=MagicMock(return_value=True)),
        "retrieval_history": {},
        "resource_manager": AsyncMock(),
        "sync_manager": MagicMock(),
        "version_manager": AsyncMock(),
    }

    await handle_delete_document(context, {"file_id": "doc", "confirm": True})

    assert "doc_n0" not in compressor.chunks
    assert "doc_n1" not in compressor.chunks
    assert "doc-large_n0" in compressor.chunks, "sibling document's chunk must survive"
    assert "doc" not in compressor.graphs
    assert "doc-large" in compressor.graphs, "sibling document's graph must survive"


# =========================================================================
# src/handlers/connector_handlers.py -- handle_sync_connector_feed persist filter
# =========================================================================


@pytest.mark.asyncio
async def test_connector_sync_persists_only_own_chunks_not_prefix_sibling():
    from src.handlers.connector_handlers import handle_sync_connector_feed

    document = SimpleNamespace(
        source_id="src-1", file_id="doc", text="hello world content", metadata={}
    )
    feed = {"name": "feed", "connector_type": "web"}

    connector_registry = MagicMock()
    connector_registry.collect_documents = MagicMock(return_value=(feed, [document]))
    connector_registry.mark_synced = MagicMock(return_value=feed)

    compressor = AsyncMock()
    compressor.ingest_file_async = AsyncMock(
        return_value=SimpleNamespace(total_tokens=100, skeleton_tokens=20, compression_ratio=5.0)
    )
    # "feed:doc-large_n0" already exists from a prior sync of a sibling
    # document; "feed:doc" is the scoped_file_id this sync call targets.
    compressor.chunks = {
        "feed:doc_n0": _Node("own"),
        "feed:doc-large_n0": _Node("sibling"),
    }
    compressor.graphs = {"feed:doc": nx.Graph()}
    compressor.file_metadata = {}

    persistence = MagicMock()
    persistence.save_document = MagicMock(return_value=True)

    context = {
        "connector_registry": connector_registry,
        "compressor": compressor,
        "resource_manager": MagicMock(
            check_connector_batch_async=AsyncMock(return_value=(True, None)),
            check_document_size_async=AsyncMock(return_value=(True, None)),
            register_document_async=AsyncMock(),
        ),
        "sync_manager": MagicMock(register_file=MagicMock()),
        "version_manager": MagicMock(add_version_async=AsyncMock()),
        "persistence": persistence,
        "retrieval_history": {},
    }

    result = json.loads(await handle_sync_connector_feed(context, {"name": "feed"}))
    assert result["results"][0]["success"] is True

    assert persistence.save_document.called
    saved_chunks = persistence.save_document.call_args.kwargs["chunks"]
    assert "feed:doc-large_n0" not in saved_chunks
    assert set(saved_chunks) == {"feed:doc_n0"}


# =========================================================================
# src/handlers/file_sync_handlers.py -- handle_refresh_document persist filter
# =========================================================================


@pytest.mark.asyncio
async def test_refresh_document_persists_only_own_chunks_not_prefix_sibling(tmp_path):
    from src.file_sync_manager import FileMetadata
    from src.handlers.file_sync_handlers import handle_refresh_document

    source_file = tmp_path / "src.txt"
    source_file.write_text("refreshed content", encoding="utf-8")

    compressor = AsyncMock()
    compressor.ingest_file_async = AsyncMock(
        return_value=SimpleNamespace(total_tokens=50, skeleton_tokens=10, compression_ratio=5.0)
    )
    compressor.chunks = {
        "doc_n0": _Node("own"),
        "doc-large_n0": _Node("sibling"),
    }
    compressor.graphs = {"doc": nx.Graph()}
    compressor.file_metadata = {}

    sync_manager = MagicMock()
    sync_manager.file_metadata = {
        "doc": FileMetadata(
            file_path=str(source_file),
            mtime=0.0,
            checksum="x",
            ingestion_time=0.0,
            size_bytes=10,
            doc_id="doc",
        )
    }
    sync_manager.update_metadata = MagicMock()
    sync_manager.export_metadata = MagicMock(return_value={})

    version_manager = AsyncMock()
    version_manager.add_version_async = AsyncMock()
    version_manager.get_version_history_async = AsyncMock(return_value=[])

    persistence = MagicMock()
    persistence.save_document = MagicMock(return_value=True)
    persistence.save_file_sync_metadata = MagicMock(return_value=True)

    context = {
        "sync_manager": sync_manager,
        "version_manager": version_manager,
        "compressor": compressor,
        "persistence": persistence,
    }

    await handle_refresh_document(context, {"file_id": "doc"})

    assert persistence.save_document.called
    saved_chunks = persistence.save_document.call_args.kwargs["chunks"]
    assert "doc-large_n0" not in saved_chunks
    assert set(saved_chunks) == {"doc_n0"}


# =========================================================================
# src/code_compressor.py -- search_code file_id filter
# =========================================================================


class _FakeCodeModel:
    def encode(self, texts):
        return [np.ones(4) for _ in texts]


def _code_chunk(chunk_id: str, text: str = "def f(): pass"):
    from src.code_compressor import CodeChunk

    return CodeChunk(
        chunk_id=chunk_id,
        chunk_type="function",
        code=text,
        name="f",
        docstring=None,
        start_line=1,
        end_line=2,
        dependencies=[],
        embedding=np.ones(4),
        importance=0.0,
    )


def test_search_code_file_id_filter_excludes_prefix_sibling():
    from src.code_compressor import CodeSemanticCompressor

    compressor = object.__new__(CodeSemanticCompressor)
    compressor.model = _FakeCodeModel()
    compressor.chunks = {
        "doc.py::run": _code_chunk("doc.py::run"),
        "doc.py.bak::run": _code_chunk("doc.py.bak::run"),
    }

    results = compressor.search_code("run", file_id="doc.py", top_k=10)
    ids = [cid for cid, _ in results]

    assert "doc.py.bak::run" not in ids
    assert "doc.py::run" in ids


# =========================================================================
# src/scar_compressor.py -- search_with_alignment file_id filter
# =========================================================================


class _FakeScarModel:
    def encode(self, texts):
        return [np.array([1.0, 0.0, 0.0]) for _ in texts]


def _scar_node(text: str = "n"):
    node = SimpleNamespace()
    node.text = text
    node.embedding = np.array([1.0, 0.0, 0.0])
    return node


def test_search_with_alignment_file_id_filter_excludes_prefix_sibling():
    from src.scar_compressor import SCAREnhancedCompressor

    base_compressor = SimpleNamespace(
        model=_FakeScarModel(),
        chunks={
            "doc_n0": _scar_node("own"),
            "doc-large_n0": _scar_node("sibling"),
        },
    )

    scar = object.__new__(SCAREnhancedCompressor)
    scar.base_compressor = base_compressor
    scar.use_alignment_guidance = False
    scar.use_learnable_compression = False

    results = scar.search_with_alignment("query", file_id="doc", top_k=10, alignment_weight=0.0)
    ids = [nid for nid, _ in results]

    assert "doc-large_n0" not in ids
    assert "doc_n0" in ids
