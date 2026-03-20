"""Phase 1 tenancy tests for scoped document identity and isolation."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.handlers.compression_handlers import (
    handle_delete_document,
    handle_get_stats,
    handle_ingest,
    handle_list_documents,
    handle_read_skeleton,
    handle_search_semantic,
)
from src.identity_scope import compose_scoped_file_id, display_file_id, parse_scoped_file_id
from src.persistence import PersistenceManager
from src.semantic_compressor import SemanticNode


def test_scoped_identity_round_trip():
    scoped_file_id = compose_scoped_file_id(
        "shared_doc",
        workspace_id="workspace/acme",
        user_id="user@example.com",
        agent_id="agent:writer",
        session_id="session 123",
    )

    parsed = parse_scoped_file_id(scoped_file_id)

    assert display_file_id(scoped_file_id) == "shared_doc"
    assert parsed == {
        "file_id": "shared_doc",
        "workspace_id": "workspace/acme",
        "user_id": "user@example.com",
        "agent_id": "agent:writer",
        "session_id": "session 123",
    }


def test_persistence_scopes_same_file_id(tmp_path: Path):
    persistence = PersistenceManager(storage_dir=str(tmp_path))
    persistence.use_chromadb = False

    for workspace_id, token_count in (("acme", 11), ("globex", 22)):
        scoped_file_id = compose_scoped_file_id("shared_doc", workspace_id=workspace_id)
        node_id = f"{scoped_file_id}_0"
        chunks = {
            node_id: SemanticNode(
                node_id=node_id,
                text=f"Doc for {workspace_id}",
                embedding=np.array([0.1, 0.2, 0.3]),
                importance=0.7,
                metadata={"tokens": token_count},
            )
        }
        graph_data = {
            "nodes": [{"id": node_id}],
            "edges": [],
            "metadata": {"workspace_id": workspace_id},
        }
        assert persistence.save_document(
            "shared_doc",
            chunks,
            graph_data,
            {"workspace_id": workspace_id},
            workspace_id=workspace_id,
        )

    acme_doc = persistence.load_document("shared_doc", workspace_id="acme")
    globex_doc = persistence.load_document("shared_doc", workspace_id="globex")

    assert acme_doc is not None
    assert globex_doc is not None
    assert acme_doc["metadata"]["workspace_id"] == "acme"
    assert globex_doc["metadata"]["workspace_id"] == "globex"
    assert persistence.list_documents(workspace_id="acme") == ["shared_doc"]
    assert persistence.list_documents(workspace_id="globex") == ["shared_doc"]

    assert persistence.delete_document("shared_doc", workspace_id="acme")
    assert persistence.load_document("shared_doc", workspace_id="acme") is None
    assert persistence.load_document("shared_doc", workspace_id="globex") is not None


@pytest.mark.asyncio
async def test_handler_scope_isolates_same_file_id(characterization_context_factory):
    context = characterization_context_factory()

    alpha_text = (
        "Acme prompt caching guide keeps static tool definitions and system instructions first. "
        "The workspace alpha document focuses on cache-stable prefixes."
    )
    beta_text = (
        "Globex retrieval guide prioritizes timeline context and experiments. "
        "The workspace beta document focuses on evaluation datasets."
    )

    await handle_ingest(
        context,
        {"text": alpha_text, "file_id": "shared_doc", "workspace_id": "alpha"},
    )
    await handle_ingest(
        context,
        {"text": beta_text, "file_id": "shared_doc", "workspace_id": "beta"},
    )

    alpha_read = json.loads(
        await handle_read_skeleton(
            context,
            {"file_id": "shared_doc", "workspace_id": "alpha"},
        )
    )
    beta_read = json.loads(
        await handle_read_skeleton(
            context,
            {"file_id": "shared_doc", "workspace_id": "beta"},
        )
    )

    assert alpha_read["file_id"] == "shared_doc"
    assert beta_read["file_id"] == "shared_doc"
    assert "acme prompt caching guide" in alpha_read["skeleton_text"].lower()
    assert "globex retrieval guide" in beta_read["skeleton_text"].lower()

    alpha_search = json.loads(
        await handle_search_semantic(
            context,
            {"query": "cache stable prefixes", "workspace_id": "alpha", "top_k": 3},
        )
    )
    assert alpha_search["results"]
    assert all("beta" not in result["summary"].lower() for result in alpha_search["results"])

    alpha_list = await handle_list_documents(context, {"workspace_id": "alpha"})
    beta_list = await handle_list_documents(context, {"workspace_id": "beta"})

    assert "[shared_doc]" in alpha_list
    assert "[shared_doc]" in beta_list
    assert "Scope: workspace=alpha" in alpha_list
    assert "Scope: workspace=beta" in beta_list

    workspace_stats = await handle_get_stats(context, {"workspace_id": "alpha"})
    scoped_doc_stats = await handle_get_stats(
        context,
        {"file_id": "shared_doc", "workspace_id": "alpha"},
    )
    assert "shared_doc (workspace=alpha)" in workspace_stats
    assert "Scope: workspace=alpha" in scoped_doc_stats

    delete_message = await handle_delete_document(
        context,
        {"file_id": "shared_doc", "workspace_id": "alpha", "confirm": True},
    )
    assert "shared_doc" in delete_message

    with pytest.raises(ValueError):
        await handle_read_skeleton(
            context,
            {"file_id": "shared_doc", "workspace_id": "alpha"},
        )

    beta_read_after_delete = json.loads(
        await handle_read_skeleton(
            context,
            {"file_id": "shared_doc", "workspace_id": "beta"},
        )
    )
    assert "globex retrieval guide" in beta_read_after_delete["skeleton_text"].lower()
