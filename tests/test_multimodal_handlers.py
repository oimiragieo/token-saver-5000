import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.handlers.multimodal_handlers import handle_ingest_multimodal, handle_search_multimodal
from src.identity_scope import compose_scoped_file_id


class FakeMultimodalService:
    def __init__(self):
        self.ingest_calls = []
        self.search_calls = []

    def ingest(self, doc_id, content_items, manifest):
        self.ingest_calls.append((doc_id, content_items, manifest))
        return {
            "doc_id": doc_id,
            "content_types": ["image", "text"],
            "asset_count": len(content_items),
            "stats": {"total_nodes": len(content_items)},
            "manifest": manifest,
        }

    def search(self, doc_id, query, **kwargs):
        self.search_calls.append((doc_id, query, kwargs))
        return {
            "doc_id": doc_id,
            "total_results": 1,
            "results": [
                {
                    "node_id": f"{doc_id}_n0",
                    "score": 0.99,
                    "modality": "text",
                    "metadata": {"source": "text"},
                    "content_preview": "Architecture overview",
                }
            ],
        }


@pytest.mark.asyncio
async def test_ingest_multimodal_uses_scoped_doc_id_and_returns_visible_doc_id(tmp_path):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"image-bytes")
    service = FakeMultimodalService()
    context = {
        "path_validator": MagicMock(validate=MagicMock(return_value=str(image_path))),
        "resource_manager": MagicMock(
            check_multimodal_batch_async=AsyncMock(return_value=(True, None))
        ),
        "multimodal_ingestion": service,
    }

    result = json.loads(
        await handle_ingest_multimodal(
            context,
            {
                "doc_id": "release_brief",
                "workspace_id": "acme",
                "text_content": "Architecture overview",
                "image_paths": [str(image_path)],
                "image_captions": {str(image_path): "Deployment diagram"},
            },
        )
    )

    scoped_doc_id = compose_scoped_file_id("release_brief", workspace_id="acme")
    assert service.ingest_calls[0][0] == scoped_doc_id
    assert result["doc_id"] == "release_brief"
    assert result["project"]["doc_id"] == "release_brief"
    context["resource_manager"].check_multimodal_batch_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_multimodal_returns_visible_doc_id():
    service = FakeMultimodalService()
    context = {
        "path_validator": MagicMock(),
        "resource_manager": MagicMock(),
        "multimodal_ingestion": service,
    }

    result = json.loads(
        await handle_search_multimodal(
            context,
            {"doc_id": "release_brief", "workspace_id": "acme", "query": "architecture"},
        )
    )

    scoped_doc_id = compose_scoped_file_id("release_brief", workspace_id="acme")
    assert service.search_calls[0][0] == scoped_doc_id
    assert service.search_calls[0][1] == "architecture"
    assert result["doc_id"] == "release_brief"
    assert result["results"][0]["modality"] == "text"


@pytest.mark.asyncio
async def test_search_multimodal_requires_image_path_for_image_queries():
    with pytest.raises(ValueError, match="image_query_path"):
        await handle_search_multimodal(
            {
                "path_validator": MagicMock(),
                "resource_manager": MagicMock(),
                "multimodal_ingestion": FakeMultimodalService(),
            },
            {"doc_id": "release_brief", "query_type": "image"},
        )
