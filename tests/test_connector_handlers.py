"""Tests for managed connector feed handlers."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.connector_registry import ConnectorRegistry


@pytest.fixture
def connector_context():
    ConnectorRegistry.reset_singleton()
    compressor = Mock()
    compressor.graphs = {"scope__w=acme__f=docs-web%3Aweb-example.com-docs": object()}
    compressor.file_metadata = {}
    compressor.chunks = {}
    compressor.ingest_file_async = AsyncMock(
        return_value=SimpleNamespace(total_tokens=100, skeleton_tokens=20, compression_ratio=5.0)
    )
    return {
        "connector_registry": ConnectorRegistry.get_registry(),
        "compressor": compressor,
        "resource_manager": Mock(
            check_connector_batch_async=AsyncMock(return_value=(True, None)),
            check_document_size_async=AsyncMock(return_value=(True, None)),
            register_document_async=AsyncMock(),
        ),
        "sync_manager": Mock(register_file=Mock(), export_metadata=Mock(return_value={})),
        "version_manager": Mock(add_version_async=AsyncMock()),
        "persistence": Mock(
            save_document=Mock(return_value=True), save_file_sync_metadata=Mock(return_value=True)
        ),
        "retrieval_history": {},
    }


@pytest.mark.asyncio
async def test_create_list_get_and_sync_connector_feed(connector_context):
    from src.handlers.connector_handlers import (
        handle_create_connector_feed,
        handle_get_connector_feed,
        handle_list_connector_feeds,
        handle_list_connector_types,
        handle_sync_connector_feed,
    )

    connector_types = json.loads(await handle_list_connector_types(connector_context, {}))
    created = json.loads(
        await handle_create_connector_feed(
            connector_context,
            {
                "name": "docs-web",
                "connector_type": "web",
                "config": {"pages": [{"url": "https://example.com/docs", "content": "Docs body"}]},
            },
        )
    )
    listed = json.loads(await handle_list_connector_feeds(connector_context, {}))
    fetched = json.loads(await handle_get_connector_feed(connector_context, {"name": "docs-web"}))
    synced = json.loads(
        await handle_sync_connector_feed(
            connector_context, {"name": "docs-web", "workspace_id": "acme"}
        )
    )

    assert connector_types["status"] == "success"
    assert created["feed"]["name"] == "docs-web"
    assert listed["total_feeds"] == 1
    assert fetched["feed"]["connector_type"] == "web"
    assert synced["ingested_documents"] == 1
    assert synced["results"][0]["success"] is True
