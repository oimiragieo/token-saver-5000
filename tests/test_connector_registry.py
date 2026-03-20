"""Tests for connector registry and managed feeds."""

from src.connector_registry import ConnectorRegistry


def setup_function():
    ConnectorRegistry.reset_singleton()


def test_lists_supported_connector_types():
    registry = ConnectorRegistry.get_registry()
    connector_types = {item["connector_type"] for item in registry.list_connector_types()}

    assert {"web", "github", "s3", "slack_export"} <= connector_types


def test_create_and_update_feed_sync_state():
    registry = ConnectorRegistry.get_registry()
    feed = registry.create_feed(
        name="docs-web",
        connector_type="web",
        config={"pages": [{"url": "https://example.com/docs", "content": "Docs body"}]},
    )
    synced = registry.mark_synced("docs-web", ["scope__x"])

    assert feed["name"] == "docs-web"
    assert synced["last_sync_document_ids"] == ["scope__x"]
