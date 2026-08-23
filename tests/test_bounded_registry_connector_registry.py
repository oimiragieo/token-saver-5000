"""Regression lock: ConnectorRegistry._feeds is bounded via BoundedDict (A1)."""

import pytest

from src.connector_registry import ConnectorRegistry


def test_connector_registry_evicts_oldest_feed_past_max_feeds():
    registry = ConnectorRegistry(max_feeds=2)

    registry.create_feed(
        name="feed-1",
        connector_type="web",
        config={"pages": [{"url": "https://example.com/1", "content": "one"}]},
    )
    registry.create_feed(
        name="feed-2",
        connector_type="web",
        config={"pages": [{"url": "https://example.com/2", "content": "two"}]},
    )
    registry.create_feed(
        name="feed-3",
        connector_type="web",
        config={"pages": [{"url": "https://example.com/3", "content": "three"}]},
    )

    with pytest.raises(ValueError, match="Unknown connector feed 'feed-1'"):
        registry.get_feed("feed-1")

    assert registry.get_feed("feed-2")["name"] == "feed-2"
    assert registry.get_feed("feed-3")["name"] == "feed-3"
