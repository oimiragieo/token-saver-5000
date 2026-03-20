"""Tests for web connector normalization."""

from src.connectors.web_connector import WebConnector


def test_collects_pages_into_documents():
    connector = WebConnector()
    documents = connector.collect_documents(
        {
            "pages": [
                {
                    "url": "https://example.com/docs/cache",
                    "title": "Caching",
                    "content": "Stable prefixes improve cache hits.",
                }
            ]
        }
    )

    assert len(documents) == 1
    assert documents[0].source_id == "https://example.com/docs/cache"
    assert "cache" in documents[0].file_id
