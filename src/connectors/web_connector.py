"""Web URL and sitemap style connector normalization."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .base import BaseConnector, ConnectorDocument, sanitize_segment


class WebConnector(BaseConnector):
    connector_type = "web"
    description = "Normalize web pages or sitemap exports into ingestible documents."

    def validate_config(self, config: dict[str, Any]) -> None:
        pages = config.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ValueError("web connector requires a non-empty 'pages' list")
        for page in pages:
            if not isinstance(page, dict):
                raise ValueError("web connector pages must be objects")
            url = page.get("url")
            if not isinstance(url, str) or not url.strip():
                raise ValueError("web connector page requires non-empty 'url'")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Invalid web URL: {url}")

    def collect_documents(self, config: dict[str, Any]) -> list[ConnectorDocument]:
        self.validate_config(config)
        documents: list[ConnectorDocument] = []
        for page in config["pages"]:
            url = page["url"].strip()
            parsed = urlparse(url)
            slug = sanitize_segment(f"{parsed.netloc}{parsed.path or '/'}")
            content = page.get("content") or f"Captured web page for {url}"
            documents.append(
                ConnectorDocument(
                    source_id=url,
                    file_id=f"web-{slug}",
                    text=str(content),
                    metadata={
                        "connector_type": self.connector_type,
                        "url": url,
                        "title": page.get("title"),
                        "source": "web",
                    },
                )
            )
        return documents
