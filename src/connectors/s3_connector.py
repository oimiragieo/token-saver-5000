"""S3 export connector normalization."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseConnector, ConnectorDocument, sanitize_segment

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


class S3Connector(BaseConnector):
    connector_type = "s3"
    description = "Normalize exported S3 objects under a bucket/prefix into documents."

    def validate_config(self, config: dict[str, Any]) -> None:
        bucket = config.get("bucket")
        objects = config.get("objects")
        if (
            not isinstance(bucket, str)
            or not _BUCKET_RE.match(bucket.strip())
            or ".." in bucket.strip()
        ):
            raise ValueError("s3 connector requires a valid 'bucket'")
        if not isinstance(objects, list) or not objects:
            raise ValueError("s3 connector requires a non-empty 'objects' list")
        for obj in objects:
            if not isinstance(obj, dict):
                raise ValueError("s3 connector objects must be objects")
            key = obj.get("key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("s3 connector objects require non-empty 'key'")

    def collect_documents(self, config: dict[str, Any]) -> list[ConnectorDocument]:
        self.validate_config(config)
        bucket = config["bucket"].strip()
        prefix = str(config.get("prefix") or "")
        documents: list[ConnectorDocument] = []
        for obj in config["objects"]:
            key = obj["key"].strip()
            content = obj.get("content") or f"Captured S3 object {bucket}/{key}"
            documents.append(
                ConnectorDocument(
                    source_id=f"s3://{bucket}/{key}",
                    file_id=f"s3-{sanitize_segment(bucket)}-{sanitize_segment(key)}",
                    text=str(content),
                    metadata={
                        "connector_type": self.connector_type,
                        "bucket": bucket,
                        "prefix": prefix,
                        "key": key,
                        "source": "s3",
                    },
                )
            )
        return documents
