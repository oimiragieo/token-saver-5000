"""Tests for S3 connector normalization."""

import pytest

from src.connectors.s3_connector import S3Connector


def test_collects_s3_objects_into_documents():
    connector = S3Connector()
    documents = connector.collect_documents(
        {
            "bucket": "example-bucket",
            "prefix": "docs/",
            "objects": [{"key": "docs/readme.md", "content": "S3 readme"}],
        }
    )

    assert len(documents) == 1
    assert documents[0].metadata["bucket"] == "example-bucket"
    assert documents[0].source_id == "s3://example-bucket/docs/readme.md"


def test_rejects_bucket_name_with_consecutive_dots():
    """Consecutive dots are invalid per AWS S3 bucket-naming rules.

    Regression lock for backlog A5 — this validator's regex alone (matching
    path_validator.py's) accepts "ab..cd"; the explicit ".." check closes it.
    """
    connector = S3Connector()
    with pytest.raises(ValueError, match="valid 'bucket'"):
        connector.collect_documents(
            {
                "bucket": "ab..cd",
                "objects": [{"key": "docs/readme.md", "content": "S3 readme"}],
            }
        )
