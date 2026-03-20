"""Tests for S3 connector normalization."""

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
