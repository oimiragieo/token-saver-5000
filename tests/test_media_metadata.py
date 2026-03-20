import pytest

from src.media_metadata import detect_media_kind, inspect_media_path, summarize_payload_bytes


def test_detect_media_kind_recognizes_supported_types():
    assert detect_media_kind("diagram.png") == "image"
    assert detect_media_kind("call.wav") == "audio"
    assert detect_media_kind("notes.pdf") == "document"
    assert detect_media_kind("archive.zip") == "unknown"


def test_inspect_media_path_reports_file_metadata(tmp_path):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-png")

    result = inspect_media_path(str(image_path))

    assert result["media_kind"] == "image"
    assert result["name"] == "diagram.png"
    assert result["size_bytes"] == len(b"fake-png")


def test_inspect_media_path_requires_existing_file(tmp_path):
    missing = tmp_path / "missing.png"
    with pytest.raises(ValueError, match="does not exist"):
        inspect_media_path(str(missing))


def test_summarize_payload_bytes_counts_text_and_binary():
    total = summarize_payload_bytes(
        [
            {"content": "hello"},
            {"content": b"world"},
            {"content": 123},
        ]
    )
    assert total == len("hello".encode("utf-8")) + len(b"world")
