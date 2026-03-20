from unittest.mock import MagicMock

import pytest

from src.multimodal_ingestion import build_multimodal_request, estimate_multimodal_payload_bytes


def test_build_multimodal_request_normalizes_supported_payloads(tmp_path):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"image-bytes")
    validator = MagicMock()
    validator.validate.return_value = str(image_path)

    content_items, manifest = build_multimodal_request(
        {
            "text_content": "Architecture overview",
            "code_content": "def deploy():\n    return True",
            "code_language": "python",
            "image_paths": [str(image_path)],
            "image_captions": {str(image_path): "Deployment diagram"},
            "image_ocr_text": {str(image_path): "Service A -> Service B"},
            "audio_items": [{"title": "Demo", "transcript": "Walkthrough transcript"}],
            "document_items": [
                {
                    "title": "Spec",
                    "text": "Document summary",
                    "image_paths": [str(image_path)],
                    "image_captions": {str(image_path): "Embedded figure"},
                }
            ],
        },
        path_validator=validator,
    )

    types = [item["type"] for item in content_items]
    assert types.count("text") >= 5
    assert "code" in types
    assert "image" in types
    assert any(entry.get("type") == "audio" for entry in manifest)
    assert any(entry.get("media_kind") == "image" for entry in manifest)


def test_build_multimodal_request_rejects_video_payloads():
    with pytest.raises(ValueError, match="not production-ready"):
        build_multimodal_request(
            {"video_items": [{"path": "demo.mp4"}]}, path_validator=MagicMock()
        )


def test_build_multimodal_request_requires_audio_transcripts():
    with pytest.raises(ValueError, match="non-empty transcript"):
        build_multimodal_request({"audio_items": [{"title": "Demo"}]}, path_validator=MagicMock())


def test_build_multimodal_request_requires_supported_content():
    with pytest.raises(ValueError, match="At least one supported multimodal payload is required"):
        build_multimodal_request({}, path_validator=MagicMock())


def test_estimate_multimodal_payload_bytes_counts_binary_and_text():
    total = estimate_multimodal_payload_bytes([{"content": "hello"}, {"content": b"world"}])
    assert total == len("hello".encode("utf-8")) + len(b"world")
